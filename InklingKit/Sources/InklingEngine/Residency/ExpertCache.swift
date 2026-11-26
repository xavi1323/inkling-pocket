import Foundation
import MLX
import InklingCore

/// One MoE layer's experts in a fixed number of slots of ordinary memory, read in from the shard files when a token
/// needs them. The slot reused is the one whose expert is used least often (counts halve every `agingPeriod` calls),
/// oldest first among equals.
///
/// Why: MLX puts every buffer it wraps into a Metal residency set, which pins (wires) it. Handing MLX the mapped
/// 9.7 GB of experts pinned nearly all of them and iOS killed the app. The GPU only ever sees these slots.
///
/// Callers must make sure no queued GPU work still reads a slot before `ensure` refills it (SparseMoE evaluates the
/// routing first, which forces every earlier step).
/// `@unchecked Sendable`: all slot state changes run on `queue`.
final class ExpertCache: @unchecked Sendable {
    static let projections = ["gate_proj", "up_proj", "down_proj"]
    static let parts = [".weight", ".scales", ".biases"]

    private struct Part: @unchecked Sendable {  // raw pointers into the mapping and the slots; read-only here
        let source: UnsafeRawPointer
        let stride: Int
        let destination: UnsafeMutableRawPointer
        /// Shard file and the tensor's offset in it: one `pread` per expert instead of a page fault per 16 KB.
        let fd: Int32
        let offset: Int64
        let url: URL
    }
    static let agingPeriod: UInt64 = 2048

    let slots: Int
    let experts: Int
    /// Keeps the mapping (the copy sources) alive.
    private let weights: MappedSafetensors
    let gate, up, down: QLinear
    /// Bytes held by the slots.
    let bytes: Int64
    private var parts: [Part] = []
    private var slotOf: [Int: Int] = [:]
    private var expertIn: [Int?]
    private var lastUse: [UInt64]
    private var uses: [Int]
    private var files: [URL: Int32] = [:]
    private var clock: UInt64 = 0
    private(set) var hits = 0
    private(set) var misses = 0
    /// Experts read ahead by `prefetch`.
    private(set) var prefetched = 0
    /// Read-ahead experts not used yet (a later hit on one counts as a useful read-ahead).
    private var unusedPrefetch: [Int: Int] = [:]  // expert → its rank in the guess
    /// Shared by the model's caches; nil = not measured.
    var profile: StreamProfile?
    /// Tests: seconds each read waits first (taken when `ensure`/`prefetch` is called).
    var readDelay: TimeInterval = 0
    /// Serialises slot changes: `ensure` (generation thread) and `prefetch` (background).
    private let queue = DispatchQueue(label: "inkling.expert-cache", qos: .userInitiated)

    /// Per-expert bytes of one layer's three projections (weights, scales, biases).
    static func bytesPerExpert(path: String, weights: MappedSafetensors, prefix: String, experts: Int) -> Int64 {
        projections.flatMap { p in parts.compactMap { weights.entry(prefix + path + ".switch_mlp." + p + $0) } }
            .map { $0.length / Int64(experts) }.reduce(0, +)
    }

    init(path: String, weights: MappedSafetensors, quant: QuantizationMap, prefix: String, slots: Int) throws {
        var built: [QLinear] = []
        var parts: [Part] = []
        var bytes: Int64 = 0
        var experts = 0
        var files: [URL: Int32] = [:]
        for projection in Self.projections {
            let module = path + ".switch_mlp." + projection
            var arrays: [MLXArray?] = []
            for suffix in Self.parts {
                let name = prefix + module + suffix
                guard let entry = weights.entry(name) else { arrays.append(nil); continue }
                let dtype = try MappedSafetensors.dtype(entry.dtype)
                experts = entry.shape[0]
                let stride = Int(entry.length) / experts
                let array = MLXArray.zeros([slots] + entry.shape.dropFirst(), dtype: dtype)
                eval(array)
                let data = array.asData(access: .noCopy)
                guard data.strides == Self.contiguousStrides(array.shape), data.data.count == slots * stride else {
                    throw ModelLoadError.unsupported("\(name): expert slots are not contiguous")
                }
                let destination = data.data.withUnsafeBytes { UnsafeMutableRawPointer(mutating: $0.baseAddress!) }
                let (url, offset) = try weights.location(name)
                if files[url] == nil {
                    let fd = open(url.path, O_RDONLY)
                    #if canImport(Darwin)
                    if fd >= 0 { _ = fcntl(fd, F_RDAHEAD, 0) }  // reads are whole experts; no read-ahead into the next
                    #endif
                    files[url] = fd
                }
                parts.append(Part(source: try weights.pointer(name), stride: stride, destination: destination,
                                  fd: files[url]!, offset: offset, url: url))
                bytes += Int64(slots * stride)
                arrays.append(array)
            }
            guard let weight = arrays[0] else { throw ModelLoadError.missingWeight(prefix + module + ".weight") }
            guard let scales = arrays[1] else {
                built.append(QLinear(weight: weight, scales: nil, biases: nil, groupSize: 0, bits: 0))
                continue
            }
            guard let spec = quant.spec(for: module) else { throw ModelLoadError.badQuantization("\(module): quantized but not in config") }
            guard spec.mode == "affine" else { throw ModelLoadError.unsupported("\(module): quantization mode \(spec.mode)") }
            built.append(QLinear(weight: weight, scales: scales, biases: arrays[2], groupSize: spec.groupSize, bits: spec.bits))
        }
        self.slots = slots
        self.experts = experts
        self.weights = weights
        (gate, up, down) = (built[0], built[1], built[2])
        self.parts = parts
        self.bytes = bytes
        self.files = files
        expertIn = Array(repeating: nil, count: slots)
        lastUse = Array(repeating: 0, count: slots)
        uses = Array(repeating: 0, count: experts)
    }

    deinit {
        for fd in files.values where fd >= 0 { close(fd) }
    }

    /// Benchmark: experts of this layer not in a slot (a random sample of `count`).
    func notResident(_ count: Int) -> [Int] {
        queue.sync { Array((0..<experts).filter { slotOf[$0] == nil }.shuffled().prefix(count)) }
    }

    /// Benchmark: `count` reads of `size` contiguous bytes each from the largest expert tensor (random expert-aligned
    /// starts), `inFlight` at a time. What one read per expert would give if each expert were stored in one piece.
    /// Returns (bytes, seconds).
    func probeContiguous(count: Int, size: Int, inFlight: Int) -> (Int, Double) {
        guard let part = parts.max(by: { $0.stride < $1.stride }) else { return (0, 0) }
        let fd = open(part.url.path, O_RDONLY)
        guard fd >= 0 else { return (0, 0) }
        defer { close(fd) }
        #if canImport(Darwin)
        _ = fcntl(fd, F_RDAHEAD, 0)
        #endif
        let span = min(size, part.stride * experts)
        let lastStart = max(experts - (span + part.stride - 1) / part.stride, 0)
        let starts = (0..<count).map { _ in part.offset + Int64(Int.random(in: 0...lastStart) * part.stride) }
        var buffer: UnsafeMutableRawPointer?
        guard posix_memalign(&buffer, 16384, max(count * span, 1)) == 0, let buffer else { return (0, 0) }
        defer { free(buffer) }
        let base = UInt(bitPattern: buffer)
        let gate = DispatchSemaphore(value: inFlight), group = DispatchGroup()
        let queue = DispatchQueue(label: "inkling.probe", qos: .userInitiated, attributes: .concurrent)
        let start = StreamProfile.now()
        for (i, offset) in starts.enumerated() {
            gate.wait()
            group.enter()
            queue.async {
                let target = UnsafeMutableRawPointer(bitPattern: base + UInt(i * span))!
                var done = 0
                while done < span {
                    let n = pread(fd, target + done, span - done, off_t(offset) + off_t(done))
                    if n <= 0 { break }
                    done += n
                }
                gate.signal()
                group.leave()
            }
        }
        group.wait()
        return (count * span, Double(StreamProfile.now() - start) / 1e9)
    }

    /// Benchmark: reads `experts` into a scratch buffer the way slots are filled (all parts at once), through new
    /// file handles; `direct` bypasses the OS file cache, `chunk` splits each part into reads of that many bytes.
    /// Returns (bytes, seconds).
    func probeRead(_ experts: [Int], direct: Bool, chunk: Int?) -> (Int, Double) {
        var fds: [URL: Int32] = [:]
        for part in parts where fds[part.url] == nil {
            let fd = open(part.url.path, O_RDONLY)
            #if canImport(Darwin)
            if fd >= 0 {
                _ = fcntl(fd, F_RDAHEAD, 0)
                if direct { _ = fcntl(fd, F_NOCACHE, 1) }
            }
            #endif
            fds[part.url] = fd
        }
        defer { for fd in fds.values where fd >= 0 { close(fd) } }
        let perExpert = parts.map(\.stride).reduce(0, +)
        var buffer: UnsafeMutableRawPointer?
        guard posix_memalign(&buffer, 16384, max(experts.count * perExpert, 1)) == 0, let buffer else { return (0, 0) }
        defer { free(buffer) }
        // (fd, file offset, destination offset, length) for every read.
        var reads: [(Int32, Int64, Int, Int)] = []
        var destination = 0
        for e in experts {
            for part in parts {
                let fd = fds[part.url]!, start = part.offset + Int64(e * part.stride)
                let step = chunk ?? part.stride
                for o in stride(from: 0, to: part.stride, by: step) {
                    reads.append((fd, start + Int64(o), destination + o, min(step, part.stride - o)))
                }
                destination += part.stride
            }
        }
        let jobs = reads, base = UInt(bitPattern: buffer)
        let start = StreamProfile.now()
        DispatchQueue.concurrentPerform(iterations: jobs.count) { i in
            let (fd, offset, to, length) = jobs[i]
            let target = UnsafeMutableRawPointer(bitPattern: base + UInt(to))!
            var done = 0
            while done < length {
                let n = pread(fd, target + done, length - done, off_t(offset) + off_t(done))
                if n <= 0 { break }
                done += n
            }
        }
        return (destination, Double(StreamProfile.now() - start) / 1e9)
    }

    /// Reads one expert's bytes of `part` into `slot` (falls back to copying from the mapping).
    private static func load(_ part: Part, expert: Int, slot: Int) {
        let destination = part.destination + slot * part.stride
        var done = 0
        if part.fd >= 0 {
            while done < part.stride {
                let n = pread(part.fd, destination + done, part.stride - done, off_t(part.offset) + off_t(expert * part.stride + done))
                if n <= 0 { break }
                done += n
            }
        }
        if done < part.stride {
            memcpy(destination + done, part.source + expert * part.stride + done, part.stride - done)
        }
    }

    static func contiguousStrides(_ shape: [Int]) -> [Int] {
        var strides = Array(repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) { strides[i] = strides[i + 1] * shape[i + 1] }
        return strides
    }

    func slot(of expert: Int) -> Int? { slotOf[expert] }

    /// Puts `experts` (at most `slots` distinct ids) into slots, reading the missing ones; returns expert → slot.
    /// Waits only for read-aheads of experts it needs (or, when read-aheads hold every free slot, for them all).
    func ensure(_ experts: [Int]) -> [Int: Int32] {
        precondition(Set(experts).count <= slots, "more experts than slots")
        guard let profile else { return ensureUnmeasured(experts) }
        return profile.time(\.expertWait) { ensureUnmeasured(experts) }
    }

    private func ensureUnmeasured(_ experts: [Int]) -> [Int: Int32] {
        let group = DispatchGroup()
        group.enter()
        var reservation: Reservation?
        while reservation == nil {
            reservation = queue.sync { reserve(experts, counting: true, group: group) }
            if reservation == nil { for g in queue.sync(execute: { Array(loading.values) }) { g.wait() } }
        }
        let r = reservation!
        for g in r.evicted { g.wait() }  // read-aheads still writing into slots we took
        read(r.jobs, counting: true, delay: readDelay)
        queue.sync { finish(r.jobs, group: group) }
        group.leave()
        for g in r.waits { g.wait() }
        return r.map
    }

    /// `experts` reordered so those already in a slot come first: running them in slot-sized groups in this order
    /// reads each missing expert once and never evicts one a later group needs.
    func residentFirst(_ experts: [Int]) -> [Int] {
        queue.sync { experts.filter { slotOf[$0] != nil } + experts.filter { slotOf[$0] == nil } }
    }

    /// Reads experts in the background (a guess at what the next step needs). Only call while no queued GPU work
    /// reads this layer's slots. Skipped when there are not enough free slots.
    /// `experts` are ranked best guess first; ranks that rarely pay off are skipped (`StreamProfile.shouldPrefetch`).
    /// Reads run at lower I/O priority than the reads a step is waiting on.
    func prefetch(_ experts: [Int]) {
        guard Set(experts).count <= slots else { return }
        var ranks: [Int: Int] = [:]
        for (rank, e) in experts.enumerated() where ranks[e] == nil && (profile?.shouldPrefetch(rank: rank) ?? true) { ranks[e] = rank }
        guard !ranks.isEmpty else { return }
        let ranked = ranks, wanted = experts.filter { ranks[$0] != nil }
        let delay = readDelay, group = DispatchGroup()
        group.enter()
        queue.async { [self] in
            guard let r = reserve(wanted, counting: false, group: group, ranks: ranked), !r.jobs.isEmpty else { group.leave(); return }
            DispatchQueue.global(qos: .utility).async { [self] in
                read(r.jobs, counting: false, delay: delay)
                queue.sync { finish(r.jobs, group: group) }
                group.leave()
            }
        }
    }

    private struct Reservation {
        var map: [Int: Int32]
        var jobs: [(expert: Int, slot: Int)]
        /// Read-aheads still filling slots in `map`.
        var waits: [DispatchGroup]
        /// Read-aheads still filling slots taken for `jobs` (wait before reading into them).
        var evicted: [DispatchGroup] = []
    }

    /// Slots being read, and the read that fills each (a slot being read is never evicted).
    private var loading: [Int: DispatchGroup] = [:]

    /// On `queue`: maps `experts` to slots, claiming the least-used free slots for missing ones (marked loading
    /// under `group`). A step (`counting`) may take a slot a read-ahead is still filling (it waits for that read
    /// first); a read-ahead never does. Nil, changing nothing, when too few slots are free.
    private func reserve(_ experts: [Int], counting: Bool, group: DispatchGroup, ranks: [Int: Int] = [:]) -> Reservation? {
        var map: [Int: Int32] = [:]
        var missing: [Int] = []
        for e in experts where map[e] == nil {
            if let s = slotOf[e] { map[e] = Int32(s) } else if !missing.contains(e) { missing.append(e) }
        }
        let held = Set(map.values.map(Int.init))
        let free = missing.isEmpty ? [] : (0..<slots).filter { !held.contains($0) && (counting || loading[$0] == nil) }.sorted { a, b in
            let ua = expertIn[a].map { uses[$0] } ?? -1, ub = expertIn[b].map { uses[$0] } ?? -1
            return ua != ub ? ua < ub : lastUse[a] < lastUse[b]
        }
        guard free.count >= missing.count else { return nil }
        clock += 1
        if counting {
            if clock % Self.agingPeriod == 0 { uses = uses.map { $0 / 2 } }
            for e in Set(experts) { uses[e] += 1 }
            hits += map.count
            let ranks = map.keys.compactMap { unusedPrefetch.removeValue(forKey: $0) }
            if !ranks.isEmpty {
                profile?.update {
                    $0.prefetchUsed += ranks.count
                    for r in ranks where r < StreamProfile.ranks { $0.usedByRank[r] += 1 }
                }
            }
        }
        var waits: [DispatchGroup] = []
        for s in map.values.map(Int.init) {
            lastUse[s] = clock
            if let g = loading[s] { waits.append(g) }
        }
        var jobs: [(expert: Int, slot: Int)] = []
        var evicted: [DispatchGroup] = []
        for (e, s) in zip(missing, free) {
            if let g = loading[s] { evicted.append(g) }
            if let old = expertIn[s] {
                slotOf[old] = nil
                unusedPrefetch.removeValue(forKey: old)
            }
            if !counting {
                let rank = ranks[e] ?? experts.firstIndex(of: e) ?? 0
                unusedPrefetch[e] = rank
                if rank < StreamProfile.ranks { profile?.update { $0.prefetchedByRank[rank] += 1 } }
            }
            expertIn[s] = e
            slotOf[e] = s
            lastUse[s] = clock
            loading[s] = group
            map[e] = Int32(s)
            jobs.append((e, s))
        }
        if counting { misses += jobs.count } else { prefetched += jobs.count }
        return Reservation(map: map, jobs: jobs, waits: waits, evicted: evicted)
    }

    /// On `queue`: the reads of `jobs` (under `group`) are done; slots since taken by another read stay marked.
    private func finish(_ jobs: [(expert: Int, slot: Int)], group: DispatchGroup) {
        for j in jobs where loading[j.slot] === group { loading[j.slot] = nil }
    }

    /// Off `queue`: copies the experts of `jobs` into their slots (all parts in parallel).
    private func read(_ jobs: [(expert: Int, slot: Int)], counting: Bool, delay: TimeInterval) {
        guard !jobs.isEmpty else { return }
        if delay > 0 { Thread.sleep(forTimeInterval: delay) }
        let parts = self.parts
        let start = StreamProfile.now()
        DispatchQueue.concurrentPerform(iterations: jobs.count * parts.count) { i in
            let job = jobs[i / parts.count], part = parts[i % parts.count]
            Self.load(part, expert: job.expert, slot: job.slot)
        }
        if let profile {
            let elapsed = StreamProfile.now() - start, bytes = Int64(jobs.count * parts.map(\.stride).reduce(0, +))
            profile.update {
                if counting { $0.misses += jobs.count } else { $0.prefetched += jobs.count }
                $0.readBytes += bytes
                $0.readTime += elapsed
            }
        }
    }
}
