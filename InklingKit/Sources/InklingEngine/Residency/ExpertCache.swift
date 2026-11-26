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
final class ExpertCache {
    static let projections = ["gate_proj", "up_proj", "down_proj"]
    static let parts = [".weight", ".scales", ".biases"]

    private struct Part: @unchecked Sendable {  // raw pointers into the mapping and the slots; read-only here
        let source: UnsafeRawPointer
        let stride: Int
        let destination: UnsafeMutableRawPointer
        /// Shard file and the tensor's offset in it: one `pread` per expert instead of a page fault per 16 KB.
        let fd: Int32
        let offset: Int64
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
                                  fd: files[url]!, offset: offset))
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

    /// Puts `experts` (at most `slots` distinct ids) into slots, copying the missing ones; returns expert → slot.
    func ensure(_ experts: [Int]) -> [Int: Int32] {
        precondition(Set(experts).count <= slots, "more experts than slots")
        clock += 1
        if clock % Self.agingPeriod == 0 { uses = uses.map { $0 / 2 } }
        for e in Set(experts) { uses[e] += 1 }
        var map: [Int: Int32] = [:]
        var missing: [Int] = []
        for e in experts where map[e] == nil {
            if let s = slotOf[e] {
                lastUse[s] = clock
                map[e] = Int32(s)
                hits += 1
            } else if !missing.contains(e) {
                missing.append(e)
            }
        }
        guard !missing.isEmpty else { return map }
        let free = (0..<slots).filter { lastUse[$0] != clock }.sorted { a, b in
            let ua = expertIn[a].map { uses[$0] } ?? -1, ub = expertIn[b].map { uses[$0] } ?? -1
            return ua != ub ? ua < ub : lastUse[a] < lastUse[b]
        }
        var jobs: [(expert: Int, slot: Int)] = []
        for (e, s) in zip(missing, free) {
            if let old = expertIn[s] { slotOf[old] = nil }
            expertIn[s] = e
            slotOf[e] = s
            lastUse[s] = clock
            map[e] = Int32(s)
            jobs.append((e, s))
        }
        misses += jobs.count
        let parts = self.parts, copies = jobs
        DispatchQueue.concurrentPerform(iterations: jobs.count * parts.count) { i in
            let job = copies[i / parts.count], part = parts[i % parts.count]
            Self.load(part, expert: job.expert, slot: job.slot)
        }
        return map
    }
}
