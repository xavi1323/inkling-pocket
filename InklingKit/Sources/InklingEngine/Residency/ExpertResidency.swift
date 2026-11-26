import Foundation
import InklingCore
#if canImport(Darwin)
import Darwin
#endif

public struct ExpertKey: Hashable, Sendable, Comparable, CustomStringConvertible {
    public var layer: Int
    public var expert: Int
    public init(layer: Int, expert: Int) { self.layer = layer; self.expert = expert }
    public static func < (a: ExpertKey, b: ExpertKey) -> Bool { (a.layer, a.expert) < (b.layer, b.expert) }
    public var description: String { "L\(layer)E\(expert)" }
}

/// Brings experts' weight pages into RAM ahead of use.
public protocol ExpertPager: Sendable {
    func willNeed(_ keys: [ExpertKey])
}

/// Hot-expert residency (E5, E10): counts routing per (layer, expert), keeps the hottest experts' pages warm within a
/// byte budget, prefetches routed ones, and halves the pinned set on a memory warning. Pinning only advises and touches
/// clean file-backed pages; iOS can still drop them, and nothing is ever written.
public final class ExpertResidency: @unchecked Sendable {
    private let lock = NSLock()
    private let expertBytes: (ExpertKey) -> Int64
    private let pager: ExpertPager?
    private var counts: [ExpertKey: Int] = [:]
    private var pinnedSet: Set<ExpertKey> = []
    private var hits = 0, lookups = 0
    public private(set) var budget: Int64

    public init(budget: Int64, expertBytes: @escaping (ExpertKey) -> Int64, pager: ExpertPager?) {
        self.budget = budget
        self.expertBytes = expertBytes
        self.pager = pager
    }

    /// Residency over a model's mapped weights; `budget` is for expert pages only.
    public convenience init(weights: MappedSafetensors, config: InklingTextConfig, budget: Int64) {
        let sizes = Self.expertSizes(weights: weights, config: config)
        self.init(budget: budget, expertBytes: { sizes[$0.layer] },
                  pager: MappedExpertPager(weights: weights, numExperts: config.numExperts))
    }

    /// Bytes of one expert (gate/up/down weight, scales, biases) per layer.
    public static func expertSizes(weights: MappedSafetensors, config: InklingTextConfig) -> [Int64] {
        (0..<config.numHiddenLayers).map { layer in
            MappedExpertPager.tensorNames(layer: layer).compactMap { weights.entry($0) }
                .map { ($0.end - $0.start) / Int64(config.numExperts) }.reduce(0, +)
        }
    }

    public func record(layer: Int, experts: [Int]) {
        lock.withLock {
            for e in experts {
                let key = ExpertKey(layer: layer, expert: e)
                counts[key, default: 0] += 1
                lookups += 1
                if pinnedSet.contains(key) { hits += 1 }
            }
        }
    }

    /// Re-picks the hottest experts that fit the budget and pages in the newly pinned ones.
    public func rebalance() {
        let added: [ExpertKey] = lock.withLock {
            let ranked = counts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            var chosen: Set<ExpertKey> = []
            var used: Int64 = 0
            for (key, _) in ranked {
                let size = expertBytes(key)
                if used + size > budget { continue }
                used += size
                chosen.insert(key)
            }
            let added = chosen.subtracting(pinnedSet).sorted()
            pinnedSet = chosen
            return added
        }
        if !added.isEmpty { pager?.willNeed(added) }
    }

    /// Pages in routed experts that are not pinned (they are likely needed again on the next token).
    public func prefetch(layer: Int, experts: [Int]) {
        let keys = lock.withLock { experts.map { ExpertKey(layer: layer, expert: $0) }.filter { !pinnedSet.contains($0) } }
        if !keys.isEmpty { pager?.willNeed(keys) }
    }

    /// Memory warning: keep only the hottest half (by bytes) and lower the budget to match.
    public func handleMemoryWarning() {
        lock.withLock {
            budget = pinnedBytesLocked / 2
        }
        rebalance()
    }

    public func resetHitRate() { lock.withLock { hits = 0; lookups = 0 } }

    public var hitRate: Double { lock.withLock { lookups == 0 ? 0 : Double(hits) / Double(lookups) } }
    public var pinned: [ExpertKey] { lock.withLock { pinnedSet.sorted() } }
    public var pinnedBytes: Int64 { lock.withLock { pinnedBytesLocked } }
    private var pinnedBytesLocked: Int64 { pinnedSet.reduce(0) { $0 + expertBytes($1) } }
}

/// `madvise(WILLNEED)` plus a read of one byte per page, on a background queue.
final class MappedExpertPager: ExpertPager, @unchecked Sendable {
    private let weights: MappedSafetensors
    private let numExperts: Int
    private let queue = DispatchQueue(label: "inkling.residency", qos: .utility)

    init(weights: MappedSafetensors, numExperts: Int) {
        self.weights = weights
        self.numExperts = numExperts
    }

    static func tensorNames(layer: Int) -> [String] {
        let base = "language_model.model.layers.\(layer).mlp.switch_mlp."
        return ["gate_proj", "up_proj", "down_proj"].flatMap { p in ["weight", "scales", "biases"].map { "\(base)\(p).\($0)" } }
    }

    func willNeed(_ keys: [ExpertKey]) {
        queue.async { [weights, numExperts] in
            let page = Int(getpagesize())
            for key in keys {
                for name in Self.tensorNames(layer: key.layer) {
                    guard let entry = weights.entry(name), let base = try? weights.pointer(name) else { continue }
                    let per = Int(entry.end - entry.start) / numExperts
                    let start = UInt(bitPattern: base) + UInt(per * key.expert)
                    let aligned = start & ~UInt(page - 1)
                    let length = Int(start - aligned) + per
                    guard let p = UnsafeMutableRawPointer(bitPattern: aligned) else { continue }
                    _ = madvise(p, length, MADV_WILLNEED)
                    var sum: UInt8 = 0
                    var offset = 0
                    while offset < length {
                        sum &+= p.load(fromByteOffset: offset, as: UInt8.self)
                        offset += page
                    }
                    withExtendedLifetime(sum) {}
                }
            }
        }
    }
}
