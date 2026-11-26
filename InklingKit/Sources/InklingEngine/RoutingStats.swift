import Foundation
import MLX

/// Keeps routed-expert id arrays from a forward pass without reading them back; drain after the pass is evaluated.
public final class RouteCollector {
    private var current: [(visit: Int, experts: MLXArray)] = []
    private var passes: [[(visit: Int, experts: MLXArray)]] = []
    public init() {}
    func add(visit: Int, experts: MLXArray) { current.append((visit, experts)) }
    /// Closes one forward pass.
    func endPass() {
        if !current.isEmpty { passes.append(current) }
        current = []
    }
    /// (visit, distinct expert ids) of every finished pass except the newest `keepingLatest` (still running on the GPU).
    public func drain(keepingLatest: Int = 0) -> [(visit: Int, experts: [Int])] {
        let count = max(0, passes.count - keepingLatest)
        let done = passes.prefix(count).flatMap { $0 }
        passes.removeFirst(count)
        return done.map { ($0.visit, Array(Set($0.experts.asType(.int32).reshaped([-1]).asArray(Int32.self).map(Int.init))).sorted()) }
    }
}

/// Which experts each layer visit picked (Benchmark; residency in Task 19 uses the same counts).
public final class RoutingStats: @unchecked Sendable {
    public let visits: [Int]
    public let numExperts: Int
    private var counts: [[Int]]
    private let lock = NSLock()

    public init(visits: [Int], numExperts: Int) {
        self.visits = visits
        self.numExperts = numExperts
        counts = Array(repeating: Array(repeating: 0, count: numExperts), count: visits.count)
    }

    /// `experts`: routed ids, any shape (e.g. [B, L, k]). Reads them back to the CPU.
    public func record(visit: Int, experts: MLXArray) {
        let ids = experts.asType(.int32).reshaped([-1]).asArray(Int32.self)
        lock.withLock { for id in ids { counts[visit][Int(id)] += 1 } }
    }

    public func hitHistogram(visit: Int) -> [Int] { lock.withLock { counts[visit] } }

    /// Share of experts reused across a looped layer's passes: |used in every pass| / |used in any pass|.
    public func passOverlap(layer: Int) -> Double {
        let sets = visits.indices.filter { visits[$0] == layer }.map { v in
            Set(hitHistogram(visit: v).enumerated().filter { $0.element > 0 }.map(\.offset))
        }
        guard let first = sets.first else { return 0 }
        let any = sets.reduce(Set<Int>()) { $0.union($1) }
        guard !any.isEmpty else { return 0 }
        return Double(sets.reduce(first) { $0.intersection($1) }.count) / Double(any.count)
    }

    public func reset() {
        lock.withLock { counts = Array(repeating: Array(repeating: 0, count: numExperts), count: visits.count) }
    }

    /// One line per visit: layer, distinct experts used, the five busiest.
    public var summary: String {
        visits.indices.map { v in
            let h = hitHistogram(visit: v)
            let top = h.enumerated().sorted { $0.element > $1.element }.prefix(5).filter { $0.element > 0 }
                .map { "\($0.offset)×\($0.element)" }.joined(separator: " ")
            return "visit \(v) (layer \(visits[v])): \(h.filter { $0 > 0 }.count) experts; top \(top)"
        }.joined(separator: "\n")
    }

    /// Pass overlap for every looped layer.
    public var overlapSummary: String {
        let looped = Set(visits.filter { l in visits.filter { $0 == l }.count > 1 }).sorted()
        return looped.map { String(format: "layer %d: %.0f%%", $0, passOverlap(layer: $0) * 100) }.joined(separator: ", ")
    }
}
