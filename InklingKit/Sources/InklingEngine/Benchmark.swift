import Foundation
import MLX
import InklingCore
import AgentCore
#if canImport(Darwin)
@preconcurrency import Darwin
#endif
#if os(iOS)
import os
#endif

/// Speed and memory numbers for one run (B3, B4).
public struct BenchmarkReport: Sendable, Equatable {
    public var enginePath = EnginePath.reference
    public var promptTokens = 0
    public var generatedTokens = 0
    public var prefillTokensPerSecond = 0.0
    public var decodeTokensPerSecond = 0.0
    public var timeToFirstToken: Duration = .zero
    public var availableMemoryBytes: Int64?
    public var footprintBytes: Int64?
    public var peakFootprintBytes: Int64?
    public var mappedWeightBytes: Int64 = 0
    public var expertHitRate: Double?
    /// Expert slots per layer (streaming), nil when the whole expert tensors are used.
    public var expertSlots: [Int]?
    public var thermalState = "unknown"
    public var routing = ""
    public var passOverlap = ""
    /// Where streaming time went (empty when not streaming).
    public var prefillProfile = ""
    public var decodeProfile = ""
    /// Decode speed with no routing waits or expert reads (streaming only).
    public var ceiling = ""
    public init() {}

    public var asText: String {
        func gb(_ b: Int64?) -> String { b.map { String(format: "%.2f GB", Double($0) / 1e9) } ?? "n/a" }
        let ttft = Double(timeToFirstToken.components.seconds) + Double(timeToFirstToken.components.attoseconds) / 1e18
        return [
            "Inkling Pocket benchmark (\(enginePath.rawValue) path)",
            String(format: "Prefill: %.1f tok/s (%d tokens)", prefillTokensPerSecond, promptTokens),
            String(format: "Decode: %.1f tok/s (%d tokens)", decodeTokensPerSecond, generatedTokens),
            String(format: "TTFT: %.2f s", ttft),
            "Memory limit (available): \(gb(availableMemoryBytes))",
            "Footprint now: \(gb(footprintBytes))",
            "Peak footprint: \(gb(peakFootprintBytes))",
            "Mapped weights: \(gb(mappedWeightBytes))",
            "Expert hit rate: \(expertHitRate.map { String(format: "%.1f%%", $0 * 100) } ?? "n/a (not streaming)")",
            "Expert slots per layer: \(expertSlots.map { "\($0.min() ?? 0)–\($0.max() ?? 0) (\($0.reduce(0, +)) total)" } ?? "all (not streaming)")",
            "Thermal: \(thermalState)",
            "Prefill time (whole prompt): \(prefillProfile.isEmpty ? "n/a" : prefillProfile)",
            "Decode time per token: \(decodeProfile.isEmpty ? "n/a" : decodeProfile)",
            "Decode ceiling (no routing waits or reads): \(ceiling.isEmpty ? "n/a" : ceiling)",
            "Pass overlap: \(passOverlap.isEmpty ? "n/a" : passOverlap)",
            "Routing:",
            routing,
        ].joined(separator: "\n")
    }
}

public enum MemoryInfo {
    /// Bytes the app may still allocate before iOS kills it (`os_proc_available_memory`), iOS only.
    public static func availableBytes() -> Int64? {
        #if os(iOS)
        return Int64(os_proc_available_memory())
        #else
        return nil
        #endif
    }

    /// (current, peak) physical footprint from `task_info(TASK_VM_INFO)`.
    public static func footprint() -> (current: Int64, peak: Int64)? {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return (Int64(info.phys_footprint), Int64(info.ledger_phys_footprint_peak))
        #else
        return nil
        #endif
    }

    public static func thermalState() -> String {
        #if canImport(Darwin)
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
        #else
        return "unknown"
        #endif
    }
}

extension InklingGenerator {
    /// Times a ~`promptTokens` prefill and `decodeTokens` greedy decode (EOS ignored), then records routing on a short pass.
    public func benchmark(promptTokens: Int = 512, decodeTokens: Int = 256,
                          progress: @escaping @Sendable (String) -> Void = { _ in }) throws -> BenchmarkReport {
        let (reference, fast) = paths()
        let model = path == .fast ? fast : reference
        let paragraph = "The quick brown fox jumps over the lazy dog while the iPad keeps every token on device. "
        var tokens = tokenizer.encode(String(repeating: paragraph, count: max(1, promptTokens / 16 + 1)))
        tokens = Array(tokens.prefix(promptTokens))
        var greedy = GenerateParameters()
        greedy.temperature = 0
        progress("Prefill \(tokens.count) tokens, decode \(decodeTokens)…")
        let gen = TokenGenerator(model: model, eos: [])
        gen.pipelined = generator.pipelined
        let profile = (reference as? Inkling2LanguageModel)?.profile
        let before = profile?.snapshot
        var afterPrefill: StreamProfile.Snapshot?
        gen.afterStep = { _ in if afterPrefill == nil { afterPrefill = profile?.snapshot } }
        try withError {
            gen.generate(prompt: tokens, sampler: Sampler(greedy), maxTokens: decodeTokens, shouldStop: { Task.isCancelled }, onToken: { _ in })
        }
        try Task.checkCancellation()
        var report = BenchmarkReport()
        report.enginePath = path
        report.promptTokens = gen.stats.promptTokens
        report.generatedTokens = gen.stats.generatedTokens
        report.prefillTokensPerSecond = gen.stats.prefillTokensPerSecond
        report.decodeTokensPerSecond = gen.stats.decodeTokensPerSecond
        report.timeToFirstToken = gen.stats.timeToFirstToken
        report.footprintBytes = MemoryInfo.footprint()?.current
        report.peakFootprintBytes = MemoryInfo.footprint()?.peak
        report.availableMemoryBytes = MemoryInfo.availableBytes()
        report.mappedWeightBytes = mappedWeightBytes
        report.thermalState = MemoryInfo.thermalState()
        report.expertHitRate = expertHitRate
        report.expertSlots = expertSlots
        if let profile, let before, let afterPrefill {
            let prefillSeconds = gen.stats.prefillTokensPerSecond > 0 ? Double(gen.stats.prefilledTokens) / gen.stats.prefillTokensPerSecond : 0
            report.prefillProfile = (afterPrefill - before).summary(tokens: 1, seconds: prefillSeconds)
            let steps = max(gen.stats.generatedTokens - 1, 1)
            let decodeSeconds = gen.stats.decodeTokensPerSecond > 0 ? Double(steps) / gen.stats.decodeTokensPerSecond : 0
            report.decodeProfile = (profile.snapshot - afterPrefill).summary(tokens: steps, seconds: decodeSeconds)
        }
        if let lm = reference as? Inkling2LanguageModel, profile != nil {
            progress("Measuring the decode ceiling…")
            lm.ignoreRouting = true
            defer { lm.ignoreRouting = false }
            var parts: [String] = []
            for (name, m) in [("fast", fast), ("reference", reference)] {
                let ceiling = TokenGenerator(model: m, eos: [])
                try withError {
                    ceiling.generate(prompt: Array(tokens.prefix(16)), sampler: Sampler(greedy), maxTokens: 48,
                                     shouldStop: { Task.isCancelled }, onToken: { _ in })
                }
                parts.append("\(name) " + String(format: "%.1f tok/s", ceiling.stats.decodeTokensPerSecond))
            }
            report.ceiling = parts.joined(separator: ", ")
        }
        if let lm = reference as? Inkling2LanguageModel {
            progress("Recording expert routing…")
            let stats = RoutingStats(visits: lm.config.layerVisits, numExperts: lm.config.numExperts)
            lm.routing = stats
            defer { lm.routing = nil }
            try withError { eval(lm(MLXArray(Array(tokens.prefix(128))).reshaped([1, -1]), cache: lm.makeCache())) }
            report.routing = stats.summary
            report.passOverlap = stats.overlapSummary
        }
        return report
    }
}
