import Foundation
import Observation
import InklingCore
import InklingEngine

extension EnginePath: Identifiable {
    public var id: String { rawValue }
    var title: String {
        switch self {
        case .turbo: "Turbo (fastest, exact)"
        case .fast: "Fast (custom Metal, exact)"
        case .reference: "Reference (plain MLX)"
        }
    }
}

/// User settings, saved in UserDefaults.
@Observable @MainActor
final class AppSettings {
    @ObservationIgnored private let defaults: UserDefaults

    var effort: ReasoningEffort { didSet { defaults.set(effort.rawValue, forKey: "effort") } }
    var temperature: Double { didSet { defaults.set(temperature, forKey: "temperature") } }
    var topP: Double { didSet { defaults.set(topP, forKey: "topP") } }
    var maxSteps: Int { didSet { defaults.set(maxSteps, forKey: "maxSteps") } }
    var contextCap: Int { didSet { defaults.set(contextCap, forKey: "contextCap") } }
    var crawlTopN: Int { didSet { defaults.set(crawlTopN, forKey: "crawlTopN") } }
    /// 0 = automatic (from os_proc_available_memory).
    var memoryBudgetOverrideGB: Double { didSet { defaults.set(memoryBudgetOverrideGB, forKey: "memoryBudgetOverrideGB") } }
    var enginePath: EnginePath { didSet { defaults.set(enginePath.rawValue, forKey: "enginePath") } }
    var keepModelInPlace: Bool { didSet { defaults.set(keepModelInPlace, forKey: "keepModelInPlace") } }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        // Faster defaults (once): a calmer temperature keeps more of turbo's guesses, and fewer pages per search mean
        // less to read before answering. Values the user changed away from the old defaults stay.
        if !defaults.bool(forKey: "fasterDefaults") {
            if abs((defaults.object(forKey: "temperature") as? Double ?? 0.7) - 0.7) < 0.001 { defaults.set(0.4, forKey: "temperature") }
            if (defaults.object(forKey: "crawlTopN") as? Int ?? 3) == 3 { defaults.set(2, forKey: "crawlTopN") }
            defaults.set(true, forKey: "fasterDefaults")
        }
        effort = ReasoningEffort(rawValue: defaults.string(forKey: "effort") ?? "") ?? .low
        temperature = defaults.object(forKey: "temperature") as? Double ?? 0.4
        topP = defaults.object(forKey: "topP") as? Double ?? 0.95
        maxSteps = defaults.object(forKey: "maxSteps") as? Int ?? 8
        contextCap = defaults.object(forKey: "contextCap") as? Int ?? 16_384
        crawlTopN = defaults.object(forKey: "crawlTopN") as? Int ?? 2
        memoryBudgetOverrideGB = defaults.object(forKey: "memoryBudgetOverrideGB") as? Double ?? 0
        // Turbo became the default; a saved Fast from before it existed moves to Turbo once (Settings can switch back).
        if defaults.string(forKey: "enginePath") == EnginePath.fast.rawValue, !defaults.bool(forKey: "turboOffered") {
            defaults.set(EnginePath.turbo.rawValue, forKey: "enginePath")
        }
        defaults.set(true, forKey: "turboOffered")
        enginePath = EnginePath(rawValue: defaults.string(forKey: "enginePath") ?? "") ?? .turbo
        keepModelInPlace = defaults.object(forKey: "keepModelInPlace") as? Bool ?? false
    }
}
