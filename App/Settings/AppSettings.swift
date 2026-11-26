import Foundation
import Observation
import InklingCore
import InklingEngine

extension EnginePath: Identifiable {
    public var id: String { rawValue }
    var title: String { self == .fast ? "Fast (custom Metal)" : "Reference (plain MLX)" }
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
        effort = ReasoningEffort(rawValue: defaults.string(forKey: "effort") ?? "") ?? .low
        temperature = defaults.object(forKey: "temperature") as? Double ?? 0.7
        topP = defaults.object(forKey: "topP") as? Double ?? 0.95
        maxSteps = defaults.object(forKey: "maxSteps") as? Int ?? 8
        contextCap = defaults.object(forKey: "contextCap") as? Int ?? 16_384
        crawlTopN = defaults.object(forKey: "crawlTopN") as? Int ?? 3
        memoryBudgetOverrideGB = defaults.object(forKey: "memoryBudgetOverrideGB") as? Double ?? 0
        enginePath = EnginePath(rawValue: defaults.string(forKey: "enginePath") ?? "") ?? .fast
        keepModelInPlace = defaults.object(forKey: "keepModelInPlace") as? Bool ?? false
    }
}
