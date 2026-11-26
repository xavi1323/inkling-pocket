import Foundation

/// How many bytes the engine may use in total: weights it maps for the GPU plus expert slots (E4).
public struct MemoryBudget {
    /// Left free for the app, the system and Metal scratch buffers.
    public static let safetyMargin: Int64 = 1_500_000_000
    /// Used where `os_proc_available_memory` does not exist (macOS tests, Linux).
    public static let fallback: Int64 = 4_000_000_000
    /// Never more than this share of physical RAM: with the per-app limit lifted (iRAM+), iOS reports most of RAM
    /// as available, but the system and its other processes need the rest (11.6 GB iPad → 7 GB).
    public static let physicalShare = 0.6

    /// `min(os_proc_available_memory() - 1.5 GB, 60% of RAM)`, or the Settings override (GB) when set.
    public static func current(overrideGB: Double?) -> Int64 {
        budget(available: MemoryInfo.availableBytes(), physical: Int64(ProcessInfo.processInfo.physicalMemory), overrideGB: overrideGB)
    }

    static func budget(available: Int64?, physical: Int64, overrideGB: Double?) -> Int64 {
        if let gb = overrideGB, gb > 0 { return Int64((gb * 1e9).rounded()) }
        guard let available else { return fallback }
        return max(0, min(available - safetyMargin, Int64(Double(physical) * physicalShare)))
    }
}
