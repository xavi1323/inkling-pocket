import Foundation

/// How many bytes the engine may keep resident (E4).
public struct MemoryBudget {
    /// Left free for the app, the system and Metal scratch buffers.
    public static let safetyMargin: Int64 = 1_500_000_000
    /// Used where `os_proc_available_memory` does not exist (macOS tests, Linux).
    public static let fallback: Int64 = 4_000_000_000

    /// `os_proc_available_memory() - 1.5 GB`, or the Settings override (GB) when set.
    public static func current(overrideGB: Double?) -> Int64 {
        if let gb = overrideGB, gb > 0 { return Int64((gb * 1e9).rounded()) }
        guard let available = MemoryInfo.availableBytes() else { return fallback }
        return max(0, available - safetyMargin)
    }
}
