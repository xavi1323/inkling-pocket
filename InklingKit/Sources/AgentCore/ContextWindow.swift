import InklingCore

public enum ContextWindow {
    /// Keeps the newest messages that fit in `cap` tokens (system prompt counted).
    /// The last user turn is always kept; the result never starts with an assistant or tool message.
    public static func fit(system: String, history: [ChatMessage], cap: Int, count: (String) -> Int) -> [ChatMessage] {
        func cost(_ m: ChatMessage) -> Int {
            count(m.content) + count(m.reasoning ?? "") + m.toolCalls.reduce(0) { $0 + count($1.name) + count(JSONValue.object($1.arguments).canonicalJSON(sortKeys: false)) } + 4
        }
        var budget = cap - count(system)
        guard let lastUser = history.lastIndex(where: { $0.role == .user }) else { return history }
        var start = history.count
        // Everything from the last user turn on is mandatory.
        for i in stride(from: history.count - 1, through: lastUser, by: -1) {
            budget -= cost(history[i])
            start = i
        }
        var i = lastUser - 1
        while i >= 0 {
            let c = cost(history[i])
            if c > budget { break }
            budget -= c
            start = i
            i -= 1
        }
        // Don't begin mid-exchange.
        while start < lastUser, history[start].role != .user { start += 1 }
        return Array(history[start...])
    }
}

/// Cuts `text` to at most `budget` tokens and marks the cut.
public func trimToBudget(_ text: String, budget: Int, count: (String) -> Int) -> String {
    if count(text) <= budget { return text }
    let marker = "\n[trimmed]"
    let target = budget - count(marker)
    var lo = 0, hi = text.count
    while lo < hi {
        let mid = (lo + hi + 1) / 2
        if count(String(text.prefix(mid))) <= target { lo = mid } else { hi = mid - 1 }
    }
    return String(text.prefix(lo)) + marker
}
