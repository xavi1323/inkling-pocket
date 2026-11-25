import Testing
import InklingCore
@testable import AgentCore

func count4(_ s: String) -> Int { (s.count + 3) / 4 }

@Test func contextWindowKeepsEverythingWhenItFits() {
    let h = [ChatMessage(role: .user, content: "a"), ChatMessage(role: .assistant, content: "b")]
    #expect(ContextWindow.fit(system: "s", history: h, cap: 1000, count: count4) == h)
}

@Test func contextWindowDropsOldestTurns() {
    let big = String(repeating: "x", count: 400)  // 100 tokens each
    let h = [
        ChatMessage(role: .user, content: big), ChatMessage(role: .assistant, content: big),
        ChatMessage(role: .user, content: big), ChatMessage(role: .assistant, content: big),
        ChatMessage(role: .user, content: "last question"),
    ]
    let fitted = ContextWindow.fit(system: "sys", history: h, cap: 250, count: count4)
    #expect(fitted.last == h.last)
    #expect(fitted.count < h.count)
    #expect(fitted.first?.role == .user)  // never starts with an orphan assistant/tool message
}

@Test func contextWindowAlwaysKeepsLastUserTurnEvenIfHuge() {
    let h = [ChatMessage(role: .user, content: String(repeating: "y", count: 10_000))]
    #expect(ContextWindow.fit(system: "s", history: h, cap: 100, count: count4) == h)
}

@Test func trimToBudgetCutsAndMarks() {
    let text = String(repeating: "abcd", count: 1000)  // 1000 tokens
    let out = trimToBudget(text, budget: 100, count: count4)
    #expect(count4(out) <= 100)
    #expect(out.hasSuffix("\n[trimmed]"))
    #expect(trimToBudget("short", budget: 100, count: count4) == "short")
}
