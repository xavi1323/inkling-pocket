import InklingCore

/// Encodes chat prompts, reusing the last prompt's tokens up to the last special token both share. Each turn resends
/// the whole conversation; only the text after the last shared special token (usually the newest messages) is
/// encoded again. Exact: specials split the text, so tokens before a cut never depend on what follows.
final class PromptEncoder {
    let tokenizer: InklingTokenizer
    private var last: (text: [UInt8], tokens: [Int32], cuts: [(offset: Int, count: Int)])?

    init(tokenizer: InklingTokenizer) { self.tokenizer = tokenizer }

    /// Tokens reused from the last prompt by the last call (for tests and stats).
    private(set) var reusedTokens = 0

    func encode(_ text: String) -> [Int32] {
        let bytes = Array(text.utf8)
        var cut: (offset: Int, count: Int)?
        if let last {
            var shared = 0
            let n = min(bytes.count, last.text.count)
            while shared < n, bytes[shared] == last.text[shared] { shared += 1 }
            cut = last.cuts.last { $0.offset <= shared }
        }
        let tokens: [Int32], cuts: [(offset: Int, count: Int)]
        if let cut, let last, let tail = String(bytes: bytes[cut.offset...], encoding: .utf8) {
            let (rest, restCuts) = tokenizer.encodeWithCuts(tail)
            tokens = Array(last.tokens.prefix(cut.count)) + rest
            cuts = last.cuts.filter { $0.offset <= cut.offset } + restCuts.map { ($0.offset + cut.offset, $0.count + cut.count) }
            reusedTokens = cut.count
        } else {
            (tokens, cuts) = tokenizer.encodeWithCuts(text)
            reusedTokens = 0
        }
        last = (bytes, tokens, cuts)
        return tokens
    }
}
