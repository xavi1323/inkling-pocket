import Foundation
import Testing
@testable import InklingCore

/// Needs the real tokenizer files: set INKLING_MODEL_DIR to the model folder (PC only; skipped on CI).
let modelDir = ProcessInfo.processInfo.environment["INKLING_MODEL_DIR"]
let hasModelDir = modelDir != nil

struct TokenCase { let text: String; let ids: [Int32] }

func tokenCases() throws -> [TokenCase] {
    let url = Bundle.module.url(forResource: "Fixtures/tokens", withExtension: "json")!
    guard case .array(let items) = try JSONValue.parse(String(contentsOf: url, encoding: .utf8)) else { return [] }
    return items.map { item in
        guard case .array(let ids)? = item["ids"] else { return TokenCase(text: "", ids: []) }
        return TokenCase(text: item["text"]!.stringValue!,
                         ids: ids.compactMap { if case .int(let v) = $0 { Int32(v) } else { nil } })
    }
}

@Suite(.enabled(if: hasModelDir, "set INKLING_MODEL_DIR to run tokenizer tests"))
struct TokenizerTests {
    static let tokenizer: InklingTokenizer? = try? InklingTokenizer(modelFolder: URL(fileURLWithPath: modelDir ?? "/"))

    @Test func encodeMatchesReference() throws {
        let tok = try #require(Self.tokenizer)
        for c in try tokenCases() {
            #expect(tok.encode(c.text) == c.ids, "text: \(c.text.debugDescription)")
        }
    }

    @Test func decodeRoundTrip() throws {
        let tok = try #require(Self.tokenizer)
        for c in try tokenCases() {
            #expect(tok.decode(c.ids) == c.text)
        }
    }

    @Test func specialTokensAreSingleIDs() throws {
        let tok = try #require(Self.tokenizer)
        #expect(tok.specialTokenID("<|content_model_end_sampling|>") == 200006)
        #expect(tok.eosID == 200006)
        #expect(tok.encode("<|end_message|>") == [200010])
    }

    @Test func streamingDecoderNeverSplitsUTF8() throws {
        let tok = try #require(Self.tokenizer)
        let ids = tok.encode("Emoji 😀👍🏽 日本語 done")
        var decoder = StreamingDecoder(tokenizer: tok)
        var pieces = ""
        for id in ids {
            let piece = decoder.append(id)
            #expect(!piece.contains("\u{FFFD}"))
            pieces += piece
        }
        pieces += decoder.flush()
        #expect(pieces == tok.decode(ids))
    }
}
