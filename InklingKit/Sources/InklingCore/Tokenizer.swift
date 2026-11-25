import Foundation

public enum TokenizerError: Error, Equatable {
    case missingFile(String)
    case badFormat(String)
}

/// tiktoken-style byte-level BPE for Inkling 2.
/// Ranks come from `tiktoken/tokenizer.model`; special tokens and the split regex from `tokenizer.json`.
public final class InklingTokenizer: @unchecked Sendable {
    private let ranks: [[UInt8]: Int32]
    private let tokenBytes: [[UInt8]]          // id -> bytes (ordinary + specials)
    private let specials: [(String, Int32)]    // longest first
    private let specialIDs: [String: Int32]
    private let splitter: NSRegularExpression  // thread-safe for matching
    public let eosID: Int32
    public var vocabularySize: Int { tokenBytes.count }

    public init(modelFolder: URL) throws {
        let modelURL = modelFolder.appending(path: "tiktoken/tokenizer.model")
        let jsonURL = modelFolder.appending(path: "tokenizer.json")
        guard let ranksText = try? String(contentsOf: modelURL, encoding: .utf8) else {
            throw TokenizerError.missingFile("tiktoken/tokenizer.model")
        }
        guard let jsonData = try? Data(contentsOf: jsonURL),
              let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw TokenizerError.missingFile("tokenizer.json")
        }

        var ranks: [[UInt8]: Int32] = [:]
        ranks.reserveCapacity(200_100)
        var maxID: Int32 = -1
        for line in ranksText.split(separator: "\n", omittingEmptySubsequences: true) {
            let parts = line.split(separator: " ")
            guard parts.count == 2, let data = Data(base64Encoded: String(parts[0])), let id = Int32(parts[1]) else {
                throw TokenizerError.badFormat("tiktoken line: \(line.prefix(40))")
            }
            ranks[[UInt8](data)] = id
            maxID = max(maxID, id)
        }

        var specialList: [(String, Int32)] = []
        for case let token as [String: Any] in json["added_tokens"] as? [Any] ?? [] {
            if let content = token["content"] as? String, let id = token["id"] as? Int {
                specialList.append((content, Int32(id)))
                maxID = max(maxID, Int32(id))
            }
        }

        var table = [[UInt8]](repeating: [], count: Int(maxID) + 1)
        for (bytes, id) in ranks { table[Int(id)] = bytes }
        for (s, id) in specialList {
            table[Int(id)] = Array(s.utf8)
            ranks.removeValue(forKey: Array(s.utf8))  // specials are never produced by BPE
        }

        guard let pre = json["pre_tokenizer"] as? [String: Any],
              let pattern = InklingTokenizer.findSplitRegex(pre) else {
            throw TokenizerError.badFormat("pre_tokenizer Split regex not found")
        }

        self.ranks = ranks
        self.tokenBytes = table
        self.specials = specialList.sorted { $0.0.utf8.count > $1.0.utf8.count }
        self.specialIDs = Dictionary(specialList, uniquingKeysWith: { a, _ in a })
        self.splitter = try NSRegularExpression(pattern: pattern)
        self.eosID = specialIDs["<|content_model_end_sampling|>"] ?? 200_006
    }

    private static func findSplitRegex(_ node: [String: Any]) -> String? {
        if node["type"] as? String == "Split", let p = node["pattern"] as? [String: Any], let r = p["Regex"] as? String {
            return r
        }
        for case let child as [String: Any] in node["pretokenizers"] as? [Any] ?? [] {
            if let r = findSplitRegex(child) { return r }
        }
        return nil
    }

    public func specialTokenID(_ token: String) -> Int32? { specialIDs[token] }

    public func tokenBytes(_ id: Int32) -> [UInt8] {
        id >= 0 && Int(id) < tokenBytes.count ? tokenBytes[Int(id)] : []
    }

    // MARK: Encode

    public func encode(_ text: String) -> [Int32] {
        var out: [Int32] = []
        var rest = Substring(text)
        if text.contains("<|") {
            while let (range, id) = earliestSpecial(in: rest) {
                encodeOrdinary(rest[rest.startIndex..<range.lowerBound], into: &out)
                out.append(id)
                rest = rest[range.upperBound...]
            }
        }
        encodeOrdinary(rest, into: &out)
        return out
    }

    private func earliestSpecial(in text: Substring) -> (Range<Substring.Index>, Int32)? {
        var best: (Range<Substring.Index>, Int32)?
        for (s, id) in specials {
            if let r = text.range(of: s), best == nil || r.lowerBound < best!.0.lowerBound {
                best = (r, id)
            }
        }
        return best
    }

    private func encodeOrdinary(_ text: Substring, into out: inout [Int32]) {
        guard !text.isEmpty else { return }
        let s = String(text)
        let ns = s as NSString
        var cursor = 0
        for match in splitter.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            if match.range.location > cursor {  // gap between matches is its own piece
                bpe(Array(ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor)).utf8), into: &out)
            }
            bpe(Array(ns.substring(with: match.range).utf8), into: &out)
            cursor = match.range.location + match.range.length
        }
        if cursor < ns.length {
            bpe(Array(ns.substring(from: cursor).utf8), into: &out)
        }
    }

    /// Byte-pair merge by lowest rank (tiktoken algorithm). Whole piece in vocab wins (ignore_merges).
    private func bpe(_ piece: [UInt8], into out: inout [Int32]) {
        guard !piece.isEmpty else { return }
        if let id = ranks[piece] { out.append(id); return }
        // boundaries[i] = start of part i; last entry = piece.count
        var boundaries = Array(0...piece.count)
        func rank(_ i: Int) -> Int32? {
            guard i + 2 < boundaries.count else { return nil }
            return ranks[Array(piece[boundaries[i]..<boundaries[i + 2]])]
        }
        var pairRanks: [Int32?] = (0..<(boundaries.count - 1)).map { rank($0) }
        while true {
            var minIndex = -1
            var minRank = Int32.max
            for (i, r) in pairRanks.enumerated() {
                if let r, r < minRank { minRank = r; minIndex = i }
            }
            if minIndex < 0 { break }
            boundaries.remove(at: minIndex + 1)
            pairRanks.remove(at: minIndex + 1)
            pairRanks[minIndex] = rank(minIndex)
            if minIndex > 0 { pairRanks[minIndex - 1] = rank(minIndex - 1) }
        }
        for i in 0..<(boundaries.count - 1) {
            let bytes = Array(piece[boundaries[i]..<boundaries[i + 1]])
            if let id = ranks[bytes] {
                out.append(id)
            } else {
                // single bytes are always in a byte-level vocab; fall back per byte just in case
                for b in bytes { if let id = ranks[[b]] { out.append(id) } }
            }
        }
    }

    // MARK: Decode

    public func decodeBytes(_ ids: [Int32]) -> [UInt8] {
        var bytes: [UInt8] = []
        for id in ids { bytes += tokenBytes(id) }
        return bytes
    }

    public func decode(_ ids: [Int32]) -> String {
        String(decoding: decodeBytes(ids), as: UTF8.self)
    }
}

/// Turns a stream of token ids into text, never emitting half a UTF-8 character.
public struct StreamingDecoder {
    private let tokenizer: InklingTokenizer
    private var pending: [UInt8] = []

    public init(tokenizer: InklingTokenizer) { self.tokenizer = tokenizer }

    public mutating func append(_ id: Int32) -> String {
        pending += tokenizer.tokenBytes(id)
        let cut = StreamingDecoder.completePrefixLength(pending)
        guard cut > 0 else { return "" }
        let text = String(decoding: pending[0..<cut], as: UTF8.self)
        pending.removeFirst(cut)
        return text
    }

    public mutating func flush() -> String {
        defer { pending.removeAll() }
        return String(decoding: pending, as: UTF8.self)
    }

    /// Length of the longest prefix that does not end inside a multi-byte sequence.
    static func completePrefixLength(_ bytes: [UInt8]) -> Int {
        let n = bytes.count
        var i = n - 1
        var back = 0
        while i >= 0 && back < 4 {
            let b = bytes[i]
            if b & 0xC0 != 0x80 {  // lead byte or ASCII
                let need: Int
                if b < 0x80 { need = 1 } else if b & 0xE0 == 0xC0 { need = 2 } else if b & 0xF0 == 0xE0 { need = 3 } else if b & 0xF8 == 0xF0 { need = 4 } else { need = 1 }
                return (n - i) >= need ? n : i
            }
            i -= 1
            back += 1
        }
        return n
    }
}
