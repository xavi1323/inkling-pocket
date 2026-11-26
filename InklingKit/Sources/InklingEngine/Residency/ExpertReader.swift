import Foundation

/// One pool of read threads for every expert read in the process: a deep queue (many `pread`s in flight, which fast
/// flash storage needs to reach full speed) of pieces of at most `pieceSize` bytes. Pieces a step is waiting for
/// (`urgent`) start before any queued read-ahead piece.
/// `@unchecked Sendable`: the queues are behind `condition`; pieces write disjoint memory their callers own.
final class ExpertReader: @unchecked Sendable {
    struct Piece: @unchecked Sendable {
        let fd: Int32
        let offset: Int64
        let destination: UnsafeMutableRawPointer
        let length: Int
        /// The same bytes in the mapping, copied when `pread` falls short.
        let fallback: UnsafeRawPointer
    }

    static let pieceSize = 1 << 20
    static let threads = 32
    static let shared = ExpertReader(threads: threads)

    private struct Item { let piece: Piece; let group: DispatchGroup }
    private let condition = NSCondition()
    private var urgent: [Item] = [], background: [Item] = []
    private var urgentHead = 0, backgroundHead = 0

    private init(threads: Int) {
        for i in 0..<threads {
            let thread = Thread { [unowned self] in work() }
            thread.name = "inkling.expert-read.\(i)"
            thread.qualityOfService = .userInitiated
            thread.start()
        }
    }

    /// `length` bytes at `offset` of `fd` into `destination`, as pieces of at most `pieceSize`.
    static func pieces(fd: Int32, offset: Int64, length: Int, destination: UnsafeMutableRawPointer,
                       fallback: UnsafeRawPointer) -> [Piece] {
        stride(from: 0, to: length, by: pieceSize).map { start in
            Piece(fd: fd, offset: offset + Int64(start), destination: destination + start,
                  length: min(pieceSize, length - start), fallback: fallback + start)
        }
    }

    /// Reads every piece; returns when all are in memory.
    func read(_ pieces: [Piece], urgent isUrgent: Bool) {
        guard !pieces.isEmpty else { return }
        let group = DispatchGroup()
        for _ in pieces { group.enter() }
        condition.lock()
        for p in pieces {
            if isUrgent { urgent.append(Item(piece: p, group: group)) } else { background.append(Item(piece: p, group: group)) }
        }
        condition.broadcast()
        condition.unlock()
        group.wait()
    }

    private func next() -> Item {
        condition.lock()
        defer { condition.unlock() }
        while true {
            if urgentHead < urgent.count {
                let item = urgent[urgentHead]
                urgentHead += 1
                if urgentHead == urgent.count { urgent.removeAll(keepingCapacity: true); urgentHead = 0 }
                return item
            }
            if backgroundHead < background.count {
                let item = background[backgroundHead]
                backgroundHead += 1
                if backgroundHead == background.count { background.removeAll(keepingCapacity: true); backgroundHead = 0 }
                return item
            }
            condition.wait()
        }
    }

    private func work() {
        while true {
            let item = next()
            Self.perform(item.piece)
            item.group.leave()
        }
    }

    static func perform(_ p: Piece) {
        var done = 0
        if p.fd >= 0 {
            while done < p.length {
                let n = pread(p.fd, p.destination + done, p.length - done, off_t(p.offset) + off_t(done))
                if n <= 0 { break }
                done += n
            }
        }
        if done < p.length { memcpy(p.destination + done, p.fallback + done, p.length - done) }
    }
}
