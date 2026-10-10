import Testing
import Foundation
@testable import MLXUI

// S3-5c — the scoped MLX buffer-cache limit.

private final class Limit: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Int
    private(set) var writes: [Int] = []
    init(_ value: Int) { _value = value }
    var value: Int { lock.lock(); defer { lock.unlock() }; return _value }
    func set(_ new: Int) { lock.lock(); _value = new; writes.append(new); lock.unlock() }
}

struct ServeMLXMemoryTests {
    private func scope(_ limit: Limit) -> ServeMLXMemory { ServeMLXMemory(read: { limit.value }, write: { limit.set($0) }) }
    private let big = 18_360_985_190                                           // MLX's default on the 18 GiB Mac

    @Test func theLimitIsLowerWhileAGenerationRunsAndRestoredAfter() {
        let limit = Limit(big)
        let memory = scope(limit)
        memory.begin()
        #expect(limit.value == ServeMLXMemory.generationCacheLimit && ServeMLXMemory.generationCacheLimit == 1_024 * 1_048_576)
        memory.end()
        #expect(limit.value == big)                                              // exactly what was there
    }

    @Test func overlappingGenerationsShareOneSavedValueAndTheLastOneOutRestores() {
        let limit = Limit(big)
        let memory = scope(limit)
        memory.begin(); memory.begin()
        memory.end()
        #expect(limit.value == ServeMLXMemory.generationCacheLimit)              // one still running
        memory.end()
        #expect(limit.value == big)
        #expect(limit.writes == [ServeMLXMemory.generationCacheLimit, big])      // written twice in all
        memory.end()                                                              // an extra end is a no-op
        #expect(limit.writes.count == 2)
    }

    @Test func itNeverRaisesALowerLimitSomeoneAlreadySet() {
        let limit = Limit(100 * 1_048_576)
        let memory = scope(limit)
        memory.begin()
        #expect(limit.value == 100 * 1_048_576)                                  // min(current, cap)
        memory.end()
        #expect(limit.value == 100 * 1_048_576)
        #expect(ServeMLXMemory.effectiveLimit(current: 10, cap: 20) == 10)
        #expect(ServeMLXMemory.effectiveLimit(current: 30, cap: 20) == 20)
    }
}
