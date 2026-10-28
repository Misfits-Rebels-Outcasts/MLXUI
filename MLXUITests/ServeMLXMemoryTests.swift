import Testing
import Foundation
import MLX
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

// MP6-3h (owner ruling, Boo, 2026-10-27) -- `end()` restores the saved limit only if the limit is still the one the helper set, so
// a limit someone else set mid-hold (EngineCache.init, on the first flow run) is left alone.
struct ServeMLXMemoryUnchangedRestoreTests {
    private func scope(_ limit: Limit) -> ServeMLXMemory { ServeMLXMemory(read: { limit.value }, write: { limit.set($0) }) }
    private let big = 18_360_985_190
    private let thirdParty = 10_000_000_000

    @Test func aLimitChangedMidHoldIsLeftAlone() {
        let limit = Limit(big)
        let memory = scope(limit)
        memory.begin()
        limit.set(thirdParty)                                      // someone else, mid-hold
        memory.end()
        #expect(limit.value == thirdParty)
        #expect(!memory.isCapActive)
    }

    @Test func anUnchangedLimitIsRestoredAndALowerExistingOneIsNeverRaised() {
        let limit = Limit(big)
        let memory = scope(limit)
        memory.begin(); memory.end()
        #expect(limit.value == big)
        let low = Limit(256 * 1_048_576)
        let lowMemory = scope(low)
        lowMemory.begin()
        #expect(low.value == 256 * 1_048_576, "a lower limit is not raised to the cap")
        lowMemory.end()
        #expect(low.value == 256 * 1_048_576)
    }

    @Test func overlappingHoldersWithAThirdPartyChangeInEitherOrderRestoreOnceOrNotAtAll() {
        // voice turn and server generation overlap; the third party changes the limit while both hold it
        for order in [(first: "voice", second: "server"), (first: "server", second: "voice")] {
            let limit = Limit(big)
            let memory = scope(limit)
            memory.begin()                                          // `order.first` starts
            memory.begin()                                          // `order.second` starts
            limit.set(thirdParty)
            memory.end()                                            // `order.first` leaves
            #expect(limit.value == thirdParty, "\(order.first) left first; one holder remains")
            memory.end()                                            // `order.second` leaves: the last one out
            #expect(limit.value == thirdParty, "\(order.second) left last: the third party's value stays")
            #expect(!memory.isCapActive)
            // and the next hold starts from what is there now
            memory.begin()
            #expect(limit.value == min(thirdParty, ServeMLXMemory.generationCacheLimit))
            memory.end()
            #expect(limit.value == thirdParty)
        }
    }

    @Test func aVoiceTurnThatThrowsOrIsCancelledLeavesAThirdPartyLimitAlone() async {
        for failure in [CancellationError() as any Error, StageError.engineFailure(stage: "t", underlying: CancellationError())] {
            let limit = Limit(big)
            let memory = scope(limit)
            let backend = MockLFMBackend()
            let third = thirdParty
            backend.onGenerate = { _ in limit.set(third); throw failure }
            _ = try? await LFMAudioEngine.voiceTurn(userSamples: [0.1], conversation: LFMConversation(), modelDirectory: URL(fileURLWithPath: "/tmp/x"),
                                                    factory: { _ in backend }, memory: memory) { _ in }
            #expect(limit.value == thirdParty)
            #expect(!memory.isCapActive)
        }
    }
}

// MP6-3j -- the real MLX limit. `end()` restores only if `read() == applied`, so MLX's own getter must return exactly what its setter
// was given. These touch the process-global `Memory.cacheLimit`: they save and restore it, and run serially (swift-testing's
// `.serialized`).
@Suite(.serialized)
struct ServeMLXMemoryRealMLXTests {
    private func real() -> ServeMLXMemory { ServeMLXMemory(read: { Memory.cacheLimit }, write: { Memory.cacheLimit = $0 }) }

    @Test func theRealLimitRoundTripsExactlyThroughBeginAndEnd() {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }
        let memory = ServeMLXMemory.shared
        memory.begin()
        #expect(Memory.cacheLimit == ServeMLXMemory.effectiveLimit(current: original), "the getter returns what the helper applied")
        #expect(memory.isCapActive)
        memory.end()
        #expect(Memory.cacheLimit == original, "end() restored the original: the cap does not stay in force")
        #expect(!memory.isCapActive)
    }

    @Test func theRealGetterReturnsAnyValueTheSetterGotExactly() {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }
        for value in [1_073_741_824, 256 * 1_048_576, 12_345_678_901, 4_294_967_296 + 4_097, original] {
            Memory.cacheLimit = value
            #expect(Memory.cacheLimit == value, "set \(value), read \(Memory.cacheLimit)")
        }
    }

    @Test func aLowerExistingRealLimitIsNeitherRaisedNorLostAtEnd() {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }
        Memory.cacheLimit = 300 * 1_048_576
        let memory = real()
        memory.begin()
        #expect(Memory.cacheLimit == 300 * 1_048_576)
        memory.end()
        #expect(Memory.cacheLimit == 300 * 1_048_576)
    }

    /// The scenario that motivated MP6-3h: `EngineCache.init` sets `Memory.cacheLimit` to its budget the first time it is built, which
    /// can be mid-turn. After the hold ends its value must still be there.
    @Test func anEngineCacheBuiltMidHoldKeepsItsLimitAfterEnd() {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }
        let memory = real()
        memory.begin()
        let budget: Int64 = 4 * 1_073_741_824
        let cache = EngineCache(budgetBytes: budget)            // sets Memory.cacheLimit = budget
        #expect(Memory.cacheLimit == Int(budget))
        memory.end()
        #expect(Memory.cacheLimit == Int(budget), "the helper left the engine cache's limit alone")
        _ = cache
    }
}
