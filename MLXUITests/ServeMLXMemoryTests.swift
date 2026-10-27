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
