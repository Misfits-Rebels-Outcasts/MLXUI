import Testing
import Foundation
@testable import MLXUI

/// PC-3 carry-over 2 — the tail is pinned. With a recording fake in place of the model, what the
/// cache already holds (reused prefix + every prefilled segment) followed by the tail handed to
/// `generateTokens` is the prompt **exactly**: no gap, no overlap, and the tail is never empty.
/// (2026-420: one-token tail mutations survived the real-model tests, which are blind to a
/// perturbed template tail.)
struct PromptCheckpointingPlanTests {

    /// A cache that remembers which prompt tokens it was fed, in order.
    private final class FakeCache {
        var fed: [Int] = []
        var calls: [(from: Int, to: Int)] = []
    }
    private struct FakeSnapshot: Sendable { let covered: [Int] }

    private static let mb = Int64(1_048_576)
    private let key = PromptCacheModelKey(directory: "/models/a", configSHA256: "c", parameters: "p", libraryVersion: "l")

    private func store() -> PromptCacheStore<FakeSnapshot> {
        PromptCacheStore<FakeSnapshot>(budget: MemoryBudget(capacityBytes: 1 << 40), copy: { $0 })
    }

    /// Runs the planner over `tokens`, where the turn probe diverges at `turn` and the system probe at `system`.
    private func plan(_ store: PromptCacheStore<FakeSnapshot>, tokens: [Int], turn: Int?, system: Int?,
                      trimmable: Bool = false) async throws -> PromptPlan<FakeCache> {
        let conversation = [ServeMessage(role: .system, text: "s"), ServeMessage(role: .user, text: "u")]
        return try await PromptCheckpointing.plan(
            store: store, key: key, messages: conversation, tokens: tokens,
            render: { probe in
                if probe.count == conversation.count + 2 {                         // the turn probe
                    guard let turn else { throw CancellationError() }
                    return Array(tokens[0..<turn]) + [-1, -2]
                }
                guard let system else { throw CancellationError() }                // the system probe
                return Array(tokens[0..<system]) + [-9]
            },
            newCache: { FakeCache() },
            restore: { hit in
                let cache = FakeCache()
                cache.fed = Array(hit.makeCopy().covered.prefix(hit.reusedLength))     // copy, then trim
                return cache
            },
            prefill: { from, to, cache in
                #expect(from == cache.fed.count, "prefill must start exactly where the cache ends (no gap, no overlap)")
                cache.calls.append((from, to))
                cache.fed += tokens[from..<to]
            },
            snapshot: { cache in (FakeSnapshot(covered: cache.fed), Int64(cache.fed.count) * Self.mb, trimmable) })
    }

    /// The pin itself.
    private func expectExact(_ plan: PromptPlan<FakeCache>, _ tokens: [Int], sourceLocation: SourceLocation = #_sourceLocation) {
        #expect((plan.cache?.fed ?? []) + plan.tailTokens == tokens,
                "cache + tail must equal the prompt exactly", sourceLocation: sourceLocation)
        #expect(!plan.tailTokens.isEmpty, "the tail is never empty", sourceLocation: sourceLocation)
        #expect(plan.outcome.reusedTokens <= (plan.cache?.fed.count ?? 0), sourceLocation: sourceLocation)
    }

    private func ids(_ n: Int) -> [Int] { Array(1000..<(1000 + n)) }

    @Test func aMissPrefillsEachSegmentAndLeavesTheGenerationPromptAsTheTail() async throws {
        let tokens = ids(1000)
        let s = store()
        let result = try await plan(s, tokens: tokens, turn: 996, system: 700)
        expectExact(result, tokens)
        #expect(result.cache?.calls.map { [$0.from, $0.to] } == [[0, 700], [700, 996]])
        #expect(result.tailTokens == Array(tokens[996...]))
        #expect(result.outcome.storedAt == [700, 996])
        #expect(result.outcome.reusedTokens == 0)
    }

    @Test func aHitReusesThePrefixThenPrefillsOnlyTheNewSegment() async throws {
        let s = store()
        let first = ids(1000)
        _ = try await plan(s, tokens: first, turn: 996, system: nil)               // stores 996
        let second = first[0..<996] + Array(2000..<2500)                           // the conversation grew
        let result = try await plan(s, tokens: Array(second), turn: 1300, system: nil)
        expectExact(result, Array(second))
        #expect(result.outcome.reusedTokens == 996)
        #expect(result.cache?.calls.map { [$0.from, $0.to] } == [[996, 1300]])
        #expect(result.tailTokens == Array(second[1300...]))
    }

    @Test func aTrimmableHitThatDivergesInsideTheSnapshotIsTrimmedBeforeUse() async throws {
        let s = store()
        let stored = ids(900)
        _ = try await plan(s, tokens: stored + [1, 2, 3, 4], turn: 900, system: nil, trimmable: true)
        var prompt = ids(1200)
        prompt[500] = -7                                                           // shares only 500 tokens
        let result = try await plan(s, tokens: prompt, turn: 1100, system: nil, trimmable: true)
        expectExact(result, prompt)
        #expect(result.outcome.reusedTokens == 500)
        #expect(result.outcome.hitStoredLength == 900)
    }

    @Test func whenTheWholePromptIsTheBoundaryTheTailIsExactlyTheLastToken() async throws {
        let tokens = ids(800)                                                      // Gemma-style: probe agrees with everything
        let result = try await plan(store(), tokens: tokens, turn: 800, system: nil)
        expectExact(result, tokens)
        #expect(result.tailTokens == [tokens[799]])
        #expect(result.cache?.calls.map { [$0.from, $0.to] } == [[0, 799]])
    }

    @Test func aHitThatAlreadyCoversTheBoundaryPrefillsNothingAndKeepsTheRest() async throws {
        let s = store()
        let first = ids(1000)
        _ = try await plan(s, tokens: first, turn: 996, system: nil)
        let again = first[0..<996] + Array(3000..<3010)
        let result = try await plan(s, tokens: Array(again), turn: 996, system: nil)   // T is no further than the hit
        expectExact(result, Array(again))
        #expect(result.cache?.calls.isEmpty == true)
        #expect(result.tailTokens == Array(again[996...]))
    }

    @Test func noCheckpointAndNoHitMeansNoCacheAndTheWholePromptAsTheInput() async throws {
        let tokens = ids(300)
        let result = try await plan(store(), tokens: tokens, turn: 100, system: nil)   // below the 256 minimum
        #expect(result.cache == nil)
        #expect(result.tailTokens == tokens)
        let failed = try await plan(store(), tokens: tokens, turn: nil, system: nil)   // both probe renders failed
        #expect(failed.cache == nil)
        #expect(failed.tailTokens == tokens)
    }

    @Test func everyShapeOfPromptKeepsTheInvariant() async throws {
        // A sweep: lengths × boundaries, hits and misses alike.
        let s = store()
        for n in [260, 300, 513, 1024, 2047] {
            let tokens = ids(n)
            for (turn, system) in [(n - 4, n / 2), (n, nil), (n - 1, n - 1), (n / 2, nil), (nil, n / 3)] as [(Int?, Int?)] {
                let result = try await plan(s, tokens: tokens, turn: turn, system: system)
                expectExact(result, tokens)
                if let cache = result.cache {
                    var cursor = result.outcome.reusedTokens
                    for call in cache.calls {
                        #expect(call.from == cursor)                                  // contiguous segments
                        cursor = call.to
                    }
                }
            }
        }
    }
}
