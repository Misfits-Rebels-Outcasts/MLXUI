import Testing
import Foundation
@testable import MLXUI

// S3.1b — a request trades unused reply room for a prompt-cache snapshot, without ever reserving more.

struct ReplyTradeTests {
    private final class FakeCache { var fed: [Int] = [] }
    private struct FakeSnapshot: Sendable { let covered: [Int] }
    private let key = PromptCacheModelKey(directory: "/models/a", configSHA256: "c", parameters: "p", libraryVersion: "l")
    private static let mb = Int64(1_048_576)
    private static let bytesPerToken = 1_000.0

    /// 100 MB budget (store cap 25 MB): 60 MB of weights, a 30 MB working set whose reply share is what could be given back.
    private struct Rig {
        let budget = MemoryBudget(capacityBytes: 100 * ReplyTradeTests.mb)
        let store: PromptCacheStore<FakeSnapshot>
        let workingSet: MemoryBudget.Reservation
        init() {
            store = PromptCacheStore<FakeSnapshot>(budget: budget, retention: .newestWins, copy: { $0 })
            _ = budget.tryReserve(bytes: 60 * ReplyTradeTests.mb)
            workingSet = budget.tryReserve(bytes: 30 * ReplyTradeTests.mb)!
        }
        func trade(replyLimit: Int) -> ReplyTrade {
            let budget = self.budget, ws = workingSet
            return ReplyTrade(replyLimit: replyLimit, bytesPerToken: ReplyTradeTests.bytesPerToken) { budget.shrink(ws, by: $0) }
        }
        /// What the ledger holds for the working set = 30 MB minus whatever was given back (the stored snapshots' bytes excluded).
        var reserved: Int64 { get async { budget.usedBytes - 60 * ReplyTradeTests.mb - (await store.totalBytes) } }
    }

    private func plan(_ rig: Rig, snapshotMB: Int64, trade: ReplyTrade?) async throws -> PromptPlan<FakeCache> {
        let tokens = Array(1000..<2000)
        let conversation = [ServeMessage(role: .system, text: "s"), ServeMessage(role: .user, text: "u")]
        return try await PromptCheckpointing.plan(
            store: rig.store, key: key, messages: conversation, tokens: tokens,
            render: { probe in
                guard probe.count == conversation.count + 2 else { throw CancellationError() }
                return Array(tokens[0..<996]) + [-1, -2]
            },
            newCache: { FakeCache() }, isTrimmable: { _ in false },
            restore: { _ in nil },
            prefill: { from, to, cache in cache.fed += tokens[from..<to] },
            snapshot: { cache in (FakeSnapshot(covered: cache.fed), snapshotMB * Self.mb, false) },
            cacheBytes: { _ in snapshotMB * Self.mb },
            replyTrade: trade)
    }

    /// A 20 MB snapshot with 10 MB free: short by exactly 10 MB = ceil(10,485,760 / 1,000) tokens.
    private static let shortfallTokens = Int((Double(10 * mb) / bytesPerToken).rounded(.up))

    @Test func theTradeHappensAndLowersBothTheReservationAndTheLimit() async throws {
        let rig = Rig()
        let trade = rig.trade(replyLimit: 32_000)
        let before = await rig.reserved
        let result = try await plan(rig, snapshotMB: 20, trade: trade)
        #expect(result.outcome.storedAt == [996])                                  // the snapshot was stored
        #expect(await rig.store.count == 1)
        #expect(trade.tradedTokens == Self.shortfallTokens)
        #expect(trade.replyLimit == 32_000 - Self.shortfallTokens)                  // maxTokens lowered by the right amount
        #expect(result.outcome.replyTokensTraded == Self.shortfallTokens)
        let after = await rig.reserved
        #expect(before - after == Int64(Double(Self.shortfallTokens) * Self.bytesPerToken))   // …and the reservation by the same
        #expect(rig.budget.usedBytes <= rig.budget.capacityBytes)                   // reserved total + snapshot ≤ capacity
    }

    @Test func noTradeWhenTheReplyLimitWouldFallBelowTheFloor() async throws {
        for limit in [Self.shortfallTokens + ReplyTrade.minimumReplyTokens - 1, 4_096, 8_192] {
            let rig = Rig()
            let trade = rig.trade(replyLimit: limit)
            let before = await rig.reserved
            let result = try await plan(rig, snapshotMB: 20, trade: trade)
            #expect(result.outcome.storedAt.isEmpty, "limit \(limit)")                 // skipped exactly as today
            #expect(trade.tradedTokens == 0 && trade.replyLimit == limit)
            let after = await rig.reserved
            #expect(after == before)                                            // nothing given back
            #expect(result.outcome.replyTokensTraded == 0)
        }
    }

    @Test func theFloorIsInclusive() async throws {
        let rig = Rig()
        let trade = rig.trade(replyLimit: Self.shortfallTokens + ReplyTrade.minimumReplyTokens)   // lands on exactly 8,192
        let result = try await plan(rig, snapshotMB: 20, trade: trade)
        #expect(result.outcome.storedAt == [996] && trade.replyLimit == ReplyTrade.minimumReplyTokens)
    }

    @Test func noTradeWhenThereIsAlreadyRoom() async throws {
        let rig = Rig()
        let trade = rig.trade(replyLimit: 32_000)
        let before = await rig.reserved
        let result = try await plan(rig, snapshotMB: 8, trade: trade)                  // 10 MB free ≥ 8 MB
        #expect(result.outcome.storedAt == [996])
        let after = await rig.reserved
        #expect(trade.tradedTokens == 0 && trade.replyLimit == 32_000 && after == before)
    }

    @Test func noTradeWhenRoomIsNotTheReasonAndNoTradeObjectChangesNothing() async throws {
        let over = Rig()
        let overTrade = over.trade(replyLimit: 32_000)
        let r1 = try await plan(over, snapshotMB: 50, trade: overTrade)                // over the store's 25 MB cap: not a room problem
        #expect(r1.outcome.storedAt.isEmpty && overTrade.tradedTokens == 0)
        let none = Rig()
        let r2 = try await plan(none, snapshotMB: 20, trade: nil)                      // no trade offered: today's behaviour
        let noneReserved = await none.reserved
        #expect(r2.outcome.storedAt.isEmpty && noneReserved == 30 * Self.mb)
    }

    @Test func reservedTotalNeverExceedsCapacityOrGrows() async throws {
        for (limit, snapshot) in [(32_000, 20), (32_000, 24), (12_000, 20), (32_000, 5), (9_000, 25)] as [(Int, Int64)] {
            let rig = Rig()
            let before = await rig.reserved
            _ = try await plan(rig, snapshotMB: snapshot, trade: rig.trade(replyLimit: limit))
            #expect(rig.budget.usedBytes <= rig.budget.capacityBytes, "limit \(limit) snapshot \(snapshot)")
            let after = await rig.reserved
            #expect(after <= before, "the reservation only ever goes down")
        }
    }

    @Test func shrinkOnlyEverLowers() {
        let budget = MemoryBudget(capacityBytes: 100 * Self.mb)
        let r = budget.tryReserve(bytes: 10 * Self.mb)!
        #expect(!budget.shrink(r, by: 0) && !budget.shrink(r, by: -5) && !budget.shrink(r, by: 11 * Self.mb))
        #expect(budget.usedBytes == 10 * Self.mb)
        #expect(budget.shrink(r, by: 4 * Self.mb) && budget.usedBytes == 6 * Self.mb)
    }

    @Test func aReplyReachingTheLoweredLimitReportsALengthStop() {
        let lowered = 32_000 - Self.shortfallTokens
        #expect(MLXChatBackend.hitReplyLimit(libraryHitLength: false, libraryGaveNoReason: true, stoppedByClient: false,
                                             completionTokens: lowered, replyLimit: lowered))
        // …and against the original limit the same reply would not have been called a length stop.
        #expect(!MLXChatBackend.hitReplyLimit(libraryHitLength: false, libraryGaveNoReason: true, stoppedByClient: false,
                                              completionTokens: lowered, replyLimit: 32_000))
        #expect(!MLXChatBackend.hitReplyLimit(libraryHitLength: false, libraryGaveNoReason: true, stoppedByClient: true,
                                              completionTokens: lowered, replyLimit: lowered))
        #expect(MLXChatBackend.hitReplyLimit(libraryHitLength: true, libraryGaveNoReason: false, stoppedByClient: false,
                                             completionTokens: 1, replyLimit: lowered))
    }
}
