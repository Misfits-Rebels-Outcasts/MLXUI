import Testing
import Foundation
@testable import MLXUI

/// PC-3 — what the prompt cache shows: `usage.prompt_tokens_details.cached_tokens`, "reused N" and
/// "cache fell back" in the request log, the pane's line, and the counters behind it.
struct PromptCacheSurfaceTests {

    // MARK: usage

    @Test func usageDefaultsToNoCacheActivity() {
        let usage = ServeUsage(promptTokens: 10, completionTokens: 2)
        #expect(usage.cachedTokens == 0)
        #expect(usage.cacheFellBack == false)
        #expect(usage.totalTokens == 12)                         // cached tokens are part of the prompt, not extra
    }

    @Test func usageTakesTheCacheOutcomeAndAFailedCachePathIsFlagged() {
        var outcome = PromptCacheOutcome(promptTokens: 2400)
        outcome.reusedTokens = 2100
        let hit = ServeUsage(promptTokens: 2400, completionTokens: 5, outcome: outcome)
        #expect(hit.cachedTokens == 2100 && hit.cacheFellBack == false)
        #expect(hit.promptTokens == 2400)                        // still the full prompt (PC-2 rule 7)

        let failed = PromptCacheOutcome(bypass: .failed, promptTokens: 2400)
        let usage = ServeUsage(promptTokens: 2400, completionTokens: 5, outcome: failed)
        #expect(usage.cachedTokens == 0 && usage.cacheFellBack == true)

        let off = ServeUsage(promptTokens: 2400, completionTokens: 5,
                             outcome: PromptCacheOutcome(bypass: .disabled, promptTokens: 2400))
        #expect(off.cacheFellBack == false)                      // a switch that is off is not a failure
    }

    // MARK: wire

    private func usageObject(_ data: Data) throws -> [String: Any] {
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try #require(object["usage"] as? [String: Any])
    }

    @Test func theUsageChunkCarriesCachedTokensOnlyWhenSomethingWasReused() throws {
        let miss = OpenAIChatDialect.usageChunk(id: "i", created: 1, model: "m",
                                                usage: ServeUsage(promptTokens: 1, completionTokens: 2))
        // byte-identical to the S1 shape (the existing golden in ServeChatTests pins the same string)
        #expect(String(decoding: miss, as: UTF8.self)
            == #"{"choices":[],"created":1,"id":"i","model":"m","object":"chat.completion.chunk","usage":{"completion_tokens":2,"prompt_tokens":1,"total_tokens":3}}"#)
        let hit = OpenAIChatDialect.usageChunk(id: "i", created: 1, model: "m",
                                               usage: ServeUsage(promptTokens: 2400, completionTokens: 2, cachedTokens: 2100))
        let usage = try usageObject(hit)
        #expect(usage["prompt_tokens"] as? Int == 2400)
        #expect((usage["prompt_tokens_details"] as? [String: Int]) == ["cached_tokens": 2100])
        #expect(usage["total_tokens"] as? Int == 2402)
    }

    // MARK: route → log → line

    private final class UsageBackend: ServeBackend, @unchecked Sendable {
        let usage: ServeUsage
        init(_ usage: ServeUsage) { self.usage = usage }
        func countPromptTokens(_ request: ServeRequest) async throws -> Int? { nil }
        func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> {
            AsyncThrowingStream { c in
                c.yield(.textDelta("SECRET-REPLY-TEXT")); c.yield(.usage(usage)); c.yield(.finish(.stop)); c.finish()
            }
        }
    }

    private func roundTrip(_ usage: ServeUsage) async throws -> (usage: [String: Any], entry: RequestLogEntry) {
        let model = "mlx-community/Qwen3-4B-4bit"
        let log = RequestLog()
        let backend = UsageBackend(usage)
        let environment = ServeEnvironment(
            servedModelIDs: { [model] },
            installed: { InstalledModelIndex(entries: [.init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
            appleFoundationReadiness: { nil }, created: 1, backendFor: { _ in backend },
            queue: ServeQueue(), requestLog: log, keepAliveSeconds: 1)
        let body = Data(#"{"model":"\#(model)","messages":[{"role":"user","content":"SECRET-PROMPT-TEXT"}]}"#.utf8)
        let head = ServeRequestHead(method: "POST", path: "/v1/chat/completions", host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(body.count), userAgent: "t/1")
        let reply = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: environment, body: { body })
        guard case .full(let response) = reply else { Issue.record("expected a full response"); throw CancellationError() }
        return (try usageObject(response.body), try #require(log.entries.last))
    }

    @Test func aCachedReplyShowsInTheResponseAndTheRequestLog() async throws {
        let (wire, entry) = try await roundTrip(ServeUsage(promptTokens: 2400, completionTokens: 5, cachedTokens: 2100))
        #expect((wire["prompt_tokens_details"] as? [String: Int]) == ["cached_tokens": 2100])
        #expect(entry.cachedTokens == 2100)
        #expect(entry.cacheFellBack == false)
        let line = RequestLogList.line(entry)
        #expect(line.contains("reused 2100"))
        #expect(!line.contains("fell back"))
        #expect(line.contains("2400→5 tok"))
    }

    @Test func aMissLooksExactlyLikeItDidBeforeTheCacheExisted() async throws {
        let (wire, entry) = try await roundTrip(ServeUsage(promptTokens: 12, completionTokens: 3))
        #expect(wire["prompt_tokens_details"] == nil)
        #expect(Set(wire.keys) == ["prompt_tokens", "completion_tokens", "total_tokens"])
        #expect(!RequestLogList.line(entry).contains("reused"))
    }

    @Test func aFailedCachePathIsVisibleInTheLogAndNotMistakenForNoHit() async throws {
        let (wire, entry) = try await roundTrip(ServeUsage(promptTokens: 2400, completionTokens: 5, cacheFellBack: true))
        #expect(wire["prompt_tokens_details"] == nil)            // nothing on the wire: the client got a normal answer
        #expect(entry.cacheFellBack == true)
        #expect(RequestLogList.line(entry).contains("cache fell back"))
        #expect(!RequestLogList.line(entry).contains("reused"))
    }

    @Test func theLogLineCarriesCountsNotContent() async throws {
        let (_, entry) = try await roundTrip(ServeUsage(promptTokens: 2400, completionTokens: 5, cachedTokens: 2100))
        let line = RequestLogList.line(entry)
        #expect(!line.contains("SECRET-PROMPT-TEXT"))             // rule 9: counts only, never the prompt
        #expect(!line.contains("SECRET-REPLY-TEXT"))                              // …nor the reply
    }

    // MARK: counters and the pane line

    @Test func theCountersSeparateHitsMissesFallbacksAndBypasses() {
        let counters = PromptCacheCounters()
        var hit = PromptCacheOutcome(promptTokens: 2400); hit.reusedTokens = 2100
        counters.record(hit)                                                   // a hit
        counters.record(PromptCacheOutcome(promptTokens: 2400))                // a miss (cache ran, nothing reused)
        counters.record(PromptCacheOutcome(bypass: .failed, promptTokens: 9))  // a fallback
        counters.record(PromptCacheOutcome(bypass: .modelNotTested, promptTokens: 9))   // bypassed: not counted
        counters.record(PromptCacheOutcome(bypass: .disabled, promptTokens: 9))
        #expect(counters.snapshot == .init(requests: 3, hits: 1, fallbacks: 1, reusedTokens: 2100))
        counters.reset()
        #expect(counters.snapshot == .init())
    }

    @Test func theSummaryLineSaysSnapshotsSizeAndFallbacks() {
        #expect(PromptCacheSummary.line(snapshots: 0, bytes: 0, fallbacks: 0) == "Prompt cache: 0 snapshots, 0.0 GB")
        #expect(PromptCacheSummary.line(snapshots: 1, bytes: 1_610_612_736, fallbacks: 0) == "Prompt cache: 1 snapshot, 1.5 GB")
        #expect(PromptCacheSummary.line(snapshots: 3, bytes: 3_221_225_472, fallbacks: 2)
                == "Prompt cache: 3 snapshots, 3.0 GB, 2 fell back to no cache")
    }
}
