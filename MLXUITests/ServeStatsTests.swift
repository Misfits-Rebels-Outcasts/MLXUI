import Testing
import Foundation
@testable import MLXUI

/// SC-3 — the serving numbers (gate E): measured once in `ServePipeline` for every dialect, aggregated by the pure
/// `ServeStats.summary`, and carrying no content (Serve rule 13).
struct ServeStatsTests {

    // MARK: golden summaries

    private func entry(model: String? = "m", at t: Double = 0, status: Int = 200, prompt: Int? = nil,
                       ttft: Double? = nil, tps: Double? = nil, reused: Int? = nil) -> RequestLogEntry {
        RequestLogEntry(time: Date(timeIntervalSince1970: t), userAgent: nil, model: model, promptTokens: prompt,
                        completionTokens: nil, durationSeconds: 1, status: status, toolsIgnored: false,
                        ttftSeconds: ttft, decodeTokensPerSecond: tps, promptTokensReused: reused)
    }

    @Test func noEntriesNoSummary() {
        #expect(ServeStats.summary(entries: []).isEmpty)
    }

    @Test func allRefusedHasCountsAndNoMedians() throws {
        let stats = ServeStats.summary(entries: [entry(at: 5, status: 503), entry(at: 9, status: 503), entry(at: 7, status: 503)])
        let m = try #require(stats["m"])
        #expect(m == ModelServeStats(requestCount: 3, unavailableCount: 3, p50TTFTSeconds: nil,
                                     p50DecodeTokensPerSecond: nil, promptTokensReusedShare: nil,
                                     lastRequestTime: Date(timeIntervalSince1970: 9)))
    }

    @Test func mixedModelsAreSummarisedSeparately() throws {
        let stats = ServeStats.summary(entries: [
            entry(model: "a", at: 1, prompt: 100, ttft: 0.5, tps: 40, reused: 50),
            entry(model: "b", at: 2, status: 503),
            entry(model: "a", at: 3, prompt: 300, ttft: 1.5, tps: 20, reused: 0),
            entry(model: "a", at: 4, status: 503),
            entry(model: nil, at: 8, status: 404),                     // names no model: belongs to none
            entry(model: "a", at: 6, prompt: 100, ttft: 9, tps: 30),   // cache n/a: not in the share
        ])
        #expect(Set(stats.keys) == ["a", "b"])
        #expect(stats["a"] == ModelServeStats(requestCount: 4, unavailableCount: 1, p50TTFTSeconds: 1.5,
                                              p50DecodeTokensPerSecond: 30, promptTokensReusedShare: 50.0 / 400.0,
                                              lastRequestTime: Date(timeIntervalSince1970: 6)))
        #expect(stats["b"] == ModelServeStats(requestCount: 1, unavailableCount: 1, p50TTFTSeconds: nil,
                                              p50DecodeTokensPerSecond: nil, promptTokensReusedShare: nil,
                                              lastRequestTime: Date(timeIntervalSince1970: 2)))
    }

    @Test func medianOfAnEvenCountIsTheMeanOfTheMiddleTwo() {
        #expect(ServeStats.median([4, 1, 3, 2]) == 2.5)
        #expect(ServeStats.median([7]) == 7)
        #expect(ServeStats.median([]) == nil)
    }

    // MARK: privacy

    /// The entry may hold exactly two strings — the client's user agent and the model id. Anything else could hold content.
    @Test func theEntryHasNoStringFieldBeyondUserAgentAndModel() {
        let probe = entry()
        var strings: [String] = []
        for child in Mirror(reflecting: probe).children {
            let type = String(describing: type(of: child.value))
            if type.contains("String") { strings.append(child.label ?? "?") }
        }
        #expect(Set(strings) == ["userAgent", "model"])
    }

    // MARK: measured in the pipeline — every dialect, streaming and not

    private final class Clock: @unchecked Sendable {
        private let lock = NSLock()
        private var t: Double = 1_000
        func now() -> Date { lock.lock(); defer { lock.unlock() }; return Date(timeIntervalSince1970: t) }
        func advance(_ s: Double) { lock.lock(); t += s; lock.unlock() }
    }

    /// Scripted: moves the injected clock, then emits. The load (if any) is reported the way a backend does, on usage.
    private final class TimedBackend: ServeBackend, @unchecked Sendable {
        let clock: Clock, loadSeconds: Double?, cacheApplied: Bool
        init(_ clock: Clock, loadSeconds: Double? = nil, cacheApplied: Bool = false) {
            self.clock = clock; self.loadSeconds = loadSeconds; self.cacheApplied = cacheApplied
        }
        /// Pull-driven, so the clock moves between the events the pipeline sees rather than before the first.
        func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> {
            let clock = clock, loadSeconds = loadSeconds, cacheApplied = cacheApplied
            let step = Counter()
            return AsyncThrowingStream {
                switch step.next() {
                case 0:
                    if let loadSeconds { clock.advance(loadSeconds) }
                    clock.advance(2)                                  // prefill
                    return .textDelta("secret")                      // first token: 2 s (+ load) after the request
                case 1:
                    clock.advance(4)                                  // decode
                    return .usage(ServeUsage(promptTokens: 100, completionTokens: 40, cachedTokens: cacheApplied ? 60 : 0,
                                             loadSeconds: loadSeconds, promptCacheApplied: cacheApplied))
                case 2: return .finish(.stop)
                default: return nil
                }
            }
        }
    }
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock(); private var n = -1
        func next() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
    }

    private struct Dialect: Sendable {
        let name: String, path: String, body: String
    }
    private static let model = "mlx-community/Qwen3-4B-4bit"
    private static let dialects = [
        Dialect(name: "chat", path: "/v1/chat/completions",
                body: #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]__S__}"#),
        Dialect(name: "messages", path: "/v1/messages",
                body: #"{"model":"\#(model)","max_tokens":16,"messages":[{"role":"user","content":"hi"}]__S__}"#),
        Dialect(name: "responses", path: "/v1/responses", body: #"{"model":"\#(model)","input":"hi"__S__}"#),
    ]
    private struct Case: Sendable, CustomTestStringConvertible {
        let dialect: Dialect, stream: Bool
        var testDescription: String { "\(dialect.name) stream=\(stream)" }
    }
    private static let cases = dialects.flatMap { d in [false, true].map { Case(dialect: d, stream: $0) } }

    private func run(_ c: Case, backend: (Clock) -> TimedBackend) async throws -> RequestLogEntry {
        let clock = Clock()
        let log = RequestLog()
        let b = backend(clock)
        let env = ServeEnvironment(
            servedModelIDs: { [Self.model] },
            installed: { InstalledModelIndex(entries: [
                .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: Self.model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
            appleFoundationReadiness: { nil }, created: 1, backendFor: { _ in b },
            queue: ServeQueue(), requestLog: log, keepAliveSeconds: 60, now: { clock.now() })
        let body = Data(c.dialect.body.replacingOccurrences(of: "__S__", with: c.stream ? #","stream":true"# : "").utf8)
        let head = ServeRequestHead(method: "POST", path: c.dialect.path, host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(body.count), userAgent: "t/1")
        let reply = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env, body: { body })
        if case .stream(_, _, let bytes) = reply { for await _ in bytes {} }
        return try #require(log.entries.last, "no log line")
    }

    @Test(arguments: cases)
    private func theNumbersAreFilledInEveryDialect(_ c: Case) async throws {
        let e = try await run(c) { TimedBackend($0, cacheApplied: true) }
        #expect(e.status == 200)
        #expect(e.ttftSeconds == 2)
        #expect(e.queueWaitSeconds == 0)
        #expect(e.loadWaitSeconds == nil)
        let rate = try #require(e.decodeTokensPerSecond)
        #expect(abs(rate - 10) < 1e-9)                                // 40 tokens over the 4 s after the first
        #expect(e.promptTokensReused == 60)
    }

    @Test(arguments: cases)
    private func loadWaitIsSetOnlyWhenTheRequestLoadedTheModel(_ c: Case) async throws {
        let loaded = try await run(c) { TimedBackend($0, loadSeconds: 7) }
        #expect(loaded.loadWaitSeconds == 7)
        #expect(loaded.ttftSeconds == 9)                              // what the client felt: load + prefill
        let warm = try await run(c) { TimedBackend($0) }
        #expect(warm.loadWaitSeconds == nil)
        #expect(warm.promptTokensReused == nil)                       // no cache in play: not applicable, not zero
    }

    @Test func aRefusedRequestHasNoGenerationNumbers() async throws {
        let log = RequestLog()
        let env = ServeEnvironment(servedModelIDs: { [] }, installed: { InstalledModelIndex(entries: []) },
                                   appleFoundationReadiness: { nil }, created: 1, backendFor: { _ in nil },
                                   queue: ServeQueue(), requestLog: log, keepAliveSeconds: 60)
        let body = Data(#"{"model":"nope","messages":[{"role":"user","content":"hi"}]}"#.utf8)
        let head = ServeRequestHead(method: "POST", path: "/v1/chat/completions", host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(body.count), userAgent: nil)
        _ = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env, body: { body })
        let e = try #require(log.entries.last)
        #expect(e.status == 404)
        #expect(e.ttftSeconds == nil && e.queueWaitSeconds == nil && e.decodeTokensPerSecond == nil
                && e.loadWaitSeconds == nil && e.promptTokensReused == nil)
    }
}
