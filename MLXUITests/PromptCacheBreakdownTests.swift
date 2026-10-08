import Testing
import Foundation
@testable import MLXUI

/// PC-3b — where does a cached first token go? A **measurement**, not a gate: enabled by
/// `MLXUI_TEST_PC_DIR` *and* `MLXUI_TEST_PC_BREAKDOWN=1`. Run it from the command line in a Release
/// build (`TEST_RUNNER_…`, see the journal) — under Xcode's debugger, or in Debug, the Swift-side
/// stages are far slower.
///
/// The prompt is the smoke helper's (`pcsmoke.py once`: 190 lines ≈ 8K tokens). It goes through
/// `MLXChatBackend` directly (no HTTP, no queue), plus the route's **pre-count** render
/// (`countPromptTokens`, S2-6), which `generate()` does not include. Prints `PCBREAK` lines.
@Suite(.serialized)
struct PromptCacheBreakdownTests {
    static let dir = ProcessInfo.processInfo.environment["MLXUI_TEST_PC_DIR"]
    static let on = dir != nil && ProcessInfo.processInfo.environment["MLXUI_TEST_PC_BREAKDOWN"] == "1"

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var value: PromptCacheOutcome?
        func set(_ v: PromptCacheOutcome) { lock.lock(); value = v; lock.unlock() }
        func get() -> PromptCacheOutcome? { lock.lock(); defer { lock.unlock() }; return value }
    }

    private static func facts(_ lines: Int) -> String {
        (0..<lines).map { i in
            "Fact \(i): the harbour of town \(i * 7 % 101) opens at \(6 + i % 5) and closes at \(17 + i % 4) on day \(i % 7) of every week, and the ferry master is called Mr \(["Alder", "Birch", "Cedar", "Dunn", "Elm"][i % 5])\(i)."
        }.joined(separator: "\n")
    }

    @Test(.enabled(if: PromptCacheBreakdownTests.on))
    func aCachedFirstTokenBreakdown() async throws {
        let directory = URL(fileURLWithPath: try #require(Self.dir))
        let backend = MLXChatBackend(directory: directory, pool: ModelContainerPool(budgetBytes: 40 << 30))
        await PromptCacheShared.store.clear()
        let system = ServeMessage(role: .system, text: "You are a precise assistant. Use only the facts below.\n" + Self.facts(190))
        func request(_ question: String) -> ServeRequest {
            var r = ServeRequest(model: "m", messages: [system, ServeMessage(role: .user, text: question)])
            r.temperature = 0; r.maxTokens = 16
            return r
        }
        func f(_ x: Double) -> String { String(format: "%.3f", x) }

        func preCount(_ label: String, _ request: ServeRequest) async throws {
            let start = ContinuousClock.now
            let n = try await backend.countPromptTokens(request)
            print("PCBREAK pre-count \(label): \(f(MLXChatBackend.seconds(ContinuousClock.now - start))) s (\(n ?? -1) tokens)")
        }
        func ttft(_ label: String, _ request: ServeRequest, cache: Bool) async throws {
            PromptCache.overrideForTesting = cache
            defer { PromptCache.overrideForTesting = nil }
            let box = Box()
            var backend = backend
            backend.promptCacheObserver = { box.set($0) }
            let start = ContinuousClock.now
            var first: Double?
            for try await event in backend.generate(request) {
                if case .textDelta = event, first == nil { first = MLXChatBackend.seconds(ContinuousClock.now - start) }
            }
            let o = box.get()
            let t = o?.timings ?? PromptCacheTimings()
            let ttft = first ?? .nan
            print("PCBREAK \(label): TTFT \(f(ttft)) s | prompt \(o?.promptTokens ?? -1) reused \(o?.reusedTokens ?? -1) stored \(o?.storedAt ?? []) | container \(f(t.container)) render \(f(t.render)) probes \(f(t.probes)) lookup \(f(t.lookup)) restore(copy+trim+eval) \(f(t.restore)) prefill \(f(t.prefill)) snapshot \(f(t.snapshot)) | tail+first-token \(f(ttft - t.total))")
        }

        let q1 = request("Which town has the earliest harbour opening? Answer in one sentence.")
        try await preCount("cold (first in process: reads tokenizer.json)", q1)
        try await preCount("warm", q1)
        try await preCount("warm", q1)
        try await ttft("warm-up (cache off; loads weights, compiles kernels)", request("Say hi."), cache: false)
        try await ttft("miss (1st, cache on)", q1, cache: true)
        for i in 1...3 { try await ttft("same prompt, hit #\(i)", q1, cache: true) }
        try await ttft("new session, same system prompt (system-prefix hit)",
                       request("And which ferry master works on day 3? Answer in one sentence."), cache: true)
        try await ttft("fresh, cache off (reference)", q1, cache: false)
    }

    /// The same prompt over the **real HTTP path** (FlyingFox, request guard, route, pre-count,
    /// queue, SSE), streamed with URLSession — what `pcsmoke.py` sees. Time to the first
    /// `delta.content`, with the route's own pre-count included.
    @MainActor
    @Test(.enabled(if: PromptCacheBreakdownTests.on))
    func anHTTPFirstTokenBreakdown() async throws {
        let directory = URL(fileURLWithPath: try #require(Self.dir))
        let modelID = "mlx-community/Qwen3-8B-4bit"
        let backend = MLXChatBackend(directory: directory, pool: ModelContainerPool(budgetBytes: 40 << 30))
        await PromptCacheShared.store.clear()
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let log = RequestLog()
        let environment = ServeEnvironment(
            servedModelIDs: { [modelID] },
            installed: { InstalledModelIndex(entries: [.init(id: "mlx-community--Qwen3-8B-4bit", hfModelId: modelID, kind: .llm, ramGB: 6.75, downloadSizeGB: 4.3)]) },
            appleFoundationReadiness: { nil }, created: 1, backendFor: { _ in backend },
            queue: ServeQueue(), requestLog: log, keepAliveSeconds: 15)
        let server = LocalServer(settings: ServeSettings(servedModelIDs: [modelID]), defaults: nil, listenPort: 0, environment: environment)
        await server.start()
        defer { Task { await server.stop() } }
        guard case .running(let port) = server.status else { Issue.record("server not running: \(server.status)"); return }

        let system = "You are a precise assistant. Use only the facts below.\n" + Self.facts(190)
        func send(_ label: String, _ question: String) async throws {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let body: [String: Any] = ["model": modelID, "stream": true, "temperature": 0, "max_tokens": 16,
                                       "stream_options": ["include_usage": true],
                                       "messages": [["role": "system", "content": system], ["role": "user", "content": question]]]
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            let start = ContinuousClock.now
            let (bytes, _) = try await URLSession.shared.bytes(for: request)
            let head = MLXChatBackend.seconds(ContinuousClock.now - start)
            var firstContent: Double?
            var cached = 0, prompt = 0
            for try await line in bytes.lines where line.hasPrefix("data:") && line != "data: [DONE]" {
                guard let object = try? JSONSerialization.jsonObject(with: Data(line.dropFirst(5).utf8)) as? [String: Any] else { continue }
                if let choice = (object["choices"] as? [[String: Any]])?.first,
                   let content = (choice["delta"] as? [String: Any])?["content"] as? String, !content.isEmpty, firstContent == nil {
                    firstContent = MLXChatBackend.seconds(ContinuousClock.now - start)
                }
                if let usage = object["usage"] as? [String: Any] {
                    prompt = usage["prompt_tokens"] as? Int ?? 0
                    cached = (usage["prompt_tokens_details"] as? [String: Int])?["cached_tokens"] ?? 0
                }
            }
            print("PCBREAK HTTP \(label): response head \(String(format: "%.3f", head)) s, first content \(String(format: "%.3f", firstContent ?? .nan)) s | prompt_tokens \(prompt) cached_tokens \(cached)")
        }
        try await send("warm-up (weights load, kernels)", "Say hi.")
        try await send("miss (1st)", "Which town has the earliest harbour opening? Answer in one sentence.")
        for i in 1...3 { try await send("same prompt, hit #\(i)", "Which town has the earliest harbour opening? Answer in one sentence.") }
        try await send("new session, same system prompt", "And which ferry master works on day 3? Answer in one sentence.")
        print("PCBREAK HTTP request log: " + log.entries.map { RequestLogList.line($0).split(separator: " ", omittingEmptySubsequences: true).suffix(8).joined(separator: " ") }.joined(separator: " || "))
    }
}
