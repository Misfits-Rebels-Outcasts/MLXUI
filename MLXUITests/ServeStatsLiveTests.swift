import Testing
import Foundation
@testable import MLXUI

/// SC-3b live check — the new numbers on a real model, through the real route, pipeline, queue and `MLXChatBackend`.
/// **Gated on env**, one process (cold first), run the host unsandboxed:
///
///     TEST_RUNNER_MLXUI_TEST_SC3_DIR=<installed checkpoint folder>
/// Prints `SC3` lines; asserts only the shape (non-zero queue wait, load wait cold-only, reuse on a repeat).
@Suite(.serialized)
struct ServeStatsLiveTests {
    static let dir = ProcessInfo.processInfo.environment["MLXUI_TEST_SC3_DIR"]
    private static let model = "mlx-community/Qwen3-8B-4bit"

    private static func facts(_ lines: Int) -> String {
        (0..<lines).map { i in
            "Fact \(i): the harbour of town \(i * 7 % 101) opens at \(6 + i % 5) and closes at \(17 + i % 4) on day \(i % 7) of every week, and the ferry master is called Mr \(["Alder", "Birch", "Cedar", "Dunn", "Elm"][i % 5])\(i)."
        }.joined(separator: "\n")
    }

    private static func send(_ prompt: String, env: ServeEnvironment) async {
        let escaped = String(data: try! JSONEncoder().encode(prompt), encoding: .utf8)!
        let body = Data(#"{"model":"\#(model)","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":\#(escaped)}]}"#.utf8)
        let head = ServeRequestHead(method: "POST", path: "/v1/chat/completions", host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(body.count), userAgent: "sc3-live")
        _ = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env, body: { body })
    }

    private static func line(_ label: String, _ e: RequestLogEntry) -> String {
        func f(_ v: Double?) -> String { v.map { String(format: "%.3f", $0) } ?? "nil" }
        return "SC3 \(label): status=\(e.status) prompt=\(e.promptTokens.map(String.init) ?? "nil") ttft=\(f(e.ttftSeconds)) queue=\(f(e.queueWaitSeconds)) load=\(f(e.loadWaitSeconds)) tps=\(f(e.decodeTokensPerSecond)) reused=\(e.promptTokensReused.map(String.init) ?? "nil")"
    }

    @Test(.enabled(if: ServeStatsLiveTests.dir != nil)) func theNumbersOnARealModel() async throws {
        let url = URL(fileURLWithPath: try #require(Self.dir))
        PromptCache.overrideForTesting = true
        defer { PromptCache.overrideForTesting = nil }
        await PromptCacheShared.store.clear()
        let backend = MLXChatBackend(directory: url, pool: ModelContainerPool(budgetBytes: 40 << 30))
        let log = RequestLog()
        let env = ServeEnvironment(
            servedModelIDs: { [Self.model] },
            installed: { InstalledModelIndex(entries: [.init(id: "mlx-community--Qwen3-8B-4bit", hfModelId: Self.model,
                                                             kind: .llm, ramGB: 5, downloadSizeGB: 4.5)]) },
            appleFoundationReadiness: { nil }, created: 1, backendFor: { _ in backend },
            queue: ServeQueue(), requestLog: log, keepAliveSeconds: 60)

        // 1–2: cold, then warm.
        await Self.send("Name three rivers in Europe.", env: env)
        await Self.send("Name three lakes in Africa.", env: env)
        // 3: two concurrent requests; the second waits its turn.
        async let a: Void = Self.send("Write one sentence about the sea. " + Self.facts(40), env: env)
        try await Task.sleep(for: .milliseconds(150))
        async let b: Void = Self.send("Write one sentence about the sky. " + Self.facts(41), env: env)
        _ = await (a, b)
        // 4: the same long prompt twice.
        let long = Self.facts(190) + "\nHow many facts are listed? Answer in one word."
        await Self.send(long, env: env)
        await Self.send(long, env: env)

        let e = log.entries
        #expect(e.count == 6)
        for (i, name) in ["cold", "warm", "concurrent-1", "concurrent-2", "long-1", "long-2"].enumerated() where i < e.count {
            print(Self.line(name, e[i]))
        }
        guard e.count == 6 else { return }
        #expect(e[0].loadWaitSeconds != nil)
        #expect(e[1].loadWaitSeconds == nil)
        #expect((e[2].queueWaitSeconds ?? 0) + (e[3].queueWaitSeconds ?? 0) > 0.05)
        #expect((e[5].promptTokensReused ?? 0) > 0)
        print("SC3 summary:", ServeStats.summary(entries: e))
    }
}
