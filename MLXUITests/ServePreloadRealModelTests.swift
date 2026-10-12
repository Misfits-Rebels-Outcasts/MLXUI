import Testing
import Foundation
@testable import MLXUI

/// SC-1 step 5 — what a preload buys on a real model: the first request's time to first token, cold (the pool loads
/// it) against preloaded (`ModelContainerPool.preload` first), through the real `MLXChatBackend`. **Gated on env**,
/// one mode per process so MLX starts clean (pass as `TEST_RUNNER_<NAME>=…`; run the host unsandboxed):
///
///     MLXUI_TEST_SC1_DIR=<installed checkpoint folder>   MLXUI_TEST_SC1_MODE=cold|preload
struct ServePreloadRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let dir = env["MLXUI_TEST_SC1_DIR"]
    static let mode = env["MLXUI_TEST_SC1_MODE"] ?? "cold"

    private static func firstRequest(_ backend: MLXChatBackend) async throws -> (ttft: Double, total: Double) {
        PromptCache.overrideForTesting = false          // isolate the load from prefix reuse
        defer { PromptCache.overrideForTesting = nil }
        var request = ServeRequest(model: "m", messages: [ServeMessage(role: .user, text: "Name three rivers in Europe.")])
        request.temperature = 0
        request.maxTokens = 24
        let start = ContinuousClock.now
        var first: Double?
        for try await event in backend.generate(request) {
            switch event {
            case .textDelta, .reasoningDelta:
                if first == nil { first = seconds(ContinuousClock.now - start) }
            case .error(let error): Issue.record("stream error: \(error)")
            default: break
            }
        }
        return (first ?? -1, seconds(ContinuousClock.now - start))
    }

    private static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }

    @Test(.enabled(if: dir != nil)) func firstRequestTTFT() async throws {
        let url = URL(fileURLWithPath: try #require(Self.dir))
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let backend = MLXChatBackend(directory: url, pool: pool)
        var preloadSeconds = 0.0
        if Self.mode == "preload" {
            let start = ContinuousClock.now
            let state = await pool.preload(directory: url)
            preloadSeconds = Self.seconds(ContinuousClock.now - start)
            #expect(state == .ready)
        }
        let first = try await Self.firstRequest(backend)
        let second = try await Self.firstRequest(backend)
        print("SC1 mode=\(Self.mode) preload=\(String(format: "%.2f", preloadSeconds))s first-request TTFT=\(String(format: "%.2f", first.ttft))s total=\(String(format: "%.2f", first.total))s | second TTFT=\(String(format: "%.2f", second.ttft))s")
    }
}
