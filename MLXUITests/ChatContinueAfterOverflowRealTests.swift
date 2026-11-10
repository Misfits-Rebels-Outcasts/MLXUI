import Foundation
import Testing
@testable import MLXUI

/// MT3 proof of smoke step 12(a), owner-run: a ~20K-token prompt really overflows Apple Intelligence, and "Continue with
/// <MLX model>" sends the unanswered message once and gets one answer. REAL: Apple Intelligence (the overflow leg, through the
/// app's `AppleFoundationRouter` and `ModelRunner`). SCRIPTED: the MLX leg (`streamingSessionForTesting`), because this test host
/// is sandboxed and cannot read the installed weights; what is proven is the hand-over (transcript, saved conversation, notice),
/// not MLX inference. Needs Apple Intelligence ready and `TEST_RUNNER_MLXUI_TEST_REAL_AFM=1`.
@MainActor
@Suite(.serialized)
struct ChatContinueAfterOverflowRealTests {
    final class OneShot: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { $0.yield("mlx reply"); $0.finish() }
        }
    }

    private func settle(_ runner: ModelRunner) async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(60))
        while runner.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(20)) }
    }

    @Test(.enabled(if: AppleFoundationRouterTests.realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realAnOverflowOnAppleIntelligenceContinuesOnAnMLXModelOnceAndAnsweredOnce() async throws {
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("mt3c-\(UUID().uuidString)")
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt3c-m-\(UUID().uuidString)")
        for d in [store, modelDir] { try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true) }
        defer { try? FileManager.default.removeItem(at: store); try? FileManager.default.removeItem(at: modelDir)
                AppleFoundationAvailability.toolChatterOverride = nil }
        AppleFoundationAvailability.toolChatterOverride = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let runner = ModelRunner()
        runner.prepare(for: AppleIntelligenceChat.entry, in: store)

        let prompt = "Summarize: " + String(repeating: "The quick brown fox jumps over the lazy dog. Chapter marker: LONGFILE-OK.\n", count: 1_200)
        // ~90K characters, about 20K tokens
        await runner.send(prompt, for: AppleIntelligenceChat.entry)
        await settle(runner)
        print("MT3-CONTINUE overflow: notice limit", runner.contextNotice?.limit as Any, "errorMessage", runner.errorMessage ?? "nil")
        #expect(runner.contextNotice != nil)
        #expect(runner.errorMessage == nil)

        let mlx = ModelEntry(
            id: "mlx-community--smoke", family: "Q", displayName: "Qwen3-8B", paramSize: "8B", paramCountB: 8, modelType: .llm,
            source: .mlx, format: "mlx", platforms: ["macos"], minMacOSVersion: nil, hfRepo: "", hfModelId: "", ramGB: 6,
            downloadSizeGB: 5, contextWindow: 32_768, license: nil, licenseUrl: nil, description: nil, summary: nil,
            descriptionSource: nil, architecture: "Qwen3ForCausalLM", languages: nil, lastUpdated: nil, taskTags: nil,
            benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil, speedEstimated: nil, communityDownloads: nil,
            communityLikes: nil, variants: nil)
        runner.modelDirectoryForTesting = { _ in modelDir }
        runner.streamingSessionForTesting = { OneShot() }
        #expect(await runner.retryUnanswered(with: mlx))
        await settle(runner)

        let users = runner.transcript.items.compactMap { item -> String? in
            if case .message(let m) = item, m.role == .user { return m.text } else { return nil }
        }
        let answers = runner.transcript.items.compactMap { item -> String? in
            if case .message(let m) = item, m.role == .assistant { return m.text } else { return nil }
        }
        let saved = runner.conversations.first { $0.id == runner.activeConversationID }
        let savedUsers = saved?.items.filter { if case .message(let m) = $0, m.role == .user { return true } else { return false } }.count
        print("MT3-CONTINUE after: users", users.count, "answers", answers, "saved users", savedUsers as Any,
              "errorMessage", runner.errorMessage ?? "nil", "notice", runner.contextNotice == nil ? "cleared" : "set",
              "model", saved?.modelID as Any)
        #expect(users.count == 1 && users.first == prompt.trimmingCharacters(in: .whitespacesAndNewlines))
        #expect(answers == ["mlx reply"])
        #expect(savedUsers == 1)
        #expect(runner.errorMessage == nil)
        #expect(runner.contextNotice == nil)
        #expect(saved?.modelID == mlx.id)
    }
}
