import Foundation
import Testing
@testable import MLXUI

/// MT4-0 — an MLX chat send is refused before the load when the message (plus the history prefilled with it) cannot fit
/// the model's usable window; Apple Intelligence is not gated here; below the limit nothing changes.
@MainActor
@Suite(.serialized)
struct ChatSendAdmissionTests {
    final class OneShot: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { $0.yield("reply"); $0.finish() }
        }
    }

    private func llm(window: Int? = 4_096) -> ModelEntry {
        ModelEntry(
            id: "mlx-community--adm", family: "Q", displayName: "Qwen3-8B", paramSize: "", paramCountB: nil, modelType: .llm, source: .mlx,
            format: "mlx", platforms: ["macos"], minMacOSVersion: nil, hfRepo: "", hfModelId: "", ramGB: 4, downloadSizeGB: 1,
            contextWindow: window, license: nil, licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil,
            architecture: nil, languages: nil, lastUpdated: nil, taskTags: nil, benchmarks: nil, speedTokensPerSec: nil,
            speedHardware: nil, speedEstimated: nil, communityDownloads: nil, communityLikes: nil, variants: nil)
    }

    private func runner(model: ModelEntry, usable: Int?, dirs: inout [URL]) throws -> ModelRunner {
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("mt40-\(UUID().uuidString)")
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt40-m-\(UUID().uuidString)")
        for d in [store, modelDir] { try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true) }
        dirs += [store, modelDir]
        let r = ModelRunner()
        r.prepare(for: model, in: store)
        r.modelDirectoryForTesting = { _ in modelDir }
        r.streamingSessionForTesting = { OneShot() }
        r.usableTokensForTesting = { _ in usable }
        return r
    }

    private func settle(_ r: ModelRunner) async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while r.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    private func users(_ r: ModelRunner) -> Int {
        r.transcript.items.filter { if case .message(let m) = $0, m.role == .user { return true } else { return false } }.count
    }

    private let words = String(repeating: "word ", count: 4_000)         // ~20 KB, ~5,000 tokens

    // MARK: pure

    @Test func fitsDoesntFitAndUnknownWindow() {
        #expect(ChatContinueFit.admits(promptTokens: 3_000, window: 4_096))
        #expect(!ChatContinueFit.admits(promptTokens: 3_100, window: 4_096))      // margin 1,024
        #expect(ChatContinueFit.admits(promptTokens: 9_999_999, window: nil))      // unknown admits
    }

    @Test func historyIsCountedWithThePrompt() {
        let prompt = String(repeating: "a", count: 8_000)                          // 2,000 tokens
        #expect(ChatContinueFit.sendTokens(prompt: prompt, history: []) == 2_000)
        let withHistory = ChatContinueFit.sendTokens(prompt: prompt, history: [String(repeating: "b", count: 4_000)])
        #expect(withHistory == 3_000)
        #expect(ChatContinueFit.admits(promptTokens: 2_000, window: 4_096))
        #expect(!ChatContinueFit.admits(promptTokens: withHistory + 200, window: 4_096))
    }

    @Test func theSentenceNamesTheWindowAndBothWaysOut() {
        let s = ChatContinueFit.sendRefusal(promptTokens: 20_000, modelName: "Qwen3-8B", window: 29_000)
        #expect(s == "This message is about 20K tokens, more than Qwen3-8B can take at once (about 26K usable on this Mac). Attach it as a file or start a new chat.")
    }

    // MARK: runner

    @Test func aMessageTooBigIsRefusedBeforeAnythingChanges() async throws {
        var dirs: [URL] = []; defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let model = llm()
        let r = try runner(model: model, usable: 4_096, dirs: &dirs)
        let sent = await r.send(words, for: model)
        await settle(r)
        #expect(sent == false)
        #expect(users(r) == 0)
        #expect(r.isRunning == false)
        #expect(r.errorMessage?.hasSuffix("Attach it as a file or start a new chat.") == true)
        #expect((r.conversations.first { $0.id == r.activeConversationID }?.items.isEmpty ?? true))
    }

    @Test func belowTheLimitASendIsUnchanged() async throws {
        var dirs: [URL] = []; defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let model = llm()
        let r = try runner(model: model, usable: 4_096, dirs: &dirs)
        #expect(await r.send("hello there", for: model))
        await settle(r)
        #expect(users(r) == 1 && r.errorMessage == nil)
    }

    @Test func anUnknownWindowAdmitsTheSend() async throws {
        var dirs: [URL] = []; defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let model = llm(window: nil)
        let r = try runner(model: model, usable: nil, dirs: &dirs)
        #expect(await r.send(words, for: model))
        await settle(r)
        #expect(users(r) == 1 && r.errorMessage == nil)
    }

    @Test func theHistoryPushesALaterSendOver() async throws {
        var dirs: [URL] = []; defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let model = llm()
        let r = try runner(model: model, usable: 4_096, dirs: &dirs)
        let first = String(repeating: "a", count: 9_000)                      // ~2,250 tokens: fits alone
        #expect(await r.send(first, for: model)); await settle(r)
        #expect(users(r) == 1)
        let second = String(repeating: "b", count: 4_000)                     // ~1,000 more: fits alone, not with the history
        #expect(await r.send(second, for: model) == false)
        #expect(users(r) == 1)
        #expect(r.errorMessage != nil)
    }

    @Test func inlineAttachmentTextCountsAndAByReferenceFileDoesNot() async throws {
        var dirs: [URL] = []; defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let model = llm()
        let typed = String(repeating: "t", count: 8_000)                      // ~2,000 tokens alone: fits
        let small = FileManager.default.temporaryDirectory.appendingPathComponent("mt40-small-\(UUID().uuidString).txt")
        let big = FileManager.default.temporaryDirectory.appendingPathComponent("mt40-big-\(UUID().uuidString).txt")
        try String(repeating: "s", count: 5_000).write(to: small, atomically: true, encoding: .utf8)   // inlined: +1,250 tokens
        try String(repeating: "g", count: 200_000).write(to: big, atomically: true, encoding: .utf8)   // by reference
        dirs += [small, big]

        let inlineRunner = try runner(model: model, usable: 4_096, dirs: &dirs)
        #expect(await inlineRunner.send(typed, attachments: [small], for: model) == false)
        #expect(users(inlineRunner) == 0)

        let refRunner = try runner(model: model, usable: 4_096, dirs: &dirs)
        #expect(await refRunner.send(typed, attachments: [big], for: model))
        await settle(refRunner)
        #expect(users(refRunner) == 1 && refRunner.errorMessage == nil)
    }

    @Test func appleIntelligenceIsNotGatedHere() async throws {
        var dirs: [URL] = []; defer { dirs.forEach { try? FileManager.default.removeItem(at: $0); AppleFoundationAvailability.toolChatterOverride = nil } }
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("mt40-a-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        dirs.append(store)
        AppleFoundationAvailability.toolChatterOverride = AppleFoundationRouterTests.FakeChatter(script: { _, _ in [.snapshot("ok")] })
        let r = ModelRunner()
        r.prepare(for: AppleIntelligenceChat.entry, in: store)
        r.usableTokensForTesting = { _ in 100 }        // would refuse anything for an MLX model
        #expect(await r.send(words, for: AppleIntelligenceChat.entry))
        await settle(r)
        #expect(users(r) == 1)
    }

    @Test func continueStillWorksAndSharesTheHelper() async throws {
        var dirs: [URL] = []; defer { dirs.forEach { try? FileManager.default.removeItem(at: $0); AppleFoundationAvailability.toolChatterOverride = nil } }
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("mt40-c-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        dirs.append(store)
        AppleFoundationAvailability.toolChatterOverride = AppleFoundationRouterTests.FakeChatter(
            script: { _, _ in throw AppleFoundationToolError.contextWindowExceeded(limit: 8192, tokenCount: nil) })
        let r = ModelRunner()
        r.prepare(for: AppleIntelligenceChat.entry, in: store)
        await r.send("a long question", for: AppleIntelligenceChat.entry); await settle(r)
        #expect(r.contextNotice != nil)
        let model = llm(window: 32_768)
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt40-cm-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        dirs.append(modelDir)
        r.modelDirectoryForTesting = { _ in modelDir }
        r.streamingSessionForTesting = { OneShot() }
        r.usableTokensForTesting = { _ in 29_000 }
        #expect(await r.retryUnanswered(with: model, usableTokens: 29_000))
        await settle(r)
        #expect(users(r) == 1 && r.errorMessage == nil && r.contextNotice == nil)
    }
}
