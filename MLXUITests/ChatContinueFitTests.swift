import Foundation
import Testing
@testable import MLXUI

/// MT3-FIX-4 — "Continue with X" is offered only for a model whose window fits the unanswered message.
@MainActor
@Suite(.serialized)
struct ChatContinueFitTests {
    private func llm(_ id: String, _ name: String, window: Int = 32_768) -> ModelEntry {
        ModelEntry(
            id: id, family: "Q", displayName: name, paramSize: "", paramCountB: nil, modelType: .llm, source: .mlx, format: "mlx",
            platforms: ["macos"], minMacOSVersion: nil, hfRepo: "", hfModelId: "", ramGB: 4, downloadSizeGB: 1, contextWindow: window,
            license: nil, licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil, architecture: nil, languages: nil,
            lastUpdated: nil, taskTags: nil, benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil, speedEstimated: nil,
            communityDownloads: nil, communityLikes: nil, variants: nil)
    }

    // MARK: pure fit

    @Test func theReplyMarginIsTwoThousandButAtMostAQuarterOfASmallWindow() {
        #expect(ChatContinueFit.replyMargin(window: 32_768) == 2_048)
        #expect(ChatContinueFit.replyMargin(window: 4_096) == 1_024)
        #expect(ChatContinueFit.replyMargin(window: 0) == 0)
    }

    @Test func aPromptFitsWhenItAndTheMarginFitTheWindow() {
        #expect(ChatContinueFit.fits(promptTokens: 20_000, window: 29_000))
        #expect(!ChatContinueFit.fits(promptTokens: 555_000, window: 29_000))
        #expect(ChatContinueFit.fits(promptTokens: 30_720, window: 32_768))
        #expect(!ChatContinueFit.fits(promptTokens: 30_721, window: 32_768))
    }

    // MARK: banner

    @Test func theBannerOffersTheFirstModelThatFits() {
        let a = llm("a", "Alpha"), b = llm("b", "Beta")
        let banner = ChatBanner.contextExceeded(limit: 8192, installed: [b, a], installedToolCapableIDs: ["a", "b"],
                                                promptTokens: 20_000, windows: ["a": 29_000, "b": 29_000])
        #expect(banner.buttons.first == .continueWith(modelID: "a", name: "Alpha"))
        #expect(banner.text == "This chat is too long for Apple Intelligence (its window is 8K tokens) — continue with Alpha or start a new chat.")
    }

    @Test func aSmallerModelSortedFirstIsSkippedForABiggerOneThatFits() {
        let small = llm("a", "Alpha"), big = llm("b", "Beta")
        let banner = ChatBanner.contextExceeded(limit: 8192, installed: [small, big], installedToolCapableIDs: ["a", "b"],
                                                promptTokens: 20_000, windows: ["a": 8_000, "b": 29_000])
        #expect(banner.buttons.first == .continueWith(modelID: "b", name: "Beta"))
    }

    @Test func aToolCapableModelStillComesBeforeAnOtherOneBothFit() {
        let plain = llm("a", "Alpha"), tools = llm("b", "Beta")
        let banner = ChatBanner.contextExceeded(limit: nil, installed: [plain, tools], installedToolCapableIDs: ["b"],
                                                promptTokens: 100, windows: ["a": 29_000, "b": 29_000])
        #expect(banner.buttons.first == .continueWith(modelID: "b", name: "Beta"))
        // Neither tool-capable: today's fallback, first by name.
        #expect(ChatBanner.contextExceeded(limit: nil, installed: [tools, plain], installedToolCapableIDs: [], promptTokens: 100,
                                           windows: [:]).buttons.first == .continueWith(modelID: "a", name: "Alpha"))
    }

    @Test func whenNoModelFitsOnlyStartANewChatAndNoContinuePromise() {
        let banner = ChatBanner.contextExceeded(limit: 8192, installed: [llm("a", "Alpha"), llm("b", "Beta")],
                                                installedToolCapableIDs: ["a", "b"], promptTokens: 555_000,
                                                windows: ["a": 29_000, "b": 29_000])
        #expect(banner.buttons == [.startNewChat])
        #expect(banner.text == "This chat is too long for Apple Intelligence (its window is 8K tokens) — start a new chat.")
        #expect(!banner.text.contains("continue with"))
    }

    @Test func aModelWithoutAFactFallsBackToItsCatalogWindowAndAnUnknownSizeFiltersNothing() {
        let small = llm("a", "Alpha", window: 4_096), big = llm("b", "Beta", window: 32_768)
        #expect(ChatBanner.contextExceeded(limit: nil, installed: [small, big], installedToolCapableIDs: ["a", "b"],
                                           promptTokens: 20_000).buttons.first == .continueWith(modelID: "b", name: "Beta"))
        #expect(ChatBanner.contextExceeded(limit: nil, installed: [small, big], installedToolCapableIDs: ["a", "b"])
            .buttons.first == .continueWith(modelID: "a", name: "Alpha"))
    }

    @Test func theFailurePathUsesTheSameFunction() {
        let banner = ChatBanner.failure(.contextWindowExceeded(limit: 8192, tokenCount: nil), installed: [llm("a", "Alpha")],
                                        installedToolCapableIDs: ["a"], promptTokens: 555_000, windows: ["a": 29_000])
        #expect(banner.buttons == [.startNewChat])
    }

    // MARK: runner

    private func appleOverflowRunner(prompt: String, store: URL) async -> ModelRunner {
        let chatter = AppleFoundationRouterTests.FakeChatter(script: { _, _ in throw AppleFoundationToolError.contextWindowExceeded(limit: 8192, tokenCount: nil) })
        AppleFoundationAvailability.toolChatterOverride = chatter
        let runner = ModelRunner()
        runner.prepare(for: AppleIntelligenceChat.entry, in: store)
        await runner.send(prompt, for: AppleIntelligenceChat.entry)
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while runner.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(10)) }
        return runner
    }

    private func userCount(_ runner: ModelRunner) -> Int {
        runner.transcript.items.filter { if case .message(let m) = $0, m.role == .user { return true } else { return false } }.count
    }

    private func savedUserCount(_ runner: ModelRunner) -> Int {
        (runner.conversations.first { $0.id == runner.activeConversationID }?.items ?? [])
            .filter { if case .message(let m) = $0, m.role == .user { return true } else { return false } }.count
    }

    final class OneShot: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { $0.yield("mlx reply"); $0.finish() }
        }
    }

    @Test func retryRefusesAMessageThatDoesNotFitAndLeavesEverythingIntact() async throws {
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("mt3f-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: store); AppleFoundationAvailability.toolChatterOverride = nil }
        let prompt = String(repeating: "word ", count: 20_000)          // ~100 KB, tens of thousands of tokens
        let runner = await appleOverflowRunner(prompt: prompt, store: store)
        #expect(runner.contextNotice != nil)
        let before = (userCount(runner), savedUserCount(runner))
        #expect(before == (1, 1))

        let small = llm("m", "Qwen3-8B", window: 8_192)
        #expect(await runner.retryUnanswered(with: small, usableTokens: 8_192) == false)
        let sentence = try #require(runner.errorMessage)
        #expect(sentence.contains("Qwen3-8B fits about 6K"))
        #expect(sentence.hasSuffix("Start a new chat or attach it as a file."))
        #expect(userCount(runner) == 1 && savedUserCount(runner) == 1)
        #expect(runner.contextNotice != nil)
    }

    @Test func retryStillWorksWhenTheMessageFits() async throws {
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("mt3f-\(UUID().uuidString)")
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt3f-m-\(UUID().uuidString)")
        for d in [store, modelDir] { try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true) }
        defer { try? FileManager.default.removeItem(at: store); try? FileManager.default.removeItem(at: modelDir)
                AppleFoundationAvailability.toolChatterOverride = nil }
        let runner = await appleOverflowRunner(prompt: "a very long question", store: store)
        runner.modelDirectoryForTesting = { _ in modelDir }
        runner.streamingSessionForTesting = { OneShot() }
        #expect(await runner.retryUnanswered(with: llm("m", "M", window: 4_096), usableTokens: 29_000))
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while runner.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(10)) }
        #expect(userCount(runner) == 1 && savedUserCount(runner) == 1)
        #expect(runner.contextNotice == nil && runner.errorMessage == nil)
    }
}
