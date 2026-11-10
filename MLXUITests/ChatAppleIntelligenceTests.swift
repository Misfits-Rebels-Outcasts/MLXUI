import Foundation
import Testing
@testable import MLXUI

/// MT3-3 — Apple Intelligence in the Chat page: the model menu row, the default router (ruling C), the banners, and the runner
/// wiring through `AppleFoundationRouter` (fixture chatter, no device).
@MainActor
@Suite(.serialized)
struct ChatAppleIntelligenceTests {

    private func llm(_ id: String, _ name: String, ram: Double, architecture: String? = "Qwen3ForCausalLM") -> ModelEntry {
        var e = makeEntry(id: id, displayName: name, modelType: .llm, ramGB: ram)
        e = ModelEntry(
            id: e.id, family: e.family, displayName: e.displayName, paramSize: e.paramSize, paramCountB: e.paramCountB,
            modelType: .llm, source: e.source, format: e.format, platforms: e.platforms, minMacOSVersion: nil,
            hfRepo: e.hfRepo, hfModelId: e.hfModelId, ramGB: ram, downloadSizeGB: e.downloadSizeGB, contextWindow: 4096,
            license: nil, licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil, architecture: architecture,
            languages: nil, lastUpdated: nil, taskTags: nil, benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil,
            speedEstimated: nil, communityDownloads: nil, communityLikes: nil, variants: nil)
        return e
    }

    // MARK: menu

    @Test func readyAppleIntelligenceIsListedFirstAndChooseable() {
        let installed = [llm("a", "Alpha", ram: 4), llm("b", "Beta", ram: 6)]
        let items = ChatModelMenu.items(installed: installed, selectedID: "b", facts: [:], apple: .make(readiness: .ready), appleContextTokens: 8192)
        #expect(items.map(\.id) == [AppleIntelligenceChat.modelID, "a", "b"])
        #expect(items[0].isEnabled && items[0].note == nil && items[0].action == nil)
        #expect(items[0].facts == "Can use tools · 8K tokens of context on this Mac")
        #expect(items.map(\.isSelected) == [false, false, true])
    }

    @Test func notReadyIsDimmedWithTheReasonAndTheExistingTurnOnAction() {
        let readiness = Readiness.needsSetup(reason: "Turn on Apple Intelligence in System Settings", action: .enableAppleIntelligence)
        let items = ChatModelMenu.items(installed: [llm("a", "Alpha", ram: 4)], selectedID: "a", facts: [:], apple: .make(readiness: readiness))
        #expect(items.first?.id == AppleIntelligenceChat.modelID)
        #expect(items.first?.isEnabled == false)
        #expect(items.first?.note == "Turn on Apple Intelligence in System Settings")
        #expect(items.first?.action == .enableAppleIntelligence)
        #expect(settingsPane(for: .enableAppleIntelligence) == .models)
        #expect(items.first?.facts == "")
    }

    @Test func otherNotReadyStatesAreDimmedWithTheirReason() {
        let ineligible = AppleIntelligenceMenuEntry.make(readiness: .unavailable(reason: "This Mac can't run Apple Intelligence"))
        #expect(ineligible == AppleIntelligenceMenuEntry(isEnabled: false, reason: "This Mac can't run Apple Intelligence", action: nil))
        let downloading = AppleIntelligenceMenuEntry.make(readiness: .needsSetup(reason: "macOS is still downloading the model", action: nil))
        #expect(downloading?.isEnabled == false && downloading?.action == nil)
        #expect(AppleIntelligenceMenuEntry.make(readiness: .needsDownload(gb: 3))?.isEnabled == false)
    }

    @Test func belowMacOS26TheRowIsAbsent() {
        #expect(AppleIntelligenceMenuEntry.make(readiness: nil) == nil)
        let items = ChatModelMenu.items(installed: [llm("a", "Alpha", ram: 4)], selectedID: "a", facts: [:], apple: nil)
        #expect(items.map(\.id) == ["a"])
    }

    @Test func theAppleEntryIsAnLLMWithTheReservedID() {
        #expect(AppleIntelligenceChat.entry.id == "apple-intelligence")
        #expect(AppleIntelligenceChat.entry.runnerKind == .llm)
        #expect(AppleIntelligenceChat.entry.contextWindow == 8192)
        #expect(AppleIntelligenceChat.isApple("apple-intelligence") && !AppleIntelligenceChat.isApple("x") && !AppleIntelligenceChat.isApple(nil))
    }

    // MARK: default router

    private func decide(apple: Readiness?, recent: [String] = [], installed: [ModelEntry] = [], tools: Set<String> = [],
                        catalog: [ModelEntry] = [], ram: Double = 16) -> ChatRouterDecision {
        ChatDefaultRouter.decide(appleReadiness: apple, recentRouterIDs: recent, installed: installed, installedToolCapableIDs: tools,
                                 catalog: catalog, totalRAMGB: ram)
    }

    @Test func rule1AppleIntelligenceWinsWhenReadyEvenOverTheLastRouter() {
        let m = llm("m", "M", ram: 4)
        #expect(decide(apple: .ready, recent: ["m"], installed: [m], tools: ["m"]) == .appleIntelligence)
        #expect(decide(apple: .ready) == .appleIntelligence)
    }

    @Test func rule2TheLastRouterThatStillWorksWinsWhenAppleIsNotReady() {
        let m1 = llm("m1", "Alpha", ram: 4), m2 = llm("m2", "Beta", ram: 4)
        let notReady = Readiness.unavailable(reason: "x")
        #expect(decide(apple: notReady, recent: ["m2", "m1"], installed: [m1, m2]) == .installedModel("m2"))
        // newest router uninstalled: the next recent one that is installed
        #expect(decide(apple: nil, recent: ["gone", "m1"], installed: [m1, m2]) == .installedModel("m1"))
        // Apple was the last router but is not ready: skipped
        #expect(decide(apple: notReady, recent: [AppleIntelligenceChat.modelID, "m2"], installed: [m1, m2]) == .installedModel("m2"))
    }

    @Test func rule3TheFirstInstalledModelWithAKnownToolFormatByName() {
        let alpha = llm("alpha", "Alpha", ram: 4), beta = llm("beta", "Beta", ram: 4), gamma = llm("gamma", "Gamma", ram: 4)
        #expect(decide(apple: nil, installed: [gamma, beta, alpha], tools: ["beta", "gamma"]) == .installedModel("beta"))
    }

    @Test func rule3bWithNoKnownToolFormatAnyInstalledModelByNameStillBeatsAnEmptyPage() {
        let alpha = llm("alpha", "Alpha", ram: 4), beta = llm("beta", "Beta", ram: 4)
        #expect(decide(apple: nil, installed: [beta, alpha]) == .installedModel("alpha"))
    }

    @Test func rule4NothingInstalledRecommendsTheSmallestFittingToolCapableCatalogModel() {
        let big = llm("big", "Big", ram: 20), mid = llm("mid", "Mid", ram: 6), small = llm("small", "Small", ram: 2.5, architecture: "Gemma3ForCausalLM"),
            tiny = llm("tiny", "Tiny", ram: 3)
        // `small` has no known tool format; `big` does not fit 16 GB.
        #expect(decide(apple: nil, catalog: [big, mid, small, tiny], ram: 16) == .recommend("tiny"))
        #expect(decide(apple: .unavailable(reason: "x"), catalog: [big, mid], ram: 8) == .recommend("mid"))
    }

    @Test func rule5NothingFitsOrExists() {
        #expect(decide(apple: nil) == .nothing)
        #expect(decide(apple: nil, catalog: [llm("big", "Big", ram: 40)], ram: 16) == .nothing)
        #expect(decide(apple: nil, catalog: [llm("g", "G", ram: 2, architecture: "GptOssForCausalLM")], ram: 16) == .nothing)
    }

    @Test func aRecommendationHonoursTheRunnableFilterAndTiesBreakByID() {
        let x = llm("x", "X", ram: 3), y = llm("y", "Y", ram: 3)
        #expect(ChatDefaultRouter.recommendation(catalog: [y, x], totalRAMGB: 16)?.id == "x")
        #expect(ChatDefaultRouter.recommendation(catalog: [x, y], totalRAMGB: 16, isRunnable: { $0.id == "y" })?.id == "y")
    }

    @Test func catalogArchitecturesMapToTheToolFormatTable() {
        #expect(CatalogToolSupport.modelType(fromArchitecture: "Qwen3_5ForConditionalGeneration") == "qwen3_5")
        #expect(CatalogToolSupport.modelType(fromArchitecture: "GptOssForCausalLM") == "gpt_oss")
        #expect(CatalogToolSupport.modelType(fromArchitecture: "LlamaForCausalLM") == "llama")
        #expect(CatalogToolSupport.modelType(fromArchitecture: nil) == nil)
        let supported = ["Qwen3_5ForConditionalGeneration", "Qwen3ForCausalLM", "Qwen2ForCausalLM", "LlamaForCausalLM", "Mistral3ForConditionalGeneration"]
        for arch in supported { #expect(CatalogToolSupport.supportsTools(architecture: arch), "\(arch)") }
        for arch in ["Gemma3ForCausalLM", "Gemma2ForCausalLM", "Gemma4ForConditionalGeneration", "GptOssForCausalLM", nil] as [String?] {
            #expect(!CatalogToolSupport.supportsTools(architecture: arch), "\(arch ?? "nil")")
        }
    }

    // MARK: banners

    @Test func theOverflowBannerNamesAnInstalledModelAndOffersTwoButtons() {
        let installed = [llm("b", "Beta", ram: 4), llm("a", "Alpha", ram: 4)]
        let banner = ChatBanner.contextExceeded(limit: 8192, installed: installed, installedToolCapableIDs: ["b"])
        #expect(banner.text == "This chat is too long for Apple Intelligence (its window is 8K tokens) — continue with Beta or start a new chat.")
        #expect(banner.buttons == [.continueWith(modelID: "b", name: "Beta"), .startNewChat])
        #expect(banner.buttonTitles == ["Continue with Beta", "Start a new chat"])
    }

    @Test func theOverflowBannerFallsBackToTheFirstInstalledByNameAndToOneButtonWithNone() {
        let installed = [llm("b", "Beta", ram: 4), llm("a", "Alpha", ram: 4)]
        #expect(ChatBanner.contextExceeded(limit: nil, installed: installed, installedToolCapableIDs: []).buttons.first == .continueWith(modelID: "a", name: "Alpha"))
        let none = ChatBanner.contextExceeded(limit: nil, installed: [], installedToolCapableIDs: [])
        #expect(none.buttons == [.startNewChat])
        #expect(none.text == "This chat is too long for Apple Intelligence — start a new chat.")
    }

    @Test func otherFailuresShowTheirSentenceAndAssetsOffersSettings() {
        for error in [AppleFoundationToolError.guardrailViolation, .unsupportedLanguage, .refusal, .rateLimited, .timeout, .failed] {
            let banner = ChatBanner.failure(error, installed: [])
            #expect(banner.text == error.userMessage)
            #expect(banner.buttons.isEmpty)
        }
        #expect(ChatBanner.failure(.assetsUnavailable, installed: []).buttons == [.openSettings(.models)])
        #expect(ChatBanner.failure(.contextWindowExceeded(limit: 8192, tokenCount: nil), installed: []).buttons == [.startNewChat])
    }

    // MARK: controller

    private func controller(readiness: @escaping () -> Readiness?, dir: URL) -> ChatController {
        ChatController(runner: ModelRunner(), conversationDirectory: dir, readAllConversations: { _ in [] },
                       factsLoader: { _ in ChatModelFacts(canUseTools: true, usableTokens: 4096) }, readinessProvider: readiness)
    }
    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("mt33-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    @Test func thePageDefaultsToAppleIntelligenceWhenReady() {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let c = controller(readiness: { .ready }, dir: dir)
        c.refreshPageModel(installedChatModels: [llm("m", "M", ram: 4)])
        #expect(c.model(for: .page)?.id == AppleIntelligenceChat.modelID)
    }

    @Test func aChosenInstalledModelIsKeptEvenWhileAppleIsReady() {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let m = llm("m", "M", ram: 4)
        let c = controller(readiness: { .ready }, dir: dir)
        c.refreshPageModel(installedChatModels: [m])
        #expect(c.selectPageModel(m))
        c.refreshPageModel(installedChatModels: [m])
        #expect(c.model(for: .page)?.id == "m")
    }

    @Test func whenAppleStopsBeingReadyThePageFallsBackToAnInstalledModel() {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        nonisolated(unsafe) var ready = true
        let m = llm("m", "M", ram: 4)
        let c = controller(readiness: { ready ? .ready : .unavailable(reason: "off") }, dir: dir)
        c.refreshPageModel(installedChatModels: [m])
        #expect(c.model(for: .page)?.id == AppleIntelligenceChat.modelID)
        ready = false
        c.refreshPageModel(installedChatModels: [m])
        #expect(c.model(for: .page)?.id == "m")
    }

    @Test func appleIntelligenceCannotBeSelectedWhileNotReady() {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let c = controller(readiness: { .needsSetup(reason: "Turn on Apple Intelligence", action: .enableAppleIntelligence) }, dir: dir)
        #expect(!c.selectPageModel(AppleIntelligenceChat.entry))
        #expect(c.model(for: .page) == nil)
        #expect(c.appleMenuEntry()?.action == .enableAppleIntelligence)
    }

    @Test func belowMacOS26NothingChangesFromMT2() {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let m1 = llm("m1", "Alpha", ram: 4)
        let c = controller(readiness: { nil }, dir: dir)
        c.refreshPageModel(installedChatModels: [m1])
        #expect(c.model(for: .page)?.id == "m1")
        #expect(c.appleMenuEntry() == nil)
    }

    @Test func withNothingInstalledAndAppleNotReadyThePageRecommendsAModel() {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let c = controller(readiness: { nil }, dir: dir)
        c.refreshPageModel(installedChatModels: [], catalog: [llm("big", "Big", ram: 20), llm("ok", "Ok", ram: 5)], totalRAMGB: 16)
        #expect(c.model(for: .page) == nil)
        #expect(c.pageRecommendation?.id == "ok")
    }

    // MARK: runner wiring

    private func appleRunner(_ chatter: AppleFoundationRouterTests.FakeChatter, store: URL) -> ModelRunner {
        AppleFoundationAvailability.toolChatterOverride = chatter
        let runner = ModelRunner()
        runner.prepare(for: AppleIntelligenceChat.entry, in: store)
        return runner
    }
    private func settle(_ runner: ModelRunner) async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while runner.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }
    private func assistantTexts(_ runner: ModelRunner) -> [String] {
        runner.transcript.items.compactMap { if case .message(let m) = $0, m.role == .assistant { return m.text } else { return nil } }
    }

    @Test func anAppleTurnAnswersThroughTheRouterAndStampsTheConversation() async throws {
        let store = tempDir(); defer { try? FileManager.default.removeItem(at: store); AppleFoundationAvailability.toolChatterOverride = nil }
        let chatter = AppleFoundationRouterTests.FakeChatter(script: { _, _ in [.snapshot("Hel"), .snapshot("Hello")] })
        let runner = appleRunner(chatter, store: store)
        await runner.send("hi", for: AppleIntelligenceChat.entry)
        await settle(runner)
        #expect(assistantTexts(runner) == ["Hello"])
        #expect(runner.conversations.first?.modelID == AppleIntelligenceChat.modelID)
        let stamp: String? = runner.transcript.items.compactMap { if case .message(let m) = $0, m.role == .assistant { return m.modelID } else { return nil } }.first
        #expect(stamp == AppleIntelligenceChat.modelID)
        #expect(runner.errorMessage == nil)
    }

    @Test func aSecondAppleTurnReusesTheLiveSessionAndAMissFolds() async throws {
        let store = tempDir(); defer { try? FileManager.default.removeItem(at: store); AppleFoundationAvailability.toolChatterOverride = nil }
        let chatter = AppleFoundationRouterTests.FakeChatter(script: { _, _ in [.snapshot("ok")] })
        let runner = appleRunner(chatter, store: store)
        await runner.send("one", for: AppleIntelligenceChat.entry); await settle(runner)
        await runner.send("two", for: AppleIntelligenceChat.entry); await settle(runner)
        #expect(chatter.sessionCount == 1)
        #expect(chatter.sessions[0].prompts.count == 2)
        #expect(chatter.sessions[0].prompts[1] == "two")
        // a deleted conversation's session is dropped
        let id = try #require(runner.activeConversationID)
        #expect(runner.appleRouter()?.liveConversationIDs == [id])
        runner.deleteConversation(id)
        #expect(runner.appleRouter()?.liveConversationIDs.isEmpty == true)
    }

    @Test func anotherRouterAnsweringInAConversationDropsTheStaleAppleSession() async throws {
        let store = tempDir(); defer { try? FileManager.default.removeItem(at: store); AppleFoundationAvailability.toolChatterOverride = nil }
        let chatter = AppleFoundationRouterTests.FakeChatter(script: { _, _ in [.snapshot("ok")] })
        let runner = appleRunner(chatter, store: store)
        await runner.send("one", for: AppleIntelligenceChat.entry); await settle(runner)
        let mlx = llm("m", "M", ram: 4)
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt33m-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        runner.modelDirectoryForTesting = { _ in modelDir }
        runner.streamingSessionForTesting = { Self.OneShot() }
        await runner.send("two", for: mlx); await settle(runner)
        #expect(runner.appleRouter()?.liveConversationIDs.isEmpty == true)
        await runner.send("three", for: AppleIntelligenceChat.entry); await settle(runner)
        #expect(chatter.sessionCount == 2)
        #expect(chatter.sessions[1].prompts[0].hasPrefix("Conversation so far:"))
    }

    final class OneShot: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { $0.yield("mlx reply"); $0.finish() }
        }
    }

    @Test func aContextOverflowSetsTheNoticeAndContinuingWithAnotherModelResendsTheMessage() async throws {
        let store = tempDir(); defer { try? FileManager.default.removeItem(at: store); AppleFoundationAvailability.toolChatterOverride = nil }
        let chatter = AppleFoundationRouterTests.FakeChatter(script: { _, _ in throw AppleFoundationToolError.contextWindowExceeded(limit: 4096, tokenCount: nil) })
        let runner = appleRunner(chatter, store: store)
        await runner.send("a very long question", for: AppleIntelligenceChat.entry); await settle(runner)
        #expect(runner.contextNotice?.limit == 4096)
        #expect(runner.contextNotice?.prompt == "a very long question")
        #expect(runner.errorMessage == nil)
        #expect(assistantTexts(runner).isEmpty)

        let mlx = llm("m", "M", ram: 4)
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt33n-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        runner.modelDirectoryForTesting = { _ in modelDir }
        runner.streamingSessionForTesting = { Self.OneShot() }
        #expect(await runner.retryUnanswered(with: mlx))
        await settle(runner)
        let users = runner.transcript.items.compactMap { item -> String? in
            if case .message(let m) = item, m.role == .user { return m.text } else { return nil }
        }
        #expect(users == ["a very long question"])         // once, not twice
        #expect(assistantTexts(runner) == ["mlx reply"])
        #expect(runner.contextNotice == nil)
    }

    @Test func aTypedFailureShowsItsSentenceAndTheBanner() async throws {
        let store = tempDir(); defer { try? FileManager.default.removeItem(at: store); AppleFoundationAvailability.toolChatterOverride = nil }
        let chatter = AppleFoundationRouterTests.FakeChatter(script: { _, _ in throw AppleFoundationToolError.guardrailViolation })
        let runner = appleRunner(chatter, store: store)
        await runner.send("x", for: AppleIntelligenceChat.entry); await settle(runner)
        #expect(runner.errorMessage == "Apple Intelligence declined this request.")
        #expect(runner.appleFailure == .guardrailViolation)
        #expect(runner.appleRouter()?.liveConversationIDs.isEmpty == true)
    }

    @Test func withoutTheSystemModelASendSaysSo() async throws {
        let store = tempDir(); defer { try? FileManager.default.removeItem(at: store) }
        AppleFoundationAvailability.toolChatterOverride = nil
        let runner = ModelRunner()
        runner.prepare(for: AppleIntelligenceChat.entry, in: store)
        await runner.send("hi", for: AppleIntelligenceChat.entry)
        #expect(runner.errorMessage == "Apple Intelligence isn't available on this Mac.")
        #expect(!runner.isRunning)
    }

    @Test func removeTrailingUserMessageOnlyRemovesAnUnansweredUserBubble() {
        var t = TranscriptBuilder()
        let emptyResult = t.removeTrailingUserMessage()
        #expect(!emptyResult)
        t.addUserMessage("q")
        let removed = t.removeTrailingUserMessage()
        #expect(removed && t.items.isEmpty)
        t.addUserMessage("q"); t.appendAssistantChunk("a")
        let answered = t.removeTrailingUserMessage()
        #expect(!answered)
    }

    @Test func theAppleBadgeNamesAppleIntelligenceAndIsNotInstalledWhenNotReady() {
        let names = [AppleIntelligenceChat.modelID: AppleIntelligenceChat.displayName]
        #expect(ChatConversationBadge.make(modelID: AppleIntelligenceChat.modelID, catalog: names, installedIDs: [AppleIntelligenceChat.modelID]).text == "Apple Intelligence")
        #expect(ChatConversationBadge.make(modelID: AppleIntelligenceChat.modelID, catalog: names, installedIDs: []).text == "Apple Intelligence · not installed")
    }

    // MARK: owner-run real test (Apple Intelligence ready + TEST_RUNNER_MLXUI_TEST_REAL_AFM=1)

    @Test(.enabled(if: AppleFoundationRouterTests.realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realAChatTurnThroughTheRunnerWithTheAppsOwnTools() async throws {
        let store = tempDir(); defer { try? FileManager.default.removeItem(at: store); AppleFoundationAvailability.toolChatterOverride = nil }
        AppleFoundationAvailability.toolChatterOverride = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let runner = ModelRunner()
        runner.prepare(for: AppleIntelligenceChat.entry, in: store)
        await runner.send("What is 12 times 12? Answer with the number only.", for: AppleIntelligenceChat.entry)
        await settle(runner)
        print("MT3-3 real reply:", assistantTexts(runner), "error:", runner.errorMessage ?? "none", "armed:", runner.enabledToolNames.count)
        #expect(runner.errorMessage == nil)
        #expect(assistantTexts(runner).joined().contains("144"))
        #expect(runner.contextNotice == nil)
    }
}
