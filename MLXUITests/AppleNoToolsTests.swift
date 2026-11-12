import Testing
import Foundation
@testable import MLXUI

/// MT4-7-FIX-6: Apple Intelligence gets NO tools on a turn that calls for none (any armed tool makes it refuse long writing).
@MainActor @Suite(.serialized)
struct AppleNoToolsTests {
    private let entry = makeEntry(id: "m", displayName: "M", modelType: .llm, ramGB: 2)

    private func offered(_ prompt: String, router: ToolExposurePolicy.Router = .appleIntelligence, turn: Set<Kind> = [],
                         enabled: [String] = ["read_file", "list_directory", "search_files", "write_file", "edit_file", "run_shell",
                                              "fetch_url", "calculator", "datetime"],
                         statuses: [String: ModelToolStatus] = [:], folder: Bool = false) -> [String] {
        ToolExposurePolicy.select(ToolExposurePolicy.Input(
            router: router, modelToolsEnabled: true, conversationAssetKinds: turn, turnAssetKinds: turn, prompt: prompt,
            statuses: statuses, enabledTools: enabled, gateAmbiguousTools: router == .appleIntelligence, foldersAttached: folder)).tools
    }

    @Test func aPlainQuestionOrWritingPromptOffersNothing() {
        #expect(offered("What is the capital of France?") == [])
        #expect(offered("write 1500 words about the history of bread") == [])
        #expect(offered("write 300 words about the sea") == [])
    }

    @Test func capabilityCuesOfferListCapabilitiesAlone() {
        for prompt in ["what can you do with my models?", "which models can transcribe audio?", "What can you do with images and audio here?",
                       "what are your capabilities", "can you use my tools?"] {
            #expect(offered(prompt) == ["list_capabilities"], "\(prompt)")
        }
        #expect(offered("write an essay about language models") == [])
    }

    @Test func aRealCueKeepsListCapabilitiesFirstAndTheCuedToolAsToday() {
        let ready = ModelToolStatus.ready(entry)
        let tools = offered("Run OCR on /x/receipt.jpg", statuses: ["ocr_image": ready])
        #expect(tools.first == "list_capabilities" && tools.contains("ocr_image") && !tools.contains("run_shell"))
        let clipboard = offered("What time is it?")
        #expect(clipboard == ["list_capabilities", "datetime"])
    }

    @Test func anAttachedImageIsAsToday() {
        let ready = ModelToolStatus.ready(entry)
        #expect(offered("What's the total?", turn: [.image], statuses: ["ocr_image": ready]) == ["list_capabilities", "ocr_image"])
    }

    @Test func anAttachedFolderOffersTheFileToolsWithListCapabilitiesFirst() {
        let tools = offered("hello", folder: true)
        #expect(tools.first == "list_capabilities" && tools.contains("read_file"))
    }

    @Test func mlxStillOffersListCapabilitiesOnAPlainTurn() {
        #expect(offered("What is the capital of France?", router: .mlx(hasVision: false)).first == "list_capabilities")
    }

    @Test func aDeclinedListCapabilitiesIsNotAddedBack() {
        let out = ToolExposurePolicy.select(ToolExposurePolicy.Input(
            router: .appleIntelligence, modelToolsEnabled: true, prompt: "what can you do", declinedThisTurn: ["list_capabilities"],
            enabledTools: [], gateAmbiguousTools: true))
        #expect(out.tools == [] && out.dropped.contains { $0.name == "list_capabilities" })
    }

    // MARK: the router

    private func collect(_ stream: AsyncThrowingStream<RouterEvent, Error>) async throws -> String {
        var text = ""
        for try await event in stream { if case .text(let t) = event { text += t } }
        return text
    }

    @Test func anEmptyToolSetBuildsASessionWithNoToolsAndAnswers() async throws {
        let chatter = AppleFoundationRouterTests.FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter)
        let reply = try await collect(router.streamTurn(RouterTurn(instructions: "Be brief.", prompt: "write 1500 words", tools: [])))
        #expect(reply == "ok")
        #expect(chatter.toolSets.count == 1 && chatter.toolSets[0].isEmpty)
        #expect((chatter.instructions.first ?? nil) == "Be brief.")        // no "not available" note when nothing was dropped
    }

    @Test func flippingTheToolSetBetweenTurnsRebuildsTheSessionAndStillAnswers() async throws {
        let chatter = AppleFoundationRouterTests.FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter)
        let id = UUID()
        let cap: [any AgentTool] = [AppleFoundationRouterTests.NamedTool(name: "list_capabilities")]
        func turn(_ prompt: String, _ tools: [any AgentTool]) -> RouterTurn {
            RouterTurn(history: .live(conversationID: id, fallback: []), prompt: prompt, tools: tools)
        }
        #expect(try await collect(router.streamTurn(turn("one", []))) == "ok")
        #expect(try await collect(router.streamTurn(turn("two", []))) == "ok")
        #expect(chatter.sessionCount == 1)                                  // unchanged (empty) set: the live session is reused
        #expect(try await collect(router.streamTurn(turn("three", cap))) == "ok")
        #expect(chatter.sessionCount == 2 && chatter.toolSets[1].map(\.name) == ["list_capabilities"])
        #expect(try await collect(router.streamTurn(turn("four", []))) == "ok")
        #expect(chatter.sessionCount == 3 && chatter.toolSets[2].isEmpty)
    }
}
