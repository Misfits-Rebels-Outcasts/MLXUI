import Foundation
import Testing
@testable import MLXUI

/// MT4-1 — `ToolExposurePolicy`: the §3.6 table for model tools and the tool-picking policy for ambiguous tools.
struct ToolExposurePolicyTests {
    private let entry = makeEntry(id: "m", displayName: "M", modelType: .llm, ramGB: 2)
    private var ready: ModelToolStatus { .ready(entry) }
    private var download: ModelToolStatus { .needsDownload(entry, gb: 1.5) }

    private func input(_ router: ToolExposurePolicy.Router = .mlx(hasVision: false), models: Bool = true,
                       conversation: Set<Kind> = [], turn: Set<Kind> = [], prompt: String = "",
                       statuses: [String: ModelToolStatus] = [:], declined: Set<String> = [],
                       enabled: [String] = [], gate: Bool = false, longText: Bool = false) -> ToolExposurePolicy.Input {
        ToolExposurePolicy.Input(router: router, modelToolsEnabled: models, conversationAssetKinds: conversation,
                                 turnAssetKinds: turn, hasOversizedTextAsset: longText, prompt: prompt, statuses: statuses,
                                 declinedThisTurn: declined, enabledTools: enabled, gateAmbiguousTools: gate)
    }

    private func reasons(_ out: ToolExposurePolicy.Output) -> [String: ToolExposurePolicy.DropReason] {
        Dictionary(out.dropped.map { ($0.name, $0.reason) }, uniquingKeysWith: { first, _ in first })
    }

    // MARK: model tools

    @Test func afmWithAnImageAndNothingInstalledOffersListCapabilitiesAndTheDownloadableImageTools() {
        let statuses = ["ocr_image": download, "describe_image": download, "upscale_image": download, "segment_image": download,
                        "estimate_depth": download]
        let out = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image], statuses: statuses))
        #expect(out.tools == ["list_capabilities", "ocr_image", "describe_image", "upscale_image", "segment_image", "estimate_depth"])
        #expect(out.tools.count <= AppleFoundationToolBudget.maxTools)
    }

    @Test func mlxWithAudioAndParakeetReadyOffersTranscribe() {
        let out = ToolExposurePolicy.select(input(turn: [.audio], statuses: ["transcribe_audio": ready]))
        #expect(out.tools == ["list_capabilities", "transcribe_audio"])
    }

    @Test func aToolDeclinedThisTurnIsDropped() {
        let out = ToolExposurePolicy.select(input(turn: [.audio], statuses: ["transcribe_audio": download], declined: ["transcribe_audio"]))
        #expect(out.tools == ["list_capabilities"])
        #expect(reasons(out)["transcribe_audio"] == .declinedThisTurn)
    }

    @Test func aToolThatCannotRunOrDoesNotExistIsDroppedWithItsReason() {
        let out = ToolExposurePolicy.select(input(turn: [.image], statuses: ["ocr_image": .cantRun(reason: "Needs 20 GB"), "describe_image": ready]))
        #expect(out.tools == ["list_capabilities", "describe_image"])
        #expect(reasons(out)["ocr_image"] == .cantRun("Needs 20 GB"))
        #expect(reasons(out)["upscale_image"] == .notAvailable)
    }

    @Test func aVisionRouterDoesNotGetDescribeImage() {
        let statuses = ["ocr_image": ready, "describe_image": ready]
        let out = ToolExposurePolicy.select(input(.mlx(hasVision: true), turn: [.image], statuses: statuses))
        #expect(out.tools == ["list_capabilities", "ocr_image"])
        #expect(reasons(out)["describe_image"] == .routerHasVision)
    }

    @Test func readyComesBeforeNeedsDownloadWithinAGroup() {
        let statuses = ["ocr_image": download, "describe_image": ready, "upscale_image": download, "segment_image": ready, "estimate_depth": download]
        let out = ToolExposurePolicy.select(input(turn: [.image], statuses: statuses))
        #expect(out.tools == ["list_capabilities", "describe_image", "segment_image", "ocr_image", "upscale_image", "estimate_depth"])
    }

    @Test func generatorsAreOfferedOnlyWhenThePromptCallsForThem() {
        let all = Dictionary(uniqueKeysWithValues: ToolExposurePolicy.generatorTools.map { ($0, ready) })
        #expect(ToolExposurePolicy.select(input(prompt: "What is the capital of France?", statuses: all)).tools == ["list_capabilities"])
        #expect(ToolExposurePolicy.select(input(prompt: "Translate this into French", statuses: all)).tools == ["list_capabilities", "translate"])
        #expect(ToolExposurePolicy.select(input(prompt: "Draw a picture of a fox and read it aloud", statuses: all)).tools
                == ["list_capabilities", "speak", "generate_image"])
        #expect(ToolExposurePolicy.select(input(prompt: "Write me a short song", statuses: all)).tools == ["list_capabilities", "generate_music"])
    }

    @Test func kindsAlreadyInTheConversationKeepTheirToolsAfterTheTurnsOwn() {
        let statuses = ["transcribe_audio": ready, "ocr_image": ready, "describe_image": ready]
        let out = ToolExposurePolicy.select(input(conversation: [.audio, .image], turn: [.audio], statuses: statuses))
        #expect(out.tools == ["list_capabilities", "transcribe_audio", "ocr_image", "describe_image"])
    }

    @Test func readAssetOnlyForAnOversizedTextAsset() {
        #expect(ToolExposurePolicy.select(input()).tools == ["list_capabilities"])
        #expect(ToolExposurePolicy.select(input(longText: true)).tools == ["list_capabilities", "read_asset"])
    }

    @Test func theBudgetTruncatesInRuleOrderSixForAppleTenForMLX() {
        let imageTools = ToolExposurePolicy.toolsForInputKind[.image] ?? []
        var statuses = Dictionary(uniqueKeysWithValues: (imageTools + ["transcribe_audio"]).map { ($0, ready) })
        statuses["transcribe_audio"] = ready
        let enabled = ["read_file", "list_directory", "search_files", "embed_text", "summarize"]
        let afm = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image, .audio], statuses: statuses, enabled: enabled))
        #expect(afm.tools == ["list_capabilities", "ocr_image", "describe_image", "upscale_image", "segment_image", "estimate_depth"])
        #expect(reasons(afm)["transcribe_audio"] == .overBudget)
        #expect(reasons(afm)["read_file"] == .overBudget)
        let mlx = ToolExposurePolicy.select(input(turn: [.image, .audio], statuses: statuses, enabled: enabled))
        #expect(mlx.tools.count == 10)
        #expect(Array(mlx.tools.prefix(7)) == ["list_capabilities", "ocr_image", "describe_image", "upscale_image", "segment_image", "estimate_depth", "transcribe_audio"])
        #expect(Array(mlx.tools.suffix(3)) == ["read_file", "list_directory", "search_files"])
    }

    @Test func withModelToolsOffTheLegacySetIsReturnedUnchanged() {
        let legacy = ["fetch_url", "calculator", "datetime", "read_clipboard", "read_file", "embed_text", "summarize", "semantic_search",
                      "run_javascript", "write_clipboard", "x1", "x2"]
        let statuses = ["ocr_image": ready]
        let out = ToolExposurePolicy.select(input(.appleIntelligence, models: false, turn: [.image], statuses: statuses, enabled: legacy))
        #expect(out.tools == legacy)
        #expect(out.dropped.isEmpty)
    }

    @Test func theOutputIsDeterministic() {
        let statuses = ["ocr_image": ready, "describe_image": download]
        let a = ToolExposurePolicy.select(input(turn: [.image], prompt: "draw a cat", statuses: statuses, enabled: ["calculator"], gate: true))
        let b = ToolExposurePolicy.select(input(turn: [.image], prompt: "draw a cat", statuses: statuses, enabled: ["calculator"], gate: true))
        #expect(a == b)
    }

    // MARK: the tool-picking policy

    private let ambiguous = ["fetch_url", "read_clipboard", "write_clipboard", "calculator", "run_javascript", "datetime"]

    private func offered(_ prompt: String, gate: Bool = true) -> [String] {
        ToolExposurePolicy.select(input(.appleIntelligence, models: false, prompt: prompt, enabled: ambiguous + ["read_file"], gate: gate)).tools
    }

    @Test func aPlainQuestionOffersNoAmbiguousTool() {
        #expect(offered("What is the capital of France?") == ["read_file"])
    }

    @Test func theTimeQuestionOffersDatetime() {
        #expect(offered("What time is it?") == ["datetime", "read_file"])
    }

    @Test func copyToTheClipboardOffersWriteClipboardNotRead() {
        #expect(offered("Copy hello-mt3 to my clipboard") == ["write_clipboard", "read_file"])
    }

    @Test func readingTheClipboardOffersReadClipboardNotWrite() {
        #expect(offered("Summarize what's on my clipboard") == ["read_clipboard", "read_file"])
    }

    @Test func aUrlOrWebsiteCueOffersFetchURL() {
        #expect(offered("summarize https://example.com/a").contains("fetch_url"))
        #expect(offered("what does openai.com say about pricing").contains("fetch_url"))
        #expect(offered("open this web page for me").contains("fetch_url"))
        #expect(!offered("tell me a story").contains("fetch_url"))
    }

    @Test func arithmeticOffersTheCalculator() {
        #expect(offered("what is 17*8").contains("calculator"))
        #expect(offered("what is 15% of 80?").contains("calculator"))
        #expect(offered("calculate the sum of 12 and 30").contains("calculator"))
        #expect(!offered("what is the capital of France?").contains("calculator"))
        #expect(!offered("tell me about the 1990s").contains("calculator"))
    }

    @Test func javascriptNeedsAScriptCue() {
        #expect(offered("run this javascript snippet").contains("run_javascript"))
        #expect(!offered("hello there").contains("run_javascript"))
    }

    @Test func gatingOffLeavesEveryToolOffered() {
        #expect(offered("What is the capital of France?", gate: false) == ambiguous + ["read_file"])
    }

    @Test func nonAmbiguousToolsAreNeverGated() {
        for name in ["read_file", "write_file", "embed_text", "summarize", "transcribe_audio", "ocr_image"] {
            #expect(ToolExposurePolicy.turnCalls(name, prompt: "hello"))
        }
    }

    // MARK: wired into the Apple Intelligence send

    @Test @MainActor func theAppleSendKeepsTheRegistryOrderAndWithholdsTheAmbiguousSix() {
        let all = ModelRunner.defaultTools()
        let plain = ModelRunner.exposedTools(all, prompt: "What is the capital of France?").map(\.name)
        for name in ["fetch_url", "read_clipboard", "write_clipboard", "calculator", "run_javascript", "datetime"] {
            #expect(!plain.contains(name))
        }
        let timed = ModelRunner.exposedTools(all, prompt: "What time is it?").map(\.name)
        #expect(timed.contains("datetime") && !timed.contains("calculator"))
        let order = all.map(\.name)
        #expect(timed == order.filter { timed.contains($0) })          // registry order kept
        #expect(timed.count <= AppleFoundationToolBudget.maxTools)
    }
}
