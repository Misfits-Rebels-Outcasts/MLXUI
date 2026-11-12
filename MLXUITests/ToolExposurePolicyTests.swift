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
        let out = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image], prompt: "Describe this image, upscale it, segment it and estimate its depth", statuses: statuses))
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
        let out = ToolExposurePolicy.select(input(turn: [.image], prompt: "describe this image and upscale it", statuses: ["ocr_image": .cantRun(reason: "Needs 20 GB"), "describe_image": ready]))
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
        let out = ToolExposurePolicy.select(input(turn: [.image], prompt: "Describe this image, upscale it, segment it and estimate its depth", statuses: statuses))
        // MT4-6b F2: the ready cued tools rank ahead of the default that still needs a download (was: ocr_image second).
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
        // MT4-6b F1: an earlier turn's image kind offers ocr_image by default and describe_image only when cued (was: + "describe_image").
        #expect(out.tools == ["list_capabilities", "transcribe_audio", "ocr_image"])
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
        let afm = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image, .audio], prompt: "Describe this image, upscale it, segment it and estimate its depth", statuses: statuses, enabled: enabled))
        #expect(afm.tools == ["list_capabilities", "ocr_image", "transcribe_audio", "describe_image", "upscale_image", "segment_image"])
        #expect(reasons(afm)["estimate_depth"] == .overBudget)
        #expect(reasons(afm)["read_file"] == .overBudget)
        let mlx = ToolExposurePolicy.select(input(turn: [.image, .audio], prompt: "Describe this image, upscale it, segment it and estimate its depth", statuses: statuses, enabled: enabled))
        #expect(mlx.tools.count == 10)
        #expect(Array(mlx.tools.prefix(7)) == ["list_capabilities", "ocr_image", "transcribe_audio", "describe_image", "upscale_image", "segment_image", "estimate_depth"])
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
        #expect(offered("What is the capital of France?") == [])
    }

    @Test func theTimeQuestionOffersDatetime() {
        #expect(offered("What time is it?") == ["datetime"])
    }

    @Test func copyToTheClipboardOffersWriteClipboardNotRead() {
        #expect(offered("Copy hello-mt3 to my clipboard") == ["write_clipboard"])
    }

    @Test func readingTheClipboardOffersReadClipboardNotWrite() {
        #expect(offered("Summarize what's on my clipboard") == ["read_clipboard"])
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
        for name in ["read_file", "write_file", "transcribe_audio", "ocr_image"] {
            #expect(ToolExposurePolicy.turnCalls(name, prompt: "hello"))
        }
    }

    // MARK: MT4-2b

    private var imageStatuses: [String: ModelToolStatus] {
        Dictionary(uniqueKeysWithValues: (ToolExposurePolicy.toolsForInputKind[.image] ?? []).map { ($0, download) })
    }

    /// (a) prompt -> the image tools offered on an image turn (Apple Intelligence, nothing generator-ready).
    @Test func anImageTurnOffersOcrByDefaultAndOnlyTheImageToolsThePromptCallsFor() {
        let table: [(String, [String])] = [
            ("", ["list_capabilities", "ocr_image"]),
            ("What does this say?", ["list_capabilities", "ocr_image"]),
            ("Upscale this photo", ["list_capabilities", "ocr_image", "upscale_image"]),
            ("Cut out the dog", ["list_capabilities", "ocr_image", "segment_image"]),
            ("Estimate the depth of this scene", ["list_capabilities", "ocr_image", "estimate_depth"]),
            ("Describe this image", ["list_capabilities", "ocr_image", "describe_image"]),
        ]
        for (prompt, expected) in table {
            let out = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image], prompt: prompt, statuses: imageStatuses))
            #expect(out.tools == expected, "prompt: \(prompt)")
        }
    }

    /// (a) called-for generators rank ahead of uncalled image tools.
    @Test func calledForGeneratorsRankAheadOfImageToolsUnderTheAppleBudget() {
        var statuses = imageStatuses
        for g in ToolExposurePolicy.generatorTools { statuses[g] = ready }
        let out = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image], prompt: "Translate the text in this image and read it aloud",
                                                  statuses: statuses))
        #expect(out.tools.contains("translate") && out.tools.contains("speak") && out.tools.contains("ocr_image"))
        #expect(out.tools.count <= AppleFoundationToolBudget.maxTools)
    }

    /// The conversation's assets include this turn's: a kind attached this turn is not re-offered in full by rule 4.
    @Test func anImageAttachedThisTurnIsNotReofferedInFullByTheConversationRule() {
        let out = ToolExposurePolicy.select(input(.appleIntelligence, conversation: [.image], turn: [.image], prompt: "What's the total on this receipt?",
                                                  statuses: imageStatuses))
        #expect(out.tools == ["list_capabilities", "ocr_image"])
    }

    /// (b) embed_text / summarize / semantic_search follow the gate.
    @Test func embedSummarizeAndSemanticSearchNeedATurnCue() {
        let tools = ["embed_text", "summarize", "semantic_search"]
        func offered(_ prompt: String, gate: Bool = true) -> [String] {
            ToolExposurePolicy.select(input(.appleIntelligence, models: false, prompt: prompt, enabled: tools, gate: gate)).tools
        }
        #expect(offered("What is the capital of France?").isEmpty)
        #expect(offered("Search my documents for the invoice").contains("semantic_search"))
        #expect(offered("Summarize this").contains("summarize"))
        #expect(offered("Give me a summary").contains("summarize"))
        #expect(offered("Embed this sentence").contains("embed_text"))
        #expect(offered("Index my notes").contains("semantic_search"))
        #expect(offered("What is the capital of France?", gate: false) == tools)
        #expect(!ToolExposurePolicy.turnCalls("embed_text", prompt: "hello"))
    }

    // MARK: MT4-6b (F5, F1, F2, F3)

    /// F5: with model tools on, rule 5 does not re-offer a name that is a model tool (the legacy `ocr_image` / `transcribe_audio`)
    /// without the status check; every other enabled tool is kept.
    @Test func ruleFiveSkipsModelToolNamesButKeepsOtherEnabledToolsWhenModelToolsAreOn() {
        let enabled = ["transcribe_audio", "ocr_image", "echo", "read_file", "list_directory", "search_files"]
        let out = ToolExposurePolicy.select(input(.appleIntelligence, prompt: "hello", statuses: ["ocr_image": .cantRun(reason: "x")], enabled: enabled))
        #expect(out.tools == ["list_capabilities", "echo", "read_file", "list_directory", "search_files"])
        // A model tool whose rule offered it is still offered once, through the rule, with its status check.
        let audio = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.audio], prompt: "hello", statuses: ["transcribe_audio": ready], enabled: enabled))
        #expect(audio.tools == ["list_capabilities", "transcribe_audio", "echo", "read_file", "list_directory", "search_files"])
        // Model tools off: the legacy set is exactly as before, the legacy names included.
        let off = ToolExposurePolicy.select(input(.appleIntelligence, models: false, enabled: enabled))
        #expect(off.tools == enabled)
    }

    /// F1: an earlier turn's image kind offers `ocr_image` by default and the other image tools only when the prompt calls for them.
    @Test func anEarlierTurnsImageKindOffersOcrByDefaultAndOnlyCuedTools() {
        let table: [(String, [String])] = [
            ("Thanks, that is helpful.", ["list_capabilities", "ocr_image"]),
            ("Upscale it", ["list_capabilities", "ocr_image", "upscale_image"]),
            ("Cut out the dog", ["list_capabilities", "ocr_image", "segment_image"]),
            ("Now describe it", ["list_capabilities", "ocr_image", "describe_image"]),
        ]
        for (prompt, expected) in table {
            let out = ToolExposurePolicy.select(input(.appleIntelligence, conversation: [.image], prompt: prompt, statuses: imageStatuses))
            #expect(out.tools == expected, "prompt: \(prompt)")
        }
        // The same turn with the image attached gives the same set (rule 4 and rule 2 share one path).
        let attached = ToolExposurePolicy.select(input(.appleIntelligence, conversation: [.image], turn: [.image], prompt: "Upscale it", statuses: imageStatuses))
        #expect(attached.tools == ["list_capabilities", "ocr_image", "upscale_image"])
    }

    /// F2: a cued, ready tool is ahead of a default that needs a download; with a ready default, or with no cue, nothing moves.
    @Test func aCuedReadyToolRanksAheadOfADefaultThatNeedsADownload() {
        let statuses: [String: ModelToolStatus] = ["ocr_image": download, "describe_image": ready]
        let cued = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image], prompt: "Describe this photo", statuses: statuses))
        #expect(cued.tools == ["list_capabilities", "describe_image", "ocr_image"])
        let uncued = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image], prompt: "What does this say?", statuses: statuses))
        #expect(uncued.tools == ["list_capabilities", "ocr_image"])
        let readyDefault = ToolExposurePolicy.select(input(.appleIntelligence, turn: [.image], prompt: "Describe this photo", statuses: ["ocr_image": ready, "describe_image": ready]))
        #expect(readyDefault.tools == ["list_capabilities", "ocr_image", "describe_image"])
    }

    /// F3: ordinary prompts that used to arm an ambiguous tool arm nothing.
    @Test func ordinaryPromptsNoLongerArmAnAmbiguousTool() {
        let negatives: [(String, String)] = [
            ("What is the time complexity of quicksort?", "datetime"), ("How do I improve my time management?", "datetime"),
            ("What day did the Second World War start?", "datetime"), ("Explain how a database index speeds up queries.", "semantic_search"),
            ("Give me a summary of Hamlet.", "summarize"), ("Is there a script that writes better emails?", "run_javascript"),
            ("Write a short story about a robot named Js.", "run_javascript"), ("Which do you prefer, 3x4 or 4x3 shelves?", "calculator"),
        ]
        for (prompt, tool) in negatives { #expect(!ToolExposurePolicy.turnCalls(tool, prompt: prompt), "\(prompt) arms \(tool)") }
    }

    /// F3 recall: the obvious phrasings still arm their tool.
    @Test func obviousPhrasingsStillArmTheirTool() {
        let positives: [(String, String)] = [
            ("What time is it?", "datetime"), ("what time is it right now", "datetime"), ("What's the date today?", "datetime"),
            ("What is today's date", "datetime"), ("What's the current time in Tokyo?", "datetime"), ("how many days until Friday?", "datetime"),
            ("What day is it?", "datetime"), ("What day of the week is it", "datetime"), ("what is the date?", "datetime"),
            ("Index my notes", "semantic_search"), ("Search my documents for the invoice", "semantic_search"), ("search the notes folder for rent", "semantic_search"),
            ("Summarize this", "summarize"), ("Give me a summary", "summarize"), ("summary of this article", "summarize"), ("TL;DR please", "summarize"),
            ("Calculate 15% of 80", "calculator"), ("what is 17*8", "calculator"), ("what is 3 x 4", "calculator"), ("12 plus 30", "calculator"),
            ("Run this script", "run_javascript"), ("run this javascript snippet", "run_javascript"), ("execute this JS", "run_javascript"),
            ("run the code below", "run_javascript"), ("node.js snippet", "run_javascript"),
        ]
        for (prompt, tool) in positives { #expect(ToolExposurePolicy.turnCalls(tool, prompt: prompt), "\(prompt) does not arm \(tool)") }
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
