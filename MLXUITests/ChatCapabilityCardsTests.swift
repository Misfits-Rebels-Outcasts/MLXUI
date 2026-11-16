import Testing
import Foundation
@testable import MLXUI

/// CD-1 — `ChatCapabilityCards` is pure; these use small fixtures and the real `ModelToolSpec.modelTools`.
@MainActor
struct ChatCapabilityCardsTests {
    private func model(_ name: String) -> ModelEntry {
        ModelEntry(
            id: name, family: "F", displayName: name, paramSize: "", paramCountB: nil, modelType: .llm, source: .mlx, format: "mlx",
            platforms: ["macos"], minMacOSVersion: nil, hfRepo: "", hfModelId: "", ramGB: 4, downloadSizeGB: 1, contextWindow: 4096,
            license: nil, licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil, architecture: nil, languages: nil,
            lastUpdated: nil, taskTags: nil, benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil, speedEstimated: nil,
            communityDownloads: nil, communityLikes: nil, variants: nil)
    }

    private var specs: [ModelToolSpec] { ModelToolSpec.modelTools }
    private var allTools: [String] { ChatCapabilityCards.table.map(\.tool) }

    private func ready(_ tools: [String]) -> [String: ModelToolStatus] {
        Dictionary(uniqueKeysWithValues: tools.map { ($0, ModelToolStatus.ready(model("M-\($0)"))) })
    }

    @Test func everyModelToolHasACopyRow() {
        let covered = Set(ChatCapabilityCards.table.map(\.tool))
        for spec in specs { #expect(covered.contains(spec.name), "model tool \(spec.name) has no card copy row") }
        #expect(covered == Set(specs.map(\.name)))
        #expect(covered.count == ChatCapabilityCards.table.count, "duplicate copy rows")
    }

    @Test func primaryIsTheFirstSixPresentInTableOrder() {
        let result = ChatCapabilityCards.cards(specs: specs, statuses: ready(allTools))
        #expect(result.primary.map(\.id) == Array(allTools.prefix(6)))
        #expect(result.more.map(\.id) == Array(allTools.dropFirst(6)))
        #expect(result.primary.map(\.id) == ["ocr_image", "transcribe_audio", "describe_image", "speak", "translate", "generate_image"])
        #expect(result.more.map(\.id) == ["upscale_image", "segment_image", "estimate_depth", "generate_music"])
    }

    @Test func noneAndMissingStatusesAreDropped() {
        var statuses = ready(allTools)
        statuses["ocr_image"] = ModelToolStatus.none
        statuses["speak"] = nil
        let result = ChatCapabilityCards.cards(specs: specs, statuses: statuses)
        let ids = (result.primary + result.more).map(\.id)
        #expect(!ids.contains("ocr_image") && !ids.contains("speak"))
        #expect(result.primary.count == 6)           // the dropped ones are replaced from the rest, in order
        #expect(result.primary.first?.id == "transcribe_audio")
        #expect(result.more.map(\.id) == ["estimate_depth", "generate_music"])
    }

    @Test func aToolMissingFromTheCopyTableOrSpecsIsDropped() {
        let extra = ModelToolSpec(name: "invented", task: "Invented", input: .text, output: .text, params: [], kinds: nil, description: "x")
        var statuses = ready(allTools)
        statuses["invented"] = .ready(model("X"))
        let result = ChatCapabilityCards.cards(specs: specs + [extra], statuses: statuses)
        #expect(!(result.primary + result.more).map(\.id).contains("invented"))
        let narrowed = ChatCapabilityCards.cards(specs: specs.filter { $0.name == "speak" }, statuses: statuses)
        #expect((narrowed.primary + narrowed.more).map(\.id) == ["speak"])
    }

    @Test func statesAndStatusLines() {
        let statuses: [String: ModelToolStatus] = [
            "ocr_image": .ready(model("GLM-OCR")),
            "transcribe_audio": .needsDownload(model("Parakeet"), gb: 0.6),
            "describe_image": .needsDownload(model("Big"), gb: 12.4),
            "speak": .cantRun(reason: "Needs about 9.0 GB of memory, and this Mac has 8.0 GB."),
        ]
        let cards = ChatCapabilityCards.cards(specs: specs, statuses: statuses).primary
        #expect(cards.map(\.id) == ["ocr_image", "transcribe_audio", "describe_image", "speak"])
        #expect(cards[0].state == .ready(modelName: "GLM-OCR"))
        #expect(cards[0].statusLine == "Ready · GLM-OCR")
        #expect(cards[1].state == .needsDownload(modelName: "Parakeet", gb: 0.6))
        #expect(cards[1].statusLine == "0.6 GB download on first use")
        #expect(cards[2].statusLine == "12 GB download on first use")
        #expect(cards[3].statusLine == "Needs about 9.0 GB of memory, and this Mac has 8.0 GB.")
        #expect(cards.map(\.isAvailable) == [true, true, true, false])
    }

    @Test func takesFileAndCopyMatchTheTable() {
        let expected: [(String, String, String, Kind?)] = [
            ("ocr_image", "Read text in an image", "Read the text in this image.", .image),
            ("transcribe_audio", "Transcribe a recording", "Transcribe this recording.", .audio),
            ("describe_image", "Describe an image", "What's in this image?", .image),
            ("speak", "Read text aloud", "Read this aloud: ", nil),
            ("translate", "Translate", "Translate into French: ", nil),
            ("generate_image", "Generate an image", "Draw ", nil),
            ("upscale_image", "Upscale an image", "Upscale this image.", .image),
            ("segment_image", "Cut out the subject", "Cut out the main subject of this image.", .image),
            ("estimate_depth", "Make a depth map", "Make a depth map of this image.", .image),
            ("generate_music", "Generate a sound", "Make a sound of ", nil),
        ]
        let result = ChatCapabilityCards.cards(specs: specs, statuses: ready(allTools))
        let all = result.primary + result.more
        #expect(all.count == expected.count)
        for (card, row) in zip(all, expected) {
            #expect(card.id == row.0)
            #expect(card.title == row.1)
            #expect(card.starterPrompt == row.2)
            #expect(card.takesFile == row.3)
            #expect(!card.symbol.isEmpty)
        }
    }

    @Test func fileKindsAgreeWithTheSpecInputs() throws {
        for row in ChatCapabilityCards.table {
            let spec = try #require(ModelToolSpec.spec(named: row.tool))
            #expect((row.takesFile != nil) == (spec.input == .image || spec.input == .audio), "\(row.tool)")
        }
    }

    @Test func menuSummaryCountsReadyAndNeedsDownloadOnly() {
        let statuses: [String: ModelToolStatus] = [
            "ocr_image": .ready(model("A")),
            "transcribe_audio": .needsDownload(model("B"), gb: 1),
            "speak": .cantRun(reason: "no"),
        ]
        let result = ChatCapabilityCards.cards(specs: specs, statuses: statuses)
        #expect(ChatCapabilityCards.menuSummary(cards: result.primary + result.more) == "2 tools")
        #expect(ChatCapabilityCards.menuSummary(cards: Array(result.primary.prefix(1))) == "1 tool")
        #expect(ChatCapabilityCards.menuSummary(cards: result.primary.filter { !$0.isAvailable }) == nil)
        #expect(ChatCapabilityCards.menuSummary(cards: []) == nil)
        let full = ChatCapabilityCards.cards(specs: specs, statuses: ready(allTools))
        #expect(ChatCapabilityCards.menuSummary(cards: full.primary + full.more) == "10 tools")
    }

    @Test func routerNoteOnlyWhenTheModelCannotUseTools() {
        #expect(ChatCapabilityCards.routerNote(modelName: "Qwen3 4B", canUseTools: true) == nil)
        #expect(ChatCapabilityCards.routerNote(modelName: "Qwen3 4B", canUseTools: false)
                == "Qwen3 4B can't use tools, so it can't run these. Switch to a model that can.")
    }
}
