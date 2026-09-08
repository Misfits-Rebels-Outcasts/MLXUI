import Testing
import Foundation
@testable import MLXUI

/// OCP-2 (`RSI/DelegateOCRPromptBacklog.md` §4) — a `Describe Image` / `OCR` flow row's prompt
/// reaches the model, and an out-of-set recognition mode is dropped at the flow boundary.
///
/// - OCP-2-1: `RealExecutor.stageConfig` gains an `engines.vlm.` branch — `StageConfig(prompt:
///   settings.firstBare())`, ported from `catflow-mlx/src/catflow/engines/vlm.py:68-100`.
/// - OCP-2-3 (RULED: validator warns, runtime ignores): `RealExecutor.sanitizedPrompt` drops a
///   `.modes` model's out-of-set value to `nil`; `PaddleOCRSDK.makeStage`'s throw stays.
///   `FlowRowInspectorView.strandedPromptMessage` is the warning half.
@Suite struct CatFlowVLMPromptTests {

    private func catalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
            .domains.flatMap { $0.allModels }
    }

    private func executor(promptSupport: @escaping @Sendable (ModelEntry) async -> PromptSupport = { _ in .none }) -> RealExecutor {
        RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "ocp2",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in StubTextStage() },
            promptSupport: promptSupport,
            installedModelIDs: [],
            catalog: [])
    }

    // MARK: - OCP-2-1: stageConfig reads the bare token for engines.vlm.*

    @Test func describeImageRowCarriesItsBareTokenAsThePrompt() throws {
        let desc = try #require(TaskCatalog.get("Describe Image"))
        let row = Row(task: "Describe Image", model: "LFM2-VL 1.6B", settings: "\"What is happening here?\"")
        #expect(try executor().stageConfig(for: desc, row: row).prompt == "What is happening here?")
    }

    @Test func describeImageRowWithNoTokenLeavesThePromptNilForTheSDKDefault() throws {
        let desc = try #require(TaskCatalog.get("Describe Image"))
        let row = Row(task: "Describe Image", model: "LFM2-VL 1.6B", settings: nil)
        #expect(try executor().stageConfig(for: desc, row: row).prompt == nil)
    }

    @Test func ocrRowCarriesItsBareTokenAsThePrompt() throws {
        let desc = try #require(TaskCatalog.get("OCR"))
        let row = Row(task: "OCR", model: "PaddleOCR-VL", settings: "table")
        #expect(try executor().stageConfig(for: desc, row: row).prompt == "table")
    }

    @Test func nonVLMRowsDoNotPickUpAVLMPrompt() throws {
        // A Transcribe row carrying a quoted token must not smuggle it in as `prompt`.
        let desc = try #require(TaskCatalog.get("Transcribe"))
        let row = Row(task: "Transcribe", model: "Whisper Small", settings: "lang=en; \"not a prompt\"")
        let config = try executor().stageConfig(for: desc, row: row)
        #expect(config.prompt == nil)
        #expect(config.language == "en")
    }

    // MARK: - OCP-2-3: the runtime "ignores" half

    private let modeSupport: @Sendable (ModelEntry) async -> PromptSupport = { _ in
        .modes(values: ["ocr", "table", "formula", "chart"], default: "ocr")
    }

    @Test func sanitizeKeepsAValidModeForAModesModel() async throws {
        let entry = try #require(try catalog().first { $0.id.lowercased().contains("paddleocr") })
        let out = await executor(promptSupport: modeSupport)
            .sanitizedPrompt(StageConfig(prompt: "table"), for: entry)
        #expect(out.prompt == "table")
    }

    @Test func sanitizeDropsAnOutOfSetValueToNil() async throws {
        let entry = try #require(try catalog().first { $0.id.lowercased().contains("paddleocr") })
        let out = await executor(promptSupport: modeSupport)
            .sanitizedPrompt(StageConfig(prompt: "the table on the right"), for: entry)
        #expect(out.prompt == nil)   // → SDK default applies; PaddleOCRSDK.makeStage never throws
    }

    @Test func sanitizeLeavesFreeTextAndNoneModelsUntouched() async throws {
        let entry = try #require(try catalog().first { $0.runnerKind == .ocr })
        let free: @Sendable (ModelEntry) async -> PromptSupport = { _ in .freeText(default: "x") }
        let none: @Sendable (ModelEntry) async -> PromptSupport = { _ in .none }
        #expect(await executor(promptSupport: free).sanitizedPrompt(StageConfig(prompt: "as CSV"), for: entry).prompt == "as CSV")
        #expect(await executor(promptSupport: none).sanitizedPrompt(StageConfig(prompt: "as CSV"), for: entry).prompt == "as CSV")
    }

    // MARK: - OCP-2-3: the warning half (FlowRowInspectorView.strandedPromptMessage)

    @Test func strandedWarningFiresForNoneAndBadModeNotForValid() {
        let modes = PromptSupport.modes(values: ["ocr", "table", "formula", "chart"], default: "ocr")
        #expect(FlowRowInspectorView.strandedPromptMessage(token: "table", display: "PaddleOCR-VL", support: modes) == nil)
        #expect(FlowRowInspectorView.strandedPromptMessage(token: "as CSV", display: "PaddleOCR-VL", support: modes)?.contains("recognition mode") == true)
        #expect(FlowRowInspectorView.strandedPromptMessage(token: "as CSV", display: "dots.ocr", support: .none)?.contains("won't reach it") == true)
        #expect(FlowRowInspectorView.strandedPromptMessage(token: "as CSV", display: "GLM-OCR", support: .freeText(default: "x")) == nil)
    }

    // MARK: - OCP-2-4: the three gallery flows now pass their prompts through

    @Test func galleryDescribeImageRowsNowReachTheModel() throws {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let describeDesc = try #require(TaskCatalog.get("Describe Image"))
        let ex = executor()

        // Distinctive substrings of each flow's `Describe Image` instruction — before OCP-2-1
        // `config.prompt` was `nil` for all of them (the quoted text never reached the model).
        for (file, needle) in [
            ("22-ScreenshotHowTo.cat", "What action is happening"),
            ("23-AltText.cat", "One-sentence alt text"),
            ("25-PhotoCull.cat", "closed eyes? Be specific"),
        ] {
            let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(file)")
            let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
            let describe = try #require(allRows(doc.rows).first { $0.task == "Describe Image" },
                                        "\(file) should have a Describe Image row")
            let prompt = try ex.stageConfig(for: describeDesc, row: describe).prompt
            #expect(prompt?.contains(needle) == true, "\(file): got \(String(describing: prompt))")
        }
    }

    @Test func galleryBareOCRRowStaysOnTheSDKDefault() throws {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/22-ScreenshotHowTo.cat")
        let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
        let ocr = try #require(allRows(doc.rows).first { $0.task == "OCR" })
        let desc = try #require(TaskCatalog.get("OCR"))
        #expect(try executor().stageConfig(for: desc, row: ocr).prompt == nil)
    }

    private func allRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + allRows($0.children) }
    }
}

/// A no-op `text → text` stage for the `RealExecutor` config tests (no model needed).
private struct StubTextStage: PipelineStage {
    let id = "stub.ocp2"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        progress(1.0)
        return input
    }
}
