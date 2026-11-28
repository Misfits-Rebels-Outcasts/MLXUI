import Foundation
import Testing
@testable import MLXUI

/// FA-R16 — the transcript is run input: no `StageConfig` carries it, and no stage but the aligner's sees it.
///
/// Owner ruling 17 (Boo, 2026-11-27), quoted in full at `AlignTranscript`: "The transcript must not be part of the engine
/// cache key. It's run input, not a load setting. […] That supersedes ruling 5". Ruling 5 had accepted a
/// `StageConfig.transcript` field; the field is gone.
///
/// No weights: each row goes through `RealExecutor` with a `makeModelStage` that records the config and stops there.
private nonisolated final class TranscriptScopeSeen: @unchecked Sendable {
    private let lock = NSLock()
    private var builds: [(config: StageConfig, transcript: String?)] = []
    func note(_ config: StageConfig, _ transcript: String?) {
        lock.lock(); builds.append((config, transcript)); lock.unlock()
    }
    var observed: [(config: StageConfig, transcript: String?)] { lock.lock(); defer { lock.unlock() }; return builds }
}

private nonisolated struct StopAtStageBuild: Error {}

nonisolated struct ForcedAlignerTranscriptScopeTests {

    private enum Input { case text, texts, audio, image }

    private struct Case {
        let task: String
        let model: String
        var settings: String?
        var tags: [String]?
        let inputs: [Input]
    }

    private static let llm = "Qwen3 8B"

    /// Every task the executor builds a stage for, except `Align`. `Translate` appears twice: the framed LLM row and the
    /// translation specialist are separate config paths.
    private static let otherRows: [Case] = {
        let framed = ["Summarize", "Translate", "Answer", "Rewrite", "Draft", "Ask", "Title", "Critique"]
            .map { Case(task: $0, model: llm, settings: "\"Antworte kurz.\"", inputs: [.text]) }
        let overLists = ["Revise", "Merge"].map { Case(task: $0, model: llm, inputs: [.texts]) }
        let deciders = ["Decide", "Classify", "Gate", "Score", "Judge", "Think"]
            .map { Case(task: $0, model: llm, settings: "\"Ist das fertig?\"", tags: ["ja", "nein"], inputs: [.texts]) }
        return framed + overLists + deciders + [
            Case(task: "Verify", model: llm, inputs: [.text, .text]),  // its frame reads two inputs
            Case(task: "Generate", model: llm, inputs: [.text]),
            Case(task: "Extract Structured", model: llm, settings: "\"Name, Stadt\"", inputs: [.text]),
            Case(task: "Text to Table", model: llm, settings: "\"Name, Stadt\"", inputs: [.text]),
            Case(task: "Translate", model: "TranslateGemma 4B", settings: "to=French", inputs: [.text]),
            Case(task: "Redact", model: "Privacy Filter", inputs: [.text]),
            Case(task: "Transcribe", model: "Whisper Large v3", settings: "lang=de", inputs: [.audio]),
            Case(task: "Speak", model: "Kokoro 82M", settings: "voice=af_heart", inputs: [.text]),
            Case(task: "Identify Speakers", model: "Sortformer 4spk v2.1", inputs: [.audio]),
            Case(task: "Clean Audio", model: "DeepFilterNet 3", inputs: [.audio]),
            Case(task: "Describe Image", model: "Gemma 3 4B", settings: "\"Beschreibe das Bild.\"", inputs: [.image]),
            Case(task: "OCR", model: "GLM-OCR", inputs: [.image]),
            Case(task: "Generate Image", model: "FLUX.2 Klein 4B", settings: "seed=7", inputs: [.text]),
            Case(task: "Generate Sound", model: "MusicGen", inputs: [.text]),
            // No `CatalogBridge` row: resolved by the catalog's own Hugging Face id, as a derived pick is.
            Case(task: "Generate Video", model: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", inputs: [.text]),
            Case(task: "Segment", model: "SAM Base", inputs: [.image]),
            Case(task: "Upscale", model: "SeedVR2 3B", settings: "scale=2", inputs: [.image]),
            Case(task: "Estimate Depth", model: "Depth Anything V2 Small", inputs: [.image]),
            Case(task: "Embed", model: "BGE-M3", inputs: [.texts]),
            Case(task: "Rerank", model: "Qwen3 Reranker 0.6B", settings: "query=Wetter", inputs: [.texts]),
        ]
    }()

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
    }

    /// What `RealExecutor` hands to `makeModelStage` for one row (the config, and `AlignTranscript.current` at that
    /// moment); the row ends at the first stage build.
    private func builds(task: String, model: String, settings: String?, tags: [String]? = nil,
                        inputs: [Asset]) async throws -> [(config: StageConfig, transcript: String?)] {
        let catalog = try loadCatalog()
        let ws = FlowWorkspace(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("fa-transcript-scope-\(UUID().uuidString)"))
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let seen = TranscriptScopeSeen()
        var executor = RealExecutor(workspace: ws, flowID: "f",
                                    blobDirectory: ws.root.appendingPathComponent("blobs"),
                                    makeModelStage: { _, config in
                                        seen.note(config, AlignTranscript.current)
                                        throw StopAtStageBuild()
                                    },
                                    installedModelIDs: Set(catalog.map(\.id)), catalog: catalog)
        executor.detectLanguage = { _ in "de" }
        let row = Row(task: task, model: model, settings: settings, tags: tags)
        do {
            _ = try await executor.execute(path: "1", row: row, inputs: inputs,
                                           transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("\(task) (\(model)): the row finished without building a stage")
        } catch is StopAtStageBuild {
            // The expected end.
        } catch {
            Issue.record("\(task) (\(model)): failed before the stage build: \(error)")
        }
        return seen.observed
    }

    private func asset(_ input: Input) throws -> Asset {
        func text(_ value: String) -> Item { Item(kind: .text, value: value, path: nil, sourceText: nil) }
        let temp = FileManager.default.temporaryDirectory
        switch input {
        case .text:
            return Asset(items: [text("Guten Morgen, Anna wohnt in Hamburg.")])
        case .texts:
            return Asset(items: [text("Das Wetter ist schön."), text("Il pleut à Lyon.")])
        case .audio:
            let url = temp.appendingPathComponent("fa-scope-\(UUID().uuidString).wav")
            try AudioWriter.writeWAV(AudioBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000), to: url)
            return Asset(items: [Item(kind: .audio, value: nil, path: url, sourceText: nil)])
        case .image:
            // Never read: the row stops at the stage build.
            let url = temp.appendingPathComponent("fa-scope-\(UUID().uuidString).png")
            return Asset(items: [Item(kind: .image, value: nil, path: url, sourceText: nil)])
        }
    }

    /// A new served model task must be added to `otherRows`. Not listed: tasks the executor doesn't serve (`Edit Image`,
    /// `Inpaint`, the latent rows, …), and a decider on a `.decision` model, which asks its engine directly and builds no
    /// stage (the six deciders are driven on an LLM).
    @Test func everyServedModelTaskIsCovered() {
        let served = TaskCatalog.allTasks().map(\.name)
            .filter { TaskModels.runnerKind(forTask: $0) != nil && TaskModels.isServedByExecutor($0) }
        let covered = Set(Self.otherRows.map(\.task)).union(["Align"])
        #expect(Set(served) == covered)
    }

    @Test func noStageConfigCarriesATranscript() {
        let fields = Mirror(reflecting: StageConfig.default).children.compactMap(\.label)
        #expect(fields.contains("language") && fields.contains("redactOnly"), "the mirror sees the fields")
        #expect(!fields.contains { $0.lowercased().contains("transcript") }, "\(fields)")
    }

    @Test func everyOtherStageBuildsOnceWithNoTranscriptSet() async throws {
        for item in Self.otherRows {
            let observed = try await builds(task: item.task, model: item.model, settings: item.settings, tags: item.tags,
                                            inputs: try item.inputs.map { try asset($0) })
            #expect(observed.count == 1, "\(item.task) (\(item.model)) builds one stage")
            #expect(observed.allSatisfy { $0.transcript == nil }, "\(item.task) (\(item.model)) must not see a transcript")
        }
        #expect(AlignTranscript.current == nil)
    }

    @Test func theAlignRowsStageIsBuiltFromTheLanguageAlone() async throws {
        let words = Asset(items: [Item(kind: .text, value: "Guten Morgen", path: nil, sourceText: nil)])
        let observed = try await builds(task: "Align", model: "Qwen3 Aligner 0.6B", settings: "lang=de",
                                        inputs: [try asset(.audio), words])
        #expect(observed.map(\.config) == [StageConfig(language: "de")])
        #expect(observed.map(\.transcript) == [nil], "set around the run, not the build")
    }
}
