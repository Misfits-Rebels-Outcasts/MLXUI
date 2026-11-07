import Foundation
import Testing
@testable import MLXUI

/// MC2-6 — TranslateGemma 4B in AI Workflows: the `Translate` row only. A `CatalogBridge` row (`.same`, explicit
/// `.llm`), an MLXUI-authored curated manifest, `taskExtraKinds["Translate"] = [.translate]`, the pool appended last, and
/// `RealExecutor`'s `.translate` branch (target from `to=`, source detected, raw input text to the stage) with the
/// Properties-tab preview sharing the same resolution.
/// Records what the run handed the stage builder.
private nonisolated final class TranslateWiringSeenBuild: @unchecked Sendable {
    private let lock = NSLock()
    private var config: StageConfig?
    private var calls = 0
    func note(_ config: StageConfig) { lock.lock(); self.config = config; calls += 1; lock.unlock() }
    var observed: StageConfig? { lock.lock(); defer { lock.unlock() }; return config }
    var callCount: Int { lock.lock(); defer { lock.unlock() }; return calls }
}

private nonisolated final class TranslateWiringSeenText: @unchecked Sendable {
    private let lock = NSLock()
    private var text: String?
    func note(_ text: String) { lock.lock(); self.text = text; lock.unlock() }
    var observed: String? { lock.lock(); defer { lock.unlock() }; return text }
}

private nonisolated struct TranslateWiringEchoStage: PipelineStage, @unchecked Sendable {
    let id = "translate-stub"
    let name = "TranslateGemma 4B"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    let seen: TranslateWiringSeenText
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        guard case .text(let text) = input else { throw StageError.kindMismatch(expected: .text, got: input.kind) }
        seen.note(text)
        return .text("Terjemahan: \(text)")
    }
}

struct TranslateGemmaWorkflowWiringTests {

    private static let display = "TranslateGemma 4B"
    private static let hfID = "mlx-community/translategemma-4b-it-4bit"
    private static let cardID = "mlx-community--translategemma-4b-it-4bit"
    private static let manifestFile = "translategemma-4b-it-4bit.json"
    private static let english = "The weather is lovely today, and the river is calm."

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
    }

    @MainActor
    private func registry() -> ModelRegistry {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return registry
    }

    @MainActor
    private func claimableIDs(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = registry()
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    // MARK: - The four tables

    @Test func table1_theRowAndTheKindAreUnchanged() throws {
        let desc = try #require(TaskCatalog.get("Translate"))
        #expect(desc.refKind == .frame, "Translate is a frame task; the dedicated model branches off before the frame")
        #expect(TaskModels.runnerKind(forTask: "Translate") == .llm)
        #expect(TaskModels.isServedByExecutor("Translate"))
    }

    /// Table 2/3: the `.translate` model is in exactly the `Translate` derived pool, and in no other task's.
    @MainActor
    @Test func table2theTranslatorIsOfferedForTranslateOnly() throws {
        let catalog = try loadCatalog()
        let card = try #require(catalog.first { $0.id == Self.cardID })
        #expect(card.runnerKind == .translate)
        #expect(ModelSupport.unsupportedReason(for: card) == nil)
        let claimable = claimableIDs(catalog)
        #expect(claimable.contains(Self.cardID))
        let translatePool = TaskModels.derivedModels(for: "Translate", catalog: catalog, claimableModelIDs: claimable)
        #expect(translatePool.contains { $0.modelEntry?.id == Self.cardID }, "Translate must offer TranslateGemma 4B")
        for task in TaskCatalog.allTasks().map(\.name) where task != "Translate" {
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(!derived.contains { $0.modelEntry?.id == Self.cardID },
                    "\(task) must not offer the translator")
        }
    }

    @MainActor
    @Test func table3thePoolAppendMovesNoSeed() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        // Appended last to Translate, and the seed is still the first LLM.
        #expect(TaskModels.models(forTask: "Translate").last == Self.display)
        #expect(TaskModels.models(forTask: "Translate").dropLast().last == "Gemma 4 E4B")
        #expect(TaskModels.defaultModel(forTask: "Translate", catalog: catalog, claimableModelIDs: claimable) == "Ministral 3B")
        // Every other LLM-keyed task still seeds Ministral 3B, and their pools never name the translator. (Classify,
        // Gate and Score are deliberately not in this list: their pools are prepended with the decision encoder "Laya
        // 0.4B", so their seed is Laya, unchanged by this ticket.)
        for task in ["Generate", "Summarize", "Answer", "Rewrite", "Draft", "Ask", "Title", "Critique",
                     "Verify", "Revise", "Merge", "Text to Table"] {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Ministral 3B",
                    "\(task)'s seed moved")
            #expect(!TaskModels.models(forTask: task).contains(Self.display), "\(task)'s pool names the translator")
        }
    }

    @Test func table4theBridgeResolvesTheNameTheCatWrites() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: Self.display))
        #expect(entry.pinnedID == Self.hfID)
        #expect(entry.candidates == [Self.hfID])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == Self.manifestFile)
        #expect(entry.modelType == .llm)
        switch CatalogBridge.resolve(Self.display, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == Self.hfID)
            #expect(slot.modelEntry?.runnerKind == .translate)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
        // The reverse lookup the derived pool uses names the row by its display.
        #expect(CatalogBridge.entry(forCandidate: Self.hfID, modelType: .llm)?.display == Self.display)
        #expect(CatalogBridge.entries.filter { $0.candidates.contains(Self.hfID) }.count == 1)
        let card = try #require(catalog.first { $0.hfModelId == Self.hfID })
        #expect(TaskModels.displayName(for: card) == Self.display)
    }

    // MARK: - The manifest

    @Test func theManifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifestFile))
        #expect(manifest.id == Self.hfID)
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "mlx-lm")
        #expect(manifest.tasks == ["Translate"])
        #expect(manifest.settings["temperature"]?.defaultValue == .number(0))   // decision 7: greedy by default
        #expect(manifest.settings["temperature"]?.min == 0.0)
        #expect(manifest.settings["max_tokens"]?.defaultValue == .number(2048)) // decisions 8
        #expect(manifest.settings["max_tokens"]?.max == 2048)
        // Licence and notes are not decoded by `CuratedManifest` (no consumers); read the file the bundle shipped.
        let raw = try String(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/\(Self.manifestFile)"),
                             encoding: .utf8)
        #expect(raw.contains("\"license\": \"Gemma Terms of Use\""))     // owner decision 11
        #expect(raw.contains("5788ec08c047f3f2e17808101b8d9566ac930d58"), "the manifest names the pinned revision")
        #expect(raw.contains("do_sample"), "the manifest says the checkpoint's sampling config is not used")
    }

    @Test func manifestResourcesMatchTheCatalogEntry() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifestFile))
        let card = try #require(try loadCatalog().first { $0.hfModelId == Self.hfID })
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    @Test func theManifestDeclaresOnlyTheLanguagesTheRowUses() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifestFile))
        // The v0.9 language declares `to=` on `Translate`; the manifest's schema must not invent a `from=` setting.
        #expect(manifest.settings.keys.sorted() == ["max_tokens", "temperature"])
        #expect(!manifest.settings.keys.contains("from"))
        #expect(!manifest.settings.keys.contains("to"))
    }

    // MARK: - The executor path

    private func executor(
        detect: @escaping @Sendable (String) -> String? = { _ in "en" },
        make: @escaping @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage
    ) throws -> RealExecutor {
        let catalog = try loadCatalog()
        let ws = FlowWorkspace(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("translategemma-wiring-\(UUID().uuidString)"))
        var exec = RealExecutor(workspace: ws, flowID: "f",
                                blobDirectory: ws.root.appendingPathComponent("blobs"),
                                makeModelStage: make, installedModelIDs: [Self.cardID], catalog: catalog)
        exec.detectLanguage = detect
        return exec
    }

    private func input(_ text: String = TranslateGemmaWorkflowWiringTests.english) -> Asset {
        Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }

    @Test func aTranslateRowRunsThroughTheDedicatedStageWithTheDetectedSource() async throws {
        let seen = TranslateWiringSeenBuild()
        let text = TranslateWiringSeenText()
        let exec = try executor(detect: { _ in "en" }) { _, config in
            seen.note(config)
            return TranslateWiringEchoStage(seen: text)
        }
        let row = Row(task: "Translate", model: Self.display, settings: "to=Malay")
        let out = try await exec.execute(path: "2", row: row, inputs: [input()],
                                         transcript: nil, context: nil, usedFlowContent: nil)
        let config = try #require(seen.observed)
        #expect(config.sourceLanguage == "en")
        #expect(config.targetLanguage == "ms")               // the row's `to=Malay`, resolved through the table
        // The manifest's sampler defaults reach the stage: greedy and the model's 2K output cap.
        #expect(config.temperature == 0)
        #expect(config.maxTokens == 2048)
        // The stage received the row's raw input text, not a rendered frame.
        #expect(text.observed == Self.english)
        #expect(out.items.first?.value == "Terjemahan: \(Self.english)")
    }

    @Test func toAcceptsANameOrACode() async throws {
        for (setting, expected) in [("to=Malay", "ms"), ("to=ms", "ms"), ("to=en_GB", "en-GB"),
                                    ("to=German", "de"),
                                    ("to=\"Chinese (Traditional)\"", "zh-Hant")] {
            let seen = TranslateWiringSeenBuild()
            let exec = try executor { _, config in
                seen.note(config)
                return TranslateWiringEchoStage(seen: TranslateWiringSeenText())
            }
            _ = try await exec.execute(path: "1", row: Row(task: "Translate", model: Self.display, settings: setting),
                                       inputs: [input()], transcript: nil, context: nil, usedFlowContent: nil)
            #expect(seen.observed?.targetLanguage == expected, "\(setting)")
        }
    }

    @Test func anUnknownTargetFailsTheRowAndNamesIt() async throws {
        let seen = TranslateWiringSeenBuild()
        let exec = try executor { _, config in
            seen.note(config)
            return TranslateWiringEchoStage(seen: TranslateWiringSeenText())
        }
        do {
            _ = try await exec.execute(path: "2", row: Row(task: "Translate", model: Self.display, settings: "to=Klingon"),
                                       inputs: [input()], transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("to=Klingon must fail the row")
        } catch {
            // The interpreter/runner wraps this as "Row 2 failed: …" (FlowRunner's `.rowFailed` mapping).
            let sentence = String(describing: error)
            #expect(sentence.contains("Klingon"))
            #expect(sentence.contains("isn't a language"))
            #expect(sentence.contains("to=Malay"))
        }
        #expect(seen.callCount == 0, "an unknown target must fail before any model stage is built")
    }

    @Test func aMissingTargetFailsTheRow() async throws {
        let exec = try executor { _, _ in TranslateWiringEchoStage(seen: TranslateWiringSeenText()) }
        for settings in [nil, ""] as [String?] {
            do {
                _ = try await exec.execute(path: "3", row: Row(task: "Translate", model: Self.display, settings: settings),
                                           inputs: [input()], transcript: nil, context: nil, usedFlowContent: nil)
                Issue.record("a TranslateGemma row without to= must fail")
            } catch {
                #expect(String(describing: error).contains("has no target language"))
            }
        }
    }

    @Test func anUndetectableSourceFailsTheRowInsteadOfGuessing() async throws {
        let seen = TranslateWiringSeenBuild()
        let exec = try executor(detect: { _ in nil }) { _, config in
            seen.note(config)
            return TranslateWiringEchoStage(seen: TranslateWiringSeenText())
        }
        do {
            _ = try await exec.execute(path: "4", row: Row(task: "Translate", model: Self.display, settings: "to=Malay"),
                                       inputs: [input("...")], transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("no detection must fail the row, never translate from a guess")
        } catch {
            let sentence = String(describing: error)
            #expect(sentence.contains("can't tell which language"))
            #expect(sentence.contains("Translate screen"))
        }
        #expect(seen.callCount == 0)
    }

    // MARK: - Preview and run agree (FV-2)

    @Test func thePreviewRendersTheSameRequestTheRowResolves() throws {
        let preview = try TranslateRowRequest.preview(settings: "to=Malay", inputText: Self.english,
                                                      detect: { _ in "en" })
        #expect(preview.sourceCode == "en")
        #expect(preview.targetCode == "ms")
        #expect(preview.sourceWasDetected)
        // The preview's text is the run's prompt: the pure renderer and the template are proven identical by MC2-3/4.
        #expect(preview.rendered == (try TranslateGemmaPrompt.render(sourceCode: "en", targetCode: "ms", text: Self.english)))
        #expect(preview.rendered.hasPrefix("<bos><start_of_turn>user"))
        #expect(preview.rendered.contains("English (en) to Malay (ms) translator"))
    }

    @Test func thePreviewUsesTheDocumentedFallbackForItsStandInText() throws {
        // The Properties tab passes "⟨the text from row 1⟩", which has no detectable language.
        let preview = try TranslateRowRequest.preview(settings: "to=Malay",
                                                      inputText: "⟨the text from row 1⟩",
                                                      detect: { _ in nil })
        #expect(!preview.sourceWasDetected)
        #expect(preview.sourceCode == TranslateRowRequest.previewFallbackSourceCode)
        #expect(preview.rendered.contains("English (en) to Malay (ms) translator"))
    }

    @Test func thePreviewRefusesTheSameBadTargetAsTheRun() {
        #expect(throws: TranslateRowRequest.Failure.unknownTarget("Klingon")) {
            _ = try TranslateRowRequest.preview(settings: "to=Klingon", inputText: Self.english, detect: { _ in "en" })
        }
        #expect(throws: TranslateRowRequest.Failure.missingTarget) {
            _ = try TranslateRowRequest.preview(settings: "", inputText: Self.english, detect: { _ in "en" })
        }
    }

    @Test func aBareChineseTargetIsRefusedNamingBothWritableCodes() {
        for raw in ["Chinese", "chinese"] {
            do {
                _ = try TranslateRowRequest.targetCode(settings: "to=\(raw)")
                Issue.record("to=\(raw) must be refused")
            } catch let failure as TranslateRowRequest.Failure {
                #expect(failure == .unknownTarget(raw))
                let message = failure.description
                #expect(message.contains("to=zh-Hans"), "\(message)")
                #expect(message.contains("to=zh-Hant"), "\(message)")
            } catch {
                Issue.record("unexpected error \(error)")
            }
        }
    }

    @Test func theTranslatePoolIsExactlyGeneratePlusTranslateGemmaAndNoOtherTaskHasIt() {
        #expect(TaskModels.models(forTask: "Translate") == TaskModels.models(forTask: "Generate") + [Self.display])
        let others = ["Generate", "Summarize", "Answer", "Rewrite", "Draft", "Ask", "Title", "Critique", "Verify",
                      "Revise", "Merge", "Text to Table", "Transcribe", "Embed", "Speak", "Rerank"]
        for task in others {
            #expect(!TaskModels.models(forTask: task).contains(Self.display), "\(task) must not offer \(Self.display)")
        }
    }

    @Test func theRowInspectorRecognisesTheTranslateModel() throws {
        let catalog = try loadCatalog()
        #expect(RealExecutor.translateBoundModel(for: Row(task: "Translate", model: Self.display), catalog: catalog)?.id
                == Self.cardID)
        // A row naming a normal LLM is not a translate-bound row, so it keeps the frame preview.
        #expect(RealExecutor.translateBoundModel(for: Row(task: "Translate", model: "Ministral 3B"), catalog: catalog) == nil)
        #expect(RealExecutor.translateBoundModel(for: Row(task: "Summarize", model: Self.display), catalog: catalog)?.id
                == Self.cardID, "the helper is about the model, not the task name")
    }
}
