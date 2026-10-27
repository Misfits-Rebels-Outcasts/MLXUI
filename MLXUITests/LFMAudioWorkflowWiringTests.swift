import Foundation
import Testing
@testable import MLXUI

/// MP6-W -- "LFM2.5-Audio 1.5B" is reachable from AI Workflows as a `Transcribe` and a `Speak` model
/// (`RSI/DelegateModelPackBacklog.md` MP6-W; MP-0 I). One `CatalogBridge` row (`.same`), one MLXUI-authored curated manifest listing both
/// tasks, a `.voice` entry in `TaskModels.taskExtraKinds` for exactly those two tasks, and a last-slot append to both pools. Voice
/// conversation is not a flow task and is not reachable. The model is a different `RunnerKind` (`.voice`) from the tasks' own (`.asr`,
/// `.tts`), so each of the four tables in CLAUDE.md gets a pin for each task.
struct LFMAudioWorkflowWiringTests {

    private static let display = "LFM2.5-Audio 1.5B"
    private static let hfID = "mlx-community/LFM2.5-Audio-1.5B-6bit"
    private static let cardID = "mlx-community--LFM2.5-Audio-1.5B-6bit"
    private static let manifest = "lfm2.5-audio-1.5b-6bit.json"

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

    // MARK: - the four tables, per task

    /// 1. `TaskCatalog`: both tasks exist, model-class engine tasks; none was added.
    @Test func table1_bothTasksAreInTheTaskCatalog() throws {
        let transcribe = try #require(TaskCatalog.get("Transcribe"))
        #expect(transcribe.refName == "engines.asr.transcribe")
        let speak = try #require(TaskCatalog.get("Speak"))
        #expect(speak.refName == "engines.tts.speak")
        #expect(TaskCatalog.get("Voice Chat") == nil && TaskCatalog.get("Converse") == nil, "voice conversation is not a flow task")
    }

    /// 2. `servedRefNamePrefixes`: the executor serves both, and the kinds the tasks themselves name are unchanged.
    @Test func table2_theExecutorServesBothTasks() {
        #expect(TaskModels.isServedByExecutor("Transcribe"))
        #expect(TaskModels.isServedByExecutor("Speak"))
        #expect(TaskModels.runnerKind(forTask: "Transcribe") == .asr)
        #expect(TaskModels.runnerKind(forTask: "Speak") == .tts)
    }

    /// 3. `taskKinds` + the derived pool: a `.voice` model is in exactly the Transcribe and Speak pools and in no other task's.
    @MainActor
    @Test func table3_theVoiceModelIsInTheTwoDerivedPoolsAndNoOther() throws {
        let catalog = try loadCatalog()
        let card = try #require(catalog.first { $0.id == Self.cardID })
        #expect(card.runnerKind == .voice)
        #expect(ModelSupport.unsupportedReason(for: card) == nil)
        let claimable = claimableIDs(catalog)
        #expect(claimable.contains(Self.cardID))
        for task in ["Transcribe", "Speak"] {
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(derived.contains { $0.displayName == Self.display }, "\(task)")
        }
        // Every other task: not in the derived pool, whatever it is.
        for task in TaskCatalog.allTasks().map(\.name) where task != "Transcribe" && task != "Speak" {
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(!derived.contains { $0.modelEntry?.id == Self.cardID }, "\(task) must not offer the voice model")
        }
    }

    /// 4. `CatalogBridge.entries`: the name the `.cat` writes resolves to the installable card, `.same`, no note.
    @Test func table4_theBridgeResolvesTheNameTheCatWrites() throws {
        let entry = try #require(CatalogBridge.entry(for: Self.display))
        #expect(entry.pinnedID == Self.hfID)
        #expect(entry.candidates == [Self.hfID])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == Self.manifest)
        let catalog = try loadCatalog()
        switch CatalogBridge.resolve(Self.display, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.id == Self.cardID)
            #expect(slot.modelEntry?.modelType == .voice)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let name, let reason, _):
            Issue.record("\(name) should resolve: \(reason)")
        }
        let card = try #require(catalog.first { $0.id == Self.cardID })
        #expect(TaskModels.displayName(for: card) == Self.display)
    }

    // MARK: - the pools

    /// Appended **last** to both pools; neither seed moves, and the model is never a default.
    @MainActor
    @Test func theVoiceModelIsLastInBothPoolsWithoutMovingEitherSeed() throws {
        #expect(TaskModels.models(forTask: "Transcribe") == ["Whisper Tiny", "Whisper Small", "Whisper Large v3", "Voxtral Mini 4B Realtime",
                                                              "Parakeet TDT 0.6B v3", "Qwen3-ASR 1.7B", Self.display])
        #expect(TaskModels.models(forTask: "Speak") == ["Kokoro 82M", "Qwen3-TTS 1.7B", "Qwen3-TTS 0.6B", Self.display])
        #expect(TaskModels.seededModels(forTask: "Transcribe").first == "Whisper Tiny")
        #expect(TaskModels.seededModels(forTask: "Speak").first == "Kokoro 82M")
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        #expect(TaskModels.defaultModel(forTask: "Transcribe", catalog: catalog, claimableModelIDs: claimable) == "Whisper Tiny")
        #expect(TaskModels.defaultModel(forTask: "Speak", catalog: catalog, claimableModelIDs: claimable) == "Kokoro 82M")
        // Only those two pools changed.
        for task in TaskCatalog.allTasks().map(\.name) where task != "Transcribe" && task != "Speak" {
            #expect(!TaskModels.models(forTask: task).contains(Self.display), "\(task)")
        }
    }

    // MARK: - a .cat row naming it

    private func issues(for text: String) throws -> [FlowIssue] {
        let registry = try DirectoryFlowRegistry.load(directory: repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models"))
        return FlowValidator.checkFlow(try CatParser.parseForValidation(text), registry: registry)
    }

    @Test func aTranscribeRowAndASpeakRowNamingItPassCheck() throws {
        let transcribe = try issues(for: """
        mlxflow 0.8
        1. Read Audio     memo.m4a
        2. Transcribe     LFM2.5-Audio 1.5B
        3. Save Text      out.txt
        """)
        #expect(transcribe.isEmpty, "\(transcribe)")
        let speak = try issues(for: """
        mlxflow 0.8
        1. Read Text      note.txt
        2. Speak          LFM2.5-Audio 1.5B
        3. Save Audio     out.wav
        """)
        #expect(speak.isEmpty, "\(speak)")
    }

    /// Settings are empty on purpose. Measured fact (journal 2026-612): `check` does **not** flag a bare token such as a Kokoro voice
    /// on this row, and the SDK ignores it (the model has no voice list). So a stray `af_heart` is accepted and has no effect.
    @Test func aStrayVoiceTokenPassesCheckAndHasNoEffect() throws {
        let found = try issues(for: """
        mlxflow 0.8
        1. Read Text      note.txt
        2. Speak          LFM2.5-Audio 1.5B; af_heart
        3. Save Audio     out.wav
        """)
        #expect(found.isEmpty, "\(found)")
        // The manifest declares no setting at all, so there is nothing for the token to mean.
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifest))
        #expect(manifest.settings.isEmpty)
    }

    // MARK: - the executor path

    private func executor(installed: Bool = true, make: @escaping @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage) throws -> RealExecutor {
        let catalog = try loadCatalog()
        let ws = FlowWorkspace(root: FileManager.default.temporaryDirectory.appendingPathComponent("lfm-wiring-\(UUID().uuidString)"))
        return RealExecutor(workspace: ws, flowID: "f", blobDirectory: ws.root.appendingPathComponent("blobs"),
                            makeModelStage: make, installedModelIDs: installed ? [Self.cardID] : [], catalog: catalog)
    }

    /// `stageConfig` tells the SDK which task asked, and the SDK returns the matching stage for each.
    @MainActor
    @Test func theExecutorsConfigPicksTheSpeechToTextOrTextToSpeechStage() throws {
        let exec = try executor { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .voice) }
        let transcribeDesc = try #require(TaskCatalog.get("Transcribe"))
        let speakDesc = try #require(TaskCatalog.get("Speak"))
        let asrConfig = try exec.stageConfig(for: transcribeDesc, row: Row(task: "Transcribe", model: Self.display, settings: ""))
        let ttsConfig = try exec.stageConfig(for: speakDesc, row: Row(task: "Speak", model: Self.display, settings: ""))
        #expect(asrConfig.flowTask == "Transcribe")
        #expect(ttsConfig.flowTask == "Speak")
        let card = try #require(try loadCatalog().first { $0.id == Self.cardID })
        let sdk = try #require(registry().bestModule(for: card)?.sdk)
        let asr = try sdk.makeStage(for: card, config: asrConfig)
        #expect(asr.accepts == .audio && asr.produces == .text && asr.id.hasPrefix("lfmaudio.asr."))
        let tts = try sdk.makeStage(for: card, config: ttsConfig)
        #expect(tts.accepts == .text && tts.produces == .audio && tts.id.hasPrefix("lfmaudio.tts."))
        // The Run UI's plain default config is still speech to text (MP6-3a), and the two tasks are two engine-cache keys.
        #expect(try sdk.makeStage(for: card, config: .default).id.hasPrefix("lfmaudio.asr."))
        #expect(asrConfig != ttsConfig)
        // A row for another model sets the same tag; no other SDK reads it (the Kokoro row still gets its voice).
        let kokoro = try exec.stageConfig(for: speakDesc, row: Row(task: "Speak", model: "Kokoro 82M", settings: "af_heart"))
        #expect(kokoro.voice == "af_heart" && kokoro.flowTask == "Speak")
    }

    /// A `Speak` row naming it reaches `makeModelStage` with this entry and the `Speak` tag; a cancelled build is a
    /// `CancellationError`, not a flow failure.
    @Test func aSpeakRowReachesTheStageFactoryAndACancelIsNotAnError() async throws {
        let seen = LFMWiringSeen()
        let exec = try executor { model, config in
            seen.set(model.id, config.flowTask)
            throw CancellationError()
        }
        let row = Row(task: "Speak", model: Self.display, settings: "")
        let input = Asset(items: [Item(kind: .text, value: "Hello there.", path: nil, sourceText: nil)])
        do {
            _ = try await exec.execute(path: "1", row: row, inputs: [input])
            Issue.record("expected a cancellation")
        } catch is CancellationError {
        } catch {
            Issue.record("a cancelled row must surface as CancellationError, got \(error)")
        }
        #expect(seen.value.0 == Self.cardID)
        #expect(seen.value.1 == "Speak")
    }

    @Test func aRowNamingItBeforeItIsInstalledSaysSoRatherThanRunning() async throws {
        let exec = try executor(installed: false) { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .voice) }
        let row = Row(task: "Speak", model: Self.display, settings: "")
        let input = Asset(items: [Item(kind: .text, value: "Hello.", path: nil, sourceText: nil)])
        do {
            _ = try await exec.execute(path: "1", row: row, inputs: [input])
            Issue.record("expected modelNotInstalled")
        } catch StageError.modelNotInstalled(let id) {
            #expect(id == Self.cardID)
        } catch {
            Issue.record("unexpected \(error)")
        }
    }

    // MARK: - Serve

    /// Voice models are not served: the Local Server's model list shows `.llm` models only (`ServeToggleVisibility.shouldShow`).
    @Test func theLocalServerDoesNotOfferAVoiceModel() {
        #expect(ServeToggleVisibility.shouldShow(state: .installed, isInstalledOnDisk: true, runnerKind: .llm, gateAvailable: true))
        #expect(!ServeToggleVisibility.shouldShow(state: .installed, isInstalledOnDisk: true, runnerKind: .voice, gateAvailable: true))
    }

    // MARK: - the manifest

    @Test func manifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifest))
        #expect(manifest.id == Self.hfID)
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "mlx-audio-swift")
        #expect(manifest.tasks == ["Transcribe", "Speak"])
        #expect(manifest.settings.isEmpty, "no invented voice or language setting")
        #expect(manifest.resources?.diskGB == 1.73)
        #expect(manifest.resources?.ramGB == 2.6)
    }

    @Test func manifestResourcesMatchTheCatalogCard() throws {
        let card = try #require(try loadCatalog().first { $0.id == Self.cardID })
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifest))
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    @Test func manifestSaysItIsMLXUIAuthoredAndTheRAMFigureAnEstimate() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/\(Self.manifest)")
        let json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        #expect(json["pinned"] as? Bool == true)
        #expect(json["manifest_version"] as? Int == 1)
        #expect(json["source"] as? String == "https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-6bit")
        #expect(json["checksum"] == nil)
        let notes = try #require(json["notes"] as? String)
        #expect(notes.contains("MLXUI-authored"))
        #expect(notes.contains("ESTIMATE"))
        #expect(notes.contains("MP6-3d"))
        #expect(notes.contains("English only"))
    }
}

nonisolated final class LFMWiringSeen: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: (String?, String?) = (nil, nil)
    func set(_ id: String, _ task: String?) { lock.withLock { _value = (id, task) } }
    var value: (String?, String?) { lock.withLock { _value } }
}
