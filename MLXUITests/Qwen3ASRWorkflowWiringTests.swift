import Foundation
import Testing
@testable import MLXUI

/// MP3-W — "Qwen3-ASR 1.7B" is reachable from AI Workflows (`RSI/DelegateModelPackBacklog.md` MP3-W; MP-0 ruled
/// 2026-10-21): one `CatalogBridge` row (`.same`), one MLXUI-authored curated manifest, and a
/// last-slot append to the `"Transcribe"` pool, so the seed ("Whisper Tiny") does not move and Qwen3-ASR is never
/// the default. No new task: `Transcribe` already exists, is served, and has an ASR runner kind.
/// Written to extend, not edit, `CatFlowCatalogBridgeTests` (whose count guard is bumped by convention).
struct Qwen3ASRWorkflowWiringTests {

    private static let display = "Qwen3-ASR 1.7B"
    private static let hfID = "mlx-community/Qwen3-ASR-1.7B-8bit"
    private static let cardID = "mlx-community--Qwen3-ASR-1.7B-8bit"

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
    }

    @MainActor
    private func claimableIDs(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    // MARK: - the four tables (CLAUDE.md), one pin each

    /// 1. `TaskCatalog`: `Transcribe` exists, audio -> text, a model-class engine task.
    @Test func table1_transcribeIsInTheTaskCatalog() throws {
        let task = try #require(TaskCatalog.get("Transcribe"))
        #expect(task.refKind == .engine)
        #expect(task.refName == "engines.asr.transcribe")
    }

    /// 2. `servedRefNamePrefixes`: the executor genuinely serves `engines.asr.*`.
    @Test func table2_theExecutorServesTranscribe() {
        #expect(TaskModels.isServedByExecutor("Transcribe"))
        #expect(TaskModels.runnerKind(forTask: "Transcribe") == .asr)
    }

    /// 3. `taskKinds` + the derived pool: Qwen3-ASR is a runnable `.asr` model per the live catalog, the registry's
    /// claim and `ModelSupport`.
    @MainActor
    @Test func table3_qwen3ASRIsInTheDerivedTranscribePool() throws {
        let catalog = try loadCatalog()
        let card = try #require(catalog.first { $0.id == Self.cardID })
        #expect(card.runnerKind == .asr)
        #expect(ModelSupport.unsupportedReason(for: card) == nil)
        let claimable = claimableIDs(catalog)
        #expect(claimable.contains(Self.cardID))
        let derived = TaskModels.derivedModels(for: "Transcribe", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.contains { $0.displayName == Self.display })
    }

    /// 4. `CatalogBridge.entries`: the name the `.cat` writes resolves to the installable card, `.same`, no note.
    @Test func table4_theBridgeResolvesTheNameTheCatWrites() throws {
        let entry = try #require(CatalogBridge.entry(for: Self.display))
        #expect(entry.pinnedID == Self.hfID)
        #expect(entry.candidates == [Self.hfID])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "qwen3-asr-1.7b-8bit.json")
        let catalog = try loadCatalog()
        switch CatalogBridge.resolve(Self.display, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.id == Self.cardID)
            #expect(slot.modelEntry?.modelType == .asr)
            #expect(slot.modelEntry?.ramGB == 7.3)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let name, let reason, _):
            Issue.record("\\(name) should resolve: \\(reason)")
        }
        // The Browse card and the bridge row agree on the name, so the menu shows what the card is called.
        let card = try #require(catalog.first { $0.id == Self.cardID })
        #expect(TaskModels.displayName(for: card) == Self.display)
    }

    // MARK: - the manifest

    /// AM-W step 6: the manifest must be in the **app bundle**, not just in `Resources/`.
    @Test func manifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "qwen3-asr-1.7b-8bit.json"))
        #expect(manifest.id == Self.hfID)
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "mlx-audio-swift")
        #expect(manifest.tasks == ["Transcribe"])
        #expect(manifest.settings.isEmpty, "T4: no invented lang or timestamps setting")
        #expect(manifest.resources?.diskGB == 2.47)
        #expect(manifest.resources?.ramGB == 7.3)
    }

    @Test func manifestResourcesMatchTheCatalogCard() throws {
        let card = try #require(try loadCatalog().first { $0.id == Self.cardID })
        let manifest = try #require(CuratedManifest.load(manifestFile: "qwen3-asr-1.7b-8bit.json"))
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    /// `CuratedManifest` does not decode `notes`, `license` or `checksum`, so read the shipped JSON itself.
    @Test func manifestSaysItIsMLXUIAuthoredAndFollowsTheChecksumConvention() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/qwen3-asr-1.7b-8bit.json")
        let json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        #expect(json["license"] as? String == "Apache-2.0")
        #expect(json["pinned"] as? Bool == true)
        #expect(json["manifest_version"] as? Int == 1)
        #expect(json["source"] as? String == "https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit")
        // MP1-W's convention for MLXUI-authored manifests (no catflow source to inherit a checksum from): none, and say so.
        #expect(json["checksum"] == nil)
        let notes = try #require(json["notes"] as? String)
        #expect(notes.contains("MLXUI-authored"))
        #expect(notes.contains("a8379a2e2f9e313c9292cdf1af4055ab56d50d55"))
        #expect(notes.contains("no checksum field"))
        // MP3-1-FIX: the manifest cites the measurement, not the MP-0 G estimate (the Parakeet precedent).
        #expect(notes.contains("a measured figure"))
        #expect(notes.contains("in-app peak 6.82 GiB = 7.32 GB"))
        #expect(notes.contains("MP-0 G still governs unmeasured entries"))
        #expect(notes.contains("SPEC-Q"), "forcing a language in flows is a spec question, not a manifest edit (T4)")
    }

    // MARK: - the pool

    /// Appended **last**, after Voxtral and Parakeet; the seed ("Whisper Tiny") is unchanged, and Qwen3-ASR is never the default.
    @MainActor
    @Test func qwen3ASRIsLastInTheTranscribePoolWithoutMovingTheSeed() throws {
        let pool = TaskModels.models(forTask: "Transcribe")
        // MP6-W appended "LFM2.5-Audio 1.5B" after it (implementer's call, pending owner confirmation: the assertion follows the append).
        #expect(pool == ["Whisper Tiny", "Whisper Small", "Whisper Large v3", "Voxtral Mini 4B Realtime",
                         "Parakeet TDT 0.6B v3", Self.display, "LFM2.5-Audio 1.5B"])
        #expect(TaskModels.seededModels(forTask: "Transcribe").first == "Whisper Tiny")
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        #expect(TaskModels.defaultModel(forTask: "Transcribe", catalog: catalog, claimableModelIDs: claimable) == "Whisper Tiny")
        // Only Transcribe changed: no other task's pool names Qwen3-ASR.
        for task in ["Speak", "Describe Image", "OCR", "Generate", "Embed"] {
            #expect(!TaskModels.models(forTask: task).contains(Self.display), "\\(task)")
        }
    }
}
