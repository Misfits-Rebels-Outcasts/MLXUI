import Foundation
import Testing
@testable import MLXUI

/// MP2-W — "Parakeet TDT 0.6B v3" is reachable from AI Workflows (`RSI/DelegateModelPackBacklog.md` MP2-W; owner
/// rulings V and AE, 2026-10-24): one `CatalogBridge` row (`.same`), one MLXUI-authored curated manifest, and a
/// last-slot append to the `"Transcribe"` pool, so the seed ("Whisper Tiny") does not move and Parakeet is never
/// the default. No new task: `Transcribe` already exists, is served, and has an ASR runner kind.
/// Written to extend, not edit, `CatFlowCatalogBridgeTests` (whose count guard is bumped by convention).
struct ParakeetWorkflowWiringTests {

    private static let display = "Parakeet TDT 0.6B v3"
    private static let hfID = "mlx-community/parakeet-tdt-0.6b-v3"
    private static let cardID = "mlx-community--parakeet-tdt-0.6b-v3"

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

    /// 3. `taskKinds` + the derived pool: Parakeet is a runnable `.asr` model per the live catalog, the registry's
    /// claim and `ModelSupport`.
    @MainActor
    @Test func table3_parakeetIsInTheDerivedTranscribePool() throws {
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
        #expect(entry.manifestFile == "parakeet-tdt-0.6b-v3.json")
        let catalog = try loadCatalog()
        switch CatalogBridge.resolve(Self.display, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.id == Self.cardID)
            #expect(slot.modelEntry?.modelType == .asr)
            #expect(slot.modelEntry?.ramGB == 6.6)
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
        let manifest = try #require(CuratedManifest.load(manifestFile: "parakeet-tdt-0.6b-v3.json"))
        #expect(manifest.id == Self.hfID)
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "mlx-audio-swift")
        #expect(manifest.tasks == ["Transcribe"])
        #expect(manifest.settings.isEmpty, "T4: the engine takes no language or timestamps setting")
        #expect(manifest.resources?.diskGB == 2.51)
        #expect(manifest.resources?.ramGB == 6.6)
    }

    @Test func manifestResourcesMatchTheCatalogCard() throws {
        let card = try #require(try loadCatalog().first { $0.id == Self.cardID })
        let manifest = try #require(CuratedManifest.load(manifestFile: "parakeet-tdt-0.6b-v3.json"))
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    /// `CuratedManifest` does not decode `notes`, `license` or `checksum`, so read the shipped JSON itself.
    @Test func manifestSaysItIsMLXUIAuthoredAndFollowsTheChecksumConvention() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/parakeet-tdt-0.6b-v3.json")
        let json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        #expect(json["license"] as? String == "CC-BY-4.0")
        #expect(json["pinned"] as? Bool == true)
        #expect(json["manifest_version"] as? Int == 1)
        #expect(json["source"] as? String == "https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3")
        // MP1-W's convention for MLXUI-authored manifests (no catflow source to inherit a checksum from): none, and say so.
        #expect(json["checksum"] == nil)
        let notes = try #require(json["notes"] as? String)
        #expect(notes.contains("MLXUI-authored"))
        #expect(notes.contains("ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"))
        #expect(notes.contains("no checksum field"))
        // MP2-R-FIX-1 (rulings AF, AI, AJ): the manifest cites the measurement, not the MP-0 G estimate.
        #expect(notes.contains("measured figure"))
        #expect(notes.contains("in-app peak 6.10 GiB = 6.55 GB"))
        #expect(notes.contains("MP-0 G still governs unmeasured entries"))
    }

    // MARK: - the pool

    /// Appended **last**, after Voxtral; the seed ("Whisper Tiny") is unchanged, and Parakeet is never the default.
    @MainActor
    @Test func parakeetIsLastInTheTranscribePoolWithoutMovingTheSeed() throws {
        let pool = TaskModels.models(forTask: "Transcribe")
        #expect(pool == ["Whisper Tiny", "Whisper Small", "Whisper Large v3", "Voxtral Mini 4B Realtime", Self.display])
        #expect(TaskModels.seededModels(forTask: "Transcribe").first == "Whisper Tiny")
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        #expect(TaskModels.defaultModel(forTask: "Transcribe", catalog: catalog, claimableModelIDs: claimable) == "Whisper Tiny")
        // Only Transcribe changed: no other task's pool names Parakeet.
        for task in ["Speak", "Describe Image", "OCR", "Generate", "Embed"] {
            #expect(!TaskModels.models(forTask: task).contains(Self.display), "\\(task)")
        }
    }
}
