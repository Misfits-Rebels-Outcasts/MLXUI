import Foundation
import Testing
@testable import MLXUI

/// MP1-W — "Gemma 4 E4B" and "Gemma 4 E4B Vision" are reachable from AI Workflows: two `CatalogBridge`
/// rows (`.same`, explicit `.llm` / `.vision`, both pinned to the one shared repo), two MLXUI-authored curated
/// manifests, and a last-slot append to `TaskModels.llmModels` and to `"Describe Image"`
/// (`RSI/DelegateModelPackBacklog.md`, MP-0 E ruled 2026-10-21). No new task: `servedRefNamePrefixes` and
/// `taskKinds` are untouched. The Qwen3.5 9B / Qwen3.5 9B Vision pair (MoC-6-2) is the precedent.
/// Written to extend, not edit, `MiMoWorkflowWiringTests` and the `CatFlowCatalogBridgeTests` pool guards.
struct Gemma4WorkflowWiringTests {

    private static let llmDisplay = "Gemma 4 E4B"
    private static let visionDisplay = "Gemma 4 E4B Vision"
    private static let hfID = "mlx-community/gemma-4-e4b-it-4bit"

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

    // MARK: - the bridge rows

    @Test func bothRowsAreSameAndPinnedToTheSharedRepo() throws {
        let llm = try #require(CatalogBridge.entry(for: Self.llmDisplay))
        let vision = try #require(CatalogBridge.entry(for: Self.visionDisplay))
        for (entry, file, type) in [(llm, "gemma-4-e4b-it-4bit.json", ModelType.llm),
                                    (vision, "gemma-4-e4b-it-4bit-vision.json", ModelType.vision)] {
            #expect(entry.pinnedID == Self.hfID)
            #expect(entry.candidates == [Self.hfID])
            #expect(entry.equivalence == .same)
            #expect(entry.manifestFile == file)
            #expect(entry.modelType == type)
        }
    }

    /// The two displays share one hfModelId, so each must resolve to **its own** card (by `modelType`), never to
    /// its sibling, and neither is notRunnable.
    @Test func eachDisplayResolvesToItsOwnCard() throws {
        let catalog = try loadCatalog()
        for (display, type, id) in [(Self.llmDisplay, ModelType.llm, "mlx-community--gemma-4-e4b-it-4bit"),
                                    (Self.visionDisplay, ModelType.vision, "mlx-community--gemma-4-e4b-it-4bit-vision")] {
            switch CatalogBridge.resolve(display, catalog: catalog) {
            case .runnable(let slot, let equivalence, let note):
                #expect(slot.modelEntry?.id == id, "\(display)")
                #expect(slot.modelEntry?.modelType == type)
                #expect(slot.modelEntry?.hfModelId == Self.hfID)
                #expect(slot.modelEntry?.ramGB == 7.77)
                #expect(equivalence == .same)
                #expect(note == nil)
            case .notRunnable(let name, let reason, _):
                Issue.record("\(name) should resolve: \(reason)")
            }
        }
    }

    /// Two rows list this repo (the Qwen3.5 pair does the same), and the reverse lookup the derived pool uses
    /// returns the first, the text one. That is the existing behaviour for a shared repo; recorded, not changed.
    @Test func theReverseLookupReturnsTheFirstOfTheTwoRows() {
        #expect(CatalogBridge.entries.filter { $0.candidates.contains(Self.hfID) }.map(\.display)
                == [Self.llmDisplay, Self.visionDisplay])
        #expect(CatalogBridge.entry(forCandidate: Self.hfID)?.display == Self.llmDisplay)
    }

    /// Two cards sharing one repo must each be named by **their own** bridge row. `TaskModels.displayName(for:)`
    /// used to take the first row listing the repo, so the Vision card was offered in the Describe Image menu as
    /// "Gemma 4 E4B", a name that resolves to the text card: picking it wrote a row `CatalogBridge` could not
    /// run as image Q&A. The Qwen3.5 9B pair (MoC-6-2) had the same defect; it is fixed by the same change.
    @Test func eachSharedRepoCardIsNamedByItsOwnBridgeRow() throws {
        let catalog = try loadCatalog()
        let pairs: [(id: String, display: String)] = [
            ("mlx-community--gemma-4-e4b-it-4bit", Self.llmDisplay),
            ("mlx-community--gemma-4-e4b-it-4bit-vision", Self.visionDisplay),
            ("mlx-community--Qwen3.5-9B-MLX-4bit", "Qwen3.5 9B"),
            ("mlx-community--Qwen3.5-9B-MLX-4bit-vision", "Qwen3.5 9B Vision"),
        ]
        for (id, display) in pairs {
            let card = try #require(catalog.first { $0.id == id })
            #expect(TaskModels.displayName(for: card) == display, "\(id)")
            // And the name round-trips: resolving it gives back the same card.
            if case .runnable(let slot, _, _) = CatalogBridge.resolve(display, catalog: catalog) {
                #expect(slot.modelEntry?.id == id, "\(display) should resolve back to \(id)")
            } else {
                Issue.record("\(display) should resolve")
            }
        }
    }

    // MARK: - the manifests

    /// AM-W step 6: both manifests must be in the **app bundle**, not just in `Resources/`.
    @Test func bothManifestsLoadFromTheAppBundle() throws {
        let llm = try #require(CuratedManifest.load(manifestFile: "gemma-4-e4b-it-4bit.json"))
        #expect(llm.id == Self.hfID)
        #expect(llm.display == Self.llmDisplay)
        #expect(llm.kind == "local")
        #expect(llm.engine == "mlx-lm")
        #expect(llm.tasks == ["Generate", "Decide"])
        let temperature = try #require(llm.settings["temperature"])
        #expect(temperature.min == 0.0)
        #expect(temperature.max == 2.0)
        #expect(temperature.defaultValue == .number(1.0))        // generation_config.json temperature 1.0
        let maxTokens = try #require(llm.settings["max_tokens"])
        #expect(maxTokens.min == 1)
        #expect(maxTokens.max == 32768)
        #expect(maxTokens.defaultValue == .number(2048))

        let vision = try #require(CuratedManifest.load(manifestFile: "gemma-4-e4b-it-4bit-vision.json"))
        #expect(vision.id == Self.hfID)
        #expect(vision.display == Self.visionDisplay)
        #expect(vision.engine == "mlx-vlm")
        #expect(vision.tasks == ["Describe Image"])
        #expect(vision.settings.isEmpty)
    }

    @Test func manifestsSayTheyAreMLXUIAuthoredAndCiteTheRevision() throws {
        for file in ["gemma-4-e4b-it-4bit.json", "gemma-4-e4b-it-4bit-vision.json"] {
            // `CuratedManifest` does not decode `notes`, so read the shipped JSON itself.
            let url = repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/\(file)")
            let json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            let notes = try #require(json["notes"] as? String)
            #expect(notes.contains("MLXUI-authored"), "\(file)")
            #expect(notes.contains("475b9088d29754a3379866cf5aeb6b41acd313c2"), "\(file)")
            #expect(!notes.lowercased().contains("audio input"), "\(file)")
        }
    }

    @Test func manifestResourcesMatchTheCatalogCards() throws {
        let catalog = try loadCatalog().filter { $0.hfModelId == Self.hfID }
        #expect(catalog.count == 2)
        for file in ["gemma-4-e4b-it-4bit.json", "gemma-4-e4b-it-4bit-vision.json"] {
            let manifest = try #require(CuratedManifest.load(manifestFile: file))
            for card in catalog {
                #expect(manifest.resources?.ramGB == card.ramGB, "\(file)")
                #expect(manifest.resources?.diskGB == card.downloadSizeGB, "\(file)")
            }
        }
    }

    // MARK: - the pools

    /// "Gemma 4 E4B" is appended **last**, after MiMo, so every llmModels-keyed task still seeds
    /// "Ministral 3B" and the new name is reachable from each task's Model menu.
    @MainActor
    @Test func gemma4IsLastInTheLLMPoolWithoutMovingTheSeed() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        let llmModelsKeyedTasks = ["Generate", "Summarize", "Translate", "Answer", "Rewrite",
                                   "Draft", "Ask", "Title", "Critique", "Verify", "Revise",
                                   "Merge", "Text to Table"]
        for task in llmModelsKeyedTasks {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Ministral 3B",
                    "\(task) should still seed Ministral 3B after \(Self.llmDisplay) was appended")
            let pool = TaskModels.models(forTask: task)
            #expect(pool.last == Self.llmDisplay, "\(task): last pool name")
            #expect(pool.dropLast().last == "MiMo V2.6 9B", "\(task): should follow MiMo V2.6 9B")
            #expect(!pool.contains(Self.visionDisplay), "\(task): the vision half belongs to Describe Image only")
        }
        let derived = TaskModels.derivedModels(for: "Generate", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.contains { $0.displayName == Self.llmDisplay }, "the Generate menu should offer \(Self.llmDisplay)")
    }

    /// "Gemma 4 E4B Vision" is appended **last** to Describe Image: the seed ("LFM2-VL 1.6B") does not move.
    @MainActor
    @Test func visionHalfIsLastInDescribeImageWithoutMovingTheSeed() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        #expect(TaskModels.models(forTask: "Describe Image")
                == ["LFM2-VL 1.6B", "Gemma 3 4B", "Qwen3.5 9B Vision", Self.visionDisplay])
        #expect(TaskModels.defaultModel(forTask: "Describe Image", catalog: catalog, claimableModelIDs: claimable) == "LFM2-VL 1.6B")
        #expect(!TaskModels.models(forTask: "Describe Image").contains(Self.llmDisplay))
        let derived = TaskModels.derivedModels(for: "Describe Image", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.contains { $0.displayName == Self.visionDisplay }, "the Describe Image menu should offer \(Self.visionDisplay)")
    }

    /// The seeds of the deciders keyed on the same array do not move, and no other task's pool gained a Gemma 4 name.
    @MainActor
    @Test func leavesEveryOtherSeedAndPoolAlone() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        for task in ["Classify", "Gate", "Score"] {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Laya 0.4B",
                    "\(task) should still seed Laya 0.4B")
        }
        let llmKeyed: Set<String> = ["Generate", "Summarize", "Translate", "Answer", "Rewrite", "Draft", "Ask", "Title",
                                     "Critique", "Verify", "Revise", "Merge", "Text to Table", "Classify", "Gate", "Score"]
        for task in TaskCatalog.catalog.keys where !llmKeyed.contains(task) && task != "Describe Image" {
            #expect(!TaskModels.models(forTask: task).contains { $0.hasPrefix("Gemma 4") }, "\(task)")
        }
    }

    /// No new task: both tasks the models join already exist and are served.
    @Test func joinsOnlyExistingTasks() {
        #expect(TaskCatalog.catalog["Generate"] != nil)
        #expect(TaskCatalog.catalog["Describe Image"] != nil)
    }
}
