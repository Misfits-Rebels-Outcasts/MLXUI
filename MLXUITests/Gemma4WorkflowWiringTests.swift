import Foundation
import Testing
@testable import MLXUI

/// MP1-W — "Gemma 4 E4B" is reachable from AI Workflows: a `CatalogBridge` row (`.same`, explicit `.llm`),
/// an MLXUI-authored curated manifest, and a last-slot append to `TaskModels.llmModels`
/// (`RSI/DelegateModelPackBacklog.md`, MP-0 E ruled 2026-10-21). No new task: `servedRefNamePrefixes` and
/// `taskKinds` are untouched. MP1-R-FIX-1 (owner ruling 2026-10-22) withdrew the "Gemma 4 E4B Vision" row, its
/// manifest and its `"Describe Image"` slot (mlx-swift-lm 3.31.4's VLM `Gemma4` cannot load the checkpoint); the
/// tests below pin that it is gone. The type-aware `displayName(for:)` fix (MP1-W) stays: it still names the
/// Qwen3.5 9B Vision card (MoC-6-2).
/// Written to extend, not edit, `MiMoWorkflowWiringTests` and the `CatFlowCatalogBridgeTests` pool guards.
struct Gemma4WorkflowWiringTests {

    private static let llmDisplay = "Gemma 4 E4B"
    private static let withdrawnVisionDisplay = "Gemma 4 E4B Vision"
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

    @Test func theRowIsSameAndPinnedToTheRepo() throws {
        let entry = try #require(CatalogBridge.entry(for: Self.llmDisplay))
        #expect(entry.pinnedID == Self.hfID)
        #expect(entry.candidates == [Self.hfID])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "gemma-4-e4b-it-4bit.json")
        #expect(entry.modelType == .llm)
    }

    @Test func theDisplayResolvesToItsCard() throws {
        let catalog = try loadCatalog()
        switch CatalogBridge.resolve(Self.llmDisplay, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.id == "mlx-community--gemma-4-e4b-it-4bit")
            #expect(slot.modelEntry?.modelType == .llm)
            #expect(slot.modelEntry?.hfModelId == Self.hfID)
            #expect(slot.modelEntry?.ramGB == 7.77)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let name, let reason, _):
            Issue.record("\(name) should resolve: \(reason)")
        }
    }

    /// MP1-R-FIX-1: the Vision twin is withdrawn everywhere a flow could reach it.
    @Test func theVisionTwinIsWithdrawn() throws {
        #expect(CatalogBridge.entry(for: Self.withdrawnVisionDisplay) == nil)
        #expect(CatalogBridge.entries.filter { $0.candidates.contains(Self.hfID) }.map(\.display) == [Self.llmDisplay])
        #expect(!TaskModels.models(forTask: "Describe Image").contains(Self.withdrawnVisionDisplay))
        #expect(CuratedManifest.load(manifestFile: "gemma-4-e4b-it-4bit-vision.json") == nil)
        #expect(!(try loadCatalog().contains { $0.id == "mlx-community--gemma-4-e4b-it-4bit-vision" }))
    }

    /// Two cards sharing one repo must each be named by **their own** bridge row. `TaskModels.displayName(for:)`
    /// used to take the first row listing the repo, so the Qwen3.5 Vision card was offered in the Describe Image
    /// menu as "Qwen3.5 9B", a name that resolves to the text card: picking it wrote a row `CatalogBridge` could
    /// not run as image Q&A (MoC-6-2). MP1-W fixed that; the Gemma 4 pair that exposed it is gone (MP1-R-FIX-1).
    @Test func eachSharedRepoCardIsNamedByItsOwnBridgeRow() throws {
        let catalog = try loadCatalog()
        let pairs: [(id: String, display: String)] = [
            ("mlx-community--gemma-4-e4b-it-4bit", Self.llmDisplay),
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

    /// MP1-W-FIX-1: the type-aware `displayName(for:)` changed what the Describe Image menu **offers** for the Qwen3.5
    /// Vision card ("Qwen3.5 9B" → "Qwen3.5 9B Vision"). It must not change what an **already saved** row resolves to.
    /// Before the fix that card was offered as "Qwen3.5 9B", so a saved `.cat` row can name it that way; `resolve`
    /// was never touched and maps that name to the Qwen3.5 9B **text** card (`modelType: .llm`), exactly as it did,
    /// so no existing flow silently changes model. The explicit "Qwen3.5 9B Vision" row still resolves to the vision card.
    @Test func aSavedRowNamingTheOldQwenVisionDisplayStillResolvesToTheSameCardAsBefore() throws {
        let catalog = try loadCatalog()
        guard case .runnable(let old, let oldEquivalence, let oldNote) = CatalogBridge.resolve("Qwen3.5 9B", catalog: catalog) else {
            Issue.record("\"Qwen3.5 9B\" should still resolve"); return
        }
        #expect(old.modelEntry?.id == "mlx-community--Qwen3.5-9B-MLX-4bit")
        #expect(old.modelEntry?.modelType == .llm)
        #expect(oldEquivalence == .same)
        #expect(oldNote == nil)
        // The forward table for that display is the one it always was.
        #expect(CatalogBridge.entry(for: "Qwen3.5 9B")?.modelType == .llm)
        #expect(CatalogBridge.entry(for: "Qwen3.5 9B")?.manifestFile == "qwen3.5-9b-4bit.json")

        guard case .runnable(let vision, _, _) = CatalogBridge.resolve("Qwen3.5 9B Vision", catalog: catalog) else {
            Issue.record("\"Qwen3.5 9B Vision\" should still resolve"); return
        }
        #expect(vision.modelEntry?.id == "mlx-community--Qwen3.5-9B-MLX-4bit-vision")
        #expect(vision.modelEntry?.modelType == .vision)
    }

    // MARK: - the manifests

    /// AM-W step 6: the manifest must be in the **app bundle**, not just in `Resources/`.
    @Test func theManifestLoadsFromTheAppBundle() throws {
        let llm = try #require(CuratedManifest.load(manifestFile: "gemma-4-e4b-it-4bit.json"))
        #expect(llm.id == Self.hfID)
        #expect(llm.display == Self.llmDisplay)
        #expect(llm.kind == "local")
        #expect(llm.engine == "mlx-lm")
        #expect(llm.tasks == ["Generate", "Decide"])
        let temperature = try #require(llm.settings["temperature"])
        #expect(temperature.min == 0.0)
        #expect(temperature.max == 2.0)
        #expect(temperature.defaultValue == .number(0.6))        // owner ruling 2026-10-21: 0.6 like the other LLM manifests
        let maxTokens = try #require(llm.settings["max_tokens"])
        #expect(maxTokens.min == 1)
        #expect(maxTokens.max == 32768)
        #expect(maxTokens.defaultValue == .number(2048))
    }

    @Test func theManifestSaysItIsMLXUIAuthoredAndCiteTheRevision() throws {
        for file in ["gemma-4-e4b-it-4bit.json"] {
            // `CuratedManifest` does not decode `notes`, so read the shipped JSON itself.
            let url = repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/\(file)")
            let json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            let notes = try #require(json["notes"] as? String)
            #expect(notes.contains("MLXUI-authored"), "\(file)")
            #expect(notes.contains("475b9088d29754a3379866cf5aeb6b41acd313c2"), "\(file)")
            #expect(!notes.lowercased().contains("audio input"), "\(file)")
            if file == "gemma-4-e4b-it-4bit.json" {
                // The reason for 0.6 rather than generation_config.json's 1.0 is recorded where the next reader looks.
                #expect(notes.contains("0.6") && notes.contains("top_k 64") && notes.contains("top_p 0.95"), "\(file)")
            }
        }
    }

    @Test func manifestResourcesMatchTheCatalogCards() throws {
        let catalog = try loadCatalog().filter { $0.hfModelId == Self.hfID }
        #expect(catalog.count == 1)
        for file in ["gemma-4-e4b-it-4bit.json"] {
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
            #expect(!pool.contains(Self.withdrawnVisionDisplay), "\(task)")
        }
        let derived = TaskModels.derivedModels(for: "Generate", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.contains { $0.displayName == Self.llmDisplay }, "the Generate menu should offer \(Self.llmDisplay)")
    }

    /// Describe Image is back to its pre-MP1 pool: the seed ("LFM2-VL 1.6B") never moved, and no Gemma 4 name is in it.
    @MainActor
    @Test func describeImageIsUnchangedByThisPack() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        #expect(TaskModels.models(forTask: "Describe Image") == ["LFM2-VL 1.6B", "Gemma 3 4B", "Qwen3.5 9B Vision"])
        #expect(TaskModels.defaultModel(forTask: "Describe Image", catalog: catalog, claimableModelIDs: claimable) == "LFM2-VL 1.6B")
        let derived = TaskModels.derivedModels(for: "Describe Image", catalog: catalog, claimableModelIDs: claimable)
        #expect(!derived.contains { $0.displayName.contains("Gemma 4") })
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
        for task in TaskCatalog.catalog.keys where !llmKeyed.contains(task) {
            #expect(!TaskModels.models(forTask: task).contains { $0.hasPrefix("Gemma 4") }, "\(task)")
        }
    }

    /// No new task: the task the model joins already exists and is served.
    @Test func joinsOnlyExistingTasks() {
        #expect(TaskCatalog.catalog["Generate"] != nil)
    }
}
