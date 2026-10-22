import Foundation
import Testing
@testable import MLXUI

/// MM-W — "MiMo V2.6 9B" is reachable from AI Workflows: a `CatalogBridge` row (`.same`, explicit
/// `.llm`), the MLXUI-authored curated manifest `mimo-v2.6-9b-optiq-4bit.json`, and a last-slot
/// append to `TaskModels.llmModels` (MM-0 A and C, owner, 2026-10-17). It needs no new task, so
/// `servedRefNamePrefixes` and `taskKinds` are untouched. Written to extend, not edit, the
/// existing Qwen3.5 / Qwen3.8 wiring tests in `CatFlowCatalogBridgeTests`.
struct MiMoWorkflowWiringTests {

    private static let display = "MiMo V2.6 9B"
    private static let hfID = "mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit"

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    @MainActor
    private func claimableIDs(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    // MARK: - the bridge row

    @Test func resolvesAsSameToTheOneCatalogCard() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: Self.display))
        #expect(entry.pinnedID == Self.hfID)
        #expect(entry.candidates == [Self.hfID])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "mimo-v2.6-9b-optiq-4bit.json")
        #expect(entry.modelType == .llm)
        switch CatalogBridge.resolve(Self.display, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == Self.hfID)
            #expect(slot.modelEntry?.runnerKind == .llm)
            #expect(slot.modelEntry?.ramGB == 10.70)       // MM-0 B
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
    }

    /// Exactly one bridge row names this repo, and the reverse lookup the derived pool uses finds it.
    @Test func theReverseLookupNamesTheRowByItsDisplay() {
        #expect(CatalogBridge.entries.filter { $0.candidates.contains(Self.hfID) }.count == 1)
        #expect(CatalogBridge.entry(forCandidate: Self.hfID)?.display == Self.display)
    }

    // MARK: - the manifest

    /// AM-W step 6: the manifest must be in the **app bundle**, not just in `Resources/`. A
    /// file-system-synchronized group carries a plain file write into the target; this proves it.
    @Test func manifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "mimo-v2.6-9b-optiq-4bit.json"))
        #expect(manifest.id == Self.hfID)
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "mlx-lm")
        #expect(manifest.tasks == ["Generate", "Decide"])
        let temperature = try #require(manifest.settings["temperature"])
        #expect(temperature.min == 0.0)
        #expect(temperature.max == 2.0)
        #expect(temperature.defaultValue == .number(0.6))           // generation_config temp 0.6
        let maxTokens = try #require(manifest.settings["max_tokens"])
        #expect(maxTokens.min == 1)
        #expect(maxTokens.max == 32768)
        #expect(maxTokens.defaultValue == .number(2048))
    }

    /// The manifest's footprint is the catalog's (MM-0 B), so the two cannot drift apart.
    @Test func manifestResourcesMatchTheCatalogEntry() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "mimo-v2.6-9b-optiq-4bit.json"))
        let card = try #require(try loadCatalog().first { $0.hfModelId == Self.hfID })
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    // MARK: - the pool

    /// MM-0 C: "MiMo V2.6 9B" is appended **last** (after "Qwen3.8 27B"), so every
    /// llmModels-keyed task still seeds "Ministral 3B" and the new name is menu-reachable.
    @MainActor
    @Test func isLastInTheLLMPoolWithoutMovingTheSeed() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        let llmModelsKeyedTasks = ["Generate", "Summarize", "Translate", "Answer", "Rewrite",
                                   "Draft", "Ask", "Title", "Critique", "Verify", "Revise",
                                   "Merge", "Text to Table"]
        for task in llmModelsKeyedTasks {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Ministral 3B",
                    "\(task) should still seed Ministral 3B after \(Self.display) was appended")
            // MP1-W then appended "Gemma 4 E4B" after it (implementer's call, pending owner confirmation), so MiMo
            // is now second-to-last; the last slot is Gemma 4's (Gemma4WorkflowWiringTests).
            let pool = TaskModels.models(forTask: task)
            #expect(pool.dropLast().last == Self.display, "\(task): \(Self.display) should be the second-to-last pool name")
            #expect(pool.dropLast(2).last == "Qwen3.8 27B", "\(task): \(Self.display) should follow Qwen3.8 27B")
        }
        let derived = TaskModels.derivedModels(for: "Generate", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.contains { $0.displayName == Self.display }, "the Generate menu should offer \(Self.display)")
    }

    /// The seeds of the deciders keyed on the same array do not move either.
    @MainActor
    @Test func leavesTheDeciderSeedsAlone() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        for task in ["Classify", "Gate", "Score"] {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Laya 0.4B",
                    "\(task) should still seed Laya 0.4B")
        }
    }

    /// `Generate` is the contract's reachability check: the existing task serves it, so no new task
    /// (and no `servedRefNamePrefixes` / `taskKinds` edit) is involved.
    @MainActor
    @Test func generateIsAnExistingTaskThatStillSeedsMinistral() throws {
        let catalog = try loadCatalog()
        #expect(TaskCatalog.catalog["Generate"] != nil)
        #expect(TaskModels.defaultModel(forTask: "Generate", catalog: catalog, claimableModelIDs: claimableIDs(catalog)) == "Ministral 3B")
    }
}
