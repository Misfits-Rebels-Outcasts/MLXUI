import Testing
import Foundation
@testable import MLXUI

/// CF-8 — the AM-W wiring for Clef Flash: the bridge row, the curated manifest, and the three decider
/// pools. The "exactly three pools" and "Laya is still the default" checks are the CF-1/C1 contract
/// expressed as CI rather than prose.
struct ClefWorkflowWiringTests {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // PipelineStudio
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    @MainActor
    private func claimable(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    // MARK: - The bridge row

    @Test func clefResolvesAsSame() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "Clef Flash"))
        #expect(entry.pinnedID == "mlx-community/clef-flash-4bit")
        #expect(entry.candidates == ["mlx-community/clef-flash-4bit"])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "clef-flash-4bit.json")
        switch CatalogBridge.resolve("Clef Flash", catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == "mlx-community/clef-flash-4bit")
            #expect(slot.modelEntry?.runnerKind == .decision)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
    }

    // MARK: - The curated manifest, read from the built app bundle

    @Test func clefManifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "clef-flash-4bit.json"))
        #expect(manifest.id == "mlx-community/clef-flash-4bit")
        #expect(manifest.display == "Clef Flash")
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "clef-mlx")
        #expect(manifest.tasks == ["Classify", "Gate", "Score"])
        #expect(manifest.settings.isEmpty, "no prose, no settings: the engine has no knobs")
        // `capabilities.constrained_decoding` is the reference's field (Q234 explains why a decider
        // carries it); `CuratedManifest` doesn't model it, so only the absence of rejected settings
        // is Swift-side meaningful.
        #expect(manifest.capabilities?.rejectedSettings == nil)
        #expect(manifest.resources?.diskGB == 6.21)
        // Gate F2's measured peak at 8,192 tokens, rounded up to one decimal (journal 2026-509):
        // 8,061 MiB = 8.45 GB, so 8.5.
        #expect(manifest.resources?.ramGB == 8.5)
        // `CuratedManifest` doesn't model `checksum`/`license`, so pin them where they live: the file.
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/clef-flash-4bit.json")
        let json = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        #expect(json["checksum"] as? String
                    == "sha256:19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
                "the joint_head.safetensors LFS oid from tree.json — identical in both HF repos")
        #expect(json["license"] as? String == "Apache-2.0")
        #expect(json["pinned"] as? Bool == true)
    }

    // MARK: - Exactly the three decider pools, with Laya still the default

    @MainActor
    @Test func clefIsInExactlyTheThreeDeciderPoolsAndLayaIsStillTheDefault() throws {
        let catalog = try loadCatalog()
        let claimable = claimable(catalog)

        for task in ["Classify", "Gate", "Score"] {
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(derived.contains { $0.modelEntry?.id == "mlx-community--clef-flash-4bit" },
                    "\(task) must offer Clef Flash")
            #expect(derived.last?.modelEntry?.id == "mlx-community--clef-flash-4bit",
                    "Clef Flash is appended **last** to \(task) (gate C1)")
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable)
                        == "Laya 0.4B",
                    "\(task) still defaults to Laya 0.4B — Clef joins the pool, never the default")
        }

        // Not an LLM task's model: `llmModels` is the generative pool (Generate/Summarize/…).
        for task in ["Generate", "Summarize", "Translate"] {
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(!derived.contains { $0.modelEntry?.id == "mlx-community--clef-flash-4bit" },
                    "\(task) must not offer a decision engine")
        }
    }

    // MARK: - The seed table itself (CF-8-FIX-1) — not just the derived pool

    /// Gate C1 as ruled: `"Clef Flash"` is appended **last** to Classify/Gate/Score in
    /// `TaskModels.taskModels`, after CLM 8B, and `"Laya 0.4B"` stays first — the rule is about the
    /// table, so the table is what this asserts. The derived-pool test above would pass even if the
    /// table lost the entry (the pool also reflects the live catalog), which is exactly the gap
    /// Review CF-8 found.
    @MainActor
    @Test func clefIsAppendedLastInTheSeedTableAndLayaStaysFirst() throws {
        for task in ["Classify", "Gate", "Score"] {
            let seed = TaskModels.seededModels(forTask: task)
            #expect(seed.first == "Laya 0.4B", "\(task): Laya 0.4B is the seed (gate D)")
            #expect(seed.contains("CLM 8B"), "\(task): CLM 8B is in the seed (CL-7)")
            #expect(seed.last == "Clef Flash", "\(task): Clef Flash is appended **last** (gate C1)")
            // The two deciders sit after every LLM, in the order they joined.
            let clmIndex = try #require(seed.firstIndex(of: "CLM 8B"))
            let clefIndex = try #require(seed.firstIndex(of: "Clef Flash"))
            #expect(clefIndex == clmIndex + 1, "\(task): Clef Flash directly follows CLM 8B")
            #expect(seed.lastIndex(of: "Laya 0.4B") == 0, "\(task): Laya appears only as the seed")
        }
        // The default is unchanged — the reason the append matters ("joins the pool, never the default").
        let catalog = try loadCatalog()
        for task in ["Classify", "Gate", "Score"] {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog,
                                            claimableModelIDs: claimable(catalog)) == "Laya 0.4B",
                    "\(task) still defaults to Laya 0.4B")
        }
    }
}
