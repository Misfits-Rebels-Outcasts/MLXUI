import Testing
import Foundation
@testable import MLXUI

/// CF-2 (`RSI/DelegateClefBacklog.md`) — the `mlx-community/clef-flash-4bit` catalog entry and its
/// gap flag. No runner exists until CF-5b, so nothing may claim the entry yet: Clef shares
/// `RunnerKind.decision` + `source: .mlx` with Laya and CLM, and only their `family ==` guards keep
/// either SDK from loading Clef's weights through the wrong engine.
///
/// `@MainActor` for the same reason `CatFlowChatModuleTests` is: the bundled `BrowserData`
/// conformance to `Decodable` is main-actor isolated under this project's
/// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, so decoding it from a nonisolated test warns
/// (and is an error in Swift 6 language mode).
@MainActor
struct ClefCatalogTests {
    private func clefEntry() -> ModelEntry {
        makeEntry(
            id: "mlx-community--clef-flash-4bit", family: "Clef", displayName: "Clef Flash",
            modelType: .decision, source: .mlx, ramGB: 9.32, downloadSizeGB: 6.21,
            hfRepo: "mlx-community", hfModelId: "mlx-community/clef-flash-4bit",
            contextWindow: 16384)
    }

    private func layaEntry() -> ModelEntry {
        makeEntry(
            id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
            modelType: .decision, source: .mlx, ramGB: 1.3, downloadSizeGB: 0.85,
            hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
    }

    private func clmEntry() -> ModelEntry {
        makeEntry(
            id: "RealityCat--CLM-v0.1-8B-MLX-8bit", family: "CLM", displayName: "CLM 8B",
            modelType: .decision, source: .mlx, ramGB: 12.2, downloadSizeGB: 8.12,
            hfRepo: "RealityCat", hfModelId: "RealityCat/CLM-v0.1-8B-MLX-8bit")
    }

    /// Shape of `RoutingTests.clmCatalogEntryDecodes`, with the CF-1 numbers
    /// (`RSI/journal/2026-490` §0: 11 files = 6,213,884,601 B → 6.21 GB; × 1.5 → 9.32 GB;
    /// `total_parameters` 9,409,813,744 from `Cloudflare/clef-flash`'s index → 9.4).
    @Test func clefCatalogEntryDecodes() throws {
        let json = """
        {
            "id": "mlx-community--clef-flash-4bit",
            "family": "Clef",
            "displayName": "Clef Flash",
            "paramSize": "9B",
            "paramCountB": 9.4,
            "modelType": "decision",
            "source": "mlx",
            "format": "mlx-4bit",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "mlx-community",
            "hfModelId": "mlx-community/clef-flash-4bit",
            "ramGB": 9.32,
            "downloadSizeGB": 6.21,
            "contextWindow": 16384,
            "license": "apache-2.0",
            "variants": [
                {
                    "quantization": "4-bit",
                    "format": "mlx-4bit",
                    "ramGB": 9.32,
                    "downloadSizeGB": 6.21,
                    "qualityPercent": 90,
                    "hfModelId": "mlx-community/clef-flash-4bit",
                    "recommended": true
                }
            ]
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        #expect(entry.modelType == .decision)
        #expect(entry.runnerKind == .decision)
        #expect(entry.source == .mlx)
        #expect(entry.family == "Clef")
        #expect(entry.paramCountB == 9.4)
        #expect(entry.ramGB == 9.32)
        #expect(entry.downloadSizeGB == 6.21)
        #expect(entry.contextWindow == 16384)
    }

    /// The entry really is in the bundled catalog's `decision` domain, after Laya and CLM.
    @Test func clefIsInTheBundledDecisionDomain() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        let decision = try #require(catalog.domains.first { $0.id == "decision" })
        let ids = decision.allModels.map(\.id)
        #expect(ids == ["aac6fef--laya-mlx", "RealityCat--CLM-v0.1-8B-MLX-8bit", "mlx-community--clef-flash-4bit"])
        let clef = try #require(decision.allModels.last)
        #expect(clef.family == "Clef")
        #expect(clef.displayName == "Clef Flash")
        #expect(clef.hfModelId == "mlx-community/clef-flash-4bit")
        #expect(clef.modelType == .decision)
        #expect(clef.format == "mlx-4bit")
        // rule 12 provenance, all from `RSI/notes/clef/hf/` @ `b5adbe2c…` (journal 2026-490 §0):
        // the CF-4 file set is 6,213,884,601 B → 6.21 GB, and 1.5× gives 9.32 GB. `paramCountB`
        // comes from the base repo's index (`Cloudflare/clef-flash` @ `8b2e5fd1…`,
        // `metadata.total_parameters` 9,409,813,744).
        #expect(clef.paramCountB == 9.4)
        #expect(clef.ramGB == 9.32)
        #expect(clef.downloadSizeGB == 6.21)
        #expect(clef.license == "apache-2.0")
        #expect(clef.minMacOSVersion == "13.0")
        #expect(clef.platforms == ["macOS 13+"])
        let variant = try #require(clef.variants?.first)
        #expect(variant.recommended == true)
        #expect(variant.ramGB == 9.32)
        #expect(variant.downloadSizeGB == 6.21)
        #expect(variant.hfModelId == "mlx-community/clef-flash-4bit")
        // The 4-bit convention: of the 25 `mlx-4bit` variants already in the catalog, 23 say 90
        // (the two exceptions are Flux and sam3, at 85). Implementer's call, pending owner
        // confirmation (journal 2026-491).
        #expect(variant.qualityPercent == 90)
    }

    /// The gap's whole purpose, and the reason CF-2 is safe before CF-8: Browse lists the card,
    /// but no flow row offers it. `TaskModels.derivedModels` filters on `ModelSupport`
    /// (`FlowKit/TaskModels.swift:386`), so this holds even for a model the registry claims —
    /// which Clef is not (`noModuleClaimsClef`), so it is doubly excluded today.
    @Test func clefIsNotOfferedByClassifyGateOrScoreYet() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: url.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap(\.allModels)
        // Worst case: pretend every entry is claimable, so only the ModelSupport gap can exclude it.
        let claimable = Set(catalog.map(\.id))
        for task in ["Classify", "Gate", "Score"] {
            let pool = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(pool.contains { $0.modelEntry?.hfModelId == "aac6fef/laya-mlx" },
                    "\(task)'s pool must be non-empty — otherwise this test proves nothing")
            #expect(!pool.contains { $0.modelEntry?.id == "mlx-community--clef-flash-4bit" },
                    "\(task) must not offer Clef Flash until CF-5b lands the runner")
        }
    }

    /// Browse must say "not yet supported" until CF-5b ports the runner.
    @Test func clefHasSupportGap() {
        #expect(ModelSupport.unsupportedReason(for: clefEntry()) != nil)
        // The flag is keyed to Clef only.
        #expect(ModelSupport.unsupportedReason(for: layaEntry()) == nil)
        #expect(ModelSupport.unsupportedReason(for: clmEntry()) == nil)
    }

    /// No module claims Clef yet: Laya's and CLM's `family ==` guards refuse it, and the
    /// registry resolves it to nothing. Laya and CLM still resolve to their own SDKs.
    @Test func noModuleClaimsClef() {
        #expect(LayaSDK().claim(clefEntry()) == .no)
        #expect(LayaUI().claim(clefEntry()) == .no)
        #expect(CLMSDK().claim(clefEntry()) == .no)
        #expect(CLMUI().claim(clefEntry()) == .no)

        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        #expect(registry.bestModule(for: clefEntry()) == nil)
        #expect(registry.bestModule(for: layaEntry())?.descriptor.id == "laya")
        #expect(registry.bestModule(for: clmEntry())?.descriptor.id == "clm")
    }
}
