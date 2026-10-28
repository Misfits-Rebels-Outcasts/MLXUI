import Testing
import Foundation
@testable import MLXUI

/// RIFE-1: the RIFE 4.25 catalog entry and the new `interpolate` kind. Facts are from the HF repo
/// (mlx-community/RIFE-4.25 @ 6b650eaa, fetched 2026-10-28). Until RIFE-3d registers a module the model installs but
/// resolves to "no runner yet", and no workflow pool offers it (RIFE-W is skipped by owner ruling, 2026-10-28).
@MainActor
struct RIFECatalogTests {

    private static let hfID = "mlx-community/RIFE-4.25"

    /// The repo's siblings (api), in the tree's order.
    private static let siblings = [".gitattributes", "README.md", "config.json", "model.safetensors"]

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func rife() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    @Test func entryFactsMatchTheRepo() throws {
        let e = try rife()
        #expect(e.id == "mlx-community--RIFE-4.25")
        #expect(e.displayName == "RIFE 4.25")
        #expect(e.family == "RIFE")
        #expect(e.modelType == .interpolate)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-fp32")
        #expect(e.license == "mit")
        // model.safetensors is 22,674,262 B (usedStorage) = 0.0227 GB; the card says 22.7 MB.
        #expect(e.downloadSizeGB == 0.023)
        // 5,664,776 fp32 parameters (safetensors header).
        #expect(e.paramCountB == 0.0057)
        #expect(e.speedTokensPerSec == nil)
        #expect(e.variants?.count == 1, "fp32 as published; no quantization (owner, 2026-10-28)")
        #expect(e.variants?.first?.quantization == "fp32")
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.variants?.first?.hfModelId == Self.hfID)
    }

    /// The 1.5x weights rule would give 0.03 GB. ramGB is a provisional estimate until RIFE-5 measures 1080p and 4K (scale 0.5).
    @Test func ramGBIsAnEstimateNotTheWeightsRule() throws {
        let e = try rife()
        #expect(e.ramGB > e.downloadSizeGB * 1.5 * 10, "not the 1.5x rule")
        #expect(e.description?.contains("RAM figure is an estimate") == true)
    }

    @Test func kindRoutesToInterpolateNotUpscale() throws {
        let e = try rife()
        #expect(e.runnerKind == .interpolate)
        #expect(e.runnerKind != .upscale, "upscale is SeedVR2's dispatch")
        #expect(ModelType.interpolate.rawValue == "interpolate")
        #expect(RunnerKind.interpolate.rawValue == "interpolate")
        #expect(ModelType.interpolate.sfSymbol == "forward.frame")
    }

    @Test func liveInItsOwnLeafInImagesVideos() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "interpolate" })
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let section = try #require(data.sidebarSections.first { $0.id == "images-videos" })
        #expect(section.domainIds.contains("interpolate"))
        #expect(data.sidebarSections.filter { $0.domainIds.contains("interpolate") }.map(\.id) == ["images-videos"])
        #expect(data.domainIndex["interpolate"] != nil)
        // The SeedVR2 leaf is untouched.
        let upscale = try #require(data.domains.first { $0.id == "upscale" })
        #expect(!upscale.allModels.contains { $0.hfModelId == Self.hfID })
    }

    /// No module claims it before RIFE-3d, and `ModelSupport` flags it so Run isn't offered. REMOVE the gap assertion in RIFE-3d.
    @Test func noModuleClaimsItYetAndTheGapFlagsIt() throws {
        let e = try rife()
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        #expect(full.bestModule(for: e) == nil)
        #expect(ModelSupport.unsupportedReason(for: e) != nil)
    }

    @Test func stageFactoryRefusesTheKindWithAnExplicitCase() throws {
        let e = try rife()
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: e, installedModelIDs: [e.id], availableRAMGB: 16)
        }
    }

    @Test func noWorkflowPoolOffersIt() throws {
        let e = try rife()
        for task in TaskCatalog.allTasks().map(\.name) {
            #expect(TaskModels.runnerKind(forTask: task) != .interpolate, "\(task) names the interpolate kind")
            let pool = TaskModels.derivedModels(for: task, catalog: [e], claimableModelIDs: [e.id])
            #expect(!pool.contains { $0.id == e.id }, "\(task) offers RIFE")
        }
    }

    /// The repo has root `config.json` + `model.safetensors` and no tokenizer; README and .gitattributes are not downloaded.
    @Test func selectorTakesTheConfigAndWeightsOnly() {
        let out = Set(ModelFileSelector.filesToDownload(siblings: Self.siblings))
        #expect(out == ["config.json", "model.safetensors"])
    }
}
