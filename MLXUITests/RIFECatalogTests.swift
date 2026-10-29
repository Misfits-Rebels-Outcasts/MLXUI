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

    /// The 1.5x weights rule would give 0.03 GB. ramGB is the measured peak of the largest run the app offers (RIFE-5c, 2026-10-29:
    /// 4K 4x at Scale 0.5, 6061 MiB = 6.36 GB, rounded up to 6.4).
    @Test func ramGBIsMeasuredNotTheWeightsRule() throws {
        let e = try rife()
        #expect(e.ramGB > e.downloadSizeGB * 1.5 * 10, "not the 1.5x rule")
        #expect(e.ramGB == 6.4)
        #expect(e.description?.contains("RAM figure is measured") == true)
        #expect(e.description?.contains("is an estimate") == false)
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

    /// The RIFE module is the only one that claims it, and the RIFE-1 `ModelSupport` "no runner yet" gap is gone now that the Run screen exists (RIFE-4).
    @Test func exactlyOneModuleClaimsItAndTheRIFE1GapIsGone() throws {
        let e = try rife()
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        #expect(full.bestModule(for: e)?.descriptor.id == "rife")
        var claimers: [String] = []
        for module in installedModules {
            let one = ModelRegistry()
            module.register(into: one)
            if one.bestModule(for: e) != nil { claimers.append(module.descriptor.id) }
        }
        #expect(claimers == ["rife"], "a chat/upscale module must not be handed an interpolate model: \(claimers)")
        #expect(ModelSupport.unsupportedReason(for: e) == nil)
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

// RIFE-2: the description and metadata.
extension RIFECatalogTests {

    /// Every field `ModelDetailView` and `ComparisonTableView` render, pinned so the only dashes are the sanctioned nulls:
    /// Speed (no tokens/s for a frame interpolator), Context, benchmarks (the card's parity figures are not quality benchmarks)
    /// and languages (a vision model).
    @Test func metadataRendersWithNoUnsanctionedDash() throws {
        let e = try rife()
        #expect(e.architecture?.hasPrefix("IFNet") == true)
        #expect(e.descriptionSource == "mlx-community/RIFE-4.25")
        #expect(e.taskTags == ["video-frame-interpolation", "frame-interpolation", "slow-motion"])
        #expect(e.lastUpdated == "2026-06-06")   // HF lastModified
        #expect(e.communityLikes == 3)
        #expect(e.communityDownloads != nil)
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        day.isLenient = false
        #expect(e.lastUpdated.flatMap { day.date(from: $0) } != nil)
        #expect(e.formattedDownloads() != "—")
        #expect(e.speedTokensPerSec == nil)
        #expect(e.contextWindow == nil)
        #expect(e.benchmarks == nil)
        #expect(e.languages == nil)
        #expect(DataNormalizer.normalizeLicense(e.license) != "—")
        #expect(!e.paramSize.isEmpty)
        #expect(e.summary?.isEmpty == false)
    }

    /// What it does, what it doesn't, why fp32, and the attribution the MIT licence asks for (hzwer). No quality claim the card
    /// doesn't make: only its own parity figure may be cited, and the description cites none.
    @Test func descriptionStatesTheModesTheAttributionAndNoInventedClaims() throws {
        let d = try #require(try rife().description)
        #expect(d.contains("2×") && d.contains("4×"))
        #expect(d.contains("hzwer"))
        #expect(d.contains("MIT"))
        #expect(d.contains("fp32"))
        #expect(d.contains("does not make the picture sharper"))
        for banned in ["FPS", "faster than", "best", "state-of-the-art", "PSNR", "SSIM"] {
            #expect(!d.contains(banned), "unsupported claim: \(banned)")
        }
    }
}
