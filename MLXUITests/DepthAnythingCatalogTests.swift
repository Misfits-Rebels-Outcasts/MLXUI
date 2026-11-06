import Testing
import Foundation
@testable import MLXUI

/// MC3-2: the Depth Anything V2 Small catalog entry, the new `depth` kind, the pinned HF file list and the
/// download selection. Facts come from `depth-anything/Depth-Anything-V2-Small-hf` at revision
/// `5426e4f0f36572d16453bbda7a8389317b1bef99` (fetched 2026-11-06; see `Fixtures/DepthAnything/repo-files.json`).
///
/// Until MC3-6 registers `DepthAnythingModule` the model installs but resolves to "no runner yet"; the module
/// claim is asserted where the module lands.
@MainActor
struct DepthAnythingCatalogTests {

    private static let hfID = "depth-anything/Depth-Anything-V2-Small-hf"

    private static var fixtureDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .appendingPathComponent("Fixtures/DepthAnything")
    }

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func depth() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    /// Trap 1: the licence lookalikes. Every other size (Base, Large), Video-Depth-Anything and DA3Mono is
    /// cc-by-nc-4.0; this entry must be the official Small checkpoint and nothing else.
    @Test func entryFactsMatchTheRepo() throws {
        // The exact HF repo the fixtures and the strict loader read from — never a mlx-community conversion.
        let fixtureURL = Self.fixtureDirectory.appendingPathComponent("repo-files.json")
        let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL)) as? [String: Any]
        #expect(fixture?["repository"] as? String == Self.hfID, "the pinned file-list fixture is the official repo")
        #expect(fixture?["revision"] as? String == "5426e4f0f36572d16453bbda7a8389317b1bef99")

        let e = try depth()
        #expect(e.id == "depth-anything--Depth-Anything-V2-Small-hf")
        #expect(e.hfModelId == "depth-anything/Depth-Anything-V2-Small-hf",
                "only the Small checkpoint is Apache-2.0; do not point this at Base/Large or an mlx-community repo")
        #expect(e.hfRepo == "depth-anything")
        #expect(e.displayName == "Depth Anything V2 Small")
        #expect(e.family == "depth-anything")
        #expect(e.modelType == .depth)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-fp32")
        #expect(e.architecture == "DepthAnythingForDepthEstimation")
        #expect(e.license == "apache-2.0", "the HF card's licence: Apache-2.0")
        #expect(e.licenseUrl == "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf")
        // model.safetensors is 99,173,660 B = 0.099 GB; with config.json + preprocessor_config.json the repo is ~0.10 GB.
        #expect(e.downloadSizeGB == 0.10)
        // 24.8 M fp32 parameters (safetensors header); the card rounds to 24.8 M.
        #expect(e.paramCountB == 0.025)
        #expect(e.paramSize == "0.025B")
        #expect(e.speedTokensPerSec == nil)
        #expect(e.speedHardware == nil)
        #expect(e.speedEstimated == nil)
        #expect(e.variants?.count == 1, "fp32 as published; no quantization (owner, 2026-11-06)")
        #expect(e.variants?.first?.quantization == "fp32")
        #expect(e.variants?.first?.format == "mlx-fp32")
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.variants?.first?.downloadSizeGB == e.downloadSizeGB)
        #expect(e.variants?.first?.hfModelId == Self.hfID)
        #expect(e.variants?.first?.recommended == true)
    }

    @Test func ramGBIsTheProvisionalWeightsEstimate() throws {
        let e = try depth()
        // MC3-1's provisional figure: 1.5 × the weights (0.10 GB), about 0.15 GB. MC3-7 books the measured
        // working set in `ModelLoadProfile`; the maintainers set the final `ramGB` from their own run.
        #expect(e.ramGB == 0.15)
        #expect(e.description?.contains("estimate") == true)
        #expect(e.description?.contains("not a measurement") == true)
    }

    @Test func kindRoutesToDepth() throws {
        let e = try depth()
        #expect(e.runnerKind == .depth)
        #expect(ModelType.depth.rawValue == "depth")
        #expect(RunnerKind.depth.rawValue == "depth")
        #expect(ModelType.depth.sfSymbol == "cube.transparent")
    }

    @Test func liveInItsOwnLeafInImagesAfterInterpolation() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "depth" })
        #expect(leaf.name == "Depth Estimation")
        #expect(leaf.sfSymbol == "cube.transparent")
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let section = try #require(data.sidebarSections.first { $0.id == "images-videos" })
        #expect(section.domainIds.contains("depth"))
        #expect(section.domainIds.last == "depth", "owner decision 3: after Frame Interpolation")
        #expect(data.sidebarSections.filter { $0.domainIds.contains("depth") }.map(\.id) == ["images-videos"])
        #expect(data.domainIndex["depth"] != nil)
        // The Frame Interpolation leaf is untouched.
        let interpolate = try #require(data.domains.first { $0.id == "interpolate" })
        #expect(!interpolate.allModels.contains { $0.hfModelId == Self.hfID })
    }

    /// Every field `ModelDetailView` and `ComparisonTableView` render, pinned so the only dashes are the
    /// sanctioned nulls: Speed (not a token generator), Context, benchmarks (the card's parity figures are not
    /// quality benchmarks) and languages (a vision model).
    @Test func metadataRendersWithNoUnsanctionedDash() throws {
        let e = try depth()
        #expect(e.descriptionSource == Self.hfID)
        #expect(e.taskTags == ["depth-estimation", "monocular-depth-estimation", "relative-depth"])
        #expect(e.lastUpdated == "2024-07-05")   // HF lastModified
        #expect(e.communityLikes == 75)
        #expect(e.communityDownloads == 1574470)
        #expect(e.formattedDownloads() == "1.6M")
        #expect(e.contextWindow == nil)
        #expect(e.benchmarks == nil)
        #expect(e.languages == nil)
        #expect(DataNormalizer.normalizeLicense(e.license) == "Apache 2.0")
        #expect(!e.paramSize.isEmpty)
        #expect(e.summary?.isEmpty == false)
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        day.isLenient = false
        #expect(e.lastUpdated.flatMap { day.date(from: $0) } != nil)
    }

    /// What it does, what it is not (relative depth, not metres), near = bright, the input limits and the
    /// Apache-2.0 notice. No quality claim the HF card doesn't make: no benchmark or superiority wording.
    @Test func descriptionStatesTheContractTheLimitsAndTheLicence() throws {
        let d = try #require(try depth().description)
        #expect(d.contains("relative"))
        #expect(d.contains("not a measurement in metres"))
        #expect(d.contains("near is bright") || d.contains("Near is bright") || d.contains("nearer the camera"))
        #expect(d.contains("8,192"))
        #expect(d.contains("3 : 1"))
        #expect(d.contains("8-bit grayscale PNG"))
        #expect(d.contains("Apache-2.0"))
        #expect(d.contains("Lihe Yang"))
        for banned in ["state-of-the-art", "SOTA", "best ", "PSNR", "RMSE", "outperforms"] {
            #expect(!d.contains(banned), "unsupported claim: \(banned)")
        }
    }

    /// The repo's siblings at the pinned revision, from the fixture. `ModelFileSelector` must take exactly the
    /// weights and the two JSON configs: README and .gitattributes are not downloaded.
    @Test func selectorTakesTheWeightsAndTwoConfigsOnly() throws {
        let url = Self.fixtureDirectory.appendingPathComponent("repo-files.json")
        let object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let files = try #require(object["files"] as? [[String: Any]])
        let siblings = files.compactMap { $0["path"] as? String }
        #expect(siblings == [".gitattributes", "README.md", "config.json", "model.safetensors", "preprocessor_config.json"])
        let selected = Set(ModelFileSelector.filesToDownload(siblings: siblings))
        #expect(selected == ["config.json", "model.safetensors", "preprocessor_config.json"])
        // The two configs carry the model's own dimensions and the reference preprocessor's rules.
        #expect(files.first { $0["path"] as? String == "model.safetensors" }?["size"] as? Int == 99173660)
    }

    @Test func noModelSupportGapAndStageFactoryRefusesTheKindExplicitly() throws {
        let e = try depth()
        #expect(ModelSupport.unsupportedReason(for: e) == nil)
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: e, installedModelIDs: [e.id], availableRAMGB: 16)
        }
    }
}
