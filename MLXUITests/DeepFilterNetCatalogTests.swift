import Testing
import Foundation
@testable import MLXUI

/// MP5-1: the DeepFilterNet 3 catalog entry and the new `enhance` kind. Facts are from the pinned HF files
/// (RSI/notes/modelpack/hf/mlx-community_DeepFilterNet-mlx @ 220d5dfb). Until MP5-3 registers a module the model
/// installs but resolves to "no runner yet", and no workflow pool offers it.
@MainActor
struct DeepFilterNetCatalogTests {

    private static let hfID = "mlx-community/DeepFilterNet-mlx"

    /// The pinned repo's siblings (api.json), in the tree's order.
    private static let siblings = [
        ".gitattributes", "README.md", "convert_deepfilternet.py",
        "v1/README.md", "v1/config.json", "v1/model.safetensors",
        "v2/README.md", "v2/config.json", "v2/model.safetensors",
        "v3/README.md", "v3/config.json", "v3/model.safetensors",
    ]

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func deepFilterNet() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    @Test func entryFactsMatchThePinnedRepo() throws {
        let e = try deepFilterNet()
        #expect(e.id == "mlx-community--DeepFilterNet-mlx")
        #expect(e.displayName == "DeepFilterNet 3")
        #expect(e.family == "DeepFilterNet")
        #expect(e.modelType == .enhance)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-fp32")
        #expect(e.license == "mit")   // api.json cardData.license
        // Pinned tree: v1+v2+v3 model.safetensors 25,585,573 B + three config.json 2,568 B = 25,588,141 B = 0.0256 GB.
        #expect(e.downloadSizeGB == 0.03)
        // MP5-1: ESTIMATE = download x 1.5 (MP-0 G). MP5-3d replaces it with the measured in-app peak.
        #expect(e.ramGB == 0.045)
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.description?.contains("RAM figure is an estimate") == true)
        #expect(e.speedTokensPerSec == nil)
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "fp32")
        // v3 model.safetensors 8,682,709 B / 4 B per fp32 weight = 2.17 M parameters (derived).
        #expect(e.paramCountB == 0.0022)
    }

    @Test func kindRoutesToEnhance() throws {
        let e = try deepFilterNet()
        #expect(e.runnerKind == .enhance)
        #expect(ModelType.enhance.rawValue == "enhance")
        #expect(RunnerKind.enhance.rawValue == "enhance")
        #expect(ModelType.enhance.sfSymbol == "waveform.badge.minus")
    }

    @Test func liveInTheAudioSectionsEnhanceLeaf() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "enhance" })
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let audio = try #require(data.sidebarSections.first { $0.id == "audio" })
        #expect(audio.domainIds.contains("enhance"))
        #expect(data.sidebarSections.filter { $0.domainIds.contains("enhance") }.map(\.id) == ["audio"])
        #expect(data.domainIndex["enhance"] != nil)
    }

    @Test func noModuleClaimsItYetAndTheGapFlagsIt() throws {
        let e = try deepFilterNet()
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        #expect(full.bestModule(for: e) == nil, "MP5-3 registers the module and removes the ModelSupport gap")
        #expect(ModelSupport.unsupportedReason(for: e) != nil)
    }

    @Test func stageFactoryRefusesTheKindWithAnExplicitCase() throws {
        let e = try deepFilterNet()
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: e, installedModelIDs: [e.id], availableRAMGB: 16)
        }
    }

    @Test func noWorkflowPoolOffersIt() throws {
        let e = try deepFilterNet()
        for task in TaskCatalog.allTasks().map(\.name) {
            #expect(TaskModels.runnerKind(forTask: task) != .enhance, "\(task) names the enhance kind")
            let pool = TaskModels.derivedModels(for: task, catalog: [e], claimableModelIDs: [e.id])
            #expect(!pool.contains { $0.id == e.id }, "\(task) offers DeepFilterNet")
        }
    }

    /// The repo has no root weights: v1, v2 and v3 each hold `config.json` + `model.safetensors`. The selector treats
    /// each as a component and takes all three (25.6 MB, harmless, backlog MP5). `InstallManager` creates the
    /// subfolders, so the engine finds `<dir>/v3/{config.json,model.safetensors}`.
    @Test func selectorTakesAllThreeVersionsAndNoDocs() {
        let out = Set(ModelFileSelector.filesToDownload(siblings: Self.siblings))
        #expect(out == [
            "v1/config.json", "v1/model.safetensors",
            "v2/config.json", "v2/model.safetensors",
            "v3/config.json", "v3/model.safetensors",
        ])
        #expect(!out.contains("convert_deepfilternet.py"))
    }
}
