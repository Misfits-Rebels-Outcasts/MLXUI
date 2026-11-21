import Testing
import Foundation
@testable import MLXUI

/// MC5-W (AM-W, SPEC-Q251): ACE-Step 1.5 is reachable from AI Workflows. The four tables: `TaskCatalog` (existing `Generate Sound`), `servedRefNamePrefixes`
/// (existing `engines.diffusion.generate_sound`), `taskKinds` (`.music`, existing), and the new `CatalogBridge` row + MLXUI-authored manifest. ACE-Step is appended LAST
/// to the pool so the default stays MusicGen.
struct AceStepWorkflowWiringTests {

    private static let display = "ACE-Step 1.5"
    private static let catalogID = "ACE-Step--Ace-Step1.5"

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    @MainActor
    private func claimableIDs(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    private func makeExecutor() -> RealExecutor {
        RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory), flowID: "mc5w",
                     blobDirectory: FileManager.default.temporaryDirectory,
                     makeModelStage: { _, _ in AceWiringStub() }, installedModelIDs: [], catalog: [])
    }

    // MARK: - the bridge row

    @Test func theBridgeEntryResolvesToTheCatalogCard() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: Self.display))
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "ace-step-1.5.json")
        switch CatalogBridge.resolve(Self.display, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.id == Self.catalogID)
            #expect(slot.modelEntry?.runnerKind == .music)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let name, let reason, _):
            Issue.record("\(name) should resolve: \(reason)")
        }
    }

    // MARK: - the manifest

    @Test func theManifestIsInTheAppBundleAndDecodes() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "ace-step-1.5.json"))
        #expect(manifest.id == "ACE-Step/Ace-Step1.5")
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "ace-step")
        #expect(manifest.tasks == ["Generate Sound"])
        #expect(Set(manifest.settings.keys) == ["seed", "steps"])
        let steps = try #require(manifest.settings["steps"])
        #expect(steps.min == 1)
        #expect(steps.max == 20)
        #expect(steps.defaultValue == .number(8))
    }

    @Test func theManifestResourcesMatchTheCatalogEntry() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "ace-step-1.5.json"))
        let card = try #require(try loadCatalog().first { $0.id == Self.catalogID })
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    // MARK: - the pool

    @MainActor
    @Test func aceStepIsLastInTheGenerateSoundPoolAndMusicGenStaysTheDefault() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        #expect(TaskModels.seededModels(forTask: "Generate Sound") == ["MusicGen", Self.display])
        #expect(TaskModels.isServedByExecutor("Generate Sound"))
        #expect(TaskModels.runnerKind(forTask: "Generate Sound") == .music)
        #expect(TaskModels.defaultModel(forTask: "Generate Sound", catalog: catalog, claimableModelIDs: claimable) == "MusicGen")
        let derived = TaskModels.derivedModels(for: "Generate Sound", catalog: catalog, claimableModelIDs: claimable).map(\.displayName)
        #expect(derived.contains(Self.display), "the Generate Sound menu offers ACE-Step 1.5")
        #expect(derived.first == "MusicGen")
    }

    // MARK: - a one-row flow

    private func issues(_ text: String) throws -> [FlowIssue] {
        FlowValidator.checkFlow(try CatParser.parseForValidation(text))
    }

    @Test func aOneRowAceStepFlowChecksClean() throws {
        let pinned = try issues("mlxflow 0.8\n1. Generate Sound   ACE-Step 1.5; steps=8; seed=7\n\nmodels:\n  ACE-Step 1.5 = ACE-Step/Ace-Step1.5\n")
        #expect(pinned.isEmpty, "\(pinned)")
        // Unpinned, it raises only the E104 pin warning every unpinned model row raises (MusicGen included), never an error.
        let unpinned = try issues("mlxflow 0.8\n1. Generate Sound   ACE-Step 1.5; steps=8; seed=7\n")
        #expect(unpinned.map(\.code) == ["E104"], "\(unpinned)")
    }

    // MARK: - refusals by name, and the settings that are honoured

    @Test func durationAndLyricsAreRefusedByName() throws {
        let desc = try #require(TaskCatalog.get("Generate Sound"))
        let ace = try #require(try loadCatalog().first { $0.id == Self.catalogID })
        for (settings, name) in [("calm; duration=45", "duration"), ("calm; lyrics=\"x\"", "lyrics"), ("calm; width=512", "width"),
                                 ("calm; quantize=4", "quantize")] {
            let row = Row(task: "Generate Sound", model: Self.display, settings: settings)
            do {
                _ = try makeExecutor().stageConfig(for: desc, row: row, model: ace)
                Issue.record("\(name) should be refused")
            } catch let refusal as FlowSettingRefusal {
                #expect(refusal.description.contains("\(name)="), "\(refusal.description)")
            }
        }
    }

    @Test func seedAndStepsReachTheStageConfigAndOutOfRangeStepsAreRefused() throws {
        let desc = try #require(TaskCatalog.get("Generate Sound"))
        let ace = try #require(try loadCatalog().first { $0.id == Self.catalogID })
        let ok = try makeExecutor().stageConfig(for: desc, row: Row(task: "Generate Sound", model: Self.display, settings: "calm; steps=12; seed=7"), model: ace)
        #expect(ok.steps == 12 && ok.seed == 7)
        for bad in ["steps=0", "steps=21", "steps=x"] {
            #expect(throws: (any Error).self) {
                _ = try makeExecutor().stageConfig(for: desc, row: Row(task: "Generate Sound", model: Self.display, settings: "calm; \(bad)"), model: ace)
            }
        }
    }

    @Test func theSDKCarriesStepsIntoTheStageAndRefusesTheRest() throws {
        let ace = try #require(try loadCatalog().first { $0.id == Self.catalogID })
        let stage = try #require(try AceStepSDK().makeStage(for: ace, config: StageConfig(seed: 3, steps: 5)) as? AceStepStage)
        #expect(stage.settings.steps == 5 && stage.settings.seed == 3)
        #expect(stage.settings.duration == 30, "a flow row is a fixed 30 s")
        let defaults = try #require(try AceStepSDK().makeStage(for: ace, config: .default) as? AceStepStage)
        #expect(defaults.settings.steps == nil)
        #expect(throws: StageError.self) { _ = try AceStepSDK().makeStage(for: ace, config: StageConfig(steps: 99)) }
    }

    @Test func theStagePutsStepsIntoTheEngineRequest() async throws {
        let box = RequestBox()
        let stage = AceStepStage(id: "t", name: "t", settings: AceStepRunSettings(seed: 1, steps: 6)) { request, _ in
            box.value = request
            return AceStepAudio(samples: [0, 0])
        }
        _ = try await stage.run(.text("calm"), progress: { _ in })
        #expect(box.value?.steps == 6)
        #expect(box.value?.duration == 30)
        #expect(box.value?.isInstrumental == true)
    }

    private final class RequestBox: @unchecked Sendable { var value: AceStepRequest? }
}

private nonisolated struct AceWiringStub: PipelineStage {
    let id = "stub.ace"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}

extension AceStepWorkflowWiringTests {
    /// The unpinned E104 warning is not ACE-Step specific: a MusicGen row gives exactly the same one.
    @Test func anUnpinnedMusicGenRowRaisesTheSameE104() throws {
        let found = FlowValidator.checkFlow(try CatParser.parseForValidation("mlxflow 0.8\n1. Generate Sound   MusicGen; steps=8; seed=7\n"))
        #expect(found.map(\.code) == ["E104"], "\(found)")
    }
}

extension AceStepWorkflowWiringTests {
    /// The exact row text the AM-T flow check writes: the model is "ACE-Step 1.5" (a dotted name) and the quoted caption is the prompt, with `steps` / `seed` as settings.
    @Test func theSmokeFlowRowParsesAsModelSettingsAndPrompt() throws {
        let doc = try CatParser.parse("mlxflow 0.8\n1. Generate Sound   ACE-Step 1.5; steps=8; seed=7 \"warm lo-fi hip hop\"\n\nmodels:\n  ACE-Step 1.5 = ACE-Step/Ace-Step1.5\n")
        let row = try #require(doc.rows.first)
        #expect(row.model == "ACE-Step 1.5")
        let settings = FlowSettings(row.settings)
        #expect(settings.value(for: "steps") == "8" && settings.value(for: "seed") == "7")
        #expect(settings.firstBare() == "warm lo-fi hip hop")
    }
}
