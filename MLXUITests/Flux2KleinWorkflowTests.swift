import Testing
import Foundation
import CryptoKit
@testable import MLXUI

/// MP7-W: FLUX.2 Klein 4B reachable from AI Workflows. Wording of the cap sentence (owner-ruled 2026-11-01), the by-name refusals in `RealExecutor.stageConfig`, the flow lease
/// estimate, the bridge row + verbatim manifest, and the four tables. No weights load here.
struct Flux2KleinWorkflowTests {

    // MARK: Fixtures

    private static func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    private static func klein() throws -> ModelEntry {
        try #require(catalog().first { $0.hfModelId == "mlx-community/FLUX.2-klein-4B-bf16" })
    }

    private func executor(installed: Set<String> = [], catalog: [ModelEntry] = [],
                          makeStage: @escaping @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { _, _ in throw StageError.unsupportedSetting(setting: "none") }) -> RealExecutor {
        RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory), flowID: "mp7w",
                     blobDirectory: FileManager.default.temporaryDirectory, makeModelStage: makeStage,
                     installedModelIDs: installed, catalog: catalog)
    }

    private func config(_ settings: String, path: String = "3") throws -> StageConfig {
        let desc = try #require(TaskCatalog.get("Generate Image"))
        return try executor().stageConfig(for: desc, row: Row(task: "Generate Image", model: "FLUX.2 Klein 4B", settings: settings), path: path, model: Self.klein())
    }

    private func refusal(_ settings: String) throws -> String? {
        do { _ = try config(settings); return nil } catch { return String(describing: error) }
    }

    // MARK: 1. Wording (owner-ruled, 2026-11-01)

    @Test func theCapSentenceIsExactlyTheOwnerWordingInTheRunWindowAndInAFlowRow() {
        func error(_ row: String) -> Flux2KleinEngineError {
            .sizeExceedsMemory(row: row, width: 4096, height: 1024, quantize: 8, neededGB: 12.3, availableGB: 11.5, largestSide: 1344)
        }
        let tail = "on this Mac: it needs about 12.3 GB at its busiest step and this Mac can give it 11.5 GB. The largest it can make here is 1344 \u{00D7} 1344 (or any other shape with no more pixels)."
        let expected = "FLUX.2 Klein can't make a 4096 \u{00D7} 1024 image at quantize 8 " + tail
        // Run window ("This image"), the engine's own label and no label: the model is the subject, no prefix.
        for label in [Flux2KleinStage.runWindowLabel, Flux2KleinEngine.modelName, ""] {
            #expect(error(label).errorDescription == expected, Comment(rawValue: label))
            #expect(String(describing: error(label)) == expected)
        }
        // A labelled row (a direct caller naming its row) is named first; the subject is still the model.
        #expect(error("Row 3").errorDescription == "Row 3: " + expected)
        // No ASCII x between the dimensions anywhere in the sentence.
        #expect(!expected.contains(" x "))
        // No size fits at all
        let none = Flux2KleinEngineError.sizeExceedsMemory(row: "", width: 512, height: 512, quantize: 8, neededGB: 3, availableGB: 1, largestSide: 0)
        #expect(none.errorDescription == "FLUX.2 Klein can't make a 512 \u{00D7} 512 image at quantize 8 on this Mac: it needs about 3.0 GB at its busiest step and this Mac can give it 1.0 GB. It can't make an image of any size here at this quantize.")
    }

    @Test func theMultipleOfSixteenSentenceIsUnchanged() throws {
        do { _ = try Flux2KleinLatents.validateSize(width: 1000, height: 1024); Issue.record("accepted") } catch {
            #expect((error as? LocalizedError)?.errorDescription == "Width 1000 isn't a multiple of 16. The nearest sizes are 992 and 1008.")
            #expect(String(describing: error) == "Width 1000 isn't a multiple of 16. The nearest sizes are 992 and 1008.")
        }
    }

    @Test func aFlowRowOverTheCapFailsAsRowNFailedWithTheModelAsSubject() async throws {
        let klein = try Self.klein()
        let sdk = Flux2KleinSDK()
        let exec = executor(installed: [klein.id], catalog: [klein]) { model, config in try sdk.makeStage(for: model, config: config) }
        let text = "mlxflow 0.8\n1. Generate Image   FLUX.2 Klein 4B; seed=1; width=16384; height=16384 \"a cat\"\n"
        let doc = try CatParser.parse(text)
        let events = try await FlowInterpreter.run(doc, executor: exec, definitions: doc.definitions, presets: doc.presets)
        let failed = try #require(events.first { $0.kind == .rowFailed })
        let message = try #require(failed.error)
        #expect(message.hasPrefix("FLUX.2 Klein can't make a 16384 \u{00D7} 16384 image at quantize 8 on this Mac: it needs about "), Comment(rawValue: message))
        // what the flow runner shows the person
        #expect(FlowError.stageFailure(row: failed.path, message: message).description.hasPrefix("Row 1 failed: FLUX.2 Klein can't make a 16384 \u{00D7} 16384 image"))
        #expect(!message.contains("sizeExceedsMemory"))
    }

    // MARK: 2. Refusals by name, accepted defaults

    @Test func theManifestDefaultsAreAccepted() throws {
        #expect(try refusal("") == nil)
        #expect(try refusal("seed=7; width=1024; height=1024; steps=4") == nil)
        #expect(try refusal("guidance=2.0") == nil)
        #expect(try refusal("guidance=2") == nil)
        #expect(try refusal("negative=\"\"") == nil)
        #expect(try refusal("preview=false") == nil)
        #expect(try refusal("quantize=4") == nil)
        #expect(try refusal("quantize=8") == nil)
        #expect(try config("quantize=4").quantize == 4)
        #expect(try config("").quantize == nil)
        #expect(try config("seed=7; width=512; height=768; steps=2; quantize=8").width == 512)
    }

    @Test func quantizeOutsideFourAndEightIsRefusedByName() throws {
        for q in ["3", "5", "6", "16", "0"] {
            #expect(try refusal("quantize=\(q)") == "FLUX.2 Klein 4B runs at quantize 4 or 8 in this build, not \(q).")
        }
        #expect(try refusal("quantize=eight") == "quantize must be a whole number, 4 or 8; the row says eight.")
    }

    @Test func nonDefaultGuidanceIsRefusedByName() throws {
        #expect(try refusal("guidance=7") == "FLUX.2 Klein 4B has no guidance setting (it is distilled to 4 steps), so guidance can only be left at 2.0, not 7.")
        #expect(try refusal("guidance=0") != nil)
    }

    @Test func aNonEmptyNegativeIsRefusedByName() throws {
        #expect(try refusal("negative=\"blurry\"") == "FLUX.2 Klein 4B can't use a negative prompt in this build; remove the negative setting.")
    }

    @Test func loraIsRefusedByName() throws {
        #expect(try refusal("lora=\"style.safetensors\"") == "FLUX.2 Klein 4B can't use LoRA weights in this build; remove the lora setting.")
    }

    @Test func controlnetIsRefusedByName() throws {
        #expect(try refusal("controlnet=\"canny\"") == "FLUX.2 Klein 4B can't use a ControlNet in this build; remove the controlnet setting.")
    }

    @Test func previewIsRefusedByName() throws {
        #expect(try refusal("preview=true") == "FLUX.2 Klein 4B can't show a preview while it denoises in this build; remove the preview setting.")
    }

    @Test func theRefusalReadsRowNFailedInAFlow() async throws {
        let klein = try Self.klein()
        let exec = executor(installed: [klein.id], catalog: [klein])
        let doc = try CatParser.parse("mlxflow 0.8\n1. Generate Image   FLUX.2 Klein 4B; negative=\"blurry\" \"a cat\"\n")
        let events = try await FlowInterpreter.run(doc, executor: exec, definitions: doc.definitions, presets: doc.presets)
        let failed = try #require(events.first { $0.kind == .rowFailed })
        #expect(FlowError.stageFailure(row: failed.path, message: failed.error ?? "").description
                == "Row 1 failed: FLUX.2 Klein 4B can't use a negative prompt in this build; remove the negative setting.")
    }

    @Test func everyRefusedSettingReadsRowNFailedThroughTheInterpreter() async throws {
        let klein = try Self.klein()
        let exec = executor(installed: [klein.id], catalog: [klein])
        let cases: [(String, String)] = [("negative=\"blurry\"", "negative"), ("lora=\"x.safetensors\"", "LoRA"), ("controlnet=\"canny\"", "ControlNet"),
                                         ("preview=true", "preview"), ("guidance=7", "guidance"), ("quantize=6", "quantize 4 or 8")]
        for (setting, word) in cases {
            // lora=/controlnet= run greedy to `;`, so the prompt goes before them
            let doc = try CatParser.parse("mlxflow 0.8\n1. Generate Image   FLUX.2 Klein 4B; seed=7; \"a cat\"; \(setting)\n")
            let events = try await FlowInterpreter.run(doc, executor: exec, definitions: doc.definitions, presets: doc.presets)
            let failed = try #require(events.first { $0.kind == .rowFailed }, Comment(rawValue: setting))
            let text = FlowError.stageFailure(row: failed.path, message: failed.error ?? "").description
            #expect(text.hasPrefix("Row 1 failed: FLUX.2 Klein 4B ") && text.contains(word), Comment(rawValue: "\(setting): \(text)"))
        }
    }

    @Test func otherDiffusionRowsAreUntouched() throws {
        let desc = try #require(TaskCatalog.get("Generate Image"))
        let flux1 = makeEntry(id: "mlx-community--Flux-1.lite-8B-MLX-Q4", displayName: "Flux-1.lite-8B", modelType: .image, source: .mlx,
                              hfRepo: "mlx-community/Flux-1.lite-8B-MLX-Q4", hfModelId: "mlx-community/Flux-1.lite-8B-MLX-Q4")
        let settings = "guidance=7; negative=\"x\"; lora=\"y\"; controlnet=\"z\"; preview=true; quantize=3"
        let withModel = try executor().stageConfig(for: desc, row: Row(task: "Generate Image", model: "Flux-1.lite-8B", settings: settings), path: "1", model: flux1)
        let withoutModel = try executor().stageConfig(for: desc, row: Row(task: "Generate Image", model: "Flux-1.lite-8B", settings: settings), path: "1")
        #expect(withModel == withoutModel)
        #expect(withModel.quantize == nil)
        // a non-diffusion row naming `quantize` is not read either
        let asr = try executor().stageConfig(for: try #require(TaskCatalog.get("Transcribe")), row: Row(task: "Transcribe", model: "Whisper Small", settings: "quantize=3"), path: "1", model: try Self.klein())
        #expect(asr.quantize == nil)
    }

    @Test func aKleinStageBuiltFromTheRowsQuantizeCarriesIt() throws {
        let klein = try Self.klein()
        let q4 = try #require(try Flux2KleinSDK().makeStage(for: klein, config: StageConfig(seed: 1, steps: 4, quantize: 4)) as? Flux2KleinStage)
        #expect(q4.settings.quantize == 4)
        let none = try #require(try Flux2KleinSDK().makeStage(for: klein, config: StageConfig(seed: 1, width: 512, height: 512)) as? Flux2KleinStage)
        #expect(none.settings.quantize == 8)
        // an unsupported value that bypassed the executor is still refused by the SDK
        #expect(throws: (any Error).self) { try Flux2KleinSDK().makeStage(for: klein, config: StageConfig(quantize: 6)) }
    }

    // MARK: 3. The flow path books the per-run estimate

    private static let gib: Int64 = 1_073_741_824
    private func makeBudget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * Self.gib, cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }
    private func makeCache(_ budget: MemoryBudget) -> EngineCache {
        EngineCache(budget: budget, countsLLMWeights: false, modelDirectory: { _ in FileManager.default.temporaryDirectory.appendingPathComponent("no-klein-here") })
    }

    @Test func theFlowLeaseOverrideIsTheRunsOwnEstimate() throws {
        let klein = try Self.klein()
        #expect(Flux2KleinBooking.leaseOverride(for: klein, config: StageConfig()) == Flux2KleinBooking.bytes(width: 1024, height: 1024, quantize: 8))
        #expect(Flux2KleinBooking.leaseOverride(for: klein, config: StageConfig(width: 1344, height: 1344, quantize: 8)) == Flux2KleinBooking.bytes(width: 1344, height: 1344, quantize: 8))
        #expect(Flux2KleinBooking.leaseOverride(for: klein, config: StageConfig(width: 512, height: 512, quantize: 4)) == Flux2KleinBooking.bytes(width: 512, height: 512, quantize: 4))
        let flux1 = makeEntry(id: "mlx-community--Flux-1.lite-8B-MLX-Q4", displayName: "Flux-1.lite-8B", modelType: .image, source: .mlx,
                              hfRepo: "mlx-community/Flux-1.lite-8B-MLX-Q4", hfModelId: "mlx-community/Flux-1.lite-8B-MLX-Q4")
        #expect(Flux2KleinBooking.leaseOverride(for: flux1, config: StageConfig(width: 1344, height: 1344)) == nil)
    }

    @Test func aFlowRowAtALargerSizeReservesMoreFromTheSharedBudget() async throws {
        let klein = try Self.klein()
        var booked: [Int64] = []
        for side in [512, 1024, 1344] {
            let budget = makeBudget(40)
            let cache = makeCache(budget)
            let config = StageConfig(seed: 1, width: side, height: side, quantize: 8)
            let stage = try await cache.leasedStage(for: klein, config: config, footprintOverride: Flux2KleinBooking.leaseOverride(for: klein, config: config)) { _, _ in KleinProbeStage() }
            booked.append(budget.usedBytes)
            #expect(budget.usedBytes == Flux2KleinBooking.bytes(width: side, height: side, quantize: 8), "\(side)")
            _ = stage
        }
        #expect(booked[0] < booked[1] && booked[1] < booked[2], "\(booked)")
        #expect(booked[2] > Flux2KleinBooking.bookedBytes, "1344 x 1344 books above the flat 6.05 GB")
        // without the override the flow would book the flat figure whatever the size (the bug this fixes)
        let budget = makeBudget(40)
        let flat = try await makeCache(budget).leasedStage(for: klein, config: StageConfig(seed: 1, width: 1344, height: 1344, quantize: 8)) { _, _ in KleinProbeStage() }
        #expect(budget.usedBytes == Flux2KleinBooking.bookedBytes)
        _ = flat
    }

    // MARK: 4. Bridge, manifest, four tables

    @Test func theBridgeRowResolvesKleinAsRequantizedAndSilent() throws {
        let entry = try #require(CatalogBridge.entry(for: "FLUX.2 Klein 4B"))
        #expect(entry.pinnedID == "black-forest-labs/FLUX.2-klein-4B")
        #expect(entry.candidates == ["mlx-community/FLUX.2-klein-4B-bf16"])
        #expect(entry.equivalence == .requantized)
        #expect(entry.manifestFile == "flux2-klein-4b.json")
        let catalog = try Self.catalog()
        guard case let .runnable(slot, equivalence, note) = CatalogBridge.resolve("FLUX.2 Klein 4B", catalog: catalog) else {
            Issue.record("FLUX.2 Klein 4B did not resolve"); return
        }
        #expect(slot.modelEntry?.id == "mlx-community--FLUX.2-klein-4B-bf16")
        #expect(equivalence == .requantized)
        #expect(note == nil, "`.requantized` is silent, as the Whisper Large v3 and LFM2-VL rows")
        #expect(TaskModels.displayName(for: try Self.klein()) == "FLUX.2 Klein 4B")
    }

    @Test func theManifestIsAByteIdenticalCopyOfCatflowMlxs() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let ours = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/CatFlow/models/flux2-klein-4b.json"))
        // The sha256 of `catflow-mlx/models/curated/flux2-klein-4b.json` when copied (MP7-W). Never edit the manifest; a different manifest is a new copy and a new hash here.
        #expect(Self.sha256(ours) == "a7539dfbe377cf1c233e5348161b674556b1c2e1290631e4ba34018533725cd6")
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: ours)
        #expect(manifest.display == "FLUX.2 Klein 4B" && manifest.id == "black-forest-labs/FLUX.2-klein-4B")
        #expect(manifest.tasks?.contains("Generate Image") == true)
    }

    @Test func theFourTablesServeKleinForGenerateImageOnly() throws {
        let klein = try Self.klein()
        // 1. TaskCatalog   2. served   3. pool   4. bridge
        #expect(TaskCatalog.get("Generate Image") != nil)
        #expect(TaskModels.isServedByExecutor("Generate Image"))
        let pool = TaskModels.derivedModels(for: "Generate Image", catalog: try Self.catalog(), claimableModelIDs: [klein.id])
        #expect(pool.map(\.displayName) == ["FLUX.2 Klein 4B"])
        #expect(TaskModels.models(forTask: "Generate Image").contains("FLUX.2 Klein 4B"))
        // the latent tasks the manifest lists are NOT served (scope C): no stage exposes them
        for task in ["Init Latent", "Encode Latent", "Decode Latent"] {
            #expect(!TaskModels.isServedByExecutor(task), Comment(rawValue: task))
            #expect(TaskModels.derivedModels(for: task, catalog: try Self.catalog(), claimableModelIDs: [klein.id]).isEmpty, Comment(rawValue: task))
        }
        #expect(!TaskModels.isServedByExecutor("Edit Image"))
    }

    // MARK: MP7-W-f: gallery 60 and 66 (owner, Boo, 2026-11-01): the exact display "Z-Image Turbo (4-bit)" is a `.substitute` onto Klein

    private static let zImage = "Z-Image Turbo (4-bit)"

    private static func galleryDoc(_ file: String) throws -> FlowDocument {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return try CatParser.parse(String(contentsOf: root.appendingPathComponent("MLXUI/Resources/Gallery/\(file)"), encoding: .utf8))
    }

    @MainActor private static func claimable(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    @Test func theExactZImageDisplayIsASubstituteOntoKleinWithTheNoteAndTheBareNameIsNot() throws {
        let catalog = try Self.catalog()
        guard case let .runnable(slot, equivalence, note) = CatalogBridge.resolve(Self.zImage, catalog: catalog) else { Issue.record("did not resolve"); return }
        #expect(slot.modelEntry?.hfModelId == "mlx-community/FLUX.2-klein-4B-bf16")
        #expect(equivalence == .substitute)
        #expect(note == "running Z-Image Turbo (4-bit) as mlx-community/FLUX.2-klein-4B-bf16", Comment(rawValue: note ?? "nil"))
        // only the EXACT name: the bare one and a near miss stay unbridged
        for other in ["Z-Image Turbo", "Z-Image Turbo (8-bit)", "z-image turbo (4-bit)"] {
            if case .runnable = CatalogBridge.resolve(other, catalog: catalog) { Issue.record("\(other) must stay unbridged") }
        }
        // the Klein card still answers to its own name, and a new Generate Image row still defaults to Klein
        #expect(TaskModels.displayName(for: try Self.klein()) == "FLUX.2 Klein 4B")
        #expect(CatalogBridge.entry(forCandidate: "mlx-community/FLUX.2-klein-4B-bf16")?.display == "FLUX.2 Klein 4B")
    }

    @Test @MainActor func aNewGenerateImageRowStillDefaultsToKleinNotZImage() throws {
        let catalog = try Self.catalog()
        let claimable = Self.claimable(catalog)
        #expect(FlowEditorModel.defaultModel(forTask: "Generate Image", catalog: catalog, claimableModelIDs: claimable) == "FLUX.2 Klein 4B")
        #expect(!TaskModels.derivedModels(for: "Generate Image", catalog: catalog, claimableModelIDs: claimable).contains { $0.displayName == Self.zImage })
        #expect(!TaskModels.models(forTask: "Generate Image").contains(Self.zImage))
    }

    @Test @MainActor func galleryFlows60And66AreNowRunnableAndTheirSettingsAreNotRefusedByKlein() throws {
        let catalog = try Self.catalog()
        let klein = try Self.klein()
        let claimable = Self.claimable(catalog)
        let desc = try #require(TaskCatalog.get("Generate Image"))
        for file in ["60-GenerateProductShot.cat", "66-SeedSweep.cat"] {
            let doc = try Self.galleryDoc(file)
            #expect(FlowRunnability.refusalReason(for: doc, catalog: catalog, installed: [klein.id], totalRAMGB: 19, claimableModelIDs: claimable) == nil, Comment(rawValue: file))
            // the settings the file writes pass Klein's refusals, so they do not now fail for an unrelated setting
            var generate: [Row] = []
            func walk(_ rows: [Row]) { for r in rows { if r.task == "Generate Image" { generate.append(r) }; walk(r.children) } }
            walk(doc.rows)
            #expect(generate.count == 1, Comment(rawValue: file))
            for row in generate {
                #expect(row.model == Self.zImage)
                let settings = (row.settings ?? "").replacingOccurrences(of: "{item}", with: "3")
                let config = try executor().stageConfig(for: desc, row: Row(task: "Generate Image", model: row.model, settings: settings), path: "1", model: klein)
                #expect(config.quantize == nil)
                #expect(config.seed != nil, Comment(rawValue: file))
            }
        }
        // not installed: refused only for the missing install, naming the model that actually loads (the preflight's own wording), not for an unknown model
        let doc60 = try Self.galleryDoc("60-GenerateProductShot.cat")
        let bare = FlowRunnability.refusalReason(for: doc60, catalog: catalog, installed: [], totalRAMGB: 19, claimableModelIDs: claimable)
        #expect(bare == nil || !(bare ?? "").contains("runnable-model table"), Comment(rawValue: bare ?? "nil"))
    }

    @Test @MainActor func galleryFlow61StaysRefusedBecauseEditImageIsUnserved() throws {
        let catalog = try Self.catalog()
        let klein = try Self.klein()
        let doc = try Self.galleryDoc("61-EditInPlace.cat")
        let reason = FlowRunnability.refusalReason(for: doc, catalog: catalog, installed: [klein.id], totalRAMGB: 19, claimableModelIDs: Self.claimable(catalog))
        #expect(reason != nil)
        #expect(!TaskModels.isServedByExecutor("Edit Image"))
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
