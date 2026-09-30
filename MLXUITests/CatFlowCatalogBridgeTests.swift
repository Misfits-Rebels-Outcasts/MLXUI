import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R2-2: the `CatalogBridge` display-name → installable-model table and its
/// two-sided golden tests. See `RSI/DelegateMergeBacklog.md` CFM-R2-2.
struct CatFlowCatalogBridgeTests {

    // MARK: - Fixture loading (repo-relative; the test bundle doesn't carry these)

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

    private func manifestURL(_ manifestFile: String) -> URL {
        repoRoot
            .appendingPathComponent("MLXUI/Resources/CatFlow/models")
            .appendingPathComponent(manifestFile)
    }

    // MARK: - Golden test 1: every candidate id exists in browser.json

    @Test func everyCandidateExistsInBrowserCatalog() throws {
        let catalog = try loadCatalog()
        let catalogIDs = Set(catalog.map(\.hfModelId))
        for entry in CatalogBridge.entries {
            for candidate in entry.candidates {
                #expect(catalogIDs.contains(candidate),
                        "\(candidate) (candidate for \(entry.display)) missing from browser.json")
            }
        }
    }

    // MARK: - Golden test 2: every pinned id exists in the copied manifests

    @Test func everyPinnedIDExistsInCopiedManifests() throws {
        for entry in CatalogBridge.entries {
            let url = manifestURL(entry.manifestFile)
            let data = try Data(contentsOf: url)
            let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
            #expect(manifest.id == entry.pinnedID,
                    "\(entry.manifestFile) pins \(manifest.id), expected \(entry.pinnedID)")
        }
    }

    // MARK: - Every display name resolves to an installable model

    @Test func allBridgeEntriesResolve() throws {
        let catalog = try loadCatalog()
        for entry in CatalogBridge.entries {
            switch CatalogBridge.resolve(entry.display, catalog: catalog) {
            case .runnable(let slot, let equivalence, _):
                #expect(slot.modelEntry?.hfModelId == entry.candidates.first)
                #expect(equivalence == entry.equivalence)
            case .notRunnable(let display, let reason, _):
                Issue.record("\(display) should resolve: \(reason)")
            }
        }
    }

    // MARK: - Substitution note surfaces for .substitute / .sameFamily

    @Test func substituteRunsWithANote() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "MusicGen"))
        #expect(entry.equivalence == .substitute)
        switch CatalogBridge.resolve("MusicGen", catalog: catalog) {
        case .runnable(_, _, let note):
            #expect(note != nil)
            #expect(note?.contains("MusicGen") == true)
        case .notRunnable:
            Issue.record("MusicGen should resolve")
        }
    }

    @Test func requantizedRunsSilently() throws {
        let entry = try #require(CatalogBridge.entry(for: "Whisper Large v3"))
        #expect(entry.equivalence == .requantized)
        #expect(entry.equivalence.note(display: "Whisper Large v3", substitutedID: "x") == nil)
    }

    // MARK: - Unknown display name → not-runnable with the model named

    @Test func unknownDisplayNameIsNotRunnable() throws {
        let catalog = try loadCatalog()
        switch CatalogBridge.resolve("Turbo Chat 5000", catalog: catalog) {
        case .notRunnable(let display, let reason, _):
            #expect(display == "Turbo Chat 5000")
            #expect(reason.contains("Turbo Chat 5000"))
        case .runnable:
            Issue.record("an unknown display name must be not-runnable")
        }
    }

    @Test func noCandidateIsNotRunnableNamingTheModel() throws {
        let catalog = try loadCatalog()
        switch CatalogBridge.resolve("Totally Missing", catalog: catalog) {
        case .notRunnable(let display, let reason, _):
            #expect(display == "Totally Missing")
            #expect(reason.contains("Totally Missing"))
        case .runnable:
            Issue.record("a display name with no table entry must be not-runnable")
        }
    }

    // MARK: - CFM-R14-FIX-1: the derived-pool resolve fallback

    /// An R14-2 unbridged pick — display name is the raw `hfModelId` — must resolve `.same`
    /// with no manifest, so the runtime (preflight / executor) doesn't refuse it.
    @Test func unbridgedHfModelIdResolvesAsSame() throws {
        let catalog = try loadCatalog()
        let unbridged = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
        switch CatalogBridge.resolve(unbridged, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == unbridged)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve via the derived-pool fallback: \(reason)")
        }
    }

    /// The FIX-7 display shape (a catalog `displayName`, not the id) also resolves.
    @Test func unbridgedDisplayNameResolvesAsSame() throws {
        let catalog = try loadCatalog()
        // Qwen3-VL's browser.json displayName is the friendly name, not the id.
        let displayName = "Qwen3-VL-4B-Instruct"
        switch CatalogBridge.resolve(displayName, catalog: catalog) {
        case .runnable(let slot, _, _):
            #expect(slot.modelEntry?.hfModelId == "mlx-community/Qwen3-VL-4B-Instruct-4bit")
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve by catalog display name: \(reason)")
        }
    }

    // MARK: - CFM-R15-1: SAM Base is in the bridge (hazard H2 ruled option (1) 2026-08-27)

    @Test func samBaseResolvesAsASubstituteOntoSam3() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "SAM Base"))
        #expect(entry.pinnedID == "mlx/sam-base")
        #expect(entry.candidates == ["mlx-community/sam3-4bit"])
        #expect(entry.equivalence == .substitute)
        #expect(entry.manifestFile == "sam-base.json")
        // The .cat stays byte-identical to the reference ("Segment SAM Base"); the different
        // model generation is *shown*, never hidden — the MusicGen precedent.
        switch CatalogBridge.resolve("SAM Base", catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == "mlx-community/sam3-4bit")
            #expect(equivalence == .substitute)
            #expect(note != nil)
            #expect(note?.contains("SAM Base") == true)
            #expect(note?.contains("mlx-community/sam3-4bit") == true)
        case .notRunnable(let display, let reason, _):
            Issue.record("SAM Base should resolve: \(reason)")
            _ = display
        }
        #expect(CatalogBridge.entries.count == 23)
    }

    // MARK: - CFM-R13-9/12: the OCR + Describe Image rows

    @Test func olmOCRResolvesWithASameFamilyNote() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "olmOCR-2 7B"))
        #expect(entry.equivalence == .sameFamily)
        switch CatalogBridge.resolve("olmOCR-2 7B", catalog: catalog) {
        case .runnable(_, _, let note):
            #expect(note != nil)   // a different repo is surfaced, never hidden
            #expect(note?.contains("olmOCR") == true)
        case .notRunnable:
            Issue.record("olmOCR-2 7B should resolve")
        }
    }

    @Test func lfm2AndGemmaResolveForDescribeImage() throws {
        let catalog = try loadCatalog()
        for display in ["LFM2-VL 1.6B", "Gemma 3 4B"] {
            switch CatalogBridge.resolve(display, catalog: catalog) {
            case .runnable(let slot, _, _):
                // The cheap one is the pool's first entry, so a bare Describe Image row
                // defaults to LFM2-VL 1.6B (0.33 GB), not Gemma 3 4B (3.38 GB).
                if display == "LFM2-VL 1.6B" {
                    #expect((slot.modelEntry?.ramGB ?? .infinity) < 1.0)
                }
            case .notRunnable(let d, let reason, _):
                Issue.record("\(d) should resolve: \(reason)")
            }
        }
    }

    // MARK: - CFM-R14-1: the three ASR gaps

    @Test func theThreeAsrGapsResolve() throws {
        let catalog = try loadCatalog()
        for display in ["Whisper Tiny", "Whisper Small", "Voxtral Mini 4B Realtime"] {
            switch CatalogBridge.resolve(display, catalog: catalog) {
            case .runnable(let slot, _, _):
                #expect(slot.modelEntry?.runnerKind == .asr)
            case .notRunnable(let d, let reason, _):
                Issue.record("\(d) should resolve: \(reason)")
            }
        }
    }

    @Test func whisperTinyAndSmallResolveFromTheCatalog() throws {
        let catalog = try loadCatalog()
        let tiny = try #require(CatalogBridge.entry(for: "Whisper Tiny"))
        #expect(tiny.pinnedID == "mlx-community/whisper-tiny")
        #expect(tiny.candidates == ["mlx-community/whisper-tiny-asr-fp16"])
        // A pinned catflow id ≠ candidate id (whisper-tiny vs the -asr-fp16 build) → the
        // requantized class runs silently, never .same.
        #expect(tiny.equivalence == .requantized)
        let small = try #require(CatalogBridge.entry(for: "Whisper Small"))
        #expect(small.pinnedID == "mlx-community/whisper-small-asr-fp16")
        #expect(small.equivalence == .same)
        let voxtral = try #require(CatalogBridge.entry(for: "Voxtral Mini 4B Realtime"))
        #expect(voxtral.pinnedID == "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
        #expect(voxtral.equivalence == .same)
    }

    @Test func whisperSmallManifestShips() throws {
        let data = try Data(contentsOf: manifestURL("whisper-small.json"))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        #expect(manifest.id == "mlx-community/whisper-small-asr-fp16")
        #expect(manifest.display == "Whisper Small")
        #expect(manifest.settings.isEmpty)
    }

    // MARK: - Manifest settings decode (the settings authority)

    @Test func manifestSettingsDecodeEnumsAndDefaults() throws {
        let data = try Data(contentsOf: manifestURL("kokoro-82m-4bit.json"))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        #expect(manifest.display == "Kokoro 82M")
        let voice = try #require(manifest.settings["voice"])
        #expect(voice.type == "enum")
        #expect(voice.values?.contains("af_heart") == true)
        #expect(voice.defaultValue == .string("af_heart"))
        let speed = try #require(manifest.settings["speed"])
        #expect(speed.min == 0.5)
        #expect(speed.max == 2.0)
        #expect(speed.defaultValue == .number(1.0))
    }

    // MARK: - OCP-3-2 / OCP-3-4: OCR prompt capabilities declared in the manifest
    //
    // SPEC-Q212 (owner, 2026-09-08): a model's prompt support is a manifest fact, not a
    // Swift-side table. The free-text OCR models declare `prompt` as a `string` setting
    // that binds positionally; `CuratedManifest` learns to decode `capabilities` so a
    // future `rejected_settings` (DeepSeek-OCR, deferred) is visible in Swift.

    @Test(arguments: ["glm-ocr-4bit.json", "olmocr-2-7b-1025-4bit.json", "dots-ocr-4bit.json"])
    func freeTextOCRManifestsDeclarePromptAsAPositionalString(_ file: String) throws {
        let data = try Data(contentsOf: manifestURL(file))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        let prompt = try #require(manifest.settings["prompt"], "\(file) must declare `prompt`")
        #expect(prompt.type == "string")
        #expect(prompt.positionalOK == true)
        #expect(prompt.values == nil)          // free text, not an enum
    }

    /// The two models catflow-mlx also carries must stay byte-identical to the synced
    /// fixture copy — the `Resources/` manifest and `Fixtures/CatFlow/registry/` are the
    /// same file from two dirs, and OCP-3-2 edited both.
    @Test(arguments: ["olmocr-2-7b-1025-4bit.json", "dots-ocr-4bit.json"])
    func mirroredOCRManifestsMatchTheSyncedFixture(_ file: String) throws {
        let resource = try Data(contentsOf: manifestURL(file))
        let fixture = try Data(contentsOf: repoRoot
            .appendingPathComponent("Fixtures/CatFlow/registry").appendingPathComponent(file))
        #expect(resource == fixture)
    }

    /// OCP-3-4: `capabilities` now reaches Swift. No shipped manifest declares
    /// `rejected_settings` yet (DeepSeek-OCR is deferred), so this pins the decode shape
    /// directly — the prerequisite `VAL-1` builds its E708 producer on.
    @Test func curatedManifestDecodesRejectedSettingsCapability() throws {
        let json = """
        {"id": "x/y", "display": "Y", "settings": {},
         "capabilities": {"variant": "z", "rejected_settings": {"prompt": "the OCR instruction is fixed in the engine"}}}
        """
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: Data(json.utf8))
        #expect(manifest.capabilities?.rejectedSettings?["prompt"]
                == "the OCR instruction is fixed in the engine")
    }

    /// An empty `capabilities: {}` (every shipped OCR manifest today) decodes to a
    /// non-nil block with no rejected settings, never an error.
    @Test func curatedManifestDecodesEmptyCapabilities() throws {
        let data = try Data(contentsOf: manifestURL("glm-ocr-4bit.json"))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        #expect(manifest.capabilities?.rejectedSettings == nil)
    }

    // MARK: - OCP-3-1 prerequisite / AM-W: PaddleOCR-VL joins the bridge
    //
    // The model already installs, runs (OCP-0) and derives into the OCR pool; it lacked a
    // bridge row, so `CuratedManifest.load` had nowhere to hang a settings manifest. The
    // entry's `display` is exactly `browser.json`'s `displayName`, so resolution is
    // byte-identical to the R14-FIX-1 no-bridge fallback it replaces (journal `2026-235`).

    @Test func paddleOCRVLResolvesAsSame() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "PaddleOCR-VL-1.5"))
        #expect(entry.pinnedID == "mlx-community/PaddleOCR-VL-1.5-4bit")
        #expect(entry.candidates == ["mlx-community/PaddleOCR-VL-1.5-4bit"])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "paddleocr-vl-1.5-4bit.json")
        switch CatalogBridge.resolve("PaddleOCR-VL-1.5", catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == "mlx-community/PaddleOCR-VL-1.5-4bit")
            #expect(slot.modelEntry?.runnerKind == .ocr)
            #expect(equivalence == .same)
            #expect(note == nil)          // .same is silent — no substitution on the row
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
    }

    /// BasicGallery flow 3 (`3-ExtractTableFromImage.cat`, gallery number 72) writes the bare
    /// name `PaddleOCR-VL-1.5`. Adding the bridge row moves resolution off the fallback and
    /// through `entry(for:)` — it must still land on the same installable model, or a shipped
    /// gallery flow flips to `.notRunnable`. (`CatFlowNotRunnableTests.flowsThatRunStayRunnable`
    /// covers the whole-flow verdict; this pins the bridge hop the flow's OCR row takes.)
    @Test func galleryFlow72OCRRowResolvesThroughTheBridge() throws {
        let catalog = try loadCatalog()
        let viaBridge = CatalogBridge.resolve("PaddleOCR-VL-1.5", catalog: catalog)
        let viaCatalog = catalog.first { $0.displayName == "PaddleOCR-VL-1.5" }
        guard case .runnable(let slot, _, _) = viaBridge else {
            Issue.record("PaddleOCR-VL-1.5 must resolve for gallery flow 72")
            return
        }
        #expect(slot.modelEntry?.id == viaCatalog?.id)   // same card the fallback used to reach
    }

    @Test func paddleOCRVLManifestShips() throws {
        let data = try Data(contentsOf: manifestURL("paddleocr-vl-1.5-4bit.json"))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        #expect(manifest.id == "mlx-community/PaddleOCR-VL-1.5-4bit")
        #expect(manifest.display == "PaddleOCR-VL-1.5")
        #expect(manifest.capabilities?.rejectedSettings == nil)
        // OCP-3-1: the `mode` enum is the one declared setting.
        let mode = try #require(manifest.settings["mode"], "OCP-3-1: the manifest must declare `mode`")
        #expect(mode.type == "enum")
        #expect(mode.values == ["ocr", "table", "formula", "chart"])
        #expect(mode.defaultValue == .string("ocr"))
        #expect(mode.positionalOK == true)
    }

    /// OCP-3-1 drift guard. `PaddleOCRSDK.promptSupport` still carries the mode list as a
    /// hardcoded Swift array (the vendored `PaddleOCRTask` names, kept out of the manifest so
    /// the `PaddleOCRVL` import stays inside `PaddleOCREngine`). The manifest now declares the
    /// same set. OCP-3-5 will make the manifest the single source; until then, an edit to one
    /// and not the other must fail here.
    @Test func paddleOCRVLManifestModeEnumMatchesTheSDK() throws {
        let data = try Data(contentsOf: manifestURL("paddleocr-vl-1.5-4bit.json"))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        let mode = try #require(manifest.settings["mode"])
        let declaredValues = try #require(mode.values)
        guard case .modes(let sdkValues, let sdkDefault) = PaddleOCRSDK().promptSupport else {
            Issue.record("PaddleOCRSDK.promptSupport should be .modes")
            return
        }
        #expect(declaredValues == sdkValues)
        #expect(mode.defaultValue == .string(sdkDefault))
    }

    // MARK: - MoC-2-4: Qwen3.5 9B joins the bridge, llmModels' last slot

    @Test func qwen35NineBResolvesAsSame() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "Qwen3.5 9B"))
        #expect(entry.pinnedID == "mlx-community/Qwen3.5-9B-MLX-4bit")
        #expect(entry.candidates == ["mlx-community/Qwen3.5-9B-MLX-4bit"])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "qwen3.5-9b-4bit.json")
        switch CatalogBridge.resolve("Qwen3.5 9B", catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == "mlx-community/Qwen3.5-9B-MLX-4bit")
            #expect(slot.modelEntry?.runnerKind == .llm)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
        #expect(CatalogBridge.entries.count == 23)
    }

    // MARK: - MoC-4-4: Qwen3 Reranker 0.6B joins the bridge, the model on MoC-3's seam

    @Test func qwen3RerankerResolvesAsSame() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "Qwen3 Reranker 0.6B"))
        #expect(entry.pinnedID == "mlx-community/Qwen3-Reranker-0.6B-4bit")
        #expect(entry.candidates == ["mlx-community/Qwen3-Reranker-0.6B-4bit"])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "qwen3-reranker-0.6b-4bit.json")
        switch CatalogBridge.resolve("Qwen3 Reranker 0.6B", catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == "mlx-community/Qwen3-Reranker-0.6B-4bit")
            #expect(slot.modelEntry?.runnerKind == .rerank)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
    }

    @Test func qwen3RerankerManifestShips() throws {
        let data = try Data(contentsOf: manifestURL("qwen3-reranker-0.6b-4bit.json"))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        #expect(manifest.id == "mlx-community/Qwen3-Reranker-0.6B-4bit")
        #expect(manifest.display == "Qwen3 Reranker 0.6B")
        #expect(manifest.settings.isEmpty)
    }

    /// The done-when this whole arc has been building toward: `defaultModel(forTask:
    /// "Rerank")` flips from `nil` (MoC-1…MoC-3) to a name, for the first time.
    @MainActor
    @Test func defaultModelForRerankIsNoLongerNil() throws {
        let catalog = try loadCatalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        #expect(TaskModels.defaultModel(forTask: "Rerank", catalog: catalog, claimableModelIDs: claimable) == "Qwen3 Reranker 0.6B")
        let derived = TaskModels.derivedModels(for: "Rerank", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.count == 1)
    }

    @Test func qwen35NineBManifestShips() throws {
        let data = try Data(contentsOf: manifestURL("qwen3.5-9b-4bit.json"))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        #expect(manifest.id == "mlx-community/Qwen3.5-9B-MLX-4bit")
        #expect(manifest.display == "Qwen3.5 9B")
        let temperature = try #require(manifest.settings["temperature"])
        #expect(temperature.min == 0.0)
        #expect(temperature.max == 2.0)
        #expect(temperature.defaultValue == .number(0.6))
        let maxTokens = try #require(manifest.settings["max_tokens"])
        #expect(maxTokens.min == 1)
        #expect(maxTokens.max == 32768)
        #expect(maxTokens.defaultValue == .number(2048))
    }

    // MARK: - MoC-6-2: Qwen3.5 9B's vision half, sharing the text entry's hfModelId

    /// The disambiguation this phase's fix exists for: both bridge entries name the same
    /// hfModelId, and each must resolve to its own catalog card, not each other's.
    @Test func qwen35NineBVisionResolvesAsSameButDistinctFromTheTextEntry() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "Qwen3.5 9B Vision"))
        #expect(entry.pinnedID == "mlx-community/Qwen3.5-9B-MLX-4bit")
        #expect(entry.candidates == ["mlx-community/Qwen3.5-9B-MLX-4bit"])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "qwen3.5-9b-vision-4bit.json")
        switch CatalogBridge.resolve("Qwen3.5 9B Vision", catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == "mlx-community/Qwen3.5-9B-MLX-4bit")
            #expect(slot.modelEntry?.runnerKind == .vision)
            #expect(slot.modelEntry?.id == "mlx-community--Qwen3.5-9B-MLX-4bit-vision")
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
        // The text entry must still resolve to the llm card, not the vision one — this is
        // the regression the `modelType` disambiguator in `resolve` guards against.
        switch CatalogBridge.resolve("Qwen3.5 9B", catalog: catalog) {
        case .runnable(let slot, _, _):
            #expect(slot.modelEntry?.runnerKind == .llm)
            #expect(slot.modelEntry?.id == "mlx-community--Qwen3.5-9B-MLX-4bit")
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
    }

    @Test func qwen35NineBVisionManifestShips() throws {
        let data = try Data(contentsOf: manifestURL("qwen3.5-9b-vision-4bit.json"))
        let manifest = try JSONDecoder().decode(CuratedManifest.self, from: data)
        #expect(manifest.id == "mlx-community/Qwen3.5-9B-MLX-4bit")
        #expect(manifest.display == "Qwen3.5 9B Vision")
        #expect(manifest.settings.isEmpty)
    }

    /// `defaultModel(forTask: "Describe Image")` must stay `LFM2-VL 1.6B` — appending the
    /// vision entry last must not move the seed (MoC-6-2's done-when).
    @MainActor
    @Test func defaultModelForDescribeImageIsUnchangedByTheVisionEntry() throws {
        let catalog = try loadCatalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        #expect(TaskModels.defaultModel(forTask: "Describe Image", catalog: catalog, claimableModelIDs: claimable) == "LFM2-VL 1.6B")
        let derived = TaskModels.derivedModels(for: "Describe Image", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.contains { $0.id == "mlx-community--Qwen3.5-9B-MLX-4bit-vision" })
    }

    // MARK: - LY-8: Laya 0.4B joins the bridge, prepended as Classify/Gate/Score's default

    @Test func layaResolvesAsSame() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "Laya 0.4B"))
        #expect(entry.pinnedID == "aac6fef/laya-mlx")
        #expect(entry.candidates == ["aac6fef/laya-mlx"])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "laya-0.4b.json")
        switch CatalogBridge.resolve("Laya 0.4B", catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == "aac6fef/laya-mlx")
            #expect(slot.modelEntry?.runnerKind == .decision)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
    }

    /// The manifest, read from the built app bundle (not the repo-relative disk copy the other
    /// `*ManifestShips` tests use) — LY-8's own verify item: `CuratedManifest.load(manifestFile:
    /// "laya-0.4b.json") != nil`. `XcodeWrite` added the file to the synchronized
    /// `Resources/CatFlow/models/` group, so it ships the same way every other manifest there
    /// does (`CuratedManifest.load`'s own doc comment).
    @Test func layaManifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "laya-0.4b.json"))
        #expect(manifest.id == "aac6fef/laya-mlx")
        #expect(manifest.display == "Laya 0.4B")
        #expect(manifest.settings.isEmpty)
    }

    /// LY-8's own verify items: `defaultModel(forTask:)` for Classify/Gate/Score is `"Laya
    /// 0.4B"` (the 2026-09-27 gate D amendment — prepended, not appended, to those pools), and
    /// Laya is a member of each derived pool (not just the seed by coincidence).
    @MainActor
    @Test func layaIsTheDefaultForClassifyGateAndScore() throws {
        let catalog = try loadCatalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        for task in ["Classify", "Gate", "Score"] {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Laya 0.4B",
                    "\(task) should default to Laya 0.4B")
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(derived.contains { $0.modelEntry?.hfModelId == "aac6fef/laya-mlx" },
                    "\(task)'s derived pool should contain Laya 0.4B")
        }
    }

    // MARK: - CL-7: CLM 8B joins the bridge, appended (never the default)

    @Test func clmResolvesAsSame() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: "CLM 8B"))
        #expect(entry.pinnedID == "RealityCat/CLM-v0.1-8B-MLX-8bit")
        #expect(entry.candidates == ["RealityCat/CLM-v0.1-8B-MLX-8bit"])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == "clm-v0.1-8b-8bit.json")
        switch CatalogBridge.resolve("CLM 8B", catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == "RealityCat/CLM-v0.1-8B-MLX-8bit")
            #expect(slot.modelEntry?.runnerKind == .decision)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
    }

    @Test func clmManifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "clm-v0.1-8b-8bit.json"))
        #expect(manifest.id == "RealityCat/CLM-v0.1-8B-MLX-8bit")
        #expect(manifest.display == "CLM 8B")
        #expect(manifest.settings.isEmpty)
    }

    /// CL-7's own verify items: `defaultModel(forTask:)` for Classify/Gate/Score stays `"Laya
    /// 0.4B"` (gate D — CLM joins the pool but is never the default), and CLM is a member of
    /// each derived pool, **last**.
    @MainActor
    @Test func clmIsLastInEachDerivedPoolWhileLayaStaysTheDefault() throws {
        let catalog = try loadCatalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        for task in ["Classify", "Gate", "Score"] {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Laya 0.4B",
                    "\(task) should still default to Laya 0.4B, unchanged by CLM joining")
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(derived.last?.modelEntry?.hfModelId == "RealityCat/CLM-v0.1-8B-MLX-8bit",
                    "\(task)'s derived pool should end with CLM 8B")
        }
    }

    /// CL-7's fourth verify item. Before this item, "CLM 8B" had no `BridgeEntry` — `resolve`
    /// would have fallen through to the "derived pick" branch (a direct `catalog.first(where:
    /// displayName == display)` match), which happens to succeed too, since the bundled
    /// catalog's own `displayName` for this entry already reads "CLM 8B". The bridge row
    /// replaces that coincidence with an explicit, reviewable `.same` equivalence — the same
    /// discipline every other entry in this table gets (`CatalogBridge.swift`'s own header:
    /// "never guess a substitution that isn't in the table"). This drives the real bundled
    /// catalog + a fake `askCLM` through `CachingExecutor(RealExecutor)` — the exact
    /// composition `AppFlowExecutorFactory.cachingContext` builds for a real Run — and
    /// asserts the row resolves through the bridge entry and fires a tag, never the generic
    /// "isn't in the runnable-model table" refusal.
    @Test func clmRowResolvesAtRunThroughTheCachingWrapperNotTheRunnableTableRefusal() async throws {
        let catalog = try loadCatalog()
        let clm = try #require(catalog.first { $0.hfModelId == "RealityCat/CLM-v0.1-8B-MLX-8bit" })

        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("clm-bridge-run-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let store = FlowCacheStore(root: base.appendingPathComponent("cache"))

        var real = RealExecutor(
            workspace: ws, flowID: "f", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .llm) },
            installedModelIDs: [clm.id], catalog: catalog)
        real.askCLM = { _, _, _, _, _ in
            var answer = LayaAnswer(type: .choice, confidence: 0.8, probabilities: [0.9, 0.1],
                                    optionLabels: ["billing", "technical"], stateTruncated: false)
            answer.choiceLabel = "billing"
            return answer
        }
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: catalog, runSeed: 0, workspace: ws, flowID: "f")

        let row = Row(id: UUID(), task: "Classify", model: "CLM 8B", settings: "Who owns this?",
                     tags: ["billing", "technical"])
        let input = Asset(items: [Item(kind: .text, value: "my invoice was charged twice", path: nil, sourceText: nil)])
        let output = try await caching.execute(path: "1", row: row, inputs: [input],
                                               transcript: nil, context: nil, usedFlowContent: nil)
        #expect(output.items.first?.value == "my invoice was charged twice")
        #expect(caching.lastTag == "billing")
    }
}
