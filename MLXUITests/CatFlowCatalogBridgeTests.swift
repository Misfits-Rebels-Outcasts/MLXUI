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
            case .runnable(let model, let equivalence, _):
                #expect(model.hfModelId == entry.candidates.first)
                #expect(equivalence == entry.equivalence)
            case .notRunnable(let display, let reason):
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
        case .notRunnable(let display, let reason):
            #expect(display == "Turbo Chat 5000")
            #expect(reason.contains("Turbo Chat 5000"))
        case .runnable:
            Issue.record("an unknown display name must be not-runnable")
        }
    }

    @Test func noCandidateIsNotRunnableNamingTheModel() throws {
        let catalog = try loadCatalog()
        switch CatalogBridge.resolve("Totally Missing", catalog: catalog) {
        case .notRunnable(let display, let reason):
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
        case .runnable(let model, let equivalence, let note):
            #expect(model.hfModelId == unbridged)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason):
            Issue.record("\(display) should resolve via the derived-pool fallback: \(reason)")
        }
    }

    /// The FIX-7 display shape (a catalog `displayName`, not the id) also resolves.
    @Test func unbridgedDisplayNameResolvesAsSame() throws {
        let catalog = try loadCatalog()
        // Qwen3-VL's browser.json displayName is the friendly name, not the id.
        let displayName = "Qwen3-VL-4B-Instruct"
        switch CatalogBridge.resolve(displayName, catalog: catalog) {
        case .runnable(let model, _, _):
            #expect(model.hfModelId == "mlx-community/Qwen3-VL-4B-Instruct-4bit")
        case .notRunnable(let display, let reason):
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
        case .runnable(let model, let equivalence, let note):
            #expect(model.hfModelId == "mlx-community/sam3-4bit")
            #expect(equivalence == .substitute)
            #expect(note != nil)
            #expect(note?.contains("SAM Base") == true)
            #expect(note?.contains("mlx-community/sam3-4bit") == true)
        case .notRunnable(let display, let reason):
            Issue.record("SAM Base should resolve: \(reason)")
            _ = display
        }
        #expect(CatalogBridge.entries.count == 16)
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
            case .runnable(let model, _, _):
                // The cheap one is the pool's first entry, so a bare Describe Image row
                // defaults to LFM2-VL 1.6B (0.33 GB), not Gemma 3 4B (3.38 GB).
                if display == "LFM2-VL 1.6B" {
                    #expect(model.ramGB < 1.0)
                }
            case .notRunnable(let d, let reason):
                Issue.record("\(d) should resolve: \(reason)")
            }
        }
    }

    // MARK: - CFM-R14-1: the three ASR gaps

    @Test func theThreeAsrGapsResolve() throws {
        let catalog = try loadCatalog()
        for display in ["Whisper Tiny", "Whisper Small", "Voxtral Mini 4B Realtime"] {
            switch CatalogBridge.resolve(display, catalog: catalog) {
            case .runnable(let model, _, _):
                #expect(model.runnerKind == .asr)
            case .notRunnable(let d, let reason):
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
}
