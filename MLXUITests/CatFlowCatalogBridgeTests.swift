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
        switch CatalogBridge.resolve("SAM Base", catalog: catalog) {
        case .notRunnable(let display, let reason):
            #expect(display == "SAM Base")
            #expect(reason.contains("SAM Base"))
        case .runnable:
            Issue.record("SAM Base must NOT resolve (hazard H2)")
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

    // MARK: - SAM Base is deliberately absent (hazard H2)

    @Test func samBaseIsNotInTheBridge() {
        #expect(CatalogBridge.entry(for: "SAM Base") == nil)
        #expect(CatalogBridge.entries.count == 11)
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
