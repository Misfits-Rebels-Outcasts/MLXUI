import Foundation
import Testing
@testable import MLXUI

/// MG-CAT1 — verifies the `jasonvassallo/mlx-musicgen-small` catalog entry (float32 music
/// model) decodes, routes to the `.music` RunnerKind, and carries **no** `ModelSupport` gap.
/// Uses an inline fixture mirroring the `browser.json` entry (the bundled catalog isn't
/// reachable from the test bundle) — the same pattern as `TernaryBonsaiResolutionTests`.
struct MusicGenTests {

    private func makeEntry() throws -> ModelEntry {
        let json = """
        {
            "id": "jasonvassallo--mlx-musicgen-small",
            "family": "MusicGen",
            "displayName": "MusicGen-small",
            "paramSize": "0.6B",
            "paramCountB": 0.615,
            "modelType": "music",
            "source": "mlx",
            "format": "mlx-f32",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "jasonvassallo",
            "hfModelId": "jasonvassallo/mlx-musicgen-small",
            "ramGB": 3.7,
            "downloadSizeGB": 2.5,
            "contextWindow": null,
            "license": "cc-by-nc-4.0",
            "licenseUrl": "https://huggingface.co/jasonvassallo/mlx-musicgen-small/blob/main/LICENSE",
            "architecture": "MusicgenForConditionalGeneration",
            "languages": ["en"],
            "lastUpdated": "2024-01-01",
            "taskTags": ["text-to-music", "music-generation"],
            "descriptionSource": "jasonvassallo/mlx-musicgen-small",
            "variants": [
                {
                    "quantization": "f32",
                    "format": "mlx-f32",
                    "ramGB": 3.7,
                    "downloadSizeGB": 2.5,
                    "qualityPercent": 100,
                    "hfModelId": "jasonvassallo/mlx-musicgen-small",
                    "recommended": true
                }
            ]
        }
        """
        return try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
    }

    /// The `music` model type decodes and routes to the `.music` RunnerKind.
    @Test func musicgenEntryResolvesToMusicRunner() throws {
        let entry = try makeEntry()
        #expect(entry.modelType == .music)
        #expect(entry.source == .mlx)
        #expect(entry.runnerKind == .music)
    }

    /// A `.music` model with `source == .mlx` carries no `ModelSupport` gap — the
    /// MusicGen engine (MG-ENG1…ENG4) will handle it, so no `UnsupportedModelView`.
    @Test func musicgenHasNoSupportGap() throws {
        let entry = try makeEntry()
        #expect(ModelSupport.unsupportedReason(for: entry) == nil)
    }

    /// The `music` sfSymbol is the music note (display sugar for the new ModelType case).
    @Test func musicModelTypeHasSfSymbol() throws {
        let entry = try makeEntry()
        #expect(ModelType.music.sfSymbol == "music.note")
        #expect(entry.modelType.sfSymbol == "music.note")
    }

    /// MG-MOD1 — the MusicGen SDK claims the entry `.exact`; Flux and the TTS modules' SDKs
    /// decline it (disjoint predicates).
    @Test func musicgenSDKClaimsExactOthersDecline() throws {
        let entry = try makeEntry()
        #expect(MusicGenSDK().claim(entry) == .exact)
        #expect(FluxSDK().claim(entry) == .no)
    }

    /// MG-MOD1 — `MusicGenStage` builds from the catalog id and reports `.text` → `.audio`.
    @Test func musicgenStageContract() throws {
        let entry = try makeEntry()
        let stage = try MusicGenSDK().makeStage(for: entry, config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .audio)
    }

    /// MG-MOD1 — `MusicGenModule` registers into a fresh registry and claims the entry.
    @MainActor @Test func musicgenModuleRegisters() throws {
        let entry = try makeEntry()
        let registry = ModelRegistry()
        MusicGenModule.register(into: registry)
        let resolved = registry.bestModule(for: entry)
        #expect(resolved != nil)
        #expect(resolved?.descriptor.id == "musicgen")
    }
}
