import Testing
import Foundation
import MLXAudioSTT
@testable import MLXUI

/// MP2-3: the Parakeet module's resolution and install selection, with no weights needed.
/// The real load and the main-actor probe are in `ParakeetRealLoadTests.swift` (gated on the install).
@MainActor
struct ParakeetModuleTests {

    private func parakeetEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--parakeet-tdt-0.6b-v3", family: "Parakeet", modelType: .asr, source: .mlx)
    }

    @Test func parakeetResolvesToParakeetModule() throws {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let resolved = try #require(registry.bestModule(for: parakeetEntry()))
        #expect(resolved.descriptor.id == "parakeet")
        #expect(resolved.sdk.claim(parakeetEntry()) == .exact)
    }

    @Test func parakeetClaimsOnlyParakeetMLXASR() {
        let sdk = ParakeetSDK()
        #expect(sdk.claim(makeEntry(modelType: .asr, source: .coreml)) == .no)
        #expect(sdk.claim(makeEntry(id: "mlx-community--whisper-small-asr-fp16", family: "whisper",
                                    modelType: .asr, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "v", family: "Voxtral", modelType: .asr, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "p", family: "Parakeet", modelType: .llm, source: .mlx)) == .no)
    }

    @Test func parakeetMakeStageProducesASRStage() throws {
        let stage = try ParakeetSDK().makeStage(for: parakeetEntry(), config: .default)
        #expect(stage is ASRStage)
        #expect(stage.accepts == .audio && stage.produces == .text)
    }

    /// T3 (readme): every ASR entry in the bundled catalog (the four existing plus Parakeet) resolves to
    /// exactly one module, and Parakeet's family string doesn't pull in, or get pulled into, another family's
    /// claimers. Whisper is the one designed overlap: `mlx-whisper` and `whisper` both claim an MLX Whisper
    /// entry and registration order breaks the tie (`App/ModelModules.swift`), so its claimer set has two.
    @Test func everyBundledASREntryResolvesToExactlyOneModule() throws {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        let asr = try #require(catalog.domains.first { $0.id == "asr" }).allModels
        #expect(asr.count == 5, "the four existing ASR entries plus Parakeet: \(asr.map(\.id))")

        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }

        for entry in asr {
            var claimers: Set<String> = []
            for module in installedModules {
                let registry = ModelRegistry()
                module.register(into: registry)
                if registry.bestModule(for: entry) != nil { claimers.insert(module.descriptor.id) }
            }
            let family = entry.family.lowercased()
            let expectedClaimers: Set<String>
            let expectedWinner: String
            if family.contains("whisper") {
                expectedClaimers = ["mlx-whisper", "whisper"]; expectedWinner = "mlx-whisper"
            } else if family.contains("voxtral") {
                expectedClaimers = ["voxtral"]; expectedWinner = "voxtral"
            } else {
                expectedClaimers = ["parakeet"]; expectedWinner = "parakeet"
            }
            #expect(claimers == expectedClaimers, "\(entry.id) is claimed by \(claimers.sorted())")
            #expect(full.bestModule(for: entry)?.descriptor.id == expectedWinner, "\(entry.id)")
        }
    }

    /// T2: the install selector for this repo's sibling list (pinned hf/tree.json, mlx-community/
    /// parakeet-tdt-0.6b-v3 @ ed2b7e8). It takes config.json, model.safetensors and tokenizer.model and
    /// drops tokenizer.vocab and vocab.txt. The loader needs only config.json + *.safetensors (the
    /// vocabulary is in config.json's joint.vocabulary), proved by `ParakeetRealLoadTests`.
    @Test func installSelectorPicksTheFilesTheLoaderReads() {
        let siblings = [".gitattributes", "README.md", "config.json", "model.safetensors",
                        "tokenizer.model", "tokenizer.vocab", "vocab.txt"]
        let picked = Set(ModelFileSelector.filesToDownload(siblings: siblings))
        #expect(picked == ["config.json", "model.safetensors", "tokenizer.model"])
    }

    // MP2-3-FIX-2 (owner ruling X, 2026-10-24): long audio is chunked at 120 s, everything else stays the model's default.
    @Test func engineChunksLongAudioAt120SecondsAndChangesNothingElse() {
        let base = STTGenerateParameters(maxTokens: 8192, temperature: 0.0, topP: 0.95, topK: 0, verbose: false,
                                         language: "en", chunkDuration: 1200.0, minChunkDuration: 1.0)
        let tuned = ParakeetEngine.generationParameters(from: base)
        #expect(tuned.chunkDuration == 120)
        #expect(ParakeetEngine.chunkSeconds == 120)
        #expect(tuned.maxTokens == base.maxTokens && tuned.temperature == base.temperature)
        #expect(tuned.topP == base.topP && tuned.topK == base.topK && tuned.language == base.language)
        #expect(tuned.minChunkDuration == base.minChunkDuration)
        #expect(tuned.repetitionPenalty == base.repetitionPenalty)
        #expect(tuned.repetitionContextSize == base.repetitionContextSize)
    }
}
