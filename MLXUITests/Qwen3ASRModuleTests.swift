import Testing
import Foundation
import MLXAudioSTT
@testable import MLXUI

/// MP3-3: the Qwen3-ASR module's resolution, claim, stage and engine parameters, with no weights needed.
/// The real load and the main-actor probe are in `Qwen3ASRRealLoadTests.swift` (gated on the install); the install
/// side effect (readme T5) is in `Qwen3ASRInstallTests.swift`.
@MainActor
struct Qwen3ASRModuleTests {

    private func qwenEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--Qwen3-ASR-1.7B-8bit", family: "Qwen3-ASR", modelType: .asr, source: .mlx)
    }

    @Test func qwen3ASRResolvesToQwen3ASRModule() throws {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let resolved = try #require(registry.bestModule(for: qwenEntry()))
        #expect(resolved.descriptor.id == "qwen3-asr")
        #expect(resolved.sdk.claim(qwenEntry()) == .exact)
    }

    /// Claims only its own family, and every other ASR family (and kind, and source) declines it.
    @Test func qwen3ASRClaimsOnlyQwen3ASRMLXASR() {
        let sdk = Qwen3ASRSDK()
        #expect(sdk.claim(makeEntry(family: "Qwen3-ASR", modelType: .asr, source: .coreml)) == .no)
        #expect(sdk.claim(makeEntry(id: "w", family: "whisper", modelType: .asr, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "v", family: "Voxtral", modelType: .asr, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "p", family: "Parakeet", modelType: .asr, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "q", family: "Qwen3-ASR", modelType: .llm, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "c", family: "Qwen3", modelType: .llm, source: .mlx)) == .no)
        for other: any ModelSDK in [WhisperSDK(), MLXWhisperSDK(), VoxtralSDK(), ParakeetSDK()] {
            #expect(other.claim(qwenEntry()) == .no, "\(other.id) must decline Qwen3-ASR")
        }
    }

    @Test func qwen3ASRMakeStageProducesASRStage() throws {
        let stage = try Qwen3ASRSDK().makeStage(for: qwenEntry(), config: .default)
        #expect(stage is ASRStage)
        #expect(stage.accepts == .audio && stage.produces == .text)
    }

    /// The module declares no task family: `TaskModels.taskFamilies` stays empty (readme T3), so a new ASR module can
    /// never pull another model into a task by family. The descriptor lists `.asr` only.
    @Test func descriptorIsASROnly() {
        #expect(Qwen3ASRModule.descriptor.modelTypes == [.asr])
        #expect(Qwen3ASRModule.descriptor.id == "qwen3-asr")
    }

    /// T3 (readme): every ASR entry in the bundled catalog (four existing, Parakeet, Qwen3-ASR) resolves to exactly one
    /// module in its own right. Whisper is the one designed overlap: `mlx-whisper` and `whisper` both claim an MLX Whisper
    /// entry and registration order breaks the tie (`App/ModelModules.swift`).
    @Test func everyBundledASREntryResolvesToExactlyOneModule() throws {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        let asr = try #require(catalog.domains.first { $0.id == "asr" }).allModels
        #expect(asr.count == 6, "whisper x3, voxtral, parakeet, qwen3-asr: \(asr.map(\.id))")

        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }

        var winners: [String: String] = [:]
        for entry in asr {
            var claimers: Set<String> = []
            for module in installedModules {
                let registry = ModelRegistry()
                module.register(into: registry)
                if registry.bestModule(for: entry) != nil { claimers.insert(module.descriptor.id) }
            }
            let family = entry.family.lowercased()
            let expected: Set<String>
            if family.contains("whisper") { expected = ["mlx-whisper", "whisper"] }
            else if family.contains("voxtral") { expected = ["voxtral"] }
            else if family.contains("parakeet") { expected = ["parakeet"] }
            else if family.contains("qwen3-asr") { expected = ["qwen3-asr"] }
            else { Issue.record("unexpected ASR family \(entry.family)"); continue }
            #expect(claimers == expected, "\(entry.id) is claimed by \(claimers.sorted())")
            winners[entry.id] = full.bestModule(for: entry)?.descriptor.id
        }
        #expect(winners["mlx-community--Qwen3-ASR-1.7B-8bit"] == "qwen3-asr")
        #expect(winners["mlx-community--parakeet-tdt-0.6b-v3"] == "parakeet")
        #expect(winners["mlx-community--Voxtral-Mini-4B-Realtime-2602-4bit"] == "voxtral")
        #expect(winners.values.allSatisfy { !$0.isEmpty })
    }

    /// Engine parameters: the model's own defaults with only `chunkDuration` replaced. Language stays `nil` (the model
    /// identifies it) and the repetition penalty stays 1.0, which keeps the library's `nextLast[0..., recentArr]` gather
    /// (Qwen3ASR.swift:1326/1528 @ 5c4c12b, the thread-bound-stream pattern, KICKOFF rule 17) from ever being built.
    @Test func engineChunksAt120SecondsKeepsLanguageAutoAndThePenaltyOff() {
        // The library's `Qwen3ASRModel.defaultGenerationParameters` at 5c4c12b (Qwen3ASR.swift:32-42).
        let base = STTGenerateParameters(maxTokens: 8192, temperature: 0.0, topP: 0.95, topK: 0, verbose: false,
                                         language: nil, chunkDuration: 1200.0, minChunkDuration: 1.0)
        let tuned = Qwen3ASREngine.generationParameters(from: base)
        #expect(tuned.chunkDuration == 120)
        #expect(Qwen3ASREngine.chunkSeconds == 120)
        #expect(tuned.language == nil, "auto language identification")
        #expect(tuned.repetitionPenalty == 1.0, "the penalty branch holds the mixed-subscript gather; it must stay off")
        #expect(tuned.maxTokens == base.maxTokens && tuned.temperature == base.temperature)
        #expect(tuned.topP == base.topP && tuned.topK == base.topK)
        #expect(tuned.minChunkDuration == base.minChunkDuration)
        #expect(tuned.repetitionContextSize == base.repetitionContextSize)
    }

    /// MP3-FIX-1: a chunk's own length can be passed as `chunkDuration`, so the library does not split it again. Nothing else moves.
    @Test func aChunkDurationOverrideChangesOnlyTheChunkDuration() {
        let base = STTGenerateParameters(maxTokens: 8192, temperature: 0.0, topP: 0.95, topK: 0, verbose: false,
                                         language: nil, chunkDuration: 1200.0, minChunkDuration: 1.0)
        let tuned = Qwen3ASREngine.generationParameters(from: base, chunkDuration: 124)
        #expect(tuned.chunkDuration == 124)
        #expect(tuned.maxTokens == 8192 && tuned.language == nil && tuned.repetitionPenalty == 1.0)
    }

    /// Whatever the base says, the engine does not turn the penalty or a forced language on by itself.
    @Test func engineDoesNotInventALanguageOrAPenalty() {
        let base = STTGenerateParameters()
        let tuned = Qwen3ASREngine.generationParameters(from: base)
        #expect(tuned.language == base.language && base.language == nil)
        #expect(tuned.repetitionPenalty == base.repetitionPenalty && base.repetitionPenalty == 1.0)
    }

    /// T5, selection half: the install fetches everything the loader reads, for this repo's pinned sibling list
    /// (mlx-community/Qwen3-ASR-1.7B-8bit @ a8379a2e, `tree.json`). The loader reads `config.json`, `vocab.json`,
    /// `merges.txt`, `tokenizer_config.json` and every `*.safetensors`; it writes `tokenizer.json` itself because the
    /// repo ships none. The rest of the selection (index, generation / preprocessor config, chat template) is fetched
    /// but unused by the Swift loader, which is harmless and is `metadataNames`' standing behaviour.
    @Test func installSelectorPicksTheFilesTheLoaderReads() {
        let siblings = [".gitattributes", "README.md", "chat_template.json", "config.json", "generation_config.json",
                        "merges.txt", "model.safetensors", "model.safetensors.index.json", "preprocessor_config.json",
                        "tokenizer_config.json", "vocab.json"]
        let picked = Set(ModelFileSelector.filesToDownload(siblings: siblings))
        #expect(picked == ["chat_template.json", "config.json", "generation_config.json", "merges.txt",
                           "model.safetensors", "model.safetensors.index.json", "preprocessor_config.json",
                           "tokenizer_config.json", "vocab.json"])
        let loaderReads: Set<String> = ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "model.safetensors"]
        #expect(loaderReads.isSubset(of: picked))
        #expect(!picked.contains("tokenizer.json"), "the repo has none; the library generates it on first load")
        #expect(!picked.contains("README.md") && !picked.contains(".gitattributes"))
    }
}
