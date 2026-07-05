import Testing
@testable import MLXUI

/// Covers `ModelFileSelector.filesToDownload` (backlog K1b-1): which repo files an install
/// fetches. Guards that the Kokoro additions (voices/, non-standard weights) work AND that
/// standard single/sharded model installs are unchanged.
struct ModelFileSelectorTests {
    private func selected(_ siblings: [String]) -> Set<String> {
        Set(ModelFileSelector.filesToDownload(siblings: siblings))
    }

    @Test func standardSingleWeightRepoUnchanged() {
        let out = selected([
            "config.json", "tokenizer.json", "model.safetensors", "README.md", ".gitattributes",
        ])
        #expect(out == ["config.json", "tokenizer.json", "model.safetensors"])
    }

    @Test func shardedWeightsIncludeShardsAndIndex() {
        let out = selected([
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ])
        #expect(out == [
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ])
    }

    @Test func strayTopLevelSafetensorsExcludedWhenStandardWeightsPresent() {
        // `model.safetensors` is present, so the non-standard fallback must NOT pull `extra`.
        let out = selected(["config.json", "model.safetensors", "extra.safetensors"])
        #expect(!out.contains("extra.safetensors"))
        #expect(out.contains("model.safetensors"))
    }

    @Test func kokoroRepoIncludesWeightsAndVoicesOnly() {
        let out = selected([
            "config.json",
            "kokoro-v1_0.safetensors",
            "voices/af_heart.safetensors",
            "voices/am_adam.safetensors",
            "voices/af_heart.pt",          // duplicate format — skip
            "samples/demo.wav",            // sample — skip
            "VOICES.md",                   // doc — skip
        ])
        #expect(out == [
            "config.json",
            "kokoro-v1_0.safetensors",
            "voices/af_heart.safetensors",
            "voices/am_adam.safetensors",
        ])
    }

    @Test func qwen3TTSIncludesSpeechTokenizerSubfolder() {
        // Qwen3-TTS ships a `speech_tokenizer/` component (weights + its own config). Without
        // it the model loads but throws "Speech tokenizer not loaded" at generate time.
        let out = selected([
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "README.md",
        ])
        #expect(out == [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
        ])
    }

    @Test func voxtralIncludesTekkenTokenizer() {
        // Voxtral ships its tokenizer as `tekken.json` (Mistral tekken format); the runner
        // reads it directly and fails with "tekken.json not found" if it's not downloaded.
        let out = selected([
            "config.json", "tekken.json", "model.safetensors", "README.md",
        ])
        #expect(out.contains("tekken.json"))
        #expect(out == ["config.json", "tekken.json", "model.safetensors"])
    }

    @Test func includesSentencePieceAndDictTokenizers() {
        // Runners that read raw tokenizer assets: SentencePiece `tokenizer.model` (MossTTS-Nano,
        // CohereTranscribe) and FireRedASR2's `dict.txt`. Both must download or the model can't
        // build its tokenizer at run time.
        let out = selected([
            "config.json", "tokenizer.model", "dict.txt", "model.safetensors", "README.md",
        ])
        #expect(out.contains("tokenizer.model"))
        #expect(out.contains("dict.txt"))
    }

    @Test func skipsDocsScriptsAndPtFiles() {
        let out = selected(["README.md", "convert.py", "notes.ipynb", "voice.pt", "config.json"])
        #expect(out == ["config.json"])
    }

    @Test func includesSeparateChatTemplate() {
        // Qwen3-VL etc. ship the chat template as its own file — must be downloaded, else the
        // tokenizer can't apply it (journal/2026-38).
        let out = selected([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "chat_template.json", "model.safetensors",
        ])
        #expect(out.contains("chat_template.json"))
    }
}
