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

    @Test func sdxlTurboRepoIncludesComponentsOnly() {
        // SD-DL1 — `stabilityai/sdxl-turbo` (diffusers layout). Every component subdir must
        // download its **fp16** weights (the fp32 twin is redundant — the engine loads fp16),
        // both tokenizers, and the JSON-only `scheduler/scheduler_config.json`. The root
        // full checkpoints and docs/scripts must NOT come down.
        let out = selected([
            "model_index.json",
            "README.md",
            "convert.py",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
            "text_encoder_2/model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer_2/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/special_tokens_map.json",
            "scheduler/scheduler_config.json",
            "sd_xl_turbo_1.0.safetensors",
            "sd_xl_turbo_1.0_fp16.safetensors",
        ])
        // fp16 component weights included…
        #expect(out.contains("unet/diffusion_pytorch_model.fp16.safetensors"))
        #expect(out.contains("text_encoder/model.fp16.safetensors"))
        #expect(out.contains("text_encoder_2/model.fp16.safetensors"))
        #expect(out.contains("vae/diffusion_pytorch_model.fp16.safetensors"))
        // …their fp32 twins excluded…
        #expect(!out.contains("unet/diffusion_pytorch_model.safetensors"))
        #expect(!out.contains("text_encoder/model.safetensors"))
        #expect(!out.contains("text_encoder_2/model.safetensors"))
        #expect(!out.contains("vae/diffusion_pytorch_model.safetensors"))
        // tokenizers + scheduler metadata included…
        #expect(out.contains("tokenizer/vocab.json"))
        #expect(out.contains("tokenizer/merges.txt"))
        #expect(out.contains("tokenizer_2/vocab.json"))
        #expect(out.contains("tokenizer_2/merges.txt"))
        #expect(out.contains("scheduler/scheduler_config.json"))
        // …and the redundant root full checkpoints + docs excluded.
        #expect(!out.contains("sd_xl_turbo_1.0.safetensors"))
        #expect(!out.contains("sd_xl_turbo_1.0_fp16.safetensors"))
        #expect(!out.contains("README.md"))
        #expect(!out.contains("convert.py"))
    }

    @Test func fp16TwinPreferenceDropsFp32Original() {
        // The fp16-preference rule on its own: when a dir ships `X.safetensors` + the
        // `X.fp16.safetensors` twin, only the fp16 file is selected.
        let out = selected([
            "unet/weights.safetensors",
            "unet/weights.fp16.safetensors",
            "config.json",
        ])
        #expect(out == ["unet/weights.fp16.safetensors", "config.json"])
    }

    @Test func diffusersLayoutSuppressesRootFallback() {
        // A diffusers repo must not hit the top-level single-file fallback — its root
        // `.safetensors` are full checkpoints duplicating the components. Kokoro-style repos
        // (no model_index.json) keep the fallback (covered by kokoroRepoIncludesWeightsAndVoicesOnly).
        let out = selected([
            "model_index.json",
            "config.json",
            "sd_xl_turbo_1.0_fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ])
        #expect(!out.contains("sd_xl_turbo_1.0_fp16.safetensors"))
        #expect(out.contains("unet/diffusion_pytorch_model.fp16.safetensors"))
    }

    @Test func musicgenRepoIncludesDecoderT5AndTokenizers() {
        // MG-DL1 — `jasonvassallo/mlx-musicgen-small` ships the decoder + T5 as two
        // non-standard top-level safetensors (both must download), the split-out
        // `t5_config.json` (new metadata name), the fast tokenizer, and configs. Docs,
        // licences and the SPM-less repo must NOT pull `spiece.model` (none ships).
        let out = selected([
            ".gitattributes",
            "LICENSE",
            "LICENSE.t5-apache-2.0",
            "NOTICE",
            "README.md",
            "config.json",
            "decoder.safetensors",
            "t5.safetensors",
            "t5_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ])
        #expect(out.contains("decoder.safetensors"))
        #expect(out.contains("t5.safetensors"))
        #expect(out.contains("config.json"))
        #expect(out.contains("t5_config.json"))
        #expect(out.contains("tokenizer.json"))
        #expect(out.contains("tokenizer_config.json"))
        // Docs / licence / VCS files excluded.
        #expect(!out.contains("README.md"))
        #expect(!out.contains("LICENSE"))
        #expect(!out.contains(".gitattributes"))
    }

    @Test func musicgenBundlesCompanionEncodecRepo() {
        // MG-DL1 decision — the jasonvassallo repo has no EnCodec weights; the installer
        // bundles the mlx-community EnCodec repo so the engine loads only installed files.
        #expect(ModelFileSelector.companionRepo(for: "jasonvassallo/mlx-musicgen-small")
                == "mlx-community/encodec-32khz-float32")
        #expect(ModelFileSelector.companionRepo(for: "mlx-community/Kokoro-82M-bf16") == nil)
    }

    @Test func wan21RepoDownloadsAllThreeWeightFiles() {
        // WAN-AM3: Wan-AI/Wan2.1-T2V-1.3B ships three top-level weight files — a .safetensors
        // DiT, a .pth VAE, and a .pth T5 encoder — plus tokenizer JSON files under
        // google/umt5-xxl/. All weight files must be selected; assets/ images must be excluded.
        let siblings = [
            ".gitattributes",
            "LICENSE.txt",
            "README.md",
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/special_tokens_map.json",
            "google/umt5-xxl/spiece.model",
            "google/umt5-xxl/tokenizer.json",
            "google/umt5-xxl/tokenizer_config.json",
            "assets/logo.png",
            "assets/video_dit_arch.jpg",
            "examples/i2v_input.JPG",
        ]
        let out = selected(siblings)
        // All three weight files selected.
        #expect(out.contains("diffusion_pytorch_model.safetensors"))
        #expect(out.contains("Wan2.1_VAE.pth"))
        #expect(out.contains("models_t5_umt5-xxl-enc-bf16.pth"))
        // Config and tokenizer files selected.
        #expect(out.contains("config.json"))
        #expect(out.contains("google/umt5-xxl/tokenizer.json"))
        #expect(out.contains("google/umt5-xxl/tokenizer_config.json"))
        #expect(out.contains("google/umt5-xxl/special_tokens_map.json"))
        #expect(out.contains("google/umt5-xxl/spiece.model"))
        // Docs, assets, and examples excluded.
        #expect(!out.contains("README.md"))
        #expect(!out.contains("LICENSE.txt"))
        #expect(!out.contains(".gitattributes"))
        #expect(!out.contains("assets/logo.png"))
        #expect(!out.contains("examples/i2v_input.JPG"))
    }

    @Test func pthExcludedWhenStandardWeightsPresent() {
        // A .pth file must NOT be pulled in when a standard model.safetensors is present —
        // it may be a legacy or test checkpoint, not required at runtime.
        let out = selected(["config.json", "model.safetensors", "old_weights.pth"])
        #expect(out.contains("model.safetensors"))
        #expect(!out.contains("old_weights.pth"))
    }

    @Test func sam3DownloadsWeightsAndProcessorConfig() {
        // SA-AM3: mlx-community/sam3-4bit ships model.safetensors + config.json +
        // processor_config.json + tokenizer files. All are covered by existing
        // metadataNames / standard single-weight paths — no new selectors needed.
        let out = selected([
            "model.safetensors",
            "config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "README.md",
            ".gitattributes",
        ])
        #expect(out.contains("model.safetensors"))
        #expect(out.contains("config.json"))
        #expect(out.contains("processor_config.json"))
        #expect(out.contains("tokenizer.json"))
        #expect(out.contains("tokenizer_config.json"))
        #expect(!out.contains("README.md"))
        #expect(!out.contains(".gitattributes"))
    }
}
