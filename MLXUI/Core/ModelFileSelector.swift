import Foundation

/// Decides which files of an HF repo to download for an install. Pure (no I/O) so it can be
/// unit-tested directly (see RSI/evals/eval-plan.md, gate G2). Used by
/// `InstallManager.resolveFiles`.
///
/// Covers these repo shapes:
/// - **standard** single (`model.safetensors`) or **sharded** (`model-*.safetensors` +
///   `model.safetensors.index.json`) weights, plus the usual config/tokenizer files;
/// - **component subfolders** that ship their own weights — Kokoro's `voices/`, Qwen3-TTS's
///   `speech_tokenizer/` — where the subfolder's `*.safetensors` (and any per-component
///   `config.json`/metadata it needs to load) must come down too, else the model fails at
///   run time (e.g. "Speech tokenizer not loaded" when `speech_tokenizer/` is missing);
/// - the **fallback**: when none of the standard weight names are present, take any
///   top-level `*.safetensors` so non-standard weight filenames still install.
enum ModelFileSelector {
    /// Small metadata files always included when present.
    static let metadataNames: Set<String> = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        // Some VLM processors (e.g. DeepSeek-OCR-2's `DeepseekVLV2Processor`) are loaded by
        // MLXVLM's factory from `processor_config.json` rather than `preprocessor_config.json`;
        // without it the load fails with `configurationFileError("processor_config.json", …)`.
        "processor_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        // SentencePiece model — some runners read the raw `.model` directly rather than via a
        // fast tokenizer.json (e.g. mlx-audio's MossTTS-Nano TTS and CohereTranscribe ASR).
        // Also shipped alongside tokenizer.json by many LLM/VLM repos; harmless when unused.
        "tokenizer.model",
        // FireRedASR2's word dictionary — its tokenizer loads `dict.txt` from the model root.
        "dict.txt",
        // Mistral "tekken" tokenizer — Voxtral (and other Mistral repos) ship the whole
        // tokenizer as `tekken.json` instead of tokenizer.json; the runner reads it directly
        // and fails with "tekken.json not found" if it's absent.
        "tekken.json",
        // Chat template — many models (e.g. Qwen3-VL) ship it as a separate file rather than
        // embedding it in tokenizer_config.json. Without it the tokenizer can't apply the chat
        // template → no vision/image placeholders → run fails (journal/2026-38).
        "chat_template.json",
        "chat_template.jinja",
        // FLUX.1-family T5 tokenizer ships as `tokenizer_2/spiece.model` (raw SentencePiece,
        // no fast tokenizer.json). Without it the T5 text encoder can't tokenize → the diffusion
        // pipeline can't build conditioning (AM3, journal/2026-67).
        "spiece.model",
        // SDXL-Turbo's scheduler ships as `scheduler/scheduler_config.json` — a JSON-only
        // component with no `.safetensors` of its own, so the dir wouldn't qualify as a
        // component without this (SD-DL1). The engine parses `alphas_cumprod` from it.
        "scheduler_config.json",
        // MusicGen splits the T5-base text encoder's config out of the main `config.json`
        // (MG-DL1). The engine builds the T5 encoder from it; without it the encoder's
        // dimensions default to the wrong T5 variant.
        "t5_config.json",
    ]

    /// A second HF repo whose files must be bundled into an install, keyed by the catalog
    /// `hfModelId`. Some models ship only part of their weights in their own repo; the rest
    /// lives in a companion repo. Bundling (rather than a runtime fetch) keeps the MLX-first
    /// A2 pattern — the runtime loads only files `InstallManager` already downloaded.
    ///
    /// MG-DL1 decision (backlog step 3): `jasonvassallo/mlx-musicgen-small` ships the decoder
    /// and T5 encoder but **no EnCodec weights**; the 32 kHz / 4-codebook codec lives in its
    /// own mlx-community repo (the same one the mlx-examples Python reference hardcodes).
    /// Its files land under `encodec/` in the installed model dir.
    static func companionRepo(for hfModelId: String) -> String? {
        switch hfModelId {
        case "jasonvassallo/mlx-musicgen-small":
            return "mlx-community/encodec-32khz-float32"
        default:
            return nil
        }
    }

    /// True when `l` (a lowercased `X.safetensors`) has an `X.fp16.safetensors` sibling —
    /// the repo ships the same weights in both fp32 and fp16 (e.g. stabilityai/sdxl-turbo's
    /// `unet/diffusion_pytorch_model[.fp16].safetensors`). The runtime loads the fp16 build
    /// (MLX reads it natively), so the fp32 twin is redundant — downloading it wastes ~15 GB
    /// and the engine would load both into memory. Kept inside the selector so the engine never
    /// has to dedupe weight files (SD-DL1).
    static func isFp16PreferredAway(_ l: String, _ lower: Set<String>) -> Bool {
        guard l.hasSuffix(".safetensors"), !l.contains(".fp16.") else { return false }
        let base = String(l.dropLast(".safetensors".count))
        return lower.contains(base + ".fp16.safetensors")
    }

    /// Returns the subset of `siblings` (repo-relative filenames) to download.
    static func filesToDownload(siblings: [String]) -> [String] {
        let lower = Set(siblings.map { $0.lowercased() })
        let hasIndex = lower.contains("model.safetensors.index.json")
        let hasSingleModel = lower.contains("model.safetensors")
        let hasStandardWeights = hasIndex || hasSingleModel
        // A `model_index.json` marks a diffusers-layout repo (component subdirs carry the
        // weights). SDXL-Turbo also ships redundant root full checkpoints
        // (`sd_xl_turbo_1.0*.safetensors`) that duplicate every component — suppress the
        // top-level fallback so they don't download (SD-DL1). Only repos without
        // `model_index.json` (e.g. Kokoro's `kokoro-v1_0.safetensors`) keep the fallback.
        let hasDiffusersLayout = lower.contains("model_index.json")

        // Subfolders that ship model weights (e.g. `speech_tokenizer/`, `voices/`) OR tokenizer
        // assets (`tokenizer/`, `tokenizer_2/`). A component whose weights don't download fails
        // when the model loads it, so we pull every `*.safetensors` under such a folder plus the
        // metadata it needs (its own `config.json` etc.). A subfolder is a component if it holds
        // a `.safetensors` **or** a recognized metadata file — FLUX's `tokenizer_2/` has only
        // `spiece.model` + tokenizer configs (no weights), and must still come down or the T5
        // encoder can't tokenize (AM3). Keyed by the lowercased directory prefix incl. trailing "/".
        let componentDirs: Set<String> = Set(lower.compactMap { l in
            guard let slash = l.lastIndex(of: "/") else { return nil }
            let basename = String(l[l.index(after: slash)...])
            if l.hasSuffix(".safetensors") || metadataNames.contains(basename) {
                return String(l[...slash])
            }
            return nil
        })

        return siblings.filter { name in
            let l = name.lowercased()

            // Skip docs, scripts, samples, and the `.pt` voice duplicates Kokoro ships.
            if l.hasSuffix(".gitattributes") || l.hasSuffix(".md") || l.hasSuffix(".py")
                || l.hasSuffix(".ipynb") || l.hasSuffix(".onnx") || l.hasSuffix(".pt")
                || l.hasSuffix(".wav") || l.hasPrefix("samples/") {
                return false
            }

            // fp16 twin present → drop the fp32 original (the runtime loads fp16 only).
            if isFp16PreferredAway(l, lower) { return false }

            if metadataNames.contains(l) { return true }

            // Files inside a weight-bearing subfolder: its `*.safetensors`, plus the metadata
            // it needs to load (matched by basename, e.g. `speech_tokenizer/config.json`).
            if let slash = l.lastIndex(of: "/"), componentDirs.contains(String(l[...slash])) {
                if l.hasSuffix(".safetensors") { return true }
                if metadataNames.contains(String(l[l.index(after: slash)...])) { return true }
            }

            // Standard single / sharded weights.
            if hasSingleModel && l == "model.safetensors" { return true }
            if hasIndex && l == "model.safetensors.index.json" { return true }
            if hasIndex && l.hasPrefix("model-") && l.hasSuffix(".safetensors") { return true }

            // Fallback: non-standard top-level weight name (e.g. kokoro-v1_0.safetensors,
            // Wan2.1_VAE.pth). Covers .safetensors and .pth (PyTorch checkpoint — some repos
            // such as Wan-AI/Wan2.1-T2V-1.3B ship weights as .pth; MLX loads them natively).
            // Suppressed for diffusers repos — their components carry the weights and the
            // root files are redundant full checkpoints.
            if !hasStandardWeights && !hasDiffusersLayout && !l.contains("/")
                && (l.hasSuffix(".safetensors") || l.hasSuffix(".pth")) { return true }

            return false
        }
    }
}
