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
    ]

    /// Returns the subset of `siblings` (repo-relative filenames) to download.
    static func filesToDownload(siblings: [String]) -> [String] {
        let lower = Set(siblings.map { $0.lowercased() })
        let hasIndex = lower.contains("model.safetensors.index.json")
        let hasSingleModel = lower.contains("model.safetensors")
        let hasStandardWeights = hasIndex || hasSingleModel

        // Subfolders that ship model weights (e.g. `speech_tokenizer/`, `voices/`). A
        // component whose weights don't download fails when the model loads it, so we pull
        // every `*.safetensors` under such a folder plus the metadata it needs (its own
        // `config.json` etc.). Keyed by the lowercased directory prefix incl. trailing "/".
        let componentDirs: Set<String> = Set(lower.compactMap { l in
            guard l.hasSuffix(".safetensors"), let slash = l.lastIndex(of: "/") else { return nil }
            return String(l[...slash])
        })

        return siblings.filter { name in
            let l = name.lowercased()

            // Skip docs, scripts, samples, and the `.pt` voice duplicates Kokoro ships.
            if l.hasSuffix(".gitattributes") || l.hasSuffix(".md") || l.hasSuffix(".py")
                || l.hasSuffix(".ipynb") || l.hasSuffix(".onnx") || l.hasSuffix(".pt")
                || l.hasSuffix(".wav") || l.hasPrefix("samples/") {
                return false
            }

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

            // Fallback: non-standard top-level weight name (e.g. kokoro-v1_0.safetensors).
            if !hasStandardWeights && !l.contains("/") && l.hasSuffix(".safetensors") { return true }

            return false
        }
    }
}
