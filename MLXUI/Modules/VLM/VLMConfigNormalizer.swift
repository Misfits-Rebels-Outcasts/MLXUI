import Foundation

/// Some `mlx-community` VLM checkpoints ship a *verbose* HuggingFace `config.json` that
/// omits special-token ids the stock `MLXVLM` config decoders require as **non-optional**.
///
/// olmOCR-2-7B-1025 (fine-tuned from Qwen2.5-VL) is one: its config carries only
/// `vision_token_id`, so `VLMModelFactory` fails to load it with
/// `configurationDecodingError("config.json", …, keyNotFound "image_token_id")`.
/// Because those ids come from the shared Qwen2.5-VL `<|…|>` vocab, they're identical
/// across every Qwen2.5-VL checkpoint — so we can safely fill in any that are missing and
/// let the stock factory decode + run the model unchanged.
///
/// Pure Foundation, no MLX. Only ever *adds* missing keys — never overwrites values the
/// checkpoint provides — so a checkpoint that already has the ids is left untouched.
/// `nonisolated` throughout so `VLMEngine`'s nonisolated load path can call it directly.
enum VLMConfigNormalizer {
    /// Canonical Qwen2.5-VL special-token ids (the `<|vision_start|>`/`<|vision_end|>`/
    /// `<|vision_pad|>`/`<|image_pad|>`/`<|video_pad|>` vocab, identical across every
    /// Qwen2.5-VL checkpoint incl. olmOCR-2). Keyed by their `config.json` field name.
    nonisolated static let qwen25VLTokenDefaults: [String: Int] = [
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "vision_token_id": 151654,
        "image_token_id": 151655,
        "video_token_id": 151656,
    ]

    /// Pure form: given a decoded `config.json` object, return a patched copy with any
    /// missing Qwen2.5-VL token ids filled in — or `nil` when nothing needs changing, so
    /// the caller can skip the rewrite. Only acts on `model_type == "qwen2_5_vl"`; a key
    /// that's absent *or* JSON `null` counts as missing.
    nonisolated static func patchedConfig(_ config: [String: Any]) -> [String: Any]? {
        guard config["model_type"] as? String == "qwen2_5_vl" else { return nil }
        var patched = config
        var changed = false
        for (key, value) in qwen25VLTokenDefaults where patched[key] == nil || patched[key] is NSNull {
            patched[key] = value
            changed = true
        }
        return changed ? patched : nil
    }

    /// Patch the installed `config.json` in place when it's a Qwen2.5-VL checkpoint missing
    /// token ids. Idempotent (a second call finds nothing to change) and best-effort: any
    /// I/O or parse failure is ignored so the stock factory surfaces its own error. The
    /// file lives in our own Application Support container, so the write is sandbox-safe.
    nonisolated static func normalizeConfigIfNeeded(at modelDir: URL) {
        let url = modelDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let patched = patchedConfig(config),
              let out = try? JSONSerialization.data(
                withJSONObject: patched, options: [.prettyPrinted, .sortedKeys])
        else { return }
        try? out.write(to: url, options: .atomic)
    }
}
