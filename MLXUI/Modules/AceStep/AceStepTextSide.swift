import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXNN

// MC5-3d: everything from (caption, lyrics, language, duration) to the two tensors the condition encoder consumes.
// Upstream is `ace-step/ACE-Step-1.5` @ ca1e85f: `constants.SFT_GEN_PROMPT` / `DEFAULT_DIT_INSTRUCTION`,
// `handler/metadata_utils._dict_to_meta_string`, `handler/prompt_utils._format_lyrics`. The golden is `Fixtures/MC5/mc5_tokens.json`.

// MARK: - Prompt strings

nonisolated enum AceStepPrompt {
    static let instruction = "Fill the audio semantic mask based on the given conditions:"
    /// The lyric language upstream uses for an instrumental song.
    static let instrumentalLanguage = "unknown"
    static let instrumentalLyrics = "[Instrumental]"
    /// Qwen3 `<|endoftext|>`.
    static let endOfText = 151_643
    static let maxTextTokens = 256
    static let maxLyricTokens = 2048

    /// `SFT_GEN_PROMPT.format(instruction, caption, metas)`. `metas` has no bpm, time signature or key (N/A) and the duration as
    /// `int(duration)` seconds: truncation toward zero, like Python's `int()`.
    static func textString(caption: String, duration: Double) -> String {
        let metas = "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: \(Int(duration)) seconds\n"
        return "# Instruction\n\(instruction)\n\n# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }

    /// `_format_lyrics(lyrics, language)`. An instrumental song still takes the lyric path, with `[Instrumental]` and `unknown` (T18).
    static func lyricString(lyrics: String, language: String, instrumental: Bool) -> String {
        let body = instrumental ? instrumentalLyrics : lyrics
        let lang = instrumental ? instrumentalLanguage : language
        return "# Languages\n\(lang)\n\n# Lyric\n\(body)<|endoftext|>"
    }
}

// MARK: - Tokenizer

/// The Qwen3-Embedding-0.6B tokenizer (`tokenizer.json`), loaded from the installed folder through the project's loader.
nonisolated struct AceStepTokenizer: @unchecked Sendable {
    private let tokenizer: any MLXLMCommon.Tokenizer
    /// Whether the loaded tokenizer's post-processor already appends `<|endoftext|>` (probed once at load, see `encode`).
    let postProcessorAppendsEndOfText: Bool

    static let folderName = "Qwen3-Embedding-0.6B"

    /// `directory` is the installed model folder (the one holding `Qwen3-Embedding-0.6B/`). Local files only.
    init(directory: URL) async throws {
        let folder = directory.appendingPathComponent(Self.folderName)
        guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path) else {
            throw AceStepWeightsError.missingFile("\(Self.folderName)/tokenizer.json")
        }
        let loaded = try await HFTokenizerLoader().load(from: folder)
        tokenizer = loaded
        let plain = loaded.encode(text: "a", addSpecialTokens: false)
        let special = loaded.encode(text: "a", addSpecialTokens: true)
        postProcessorAppendsEndOfText = special.count == plain.count + 1 && special.last == AceStepPrompt.endOfText
    }

    /// HF `AutoTokenizer(text, truncation=True, max_length)` for batch 1 with no padding.
    ///
    /// Readme T25: Qwen3-Embedding's `tokenizer.json` post-processor (`TemplateProcessing`) appends one `<|endoftext|>` (151643)
    /// after the template's own literal one, on both the text and the lyric sequence. swift-transformers applies that
    /// post-processor too (probed at load: `postProcessorAppendsEndOfText`); if a tokenizer ever did not, this wrapper appends it,
    /// so the result is the same either way and never doubled.
    ///
    /// Truncation follows HF: it reserves room for the special token, so a prompt longer than `maxLength` is cut to its first
    /// `maxLength - 1` content ids and the appended `<|endoftext|>` is the last id (the golden's long text ends `[.., 448, 9450, 151643]`
    /// with exactly 256 ids). This corrects the readme's "the EOS is cut with the rest".
    func encode(_ string: String, maxLength: Int) -> [Int] {
        var ids = tokenizer.encode(text: string, addSpecialTokens: true)
        if !postProcessorAppendsEndOfText { ids.append(AceStepPrompt.endOfText) }
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength - 1)) + [AceStepPrompt.endOfText]
        }
        return ids
    }

    func encodeText(caption: String, duration: Double) -> [Int] {
        encode(AceStepPrompt.textString(caption: caption, duration: duration), maxLength: AceStepPrompt.maxTextTokens)
    }

    func encodeLyrics(lyrics: String, language: String, instrumental: Bool) -> [Int] {
        encode(AceStepPrompt.lyricString(lyrics: lyrics, language: language, instrumental: instrumental),
               maxLength: AceStepPrompt.maxLyricTokens)
    }
}

// MARK: - Text encoder (Qwen3-Embedding-0.6B)

/// What the condition encoder (MC5-3e) consumes from the text side.
nonisolated struct AceStepTextInputs {
    var textIds: [Int]
    /// All ones (batch 1, no padding).
    var textMask: MLXArray
    var lyricIds: [Int]
    var lyricMask: MLXArray
    /// Last hidden state of the Qwen3-Embedding trunk after its final norm, `[1, Lt, 1024]`, bf16.
    var textHidden: MLXArray
    /// `embed_tokens.weight[lyricIds]`, `[1, Ll, 1024]`, bf16 (the lyric path uses only the embedding table, T18).
    var lyricEmbeds: MLXArray
}

/// Qwen3-Embedding-0.6B built through `MLXEmbedders` (`EmbedderTypeRegistry` `qwen3` -> `Qwen3Model`) from the installed folder.
///
/// Offline by construction: nothing here resolves a repo id (no `loadContainer`, no Hub). The config is read from
/// `Qwen3-Embedding-0.6B/config.json`, the model is built from the registry, and the weights come from the 3c reader, so a
/// missing folder or file throws `AceStepWeightsError.missingFile` instead of reaching for the network.
///
/// **One copy of the embedding table.** The reader returns `embed_tokens.weight` once; the same `MLXArray` is installed in the
/// model's `Embedding` and kept for the lyric gather, so the lyric table (0.311 GB) is not loaded a second time.
nonisolated final class AceStepTextEncoder: @unchecked Sendable {
    private let model: any EmbeddingModel
    /// `[151669, 1024]` bf16, shared with the model's own embedding.
    let embedTable: MLXArray

    private init(model: any EmbeddingModel, embedTable: MLXArray) {
        self.model = model
        self.embedTable = embedTable
    }

    static func load(directory: URL) async throws -> AceStepTextEncoder {
        let configURL = directory.appendingPathComponent(AceStepTokenizer.folderName).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw AceStepWeightsError.missingFile("\(AceStepTokenizer.folderName)/config.json")
        }
        let configData = try Data(contentsOf: configURL)
        let model = try await EmbedderTypeRegistry.shared.createModel(configuration: configData, modelType: "qwen3")
        // Every tensor of the file, once (embed_tokens.weight included); bf16 kept. The raw dictionary and the sanitised one are scoped so neither outlives the model update
        // (MC5-T-mem2: nothing holds both across the build, and neither is alive when the caller returns).
        let table: MLXArray
        do {
            let raw = try AceStepWeights.readQwen(directory: directory, prefixes: [""])
            guard let t = raw[AceStepWeights.qwenEmbedTokensName] else {
                throw AceStepWeightsError.missingTensor(AceStepWeights.qwenEmbedTokensName)
            }
            table = t
            let weights = model.sanitize(weights: raw)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        }
        eval(model)
        return AceStepTextEncoder(model: model, embedTable: table)
    }

    /// The trunk's last hidden state for `ids`: causal attention, no mask (batch 1), post-final-norm, no pooling. `[1, L, 1024]`.
    func hiddenStates(ids: [Int]) throws -> MLXArray {
        let input = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let output = model(input, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
        guard let hidden = output.hiddenStates else { throw AceStepWeightsError.missingTensor("hiddenStates") }
        return hidden
    }

    /// `embed_tokens.weight[ids]` as `[1, L, 1024]`: a gather over the table, exact (no compute).
    func lyricEmbeddings(ids: [Int]) -> MLXArray {
        let rows = take(embedTable, MLXArray(ids.map { Int32($0) }), axis: 0)
        return rows.expandedDimensions(axis: 0)
    }

    func inputs(textIds: [Int], lyricIds: [Int]) throws -> AceStepTextInputs {
        let hidden = try hiddenStates(ids: textIds)
        let lyric = lyricEmbeddings(ids: lyricIds)
        let out = AceStepTextInputs(
            textIds: textIds, textMask: MLXArray.ones([1, textIds.count], dtype: .int32),
            lyricIds: lyricIds, lyricMask: MLXArray.ones([1, lyricIds.count], dtype: .int32),
            textHidden: hidden, lyricEmbeds: lyric)
        eval(hidden, lyric)
        return out
    }
}

// MARK: - text_projector

nonisolated enum AceStepTextProjector {
    static let weightName = "encoder.text_projector.weight"

    /// Bias-free `Linear(1024 -> 2048)`: `x @ W.T` with `W` `[2048, 1024]`, `x` `[1, L, 1024]` -> `[1, L, 2048]`. The dtype follows the
    /// inputs (bf16 in, bf16 out), as a `Linear` does.
    static func project(_ hidden: MLXArray, weight: MLXArray) -> MLXArray {
        matmul(hidden, weight.transposed())
    }
}
