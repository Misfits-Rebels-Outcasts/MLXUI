import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

/// The DeepSeek-OCR-2 vision-language model — a `VLMModel` assembling SAM ViT → Qwen2-0.5B encoder →
/// linear projector (per image tile) → DeepSeek-V2 MoE LM. Ported from `deepseekocr_2.py`'s `Model`.
///
/// **Tile smuggling through `LMInput`:** the processor packs every tile's pixels into one flat 1-D
/// `ProcessedImage.pixels` buffer (local 768² patches first, then the 1024² global view) with a
/// matching `frames: [THW]` describing each tile's `(1, h, w)`. `prepare` walks `frames`, slices the
/// buffer, runs each tile through SAM→Qwen2→projector, concatenates `[locals…, global, view_separator]`,
/// and splices the features into the text embeddings at every `<image>` (image_token_index) position —
/// exactly matching the `num_patches·144 + 256 + 1` token count the processor emits.
/// `nonisolated` (MainActor default).
nonisolated final class DeepSeekOCR: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "sam_model") var samModel: SAMEncoder
    @ModuleInfo(key: "vision_model") var visionModel: DeepSeekOCRVisionModel
    @ModuleInfo(key: "language_model") var languageModel: DeepSeekOCRLanguage
    @ModuleInfo(key: "projector") var projector: DeepSeekOCRProjector
    @ParameterInfo(key: "view_separator") var viewSeparator: MLXArray

    let config: DeepSeekOCRConfig

    var vocabularySize: Int { config.vocabSize }
    var kvHeads: [Int] { languageModel.kvHeads }
    var loraLayers: [Module] { languageModel.model.layers }

    init(_ config: DeepSeekOCRConfig) {
        self.config = config
        self._samModel.wrappedValue = SAMEncoder(config.vision)
        self._visionModel.wrappedValue = DeepSeekOCRVisionModel(config.vision)
        self._languageModel.wrappedValue = DeepSeekOCRLanguage(config)
        self._projector.wrappedValue = DeepSeekOCRProjector(config.projector)
        self._viewSeparator.wrappedValue = MLXArray.zeros([config.projector.nEmbed])
        super.init()
    }

    /// Run every tile (from the flat buffer + `frames`) through SAM→Qwen2→projector and concatenate
    /// `[tile features…, view_separator]`.
    private func visionFeatures(pixels: MLXArray, frames: [THW], dtype: DType) -> MLXArray {
        let flat = pixels.asType(dtype)
        var features: [MLXArray] = []
        var offset = 0
        for frame in frames {
            let (_, h, w) = frame.values
            let count = h * w * 3
            let tile = flat[offset ..< (offset + count)].reshaped(1, h, w, 3)
            offset += count
            let sam = samModel(tile)                 // (1, h/64, w/64, 896)
            let encoded = visionModel(sam)           // (1, numQueries, 896)
            features.append(projector(encoded)[0])   // (numQueries, nEmbed)
        }
        features.append(viewSeparator.expandedDimensions(axis: 0))   // (1, nEmbed)
        return concatenated(features, axis: 0)                        // (total, nEmbed)
    }

    private func inputEmbeddings(inputIds: MLXArray, image: LMInput.ProcessedImage?) -> MLXArray {
        guard let image, let frames = image.frames, !frames.isEmpty else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }
        let dtype = samModel.patchEmbed.proj.weight.dtype
        let inputEmbeds = languageModel.model.embedTokens(inputIds)   // (L, D)
        let features = visionFeatures(pixels: image.pixels, frames: frames, dtype: dtype)
        return Self.merge(inputIds: inputIds, inputEmbeds: inputEmbeds,
                          imageFeatures: features, imageTokenId: config.imageTokenIndex)
    }

    /// Splice `imageFeatures` into `inputEmbeds` at `imageTokenId` positions, in order (batch = 1).
    /// Same cumulative-count gather as the dots.ocr merge.
    static func merge(
        inputIds: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray, imageTokenId: Int
    ) -> MLXArray {
        let ids = inputIds.reshaped(-1)
        let L = ids.dim(0)
        let D = inputEmbeds.dim(-1)
        let embeds = inputEmbeds.reshaped(L, D)
        let mask = ids .== imageTokenId
        let counts = cumsum(mask.asType(.int32), axis: 0)
        let featureIdx = MLX.where(mask, counts - 1, MLXArray(Int32(0)))
        let gathered = imageFeatures[featureIdx]
        let merged = MLX.where(expandedDimensions(mask, axis: -1), gathered, embeds)
        return merged.reshaped(1, L, D)
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let embeddings = inputEmbeddings(inputIds: input.text.tokens, image: input.image)
        let logits = languageModel(nil, cache: cache, inputEmbedding: embeddings)
        return .logits(LMOutput(logits: logits))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // 1) drop position_ids + remap HF keys; 2) join MoE experts for SwitchGLU; 3) SAM conv transpose.
        var remapped = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") { continue }
            remapped[DeepSeekOCRWeights.remapKey(key)] = value
        }
        remapped = DeepSeekOCRLanguage.sanitizeExperts(
            remapped, numLayers: config.numHiddenLayers, numExperts: config.nRoutedExperts)
        return SAMEncoder.sanitize(remapped)
    }
}
