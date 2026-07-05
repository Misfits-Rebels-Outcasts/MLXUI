import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

/// The dots.ocr / dots.mocr vision-language model — a `VLMModel` assembled from the ported
/// `dots_vit` tower + Qwen2 decoder. Registered into `VLMTypeRegistry` (Slice 5) so the standard
/// `VLMEngine`/`MLXLMCommon.generate` path drives it. Flow mirrors mlx-swift-lm's own VLMs
/// (e.g. Qwen25VL): embed tokens, run the vision tower, splice image features into the embeddings
/// at `image_token_id` positions, then run the decoder over the merged embeddings.
///
/// `nonisolated` (app target defaults to MainActor; MLX `Module` is nonisolated).
nonisolated final class DotsOCR: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") var visionTower: DotsVisionModel
    @ModuleInfo(key: "language_model") var languageModel: DotsLanguageModel

    let config: DotsOCRConfig

    var vocabularySize: Int { config.vocabSize }
    var kvHeads: [Int] { languageModel.kvHeads }
    var loraLayers: [Module] { languageModel.model.layers }

    init(_ config: DotsOCRConfig) {
        self.config = config
        self._visionTower.wrappedValue = DotsVisionModel(config.visionConfig)
        self._languageModel.wrappedValue = DotsLanguageModel(config)
        super.init()
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?) -> MLXArray {
        guard let pixelValues, let frames, !frames.isEmpty else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }
        let inputEmbeds = languageModel.model.embedTokens(inputIds)          // (L, D)
        let gridTHW = frames.map { (t: $0.t, h: $0.h, w: $0.w) }
        let imageFeatures = visionTower(pixelValues, gridTHW: gridTHW)       // (numImageTokens, D)
        return Self.merge(inputIds: inputIds, inputEmbeds: inputEmbeds,
                          imageFeatures: imageFeatures, imageTokenId: config.imageTokenId)
    }

    /// Splice `imageFeatures` into `inputEmbeds` at `imageTokenId` positions (batch = 1). Mirrors
    /// `dots_ocr.py`'s `merge_input_ids_with_image_features` (cumulative-count gather + masked select).
    static func merge(
        inputIds: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray, imageTokenId: Int
    ) -> MLXArray {
        let ids = inputIds.reshaped(-1)                              // (L,)
        let L = ids.dim(0)
        let D = inputEmbeds.dim(-1)
        let embeds = inputEmbeds.reshaped(L, D)                      // (L, D)

        let mask = ids .== imageTokenId                             // (L,) bool
        let counts = cumsum(mask.asType(.int32), axis: 0)          // 1-based running count
        let featureIdx = MLX.where(mask, counts - 1, MLXArray(Int32(0)))  // (L,) → row in features
        let gathered = imageFeatures[featureIdx]                    // (L, D)
        let merged = MLX.where(expandedDimensions(mask, axis: -1), gathered, embeds)
        return merged.reshaped(1, L, D)
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let dtype = visionTower.patchEmbed.patchifier.proj.weight.dtype
        var pixels: MLXArray?
        var frames: [THW] = []
        if let p = input.image?.pixels, let f = input.image?.frames {
            pixels = p.asType(dtype)
            frames.append(contentsOf: f)
        }
        let embeddings = inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: pixels,
            frames: frames.isEmpty ? nil : frames)
        let result = languageModel(nil, cache: cache, inputEmbedding: embeddings)
        return .logits(result)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // 1) remap HF checkpoint keys to our module layout; 2) vision conv-weight transpose + drop position_ids.
        var remapped = [String: MLXArray]()
        for (key, value) in weights {
            remapped[DotsOCRWeights.remapKey(key)] = value
        }
        return DotsVisionModel.sanitize(remapped)
    }
}
