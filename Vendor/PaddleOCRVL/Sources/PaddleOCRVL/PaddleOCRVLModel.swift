import Foundation
import MLX
import MLXNN

public class PaddleOCRVLModel: Module {
    @ModuleInfo(key: "vision_model") public var visionModel: NaViTVisionEncoder
    @ModuleInfo(key: "multi_modal_projector") var projector: MultiModalProjector
    @ModuleInfo(key: "model") public var languageModel: ERNIEModelInner
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    let config: PaddleOCRVLConfig

    public init(config: PaddleOCRVLConfig) {
        self.config = config

        self._visionModel.wrappedValue = NaViTVisionEncoder(config: config.visionConfig)
        self._projector.wrappedValue = MultiModalProjector(config: config)
        self._languageModel.wrappedValue = ERNIEModelInner(config: config.textConfig)
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.vocabSize,
            bias: false
        )

        super.init()
    }

    public func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let patchSize = config.visionConfig.patchSize
        let gridH = pixelValues.dim(1) / patchSize
        let gridW = pixelValues.dim(2) / patchSize
        let visionFeatures = visionModel.getImageFeatures(pixelValues)
        // Projector merges each 2×2 patch block → (gridH/2)·(gridW/2) tokens.
        return projector(visionFeatures, gridH: gridH, gridW: gridW)
    }

    public func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray,
        imageFeatures: MLXArray
    ) -> MLXArray {
        let inputsEmbeds = languageModel.getEmbedding(inputIds)
        let imageTokenMask = inputIds .== config.visionTokenId

        return mergeEmbeddings(
            inputsEmbeds: inputsEmbeds,
            imageFeatures: imageFeatures,
            mask: imageTokenMask
        )
    }

    private func mergeEmbeddings(
        inputsEmbeds: MLXArray,
        imageFeatures: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        let seqLen = inputsEmbeds.dim(1)
        let numImageTokens = imageFeatures.dim(1)

        let maskInt = mask.asType(.int32)
        let firstTrueIdxArray = argMax(maskInt, axis: 1)
        let firstTrueIdx = firstTrueIdxArray[0].item(Int.self)

        let prePad = firstTrueIdx
        let postPad = seqLen - firstTrueIdx - numImageTokens

        var alignedImageFeatures = imageFeatures
        if prePad > 0 || postPad > 0 {
            let paddingWidths: [IntOrPair] = [[0, 0], [prePad, max(0, postPad)], [0, 0]]
            alignedImageFeatures = padded(imageFeatures, widths: paddingWidths)
        }

        if alignedImageFeatures.dim(1) > seqLen {
            alignedImageFeatures = alignedImageFeatures[0..., 0..<seqLen, 0...]
        } else if alignedImageFeatures.dim(1) < seqLen {
            let extraPad = seqLen - alignedImageFeatures.dim(1)
            let extraPadWidths: [IntOrPair] = [[0, 0], [0, extraPad], [0, 0]]
            alignedImageFeatures = padded(alignedImageFeatures, widths: extraPadWidths)
        }

        let expandedMask = mask.expandedDimensions(axis: -1)

        eval(expandedMask)
        eval(alignedImageFeatures)
        eval(inputsEmbeds)

        return MLX.which(expandedMask, alignedImageFeatures, inputsEmbeds)
    }

    public func forward(
        inputIds: MLXArray,
        pixelValues: MLXArray?,
        cache: [KVCache]?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        var inputsEmbeds: MLXArray

        if let pixelValues = pixelValues {
            let imageFeatures = getImageFeatures(pixelValues)
            eval(imageFeatures)
            inputsEmbeds = mergeInputIdsWithImageFeatures(
                inputIds: inputIds,
                imageFeatures: imageFeatures
            )
        } else {
            inputsEmbeds = languageModel.getEmbedding(inputIds)
        }

        let hiddenStates = languageModel.forward(inputsEmbeds, cache: cache, positionIds: positionIds)
        return lmHead(hiddenStates)
    }

    public func forwardGeneration(
        inputIds: MLXArray,
        cache: [KVCache]?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let embeds = languageModel.getEmbedding(inputIds)
        let hiddenStates = languageModel.forward(embeds, cache: cache, positionIds: positionIds)
        return lmHead(hiddenStates)
    }

    public func newCache() -> [KVCache] {
        (0..<config.textConfig.numHiddenLayers).map { _ in KVCache() }
    }
}

extension PaddleOCRVLModel {
    public static func load(from directory: URL) throws -> PaddleOCRVLModel {
        let config = try PaddleOCRVLConfig.load(from: directory)
        let model = PaddleOCRVLModel(config: config)
        try loadWeights(for: model, from: directory, config: config)
        return model
    }

    private static func loadWeights(
        for model: PaddleOCRVLModel, from directory: URL, config: PaddleOCRVLConfig
    ) throws {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        if safetensorFiles.isEmpty {
            throw PaddleOCRVLError.modelLoadFailed("No safetensors files found in \(directory.path)")
        }

        var allWeights: [String: MLXArray] = [:]

        for file in safetensorFiles {
            let weights = try MLX.loadArrays(url: file)
            for (key, value) in weights {
                allWeights[key] = value
            }
        }

        let sanitizedWeights = sanitizeWeights(allWeights)

        // Quantized checkpoints (e.g. mlx-community 4-bit) store `{path}.scales`/`.biases` for
        // their quantized Linear/Embedding layers. A plain `Linear` can't accept those keys
        // (UpdateError.unhandledKeys), so swap each such leaf module to its quantized form
        // BEFORE loading. Driven by which weights actually carry `.scales`, so a checkpoint
        // that quantizes only some layers (e.g. the vision tower) still loads correctly.
        if config.quantization != nil || sanitizedWeights.keys.contains(where: { $0.hasSuffix(".scales") }) {
            let groupSize = config.quantization?.groupSize ?? 64
            let bits = config.quantization?.bits ?? 4
            quantize(model: model, filter: { path, _ in
                sanitizedWeights["\(path).scales"] != nil ? (groupSize: groupSize, bits: bits) : nil
            })
        }

        let parameters = ModuleParameters.unflattened(sanitizedWeights)
        try model.update(parameters: parameters, verify: .noUnusedKeys)

        loadSpecialWeights(for: model, from: allWeights, config: config)
    }

    private static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Position embedding is handled separately (dequantized into a raw table); skip its
            // weight/scales/biases here so they don't reach the module tree.
            if key.hasPrefix("visual.embeddings.position_embedding") { continue }
            if key.contains("rotary_emb.inv_freq") { continue }

            var newKey = key
            var adjustedValue = value

            // Prefix remaps: PaddleOCR-VL 1.5 checkpoint naming → this package's module tree.
            if key.hasPrefix("visual.embeddings.patch_embedding") {
                newKey = newKey.replacingOccurrences(
                    of: "visual.embeddings.patch_embedding", with: "vision_model.patch_embed.proj")
            } else if key.hasPrefix("visual.projector") {
                newKey = newKey.replacingOccurrences(of: "visual.projector", with: "multi_modal_projector")
            } else if key.hasPrefix("visual.") {
                newKey = newKey.replacingOccurrences(of: "visual.", with: "vision_model.")
            } else if key.hasPrefix("language_model.model.") {
                newKey = newKey.replacingOccurrences(of: "language_model.model.", with: "model.")
            } else if key.hasPrefix("language_model.lm_head.") {
                newKey = newKey.replacingOccurrences(of: "language_model.lm_head.", with: "lm_head.")
            }

            // Vision attention output projection: checkpoint `out_proj` → module `proj`.
            // (The language model uses `o_proj`, so this never collides with it.)
            if newKey.contains(".self_attn.out_proj") {
                newKey = newKey.replacingOccurrences(of: ".self_attn.out_proj", with: ".self_attn.proj")
            }

            // Conv patch-embed weight: transpose PyTorch [out,in,kH,kW] → MLX [out,kH,kW,in] only
            // when it isn't already channels-last (this checkpoint ships it channels-last).
            if newKey.contains("patch_embed"), key.contains("weight"), value.ndim == 4 {
                let shape = value.shape
                if shape[1] != shape[2] && shape[2] == shape[3] {
                    adjustedValue = value.transposed(0, 2, 3, 1)
                }
            }

            result[newKey] = adjustedValue
        }

        return result
    }

    private static func loadSpecialWeights(
        for model: PaddleOCRVLModel,
        from weights: [String: MLXArray],
        config: PaddleOCRVLConfig
    ) {
        // Learned vision position table. On 1.5 it's a quantized nn.Embedding
        // (`visual.embeddings.position_embedding.{weight,scales,biases}`) — dequantize to a plain
        // [numPositions, hidden] table the encoder can interpolate. No class token exists.
        let prefix = "visual.embeddings.position_embedding"
        if let posWeight = weights["\(prefix).weight"] {
            if let scales = weights["\(prefix).scales"], let biases = weights["\(prefix).biases"] {
                let groupSize = config.quantization?.groupSize ?? 64
                let bits = config.quantization?.bits ?? 4
                model.visionModel.positionEmbedding = dequantized(
                    posWeight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
            } else {
                model.visionModel.positionEmbedding = posWeight
            }
        } else if let legacy = weights["visual.pos_embed"] {
            model.visionModel.positionEmbedding = legacy
        }
        model.visionModel.classEmbedding = nil
    }
}
