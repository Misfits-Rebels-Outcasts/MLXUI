import Foundation
import MLX
import MLXNN

/// The strict checkpoint reader: reads `config.json` + `model.safetensors`, converts the PyTorch layouts to the
/// layouts this port's modules expect, and installs the tensors into a module tree with no missing, unused or
/// wrongly-shaped tensor. Trap 3: a silently skipped CLS/register/mask token or layer-scale weight would load
/// with a plausible picture and the wrong numbers.
nonisolated enum DepthAnythingWeights {

    static func loadConfig(directory: URL) throws -> DepthAnythingConfig {
        let url = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DepthAnythingError.missingFile("config.json")
        }
        let config: DepthAnythingConfig
        do {
            config = try JSONDecoder().decode(DepthAnythingConfig.self, from: Data(contentsOf: url))
        } catch {
            throw DepthAnythingError.unsupportedConfig("config.json could not be decoded: \(error)")
        }
        try config.validate()
        return config
    }

    static func loadArrays(directory: URL) throws -> [String: MLXArray] {
        let url = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DepthAnythingError.missingFile("model.safetensors")
        }
        do {
            return try MLX.loadArrays(url: url)
        } catch {
            throw DepthAnythingError.badWeights("model.safetensors could not be read: \(error)")
        }
    }

    /// Install every tensor under `prefix` into `module`, strictly.
    ///
    /// Every parameter the module has must be set and every tensor in the file under `prefix` must be used.
    /// The first offender is named with its published safetensors key, so a port that forgot a token fails
    /// with "missing tensor 'backbone.embeddings.clsToken'" rather than a generic shape error.
    @discardableResult
    static func install(weights: [String: MLXArray], prefix: String, into module: Module,
                        config: DepthAnythingConfig) throws -> Int {
        let expected = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.shape) })
        var converted: [String: MLXArray] = [:]
        for (name, array) in weights where name.hasPrefix(prefix) {
            let local = String(name.dropFirst(prefix.count))
            let key = DepthAnythingOps.moduleKey(local)
            guard expected[key] != nil else {
                throw DepthAnythingError.badWeights("unused tensor '\(name)'")
            }
            converted[key] = DepthAnythingOps.convertedWeight(name: name, array: array, config: config)
        }
        if let missing = expected.keys.sorted().first(where: { converted[$0] == nil }) {
            throw DepthAnythingError.badWeights("missing tensor '\(prefix)\(missing)'")
        }
        for key in expected.keys.sorted() {
            guard let loaded = converted[key] else { continue }
            guard loaded.shape == expected[key] else {
                throw DepthAnythingError.badWeights(
                    "tensor '\(prefix)\(key)' has shape \(loaded.shape), expected \(expected[key] ?? [])")
            }
        }
        try module.update(parameters: ModuleParameters.unflattened(converted), verify: .all)
        try MLXErrorGuard.evaluate(module.parameters().flattened().map(\.1))
        return converted.count
    }
}

// MARK: - Embeddings

nonisolated final class DINOv2PatchEmbeddings: Module {
    let projection: Conv2d

    init(config: DepthAnythingConfig) {
        let backbone = config.backboneConfig
        self.projection = Conv2d(inputChannels: backbone.numChannels,
                                 outputChannels: backbone.hiddenSize,
                                 kernelSize: IntOrPair((backbone.patchSize, backbone.patchSize)),
                                 stride: IntOrPair((backbone.patchSize, backbone.patchSize)),
                                 padding: IntOrPair((0, 0)),
                                 bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(x) }
}

/// CLS token, mask token, patch embeddings and the interpolated position embeddings (Trap 4).
nonisolated final class DINOv2Embeddings: Module {
    let clsToken: MLXArray
    let maskToken: MLXArray
    let positionEmbeddings: MLXArray
    let patchEmbeddings: DINOv2PatchEmbeddings

    private let patchSize: Int
    private let pretrainedGrid: Int

    init(config: DepthAnythingConfig) {
        let backbone = config.backboneConfig
        self.clsToken = MLXArray.zeros([1, 1, backbone.hiddenSize])
        self.maskToken = MLXArray.zeros([1, backbone.hiddenSize])
        let grid = backbone.imageSize / backbone.patchSize
        self.positionEmbeddings = MLXArray.zeros([1, grid * grid + 1, backbone.hiddenSize])
        self.patchEmbeddings = DINOv2PatchEmbeddings(config: config)
        self.patchSize = backbone.patchSize
        self.pretrainedGrid = grid
        super.init()
    }

    /// The reference's `interpolate_pos_encoding`: bicubic over the patch grid, `align_corners=False`, CLS kept.
    func interpolatedPositionEmbeddings(patchHeight: Int, patchWidth: Int) -> MLXArray {
        if patchHeight == pretrainedGrid && patchWidth == pretrainedGrid { return positionEmbeddings }
        let classPosition = positionEmbeddings[0..., 0 ..< 1, 0...]
        let patchPosition = positionEmbeddings[0..., 1..., 0...]
        let grid = patchPosition.reshaped(1, pretrainedGrid, pretrainedGrid, -1)
        let resized = DepthAnythingOps.resizeBicubic(grid, outH: patchHeight, outW: patchWidth, alignCorners: false)
        let flat = resized.reshaped(1, patchHeight * patchWidth, -1)
        return concatenated([classPosition, flat], axis: 1)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let height = pixelValues.dim(1)
        let width = pixelValues.dim(2)
        let patchHeight = height / patchSize
        let patchWidth = width / patchSize
        let patches = patchEmbeddings(pixelValues).reshaped(1, patchHeight * patchWidth, -1)
        let classToken = broadcast(clsToken, to: [1, 1, clsToken.dim(2)])
        var embeddings = concatenated([classToken, patches], axis: 1)
        embeddings = embeddings + interpolatedPositionEmbeddings(patchHeight: patchHeight, patchWidth: patchWidth)
        return embeddings
    }
}

// MARK: - Encoder

nonisolated final class DINOv2SelfAttention: Module {
    let query: Linear
    let key: Linear
    let value: Linear
    private let numHeads: Int
    private let headDim: Int
    private let scale: Float

    init(config: DepthAnythingConfig) {
        let backbone = config.backboneConfig
        self.numHeads = backbone.numAttentionHeads
        self.headDim = backbone.hiddenSize / backbone.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self.query = Linear(backbone.hiddenSize, backbone.hiddenSize, bias: backbone.qkvBias)
        self.key = Linear(backbone.hiddenSize, backbone.hiddenSize, bias: backbone.qkvBias)
        self.value = Linear(backbone.hiddenSize, backbone.hiddenSize, bias: backbone.qkvBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        func heads(_ tensor: MLXArray) -> MLXArray {
            tensor.reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        }
        let q = heads(query(x))
        let k = heads(key(x))
        let v = heads(value(x))
        let scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let probabilities = softmax(scores, axis: -1)
        let context = matmul(probabilities, v)
        return context.transposed(0, 2, 1, 3).reshaped(batch, sequence, numHeads * headDim)
    }
}

nonisolated final class DINOv2SelfOutput: Module {
    let dense: Linear

    init(config: DepthAnythingConfig) {
        self.dense = Linear(config.backboneConfig.hiddenSize, config.backboneConfig.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

nonisolated final class DINOv2Attention: Module {
    let attention: DINOv2SelfAttention
    let output: DINOv2SelfOutput

    init(config: DepthAnythingConfig) {
        self.attention = DINOv2SelfAttention(config: config)
        self.output = DINOv2SelfOutput(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { output(attention(x)) }
}

nonisolated final class DINOv2LayerScale: Module {
    let lambda1: MLXArray

    init(config: DepthAnythingConfig) {
        self.lambda1 = MLXArray.full([config.backboneConfig.hiddenSize],
                                     values: MLXArray(config.backboneConfig.layerscaleValue))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x * lambda1 }
}

nonisolated final class DINOv2MLP: Module {
    let fc1: Linear
    let fc2: Linear

    init(config: DepthAnythingConfig) {
        let backbone = config.backboneConfig
        let hidden = Int(Float(backbone.hiddenSize) * Float(backbone.mlpRatio))
        self.fc1 = Linear(backbone.hiddenSize, hidden)
        self.fc2 = Linear(hidden, backbone.hiddenSize)
        super.init()
    }

    /// The reference uses `ACT2FN["gelu"]`, the exact erf form (not the tanh approximation).
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

nonisolated final class DINOv2Layer: Module {
    let norm1: LayerNorm
    let attention: DINOv2Attention
    let layerScale1: DINOv2LayerScale
    let norm2: LayerNorm
    let mlp: DINOv2MLP
    let layerScale2: DINOv2LayerScale

    init(config: DepthAnythingConfig) {
        let backbone = config.backboneConfig
        self.norm1 = LayerNorm(dimensions: backbone.hiddenSize, eps: backbone.layerNormEps)
        self.attention = DINOv2Attention(config: config)
        self.layerScale1 = DINOv2LayerScale(config: config)
        self.norm2 = LayerNorm(dimensions: backbone.hiddenSize, eps: backbone.layerNormEps)
        self.mlp = DINOv2MLP(config: config)
        self.layerScale2 = DINOv2LayerScale(config: config)
        super.init()
    }

    func callAsFunction(_ hiddenState: MLXArray) -> MLXArray {
        let attentionOutput = layerScale1(attention(norm1(hiddenState)))
        let hidden = hiddenState + attentionOutput
        let mlpOutput = layerScale2(mlp(norm2(hidden)))
        return hidden + mlpOutput
    }
}

nonisolated final class DINOv2Encoder: Module {
    let layer: [DINOv2Layer]

    init(config: DepthAnythingConfig) {
        self.layer = (0 ..< config.backboneConfig.numHiddenLayers).map { _ in DINOv2Layer(config: config) }
        super.init()
    }
}

/// DINOv2 ViT-S/14 with `reshape_hidden_states: false`: the four `out_indices` feature maps come back as
/// `[1, patchCount + 1, 384]` tokens (CLS first), each after the backbone's final LayerNorm, exactly as
/// `transformers`' `Dinov2Backbone` hands them to the neck.
nonisolated final class DINOv2Backbone: Module {
    let embeddings: DINOv2Embeddings
    let encoder: DINOv2Encoder
    let layernorm: LayerNorm

    private let outIndices: [Int]

    init(config: DepthAnythingConfig) {
        let backbone = config.backboneConfig
        self.embeddings = DINOv2Embeddings(config: config)
        self.encoder = DINOv2Encoder(config: config)
        self.layernorm = LayerNorm(dimensions: backbone.hiddenSize, eps: backbone.layerNormEps)
        self.outIndices = backbone.outIndices
        super.init()
    }

    /// `pixelValues`: `[1, H, W, 3]` float32, already preprocessed (normalised, NCHW transposed to NHWC).
    /// Returns the four feature maps in `out_indices` order, each `[1, patchHeight * patchWidth + 1, 384]`.
    func callAsFunction(_ pixelValues: MLXArray) -> [MLXArray] {
        var hidden = embeddings(pixelValues)
        var features: [MLXArray] = []
        for index in 0 ..< encoder.layer.count {
            hidden = encoder.layer[index](hidden)
            if outIndices.contains(index + 1) {
                features.append(layernorm(hidden))
            }
        }
        return features
    }

    /// Strict load of the `backbone.` tensors from an installed folder.
    static func load(directory: URL, config: DepthAnythingConfig) throws -> DINOv2Backbone {
        let weights = try DepthAnythingWeights.loadArrays(directory: directory)
        let backbone = DINOv2Backbone(config: config)
        try DepthAnythingWeights.install(weights: weights, prefix: "backbone.", into: backbone, config: config)
        return backbone
    }

    /// Strict load of the `backbone.` tensors from an already-read checkpoint (the full-model loader reuses this).
    static func load(weights: [String: MLXArray], config: DepthAnythingConfig) throws -> DINOv2Backbone {
        let backbone = DINOv2Backbone(config: config)
        try DepthAnythingWeights.install(weights: weights, prefix: "backbone.", into: backbone, config: config)
        return backbone
    }
}
