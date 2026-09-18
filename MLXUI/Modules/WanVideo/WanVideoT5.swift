import Foundation
import MLX
import MLXNN

// MARK: - Config

nonisolated struct WanT5Config: Sendable {
    var dModel: Int = 4096
    var dFF: Int = 10240
    var heads: Int = 64
    var vocab: Int = 256_384
    var layers: Int = 24
    var relBuckets: Int = 32
    var relMaxDist: Int = 128
    var headDim: Int { dModel / heads }
}

// MARK: - Encoder

/// UMT5-XXL text encoder for WAN 2.1. Weights from
/// `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/text_encoder/model-*.safetensors`.
/// Relative position bias is shared across all blocks (stored in block 0 in diffusers format).
nonisolated final class WanT5Encoder: Module {
    let c: WanT5Config
    @ModuleInfo(key: "embedding")  var embedding:  Embedding
    @ModuleInfo(key: "blocks")     var blocks:     [WanT5Block]
    @ModuleInfo(key: "finalNorm")  var finalNorm:  WanT5LayerNorm
    @ModuleInfo(key: "relBias")    var relBias:    Embedding   // shared, extracted from block 0

    init(config: WanT5Config = WanT5Config()) {
        self.c = config
        self._embedding.wrappedValue  = Embedding(embeddingCount: config.vocab, dimensions: config.dModel)
        self._blocks.wrappedValue     = (0 ..< config.layers).map { _ in WanT5Block(config: config) }
        self._finalNorm.wrappedValue  = WanT5LayerNorm(dim: config.dModel)
        self._relBias.wrappedValue    = Embedding(embeddingCount: config.relBuckets, dimensions: config.heads)
        super.init()
    }

    // MARK: Sanitize

    /// Remap diffusers UMT5-XXL parameter names to this module's property paths.
    static func sanitize(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, val) in raw {
            guard !key.hasPrefix("decoder.") else { continue }
            if let k = remap(key) { out[k] = val }
        }
        return out
    }

    private static func remap(_ key: String) -> String? {
        switch key {
        case "shared.weight":                   return "embedding.weight"
        case "encoder.final_layer_norm.weight": return "finalNorm.weight"
        default: break
        }
        guard key.hasPrefix("encoder.block.") else { return nil }
        let after = String(key.dropFirst("encoder.block.".count))
        guard let dotIdx = after.firstIndex(of: ".") else { return nil }
        let idx  = String(after[after.startIndex ..< dotIdx])
        let rest = String(after[after.index(after: dotIdx)...])
        switch rest {
        case "layer.0.layer_norm.weight":
            return "blocks.\(idx).norm1.weight"
        case "layer.1.layer_norm.weight":
            return "blocks.\(idx).norm2.weight"
        case "layer.0.SelfAttention.q.weight":
            return "blocks.\(idx).attn.q.weight"
        case "layer.0.SelfAttention.k.weight":
            return "blocks.\(idx).attn.k.weight"
        case "layer.0.SelfAttention.v.weight":
            return "blocks.\(idx).attn.v.weight"
        case "layer.0.SelfAttention.o.weight":
            return "blocks.\(idx).attn.o.weight"
        case "layer.0.SelfAttention.relative_attention_bias.weight":
            // Only block 0 has this in the diffusers checkpoint; hoist it to the encoder.
            return "relBias.weight"
        case "layer.1.DenseReluDense.wi_0.weight":
            return "blocks.\(idx).ffn.wi0.weight"
        case "layer.1.DenseReluDense.wi_1.weight":
            return "blocks.\(idx).ffn.wi1.weight"
        case "layer.1.DenseReluDense.wo.weight":
            return "blocks.\(idx).ffn.wo.weight"
        default:
            return nil
        }
    }

    // MARK: Forward

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var h    = embedding(tokens)
        let bias = relPositionBias(seqLen: tokens.dim(1))
        for block in blocks { h = block(h, posBias: bias) }
        return finalNorm(h)
    }

    // MARK: Relative position bias [1, H, S, S]

    private func relPositionBias(seqLen: Int) -> MLXArray {
        // ctx: [S, 1], mem: [1, S] → relative: [S, S]
        let ctx = MLXArray(Int32(0) ..< Int32(seqLen)).reshaped([seqLen, 1])
        let mem = MLXArray(Int32(0) ..< Int32(seqLen)).reshaped([1, seqLen])
        let rel = mem - ctx
        let bkt = relBucket(rel)
        // relBias(bkt): [S, S, H] → [H, S, S] → [1, H, S, S]
        return relBias(bkt).transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    private func relBucket(_ relPos: MLXArray) -> MLXArray {
        let half    = c.relBuckets / 2
        var buckets = MLX.where(relPos .> 0, MLXArray(Int32(half)), MLXArray(Int32(0))).asType(.int32)
        let absPos  = abs(relPos)
        let maxEx   = half / 2
        let isSmall = absPos .< maxEx
        let log_ratio = log(absPos.asType(.float32) / Float(maxEx) + 1e-8)
            / log(Float(c.relMaxDist) / Float(maxEx))
        let large = MLXArray(Int32(maxEx)) + (log_ratio * Float(half - maxEx)).asType(.int32)
        let largeClamped = minimum(large, MLXArray(Int32(half - 1)))
        buckets = buckets + MLX.where(isSmall, absPos.asType(.int32), largeClamped)
        return buckets
    }
}

// MARK: - Modules

nonisolated final class WanT5LayerNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    init(dim: Int = 4096) {
        self._weight.wrappedValue = MLXArray.ones([dim])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ms  = pow(x.asType(.float32), 2).mean(axis: -1, keepDims: true)
        return weight.asType(x.dtype) * (x * rsqrt(ms + 1e-6).asType(x.dtype))
    }
}

nonisolated final class WanT5Block: Module {
    @ModuleInfo(key: "norm1") var norm1: WanT5LayerNorm
    @ModuleInfo(key: "attn")  var attn:  WanT5SelfAttn
    @ModuleInfo(key: "norm2") var norm2: WanT5LayerNorm
    @ModuleInfo(key: "ffn")   var ffn:   WanT5FFN
    init(config: WanT5Config) {
        self._norm1.wrappedValue = WanT5LayerNorm(dim: config.dModel)
        self._attn.wrappedValue  = WanT5SelfAttn(config: config)
        self._norm2.wrappedValue = WanT5LayerNorm(dim: config.dModel)
        self._ffn.wrappedValue   = WanT5FFN(config: config)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, posBias: MLXArray) -> MLXArray {
        let h = x + attn(norm1(x), posBias: posBias)
        return h + ffn(norm2(h))
    }
}

nonisolated final class WanT5SelfAttn: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    let nH, hD, D: Int

    init(config: WanT5Config) {
        nH = config.heads; hD = config.dModel / config.heads; D = config.dModel
        self._q.wrappedValue = Linear(D, D, bias: false)
        self._k.wrappedValue = Linear(D, D, bias: false)
        self._v.wrappedValue = Linear(D, D, bias: false)
        self._o.wrappedValue = Linear(D, D, bias: false)
        super.init()
    }

    private func split(_ x: MLXArray) -> MLXArray {
        x.reshaped([x.dim(0), -1, nH, hD]).transposed(0, 2, 1, 3)
    }
    private func merge(_ x: MLXArray) -> MLXArray {
        x.transposed(0, 2, 1, 3).reshaped([x.dim(0), -1, D])
    }

    func callAsFunction(_ x: MLXArray, posBias: MLXArray) -> MLXArray {
        let scores = matmul(split(q(x)), split(k(x)).transposed(0, 1, 3, 2)) + posBias
        let w = softmax(scores.asType(.float32), axis: -1).asType(x.dtype)
        return o(merge(matmul(w, split(v(x)))))
    }
}

nonisolated final class WanT5FFN: Module {
    @ModuleInfo(key: "wi0") var wi0: Linear
    @ModuleInfo(key: "wi1") var wi1: Linear
    @ModuleInfo(key: "wo")  var wo:  Linear
    init(config: WanT5Config) {
        self._wi0.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wi1.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wo.wrappedValue  = Linear(config.dFF, config.dModel, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // GeGLU: gelu(wi0(x)) * wi1(x)
        let gate = wi0(x)
        let gelu = gate * (0.5 * (1.0 + tanh(0.7978845608 * (gate + 0.044715 * pow(gate, 3)))))
        return wo(gelu * wi1(x))
    }
}
