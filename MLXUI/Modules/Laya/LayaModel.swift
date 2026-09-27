import Foundation
import MLX
import MLXFast
import MLXNN

/// Laya's decision head, ported from `laya_mlx/model.py`: `HeadAttention`, `HeadLayer`,
/// `DecisionHead`, `DecisionModel`, `sanitize_weights`. Runs on top of the shared
/// `ModernBERTEncoder` (Gate C, `Core/ModernBERTEncoder.swift`) with sliding-window attention
/// turned on.

/// `model.py::HeadAttention` — standard (non-RoPE) multi-head self-attention over the decision
/// head's hidden state, masked by the same boolean `attention_mask` every layer (no sliding
/// window inside the head — only the encoder has one).
nonisolated private final class LayaHeadAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dims: Int) {
        numHeads = max(1, dims / 64)
        headDim = dims / numHeads
        scale = pow(Float(headDim), -0.5)
        _inProj.wrappedValue = Linear(dims, 3 * dims)
        _outProj.wrappedValue = Linear(dims, dims)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (b, length) = (x.dim(0), x.dim(1))
        let qkv = inProj(x).reshaped(b, length, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        // See the matching comment in `Core/ModernBERTEncoder.swift`'s `ModernBERTAttention` —
        // a bare-Int subscript here is a 0-dim-index `Gather` that Metal API Validation aborts
        // on for real, confirmed live via a debugger-attached crash (`axes_ = [0]` at the
        // failing `Gather::eval_gpu` frame). Range-slice + squeeze avoids `Gather` entirely.
        let q = qkv[0 ..< 1].squeezed(axis: 0)
        let k = qkv[1 ..< 2].squeezed(axis: 0)
        let v = qkv[2 ..< 3].squeezed(axis: 0)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return outProj(out.transposed(0, 2, 1, 3).reshaped(b, length, numHeads * headDim))
    }
}

/// `model.py::HeadLayer` — pre-norm self-attention + a **ReLU** FFN of width `4 * dims`: the
/// PyTorch `TransformerEncoderLayer` default, even though the encoder and scorer use GELU
/// (flagged in the Python source; carried over verbatim, not "fixed").
nonisolated private final class LayaHeadLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: LayaHeadAttention
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dims: Int) {
        _selfAttn.wrappedValue = LayaHeadAttention(dims: dims)
        _norm1.wrappedValue = LayerNorm(dimensions: dims)
        _norm2.wrappedValue = LayerNorm(dimensions: dims)
        _linear1.wrappedValue = Linear(dims, 4 * dims)
        _linear2.wrappedValue = Linear(4 * dims, dims)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let h = x + selfAttn(norm1(x), mask: mask)
        return h + linear2(relu(linear1(norm2(h))))
    }
}

/// `model.py::DecisionHead` — a stack of `head_layers` (from `rl_agent_config.json`, default 2).
nonisolated private final class LayaDecisionHead: Module {
    fileprivate let layers: [LayaHeadLayer]

    init(dims: Int, count: Int) {
        layers = (0 ..< count).map { _ in LayaHeadLayer(dims: dims) }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h, mask: mask) }
        return h
    }
}

/// `model.py::DecisionModel`. `predict(...)` mirrors its `__call__`: encoder → per-question-type
/// embedding → decision head → gather at marker positions → scorer → masked logits; the action
/// head runs too (its weights load strictly) but its output isn't used (§0 of
/// `RSI/DelegateLayaBacklog.md`: "no signal" — `LayaEngine` never reads it).
nonisolated final class LayaDecisionModel: Module {
    @ModuleInfo(key: "encoder") fileprivate var encoder: ModernBERTEncoder
    @ModuleInfo(key: "head") fileprivate var head: LayaDecisionHead
    @ModuleInfo(key: "type_emb") var typeEmb: Embedding
    @ModuleInfo(key: "scorer") var scorer: Sequential
    @ModuleInfo(key: "act_head") var actHead: Sequential
    @ParameterInfo(key: "temperature") var temperature: MLXArray

    init(encoderConfig: ModernBERTEncoderConfig, headLayers: Int, actCostCount: Int) {
        let dims = encoderConfig.hiddenSize
        _encoder.wrappedValue = ModernBERTEncoder(encoderConfig, slidingWindowEnabled: true)
        _head.wrappedValue = LayaDecisionHead(dims: dims, count: headLayers)
        _typeEmb.wrappedValue = Embedding(embeddingCount: 3, dimensions: dims)
        _scorer.wrappedValue = Sequential(layers: [
            LayerNorm(dimensions: dims), Linear(dims, dims), GELU(), Linear(dims, 1),
        ])
        _actHead.wrappedValue = Sequential(layers: [
            Linear(dims + 4, 256), GELU(), Linear(256, actCostCount + 1),
        ])
        // Checkpoint buffer; calibration uses the JSON config (rl_agent_config.json), not this.
        _temperature.wrappedValue = MLXArray.ones([3])
    }

    /// Returns `(logits, action)`. `logits` is `[B, maxOptions]` fp32, masked to `-1e4` where
    /// a row has no option at that slot. `action` is unused downstream (see header comment).
    func callAsFunction(
        inputIds: MLXArray, attentionMask: MLXArray, markerPos: MLXArray, markerMask: MLXArray,
        qtype: MLXArray
    ) -> (logits: MLXArray, action: MLXArray) {
        var h = encoder(inputIds: inputIds, attentionMask: attentionMask)
        h = h + typeEmb(qtype).expandedDimensions(axis: 1)
        let headMask = attentionMask.asType(.bool).expandedDimensions(axes: [1, 2])
        h = head(h, mask: headMask)
        // `takeAlong`'s own broadcasting handles a [B, count, 1] index against `h`'s
        // [B, L, D] — this is the standard take_along_axis usage (NumPy and MLX both expect
        // the non-gather axes to stay size-1 and broadcast internally). Keep this narrow.
        let clampedMarkerPos = MLX.maximum(markerPos, MLXArray(Int32(0)))
        let gatherIndex = clampedMarkerPos.expandedDimensions(axis: -1)
        let markers = takeAlong(h, gatherIndex, axis: 1)
        var logits = scorer(markers).squeezed(axis: -1).asType(.float32)
        logits = which(markerMask, logits, MLXArray(Float(-1e4)))
        let p = softmax(logits, axis: -1)
        let k = MLX.maximum(markerMask.asType(.float32).sum(axis: -1), MLXArray(Float(2)))
        let entropy = -(p * log(MLX.maximum(p, MLXArray(Float(1e-9))))).sum(axis: -1) / log(k)
        let sortedP = sorted(p, axis: -1)
        let width = sortedP.dim(-1)
        // Bare-Int subscripts here are the same 0-dim-index `Gather` that Metal API
        // Validation aborts on for real (see `LayaHeadAttention` above) — range-slice + squeeze
        // instead.
        let top1 = sortedP[0..., (width - 1) ..< width].squeezed(axis: -1)
        let top2 = sortedP[0..., (width - 2) ..< (width - 1)].squeezed(axis: -1)
        let features = stacked([top1, top1 - top2, entropy, k / 255.0], axis: -1)
        let pooled = concatenated([h[0..., 0 ..< 1].squeezed(axis: 1).asType(.float32), features], axis: -1)
        let actHead0Dtype = (actHead.layers[0] as? Linear)?.weight.dtype ?? pooled.dtype
        let action = actHead(pooled.asType(actHead0Dtype)).asType(.float32)
        return (logits, action)
    }
}

/// `model.py::sanitize_weights` — map upstream PyTorch parameter names to this MLX module tree.
/// The encoder's own keys already match (`ModernBERTEncoder`'s layout mirrors the raw HF
/// checkpoint one-for-one); only the decision head's flattened `nn.Sequential`-style names
/// need remapping onto MLXNN's `Sequential.layers` array.
nonisolated func layaSanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (rawName, value) in weights {
        var name = rawName
        name = name.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
        name = name.replacingOccurrences(of: ".in_proj_bias", with: ".in_proj.bias")
        for prefix in ["scorer", "act_head"] {
            let dotPrefix = prefix + "."
            let layersPrefix = prefix + ".layers."
            if name.hasPrefix(dotPrefix), !name.hasPrefix(layersPrefix) {
                name = layersPrefix + String(name.dropFirst(dotPrefix.count))
            }
        }
        result[name] = value
    }
    return result
}
