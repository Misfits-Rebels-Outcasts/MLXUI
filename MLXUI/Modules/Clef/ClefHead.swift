import Foundation
import MLX
import MLXNN

/// Clef Flash's **joint schema head** — a from-reference port of `clef_mlx.py` (`:257–430`,
/// `:542–580`) from `mlx-community/clef-flash-4bit` @ `b5adbe2c…` (Apache-2.0), fetched to
/// `RSI/notes/clef/hf/` (rule 18). Port map: `RSI/notes/clef/port-map.md` §2.
///
/// | reference | here |
/// |---|---|
/// | `MultiheadAttention` `:260–284` | ``ClefMultiheadAttention`` (packed `in_proj`, not three `Linear`s) |
/// | `FeedForward` `:287–294` | ``ClefFeedForward`` (`fc1`/`fc2` after `sanitize`) |
/// | `EvidenceRoutingLayer` `:297–309` | ``ClefEvidenceRoutingLayer`` |
/// | `TransformerDecoderLayer` `:312–329` | ``ClefTransformerDecoderLayer`` (norm-first) |
/// | `_l2norm` `:332–333` | ``ClefHead/l2norm(_:)`` |
/// | `JointSchemaHead` `:336–430` | ``ClefHead`` |
/// | `sanitize` `:360–368` | ``ClefHead/sanitize(_:)`` |
/// | `load`'s head half `:574–579` | ``ClefHead/load(configURL:weightsURL:)`` |
///
/// Compute dtype is **bfloat16**, as the reference's `load(head_dtype=mx.bfloat16)` + `set_dtype`
/// (`:551`, `:578`) make it; the caller softmaxes in float32 (`:489`, `:509`). Nothing here adds
/// an attention mask — the reference has none (port map §2.4).
/// The head's `lexical_lookup` seam (`:370`, `:449–457`) is a parameter: CF-5b supplies the
/// dequantized `lm_head` rows, CF-5a's tests supply the golden's vectors.

nonisolated enum ClefHeadError: Error, CustomStringConvertible {
    case invalidConfig(String)
    case loadFailed(String)

    var description: String {
        switch self {
        case .invalidConfig(let message): return "Clef head config error: \(message)"
        case .loadFailed(let message): return "Clef head weights error: \(message)"
        }
    }
}

/// `joint_head_config.json` — the six numbers `JointSchemaHead.__init__` takes (`:337`).
nonisolated struct ClefHeadConfig: Codable, Equatable, Sendable {
    let hiddenSize: Int
    let width: Int
    let routingLayers: Int
    let layers: Int
    let heads: Int
    let feedforward: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case width
        case routingLayers = "routing_layers"
        case layers
        case heads
        case feedforward
    }
}

/// `MultiheadAttention` (`:260–284`): `torch.nn.MultiheadAttention(batch_first=True)` with its
/// **packed** `in_proj_weight`/`in_proj_bias`. The packing is ported literally — splitting it into
/// three `Linear`s would rename the checkpoint's keys and break the strict load (port map §2.3).
nonisolated final class ClefMultiheadAttention: Module {
    let heads: Int
    let headDim: Int

    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(width: Int, heads: Int) {
        self.heads = heads
        self.headDim = width / heads
        _inProjWeight.wrappedValue = MLXArray.zeros([3 * width, width])
        _inProjBias.wrappedValue = MLXArray.zeros([3 * width])
        _outProj.wrappedValue = Linear(width, width)
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let d = inProjWeight.dim(1)
        let weight = inProjWeight
        let bias = inProjBias
        let query = matmul(q, weight[0 ..< d].transposed()) + bias[0 ..< d]
        let key = matmul(k, weight[d ..< 2 * d].transposed()) + bias[d ..< 2 * d]
        let value = matmul(v, weight[2 * d ..< 3 * d].transposed()) + bias[2 * d ..< 3 * d]

        let batch = query.dim(0)
        let queryLength = query.dim(1)
        let keyLength = key.dim(1)
        let queryHeads = query.reshaped(batch, queryLength, heads, headDim).transposed(0, 2, 1, 3)
        let keyHeads = key.reshaped(batch, keyLength, heads, headDim).transposed(0, 2, 1, 3)
        let valueHeads = value.reshaped(batch, keyLength, heads, headDim).transposed(0, 2, 1, 3)

        // `scale=hd ** -0.5`, and no mask anywhere in the head (port map §2.3/§2.4).
        let attended = MLXFast.scaledDotProductAttention(
            queries: queryHeads, keys: keyHeads, values: valueHeads,
            scale: pow(Float(headDim), -0.5), mask: nil)
        return outProj(attended.transposed(0, 2, 1, 3).reshaped(batch, queryLength, d))
    }
}

/// `FeedForward` (`:287–294`): `fc2(gelu(fc1(x)))`. MLXNN's free `gelu(_:)` is the exact/erf form
/// the reference uses (`nn.gelu`); never `geluApproximate` (port map §2.4).
nonisolated final class ClefFeedForward: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(width: Int, feedforward: Int) {
        _fc1.wrappedValue = Linear(width, feedforward)
        _fc2.wrappedValue = Linear(feedforward, width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

/// `EvidenceRoutingLayer` (`:297–309`).
nonisolated final class ClefEvidenceRoutingLayer: Module {
    @ModuleInfo(key: "query_norm") var queryNorm: LayerNorm
    @ModuleInfo(key: "memory_norm") var memoryNorm: LayerNorm
    @ModuleInfo(key: "attention") var attention: ClefMultiheadAttention
    @ModuleInfo(key: "feedforward_norm") var feedforwardNorm: LayerNorm
    @ModuleInfo(key: "feedforward") var feedforward: ClefFeedForward

    init(width: Int, heads: Int, feedforward: Int) {
        _queryNorm.wrappedValue = LayerNorm(dimensions: width)
        _memoryNorm.wrappedValue = LayerNorm(dimensions: width)
        _attention.wrappedValue = ClefMultiheadAttention(width: width, heads: heads)
        _feedforwardNorm.wrappedValue = LayerNorm(dimensions: width)
        _feedforward.wrappedValue = ClefFeedForward(width: width, feedforward: feedforward)
    }

    func callAsFunction(_ queries: MLXArray, _ memory: MLXArray) -> MLXArray {
        let m = memoryNorm(memory)
        let attended = queries + attention(queryNorm(queries), m, m)
        return attended + feedforward(feedforwardNorm(attended))
    }
}

/// `TransformerDecoderLayer(norm_first=True, activation='gelu')` (`:312–329`).
nonisolated final class ClefTransformerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: ClefMultiheadAttention
    @ModuleInfo(key: "multihead_attn") var multiheadAttn: ClefMultiheadAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    init(width: Int, heads: Int, feedforward: Int) {
        _selfAttn.wrappedValue = ClefMultiheadAttention(width: width, heads: heads)
        _multiheadAttn.wrappedValue = ClefMultiheadAttention(width: width, heads: heads)
        _linear1.wrappedValue = Linear(width, feedforward)
        _linear2.wrappedValue = Linear(feedforward, width)
        _norm1.wrappedValue = LayerNorm(dimensions: width)
        _norm2.wrappedValue = LayerNorm(dimensions: width)
        _norm3.wrappedValue = LayerNorm(dimensions: width)
    }

    func callAsFunction(_ x: MLXArray, _ memory: MLXArray) -> MLXArray {
        var out = x
        let h = norm1(out)
        out = out + selfAttn(h, h, h)
        out = out + multiheadAttn(norm2(out), memory, memory)
        return out + linear2(gelu(linear1(norm3(out))))
    }
}

/// `JointSchemaHead` (`:336–430`).
nonisolated final class ClefHead: Module {
    let config: ClefHeadConfig

    @ModuleInfo(key: "hidden_norm") var hiddenNorm: LayerNorm
    @ModuleInfo(key: "memory_projection") var memoryProjection: Linear
    @ModuleInfo(key: "question_projection") var questionProjection: Linear
    @ModuleInfo(key: "option_question_projection") var optionQuestionProjection: Linear
    @ModuleInfo(key: "global_projection") var globalProjection: Linear
    @ModuleInfo(key: "option_context_projection") var optionContextProjection: Linear
    @ModuleInfo(key: "option_lexical_projection") var optionLexicalProjection: Linear
    @ModuleInfo(key: "type_embedding") var typeEmbedding: Embedding
    @ModuleInfo(key: "evidence_layers") var evidenceLayers: [ClefEvidenceRoutingLayer]
    @ModuleInfo(key: "option_summary_norm") var optionSummaryNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [ClefTransformerDecoderLayer]
    @ModuleInfo(key: "field_norm") var fieldNorm: LayerNorm
    @ModuleInfo(key: "option_norm") var optionNorm: LayerNorm
    @ModuleInfo(key: "scorer1") var scorer1: Linear
    @ModuleInfo(key: "scorer2") var scorer2: Linear

    @ParameterInfo(key: "prior_logit_scale") var priorLogitScale: MLXArray
    @ParameterInfo(key: "joint_logit_scale") var jointLogitScale: MLXArray
    @ParameterInfo(key: "residual_gate") var residualGate: MLXArray

    init(_ config: ClefHeadConfig) {
        self.config = config
        let hidden = config.hiddenSize
        let width = config.width

        _hiddenNorm.wrappedValue = LayerNorm(dimensions: hidden)
        _memoryProjection.wrappedValue = Linear(hidden, width, bias: false)
        _questionProjection.wrappedValue = Linear(hidden, width, bias: false)
        _optionQuestionProjection.wrappedValue = Linear(hidden, width, bias: false)
        _globalProjection.wrappedValue = Linear(hidden, width, bias: false)
        _optionContextProjection.wrappedValue = Linear(hidden, width, bias: false)
        _optionLexicalProjection.wrappedValue = Linear(hidden, width, bias: false)
        _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
        _evidenceLayers.wrappedValue = (0 ..< config.routingLayers).map { _ in
            ClefEvidenceRoutingLayer(width: width, heads: config.heads, feedforward: config.feedforward)
        }
        _optionSummaryNorm.wrappedValue = LayerNorm(dimensions: width)
        _layers.wrappedValue = (0 ..< config.layers).map { _ in
            ClefTransformerDecoderLayer(width: width, heads: config.heads, feedforward: config.feedforward)
        }
        _fieldNorm.wrappedValue = LayerNorm(dimensions: width)
        _optionNorm.wrappedValue = LayerNorm(dimensions: width)
        _scorer1.wrappedValue = Linear(width * 4, width)
        _scorer2.wrappedValue = Linear(width, 1)

        _priorLogitScale.wrappedValue = MLXArray(0.0)
        _jointLogitScale.wrappedValue = MLXArray(0.0)
        _residualGate.wrappedValue = MLXArray(0.0)
    }

    /// `_l2norm` (`:332–333`): `x / max(||x||, 1e-12)` — note the **max**, not an epsilon added to
    /// the norm (Laya's head adds; this one clamps).
    static func l2norm(_ x: MLXArray, eps: Float = 1e-12) -> MLXArray {
        let norm = sqrt((x * x).sum(axis: -1, keepDims: true))
        return x / maximum(norm, MLXArray(eps).asType(x.dtype))
    }

    /// `JointSchemaHead.sanitize` (`:360–368`) — the file's torch-`Sequential` names to this
    /// module's property names. Applied before the strict load.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            name = name.replacingOccurrences(of: ".feedforward.0.", with: ".feedforward.fc1.")
            name = name.replacingOccurrences(of: ".feedforward.3.", with: ".feedforward.fc2.")
            name = name.replacingOccurrences(of: "residual_scorer.0.", with: "scorer1.")
            name = name.replacingOccurrences(of: "residual_scorer.3.", with: "scorer2.")
            out[name] = value
        }
        return out
    }

    /// `load`'s head half (`:574–579`): `joint_head_config.json` + `joint_head.safetensors`,
    /// sanitized, cast to bfloat16 (`head.set_dtype`), loaded with `strict=True` — every key must
    /// land, so a renamed or missing tensor throws instead of silently scoring with zeros.
    static func load(configURL: URL, weightsURL: URL) throws -> ClefHead {
        let data: Data
        do {
            data = try Data(contentsOf: configURL)
        } catch {
            throw ClefHeadError.invalidConfig("cannot read \(configURL.lastPathComponent): \(error)")
        }
        let config = try JSONDecoder().decode(ClefHeadConfig.self, from: data)

        let head = ClefHead(config)
        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: weightsURL)
        } catch {
            throw ClefHeadError.loadFailed("cannot read \(weightsURL.lastPathComponent): \(error)")
        }
        let sanitized = sanitize(raw).mapValues { $0.asType(.bfloat16) }
        do {
            try head.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        } catch {
            throw ClefHeadError.loadFailed("strict load failed: \(error)")
        }
        return head
    }

    /// `JointSchemaHead.__call__` (`:370–430`). `hidden` is `(L, hidden_size)` — the backbone's
    /// final-norm states for one record — and `lexicalLookup` maps a span's token ids to rows of
    /// the output embedding (the reference's `_lexical`, `:449–457`). Returns one float32 logits
    /// vector per question, in record order, unnormalised.
    func callAsFunction(
        hidden: MLXArray,
        record: ClefEncodedRecord,
        lexicalLookup: (MLXArray) -> MLXArray
    ) -> [MLXArray] {
        let width = config.width
        let h = hiddenNorm(hidden.asType(.bfloat16))  // (L, H)
        let memory = memoryProjection(h).expandedDimensions(axis: 0)  // (1, L, W)
        let globalVector = h[h.dim(0) - 1]  // (H,)

        let questionVectors = record.questions.map { h[$0.questionSpan].mean(axis: 0) }  // [(H,)]
        let typeIDs = MLXArray(record.questions.map { Int32($0.type) })

        var optionContexts: [[MLXArray]] = []
        var lexicalOptions: [[MLXArray]] = []
        var counts: [Int] = []
        for question in record.questions {
            optionContexts.append(question.optionSpans.map { h[$0].mean(axis: 0) })
            lexicalOptions.append(
                question.optionSpans.map { span in
                    let ids = MLXArray(record.inputIDs[span].map(Int32.init))
                    return lexicalLookup(ids).asType(.bfloat16).mean(axis: 0)
                })
            counts.append(question.optionSpans.count)
        }

        // (1, Σ options, W) — every option of every question in one evidence sequence. The shared
        // question vector stays `(W,)`: the reference's `q[i]` is `(1, W)` and broadcasts over rows,
        // but an explicit `expandedDimensions` here would make each row `(1, W)` and `stacked` would
        // yield a 4-D tensor whose attention reshape cannot succeed.
        var optionQueries: [MLXArray] = []
        for (index, _) in record.questions.enumerated() {
            let questionVector = optionQuestionProjection(questionVectors[index])  // (W,)
            for (context, lexical) in zip(optionContexts[index], lexicalOptions[index]) {
                optionQueries.append(
                    optionContextProjection(context) + optionLexicalProjection(lexical) + questionVector)
            }
        }
        var routed = stacked(optionQueries, axis: 0).expandedDimensions(axis: 0)
        for layer in evidenceLayers { routed = layer(routed, memory) }
        routed = routed.squeezed(axis: 0)

        var splitOptions: [MLXArray] = []
        var cursor = 0
        for count in counts {
            splitOptions.append(routed[cursor ..< cursor + count])
            cursor += count
        }

        let baseFields = questionProjection(stacked(questionVectors, axis: 0))  // (nQ, W)
        let scale = MLXArray(sqrt(Float(width))).asType(baseFields.dtype)
        var summaries: [MLXArray] = []
        for (index, options) in splitOptions.enumerated() {
            let weights = softmax(matmul(options, baseFields[index]) / scale, axis: 0)
            summaries.append((weights.expandedDimensions(axis: 1) * options).sum(axis: 0))
        }

        var fields = baseFields
            + optionSummaryNorm(stacked(summaries, axis: 0))
            + globalProjection(globalVector).expandedDimensions(axis: 0)
            + typeEmbedding(typeIDs)
        fields = fields.expandedDimensions(axis: 0)  // (1, nQ, W)
        for layer in layers { fields = layer(fields, memory) }
        let finalFields = fieldNorm(fields.squeezed(axis: 0))  // (nQ, W)

        let ceiling = MLXArray(Float(log(100.0))).asType(priorLogitScale.dtype)
        let priorScale = exp(minimum(priorLogitScale, ceiling))
        let jointScale = exp(minimum(jointLogitScale, ceiling))
        let gate = sigmoid(residualGate)
        let cosineFloor = MLXArray(Float(1e-8)).asType(h.dtype)

        var logits: [MLXArray] = []
        for (index, options) in splitOptions.enumerated() {
            let anchor = Self.l2norm(questionVectors[index] + globalVector)
            let prior = priorScale * matmul(Self.l2norm(stacked(lexicalOptions[index], axis: 0)), anchor)
            let normed = optionNorm(options)
            let rf = broadcast(finalFields[index].expandedDimensions(axis: 0), to: normed.shape)
            let numerator = (rf * normed).sum(axis: -1)
            let denominator = sqrt((rf * rf).sum(axis: -1)) * sqrt((normed * normed).sum(axis: -1))
            let cosine = numerator / maximum(denominator, cosineFloor)
            let features = concatenated([rf, normed, rf * normed, abs(rf - normed)], axis: -1)
            let residual = scorer2(gelu(scorer1(features)))[0..., 0]
            logits.append(prior + gate * (jointScale * cosine + residual))
        }
        return logits.map { $0.asType(.float32) }
    }
}
