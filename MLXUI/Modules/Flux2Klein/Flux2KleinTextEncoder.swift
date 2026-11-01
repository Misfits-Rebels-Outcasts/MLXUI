import Foundation
import MLX
import MLXFast
import MLXNN

// FLUX.2 Klein conditioner: Qwen3-4B run only as far as the deepest tap, with the hidden states after layers 9, 18 and 27 concatenated (MP7-3c).
// Structure and tap rule follow VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin) `Qwen3Model.forwardWithHiddenStates` and
// `KleinEmbeddingExtractor`; the numbers are checked against diffusers 0.37.1 (`Flux2KleinPipeline._get_qwen3_prompt_embeds`: a tiny-config fp32
// golden, MLXUITests/Fixtures/Flux2Klein/g3c_textencoder.safetensors).
//
// Tap indexing. `hidden_states[k]` is the output AFTER k decoder layers (index 0 = the embeddings), so taps (9, 18, 27) are the outputs of the layers
// numbered 8, 17 and 26 (0-based). The tap at 27 is PRE-final-norm: HF applies the final norm only to the last entry (index 36), which this port never
// reaches. So the port builds and runs layers 0...26 only and has no final norm and no lm_head (tied to the embedding in the checkpoint).
//
// Mask. Attention is causal AND blocks every padded KEY (right padding, id 151643). A pad query row therefore attends only to the real tokens before it.
// The DiT reads all 512 text rows (the transformer takes no text mask), so the pad rows are model input and have to match HF's masked values.
//
// Memory. The weights are streamed: each layer's bf16 tensors are assigned, quantized (when `bits` is set), evaluated and the bf16 copies released
// before the next layer is touched. `bits == nil` keeps the checkpoint dtype (a reference path for tests, ~6.2 GB of bf16 layers).

nonisolated enum Flux2KleinTextEncoderError: Error, Equatable, LocalizedError {
    case unsupportedBits(Int)
    case missingWeight(String)
    case unexpectedWeight(String)
    case quantizationFailed(String)
    case noWeightFiles(String)
    case badInput(String)

    var errorDescription: String? {
        switch self {
        case let .unsupportedBits(bits):
            // MP7-3h (owner done-when g): the encoder's own sentence mentions only its 8-bit floor. The row-level `quantize` choice is validated, in a sentence
            // about the image model, before the encoder is ever asked (Flux2KleinEngine.validate).
            return "The FLUX.2 Klein text encoder always runs at 8 bits or more; \(bits) bits is not a width it can load."
        case let .missingWeight(name):
            return "The FLUX.2 Klein text encoder is missing the weight \(name). Reinstall the model."
        case let .unexpectedWeight(name):
            return "The FLUX.2 Klein text encoder has a weight it does not know: \(name). The checkpoint is not the one this build expects."
        case let .quantizationFailed(what):
            return "FLUX.2 Klein could not quantize its text encoder (\(what))."
        case let .noWeightFiles(dir):
            return "No text-encoder weight files (model-*.safetensors) were found in \(dir). Reinstall the model."
        case let .badInput(why):
            return "FLUX.2 Klein text encoder input: \(why)."
        }
    }
}

/// MP7-3c-FIX-1, owner ruling 2026-10-31: the text encoder never goes below 8 bits. A `quantize` setting below 8 applies to the image model (DiT) only,
/// and the row says so with `textEncoderNote`. Later slices (Run UI, flow row text) use these two constants, not their own.
nonisolated enum Flux2KleinQuantization {
    static let textEncoderMinimumBits = 8
    static let textEncoderNote = "The text encoder stays 8-bit; a lower quantize setting applies to the image model only."

    /// The width the text encoder is actually loaded at for a requested `quantize`: nil (unquantized) stays nil, anything else is at least 8.
    static func textEncoderBits(for quantize: Int?) -> Int? { quantize.map { max($0, textEncoderMinimumBits) } }
}

nonisolated struct Flux2KleinQwen3Shape: Equatable, Sendable {
    var hidden: Int
    var intermediate: Int
    var heads: Int
    var kvHeads: Int
    var headDim: Int
    var eps: Float
    var ropeTheta: Float
    var vocab: Int
    /// Tap positions as "after k layers" (1-based count); the deepest one decides how many layers exist.
    var taps: [Int]

    var layerCount: Int { taps.max() ?? 0 }

    /// The shape of the real checkpoint (already validated by `Flux2KleinTextEncoderConfig.decode`).
    init(_ c: Flux2KleinTextEncoderConfig, taps: [Int] = Flux2KleinDims.tapLayers) {
        self.init(hidden: c.hiddenSize, intermediate: c.intermediateSize, heads: c.numAttentionHeads, kvHeads: c.numKeyValueHeads,
                  headDim: c.headDim, eps: Float(c.rmsNormEps), ropeTheta: Float(c.ropeTheta), vocab: c.vocabSize, taps: taps)
    }

    init(hidden: Int, intermediate: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float, ropeTheta: Float, vocab: Int, taps: [Int]) {
        self.hidden = hidden; self.intermediate = intermediate; self.heads = heads; self.kvHeads = kvHeads
        self.headDim = headDim; self.eps = eps; self.ropeTheta = ropeTheta; self.vocab = vocab; self.taps = taps
    }
}

nonisolated final class Flux2KleinQwen3Attention: Module {
    let shape: Flux2KleinQwen3Shape
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ s: Flux2KleinQwen3Shape) {
        shape = s
        _qProj.wrappedValue = Linear(s.hidden, s.heads * s.headDim, bias: false)
        _kProj.wrappedValue = Linear(s.hidden, s.kvHeads * s.headDim, bias: false)
        _vProj.wrappedValue = Linear(s.hidden, s.kvHeads * s.headDim, bias: false)
        _oProj.wrappedValue = Linear(s.heads * s.headDim, s.hidden, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: s.headDim, eps: s.eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: s.headDim, eps: s.eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1)
        let s = shape
        // q/k get a per-head RMSNorm BEFORE the rotary embedding (Qwen3).
        var q = qNorm(qProj(x).reshaped(b, n, s.heads, s.headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, n, s.kvHeads, s.headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, n, s.kvHeads, s.headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: s.headDim, traditional: false, base: s.ropeTheta, scale: 1, offset: 0)
        k = MLXFast.RoPE(k, dimensions: s.headDim, traditional: false, base: s.ropeTheta, scale: 1, offset: 0)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(s.headDim).squareRoot(), mask: mask)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(b, n, s.heads * s.headDim))
    }
}

nonisolated final class Flux2KleinQwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ s: Flux2KleinQwen3Shape) {
        _gate.wrappedValue = Linear(s.hidden, s.intermediate, bias: false)
        _up.wrappedValue = Linear(s.hidden, s.intermediate, bias: false)
        _down.wrappedValue = Linear(s.intermediate, s.hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

nonisolated final class Flux2KleinQwen3Layer: Module {
    @ModuleInfo(key: "self_attn") var attention: Flux2KleinQwen3Attention
    @ModuleInfo(key: "mlp") var mlp: Flux2KleinQwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    /// The 11 tensors a layer owns (q/k/v/o, q/k norm, gate/up/down, two layer norms), by their name inside the layer.
    static let expectedTensorNames: Set<String> = [
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]

    init(_ s: Flux2KleinQwen3Shape) {
        _attention.wrappedValue = Flux2KleinQwen3Attention(s)
        _mlp.wrappedValue = Flux2KleinQwen3MLP(s)
        _inputNorm.wrappedValue = RMSNorm(dimensions: s.hidden, eps: s.eps)
        _postNorm.wrappedValue = RMSNorm(dimensions: s.hidden, eps: s.eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask)
        return h + mlp(postNorm(h))
    }
}

/// The text conditioner: embeddings + layers 0...(deepest tap - 1). No final norm, no lm_head.
nonisolated final class Flux2KleinTextEncoder {
    let shape: Flux2KleinQwen3Shape
    /// The width actually loaded (after the 8-bit floor); nil = unquantized.
    let bits: Int?
    private(set) var embedTokens: Embedding
    private(set) var layers: [Flux2KleinQwen3Layer]

    static let supportedBits: Set<Int> = [3, 4, 5, 6, 8]
    static let maxLength = Flux2KleinPrompt.maxLength

    private init(shape: Flux2KleinQwen3Shape, bits: Int?, embedTokens: Embedding, layers: [Flux2KleinQwen3Layer]) {
        self.shape = shape; self.bits = bits; self.embedTokens = embedTokens; self.layers = layers
    }

    // MARK: Forward

    /// `ids` and `attentionMask` are `[B, S]` (Int32, mask 1 = real token). Returns the tapped hidden states concatenated on the last axis,
    /// `[B, S, taps x hidden]`, lazily; the caller evaluates it.
    func callAsFunction(ids: MLXArray, attentionMask: MLXArray) throws -> MLXArray {
        guard ids.ndim == 2, attentionMask.shape == ids.shape else {
            throw Flux2KleinTextEncoderError.badInput("ids and mask must both be [batch, length]")
        }
        var h = embedTokens(ids)
        let mask = Self.additiveMask(attentionMask, dtype: h.dtype)
        var taps: [MLXArray] = []
        for (index, layer) in layers.enumerated() {
            try Task.checkCancellation()
            h = layer(h, mask: mask)
            // "after k layers" with k = index + 1; the tap is read BEFORE any final norm (there is none here).
            if shape.taps.contains(index + 1) { taps.append(h) }
        }
        return concatenated(taps, axis: -1)
    }

    /// Causal mask OR a blocked padded key, as an additive `[B, 1, S, S]` in the activation dtype (the reference uses -1e9 as well).
    static func additiveMask(_ attentionMask: MLXArray, dtype: DType) -> MLXArray {
        let n = attentionMask.dim(1)
        let positions = MLXArray(Array(0 ..< Int32(n)))
        let causalBlocked = positions.reshaped(1, n) .> positions.reshaped(n, 1)            // key after query
        let padBlocked = (attentionMask .== 0).reshaped(attentionMask.dim(0), 1, 1, n)       // key is padding
        let blocked = logicalOr(causalBlocked.reshaped(1, 1, n, n), padBlocked)
        return which(blocked, MLXArray(-1e9).asType(dtype), MLXArray(0.0).asType(dtype))
    }

    /// The Run-UI-shaped call: a tokenized prompt -> `[1, 512, 7680]`, evaluated (so it may cross an actor hop).
    func embeddings(for prompt: Flux2KleinEncodedPrompt) throws -> MLXArray {
        let ids = MLXArray(prompt.ids, [1, prompt.ids.count])
        let mask = MLXArray(prompt.attentionMask, [1, prompt.attentionMask.count])
        let out = try self(ids: ids, attentionMask: mask)
        eval(out)
        return out
    }

    // MARK: Loading

    /// Where the weights come from: one closure per shard, called in order. Each returns the (lazy) tensors of that file.
    typealias Shard = () throws -> [String: MLXArray]

    /// Loads `text_encoder/` of an installed Klein model. Needs only `config.json` and the `model*.safetensors` shards; it does NOT need
    /// `model.safetensors.index.json` (the installer does not fetch it), so layer membership is read from the tensors' own names.
    static func load(textEncoderDirectory dir: URL, bits: Int?, groupSize: Int = 64, testOnlyKeepEmbeddingBF16: Bool = false,
                     stage: (String) -> Void = { _ in }) throws -> Flux2KleinTextEncoder {
        let config = try Flux2KleinTextEncoderConfig.decode(from: Data(contentsOf: dir.appendingPathComponent("config.json")))
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent.hasPrefix("model") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw Flux2KleinTextEncoderError.noWeightFiles(dir.path) }
        let shards: [Shard] = files.map { url in { try loadArrays(url: url) } }
        return try load(shape: Flux2KleinQwen3Shape(config), prefix: "model.", shards: shards, bits: bits, groupSize: groupSize,
                    testOnlyKeepEmbeddingBF16: testOnlyKeepEmbeddingBF16, stage: stage)
    }

    /// The streaming loader. Tensors are matched by name (`<prefix>embed_tokens.weight`, `<prefix>layers.N.…`); everything else (deeper layers,
    /// final norm, lm_head) is ignored and released as soon as its shard is read. A layer is built the moment all of its 11 tensors have
    /// arrived, so a layer that straddles two shards (layer 20 of the real checkpoint) simply waits for its second half.
    /// `testOnlyKeepEmbeddingBF16` (MP7-3d-FIX-1, implementer's call, pending owner confirmation): measurement-only; leaves the embedding table in the checkpoint dtype
    /// while the layers are still quantized. Defaults to false, and no production caller passes it, so the shipping path (embedding quantized with the layers) is unchanged.
    static func load(shape: Flux2KleinQwen3Shape, prefix: String, shards: [Shard], bits: Int?, groupSize: Int = 64, testOnlyKeepEmbeddingBF16: Bool = false,
                     stage: (String) -> Void = { _ in }) throws -> Flux2KleinTextEncoder {
        if let bits, !supportedBits.contains(bits) { throw Flux2KleinTextEncoderError.unsupportedBits(bits) }
        // The floor: whatever `quantize` the row asked for, the encoder is built at 8 bits or more (Flux2KleinQuantization).
        let bits = Flux2KleinQuantization.textEncoderBits(for: bits)
        let wanted = shape.layerCount
        var layers: [Flux2KleinQwen3Layer?] = Array(repeating: nil, count: wanted)
        var pending: [Int: [String: MLXArray]] = [:]       // layer -> local name -> tensor, until complete
        var embedWeight: MLXArray?
        var embed: Embedding?
        let layerPrefix = prefix + "layers."

        func quantizeIfAsked<M: Module>(_ module: M) {
            if let bits { quantize(model: module, groupSize: groupSize, bits: bits) }
        }

        for (shardIndex, shard) in shards.enumerated() {
            try Task.checkCancellation()
            var tensors = try shard()
            stage("shard \(shardIndex) opened (\(tensors.count) tensors)")
            for key in Array(tensors.keys) {
                if key == prefix + "embed_tokens.weight" {
                    embedWeight = tensors.removeValue(forKey: key)
                } else if key.hasPrefix(layerPrefix) {
                    let rest = key.dropFirst(layerPrefix.count)
                    guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else { tensors.removeValue(forKey: key); continue }
                    if index < wanted {
                        pending[index, default: [:]][String(rest[rest.index(after: dot)...])] = tensors.removeValue(forKey: key)
                    } else {
                        tensors.removeValue(forKey: key)           // beyond the deepest tap: never built, never evaluated
                    }
                } else {
                    tensors.removeValue(forKey: key)               // final norm, lm_head
                }
            }
            tensors.removeAll()

            if embed == nil, let weight = embedWeight {
                let e = Embedding(embeddingCount: shape.vocab, dimensions: shape.hidden)
                e.update(parameters: ModuleParameters.unflattened(["weight": weight]))
                embedWeight = nil
                var built: Embedding = e
                if let bits, !testOnlyKeepEmbeddingBF16 {
                    guard let q = quantizeSingle(layer: e, groupSize: groupSize, bits: bits) as? Embedding else {
                        throw Flux2KleinTextEncoderError.quantizationFailed("embedding table, \(bits) bits, group \(groupSize)")
                    }
                    built = q
                }
                eval(built)
                Memory.clearCache()
                embed = built
                stage("embedding ready")
            }
            for index in pending.keys.sorted() {
                let have = Set(pending[index]?.keys.map { $0 } ?? [])
                // Anything not in the expected 11 (an extra or a renamed tensor) is refused by name, whether or not the layer is complete.
                if let extra = have.subtracting(Flux2KleinQwen3Layer.expectedTensorNames).sorted().first {
                    throw Flux2KleinTextEncoderError.unexpectedWeight("\(layerPrefix)\(index).\(extra)")
                }
                guard have.count == Flux2KleinQwen3Layer.expectedTensorNames.count else { continue }
                try Task.checkCancellation()
                let layer = Flux2KleinQwen3Layer(shape)
                layer.update(parameters: ModuleParameters.unflattened(pending[index] ?? [:]))
                pending.removeValue(forKey: index)
                quantizeIfAsked(layer)
                eval(layer.parameters().flattened().map { $0.1 })
                Memory.clearCache()
                layers[index] = layer
                stage("layer \(index) ready")
            }
        }

        guard let embed else { throw Flux2KleinTextEncoderError.missingWeight(prefix + "embed_tokens.weight") }
        var built: [Flux2KleinQwen3Layer] = []
        for (index, layer) in layers.enumerated() {
            guard let layer else {
                // Name every tensor that never arrived (all 11 when the layer is absent altogether).
                let have = Set(pending[index]?.keys.map { $0 } ?? [])
                let missing = Flux2KleinQwen3Layer.expectedTensorNames.subtracting(have).sorted().map { "\(layerPrefix)\(index).\($0)" }
                throw Flux2KleinTextEncoderError.missingWeight(missing.joined(separator: ", "))
            }
            built.append(layer)
        }
        return Flux2KleinTextEncoder(shape: shape, bits: bits, embedTokens: embed, layers: built)
    }
}

/// The hand-off the staged engine (3h) will use: prompt text in, `[1, 512, 7680]` out, encoder freed before returning.
nonisolated struct Flux2KleinTextEmbeddings: @unchecked Sendable {
    /// Evaluated, `[1, 512, 7680]`.
    let array: MLXArray
    let realLength: Int
    let cutTokenCount: Int
}

nonisolated enum Flux2KleinTextEncoding {
    /// Tokenize with the installed `tokenizer/`, load the streamed encoder from `text_encoder/` at `max(quantize, 8)`, encode, drop the encoder, clear MLX's buffer cache.
    /// `quantize` is the ROW's setting and is not optional (MP7-3h, owner done-when g): the unquantized bf16 encoder (6.85 GB) stays reachable only through the test-facing
    /// `Flux2KleinTextEncoder.load(... bits: nil)`. The caller has already validated `quantize` for the whole row (`Flux2KleinEngine.validate`).
    /// `@concurrent`: a nonisolated async function runs on the caller's executor, and a 27-layer forward on the main actor beachballs.
    /// The result is evaluated before it leaves this function (MixedSubscript rule 17b).
    @concurrent static func embed(prompt: String, modelDirectory: URL, quantize: Int) async throws -> Flux2KleinTextEmbeddings {
        // Whatever happens (done, thrown, cancelled), MLX's buffer cache is emptied on the way out.
        defer { Memory.clearCache() }
        let tokenizer = try await HFTokenizerLoader().load(from: modelDirectory.appendingPathComponent("tokenizer"))
        let encoded = try Flux2KleinPrompt.encode(prompt, tokenizer: tokenizer)
        let result: Flux2KleinTextEmbeddings
        do {   // the encoder is released at the end of this block, BEFORE the function-level defer empties the cache
            let encoder = try Flux2KleinTextEncoder.load(textEncoderDirectory: modelDirectory.appendingPathComponent("text_encoder"), bits: quantize)
            result = Flux2KleinTextEmbeddings(array: try encoder.embeddings(for: encoded), realLength: encoded.realLength,
                                              cutTokenCount: encoded.cutTokenCount)
        }
        return result
    }
}
