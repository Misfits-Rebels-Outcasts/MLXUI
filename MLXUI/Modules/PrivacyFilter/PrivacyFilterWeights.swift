import Foundation
import MLX

/// PF-5, Trap 6 — the 140 tensors of `openai/privacy-filter`'s `model.safetensors`, loaded
/// **strictly**: every expected key must be present exactly once with the expected shape and dtype,
/// and no key may be left unused. The first failure is reported by name in one plain sentence.
///
/// The checkpoint's raw layout is the HF module's, which is also the layout the forward pass wants:
///
/// | key | shape | dtype |
/// |---|---|---|
/// | `model.embed_tokens.weight` | `[vocab, hidden]` | bf16 |
/// | `model.layers.N.input_layernorm.weight` | `[hidden]` | bf16 |
/// | `model.layers.N.self_attn.{q,k,v,o}_proj.{weight,bias}` | `[896,640]+[896]`, `[128,640]+[128]` ×2, `[640,896]+[640]` | bf16 |
/// | `model.layers.N.self_attn.sinks` | `[heads]` | **fp32** (`_keep_in_fp32_modules_strict`) |
/// | `model.layers.N.post_attention_layernorm.weight` | `[hidden]` | bf16 |
/// | `model.layers.N.mlp.router.{weight,bias}` | `[experts,hidden]`, `[experts]` | bf16 |
/// | `model.layers.N.mlp.experts.gate_up_proj{,_bias}` | `[experts,hidden,2*inter]`, `[experts,2*inter]` | bf16 |
/// | `model.layers.N.mlp.experts.down_proj{,_bias}` | `[experts,inter,hidden]`, `[experts,hidden]` | bf16 |
/// | `model.norm.weight` | `[hidden]` | bf16 |
/// | `score.{weight,bias}` | `[labels,hidden]`, `[labels]` | bf16 |
///
/// The expert and `score` parameters are plain `nn.Parameter`s in the reference (not `nn.Linear`),
/// so they are stored already transposed for `matmul` — the loader keeps them exactly as they are.
nonisolated struct PrivacyFilterWeights: @unchecked Sendable {

    nonisolated struct Layer: @unchecked Sendable {
        let inputLayerNorm: MLXArray          // [hidden] bf16
        let postAttentionLayerNorm: MLXArray  // [hidden] bf16
        let qProjWeight: MLXArray             // [heads*headDim, hidden] bf16
        let qProjBias: MLXArray               // [heads*headDim] bf16
        let kProjWeight: MLXArray             // [kvHeads*headDim, hidden] bf16
        let kProjBias: MLXArray               // [kvHeads*headDim] bf16
        let vProjWeight: MLXArray             // [kvHeads*headDim, hidden] bf16
        let vProjBias: MLXArray               // [kvHeads*headDim] bf16
        let oProjWeight: MLXArray             // [hidden, heads*headDim] bf16
        let oProjBias: MLXArray               // [hidden] bf16
        let sinks: MLXArray                   // [heads] fp32
        let routerWeight: MLXArray            // [experts, hidden] bf16
        let routerBias: MLXArray              // [experts] bf16
        let gateUpProj: MLXArray              // [experts, hidden, 2*intermediate] bf16
        let gateUpProjBias: MLXArray          // [experts, 2*intermediate] bf16
        let downProj: MLXArray                // [experts, intermediate, hidden] bf16
        let downProjBias: MLXArray            // [experts, hidden] bf16
    }

    let embedTokens: MLXArray   // [vocab, hidden] bf16
    let norm: MLXArray          // [hidden] bf16
    let layers: [Layer]
    let scoreWeight: MLXArray   // [numLabels, hidden] bf16
    let scoreBias: MLXArray     // [numLabels] bf16

    /// Read every `*.safetensors` file in `directory` and check it against `config` exactly.
    ///
    /// `MLX.loadArrays` is used (not a hand-rolled parser); the arrays it returns are on the CPU
    /// stream and are evaluated here before any of them crosses to another thread (Trap 7).
    static func load(from directory: URL, config: PrivacyFilterConfig) throws -> PrivacyFilterWeights {
        let files: [URL]
        do {
            files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            throw PrivacyFilterConfig.ModelError.fileUnreadable(directory.lastPathComponent)
        }
        guard !files.isEmpty else {
            throw PrivacyFilterConfig.ModelError.fileMissing("model.safetensors")
        }

        var loaded: [String: MLXArray] = [:]
        for file in files {
            let arrays: [String: MLXArray]
            do {
                arrays = try MLX.loadArrays(url: file)
            } catch {
                throw PrivacyFilterConfig.ModelError.fileUnreadable(file.lastPathComponent)
            }
            for (key, value) in arrays {
                if loaded[key] != nil {
                    throw PrivacyFilterConfig.ModelError.weightsTensorDuplicated(key)
                }
                loaded[key] = value
            }
        }

        let specs = tensorSpecs(config)
        let expectedKeys = Set(specs.map(\.name))
        // Missing first, then unused, then shape/dtype — each names the first offending key.
        for spec in specs where loaded[spec.name] == nil {
            throw PrivacyFilterConfig.ModelError.weightsTensorMissing(spec.name)
        }
        for key in loaded.keys.sorted() where !expectedKeys.contains(key) {
            throw PrivacyFilterConfig.ModelError.weightsTensorUnused(key)
        }
        for spec in specs {
            guard let array = loaded[spec.name] else { continue }  // unreachable after the loop above
            guard array.shape == spec.shape else {
                throw PrivacyFilterConfig.ModelError.weightsTensorShape(
                    key: spec.name, expected: spec.shape, got: array.shape)
            }
            guard array.dtype == spec.dtype else {
                throw PrivacyFilterConfig.ModelError.weightsTensorDType(
                    key: spec.name, expected: spec.dtypeName, got: spec.dtypeName(of: array.dtype))
            }
        }

        func array(_ key: String) -> MLXArray { loaded[key] ?? MLXArray.zeros([0]) }

        var layers: [Layer] = []
        layers.reserveCapacity(config.numLayers)
        for index in 0 ..< config.numLayers {
            let p = "model.layers.\(index)."
            let a = "self_attn."
            let m = "mlp."
            let e = "mlp.experts."
            layers.append(Layer(
                inputLayerNorm: array(p + "input_layernorm.weight"),
                postAttentionLayerNorm: array(p + "post_attention_layernorm.weight"),
                qProjWeight: array(p + a + "q_proj.weight"),
                qProjBias: array(p + a + "q_proj.bias"),
                kProjWeight: array(p + a + "k_proj.weight"),
                kProjBias: array(p + a + "k_proj.bias"),
                vProjWeight: array(p + a + "v_proj.weight"),
                vProjBias: array(p + a + "v_proj.bias"),
                oProjWeight: array(p + a + "o_proj.weight"),
                oProjBias: array(p + a + "o_proj.bias"),
                sinks: array(p + a + "sinks"),
                routerWeight: array(p + m + "router.weight"),
                routerBias: array(p + m + "router.bias"),
                gateUpProj: array(p + e + "gate_up_proj"),
                gateUpProjBias: array(p + e + "gate_up_proj_bias"),
                downProj: array(p + e + "down_proj"),
                downProjBias: array(p + e + "down_proj_bias")))
        }

        let weights = PrivacyFilterWeights(
            embedTokens: array("model.embed_tokens.weight"),
            norm: array("model.norm.weight"),
            layers: layers,
            scoreWeight: array("score.weight"),
            scoreBias: array("score.bias"))

        // Evaluate on the loading thread: every array is then materialised before it is handed to
        // whatever thread runs the forward pass (Trap 7), and malformed data fails here, by name.
        do {
            try MLXErrorGuard.evaluate(weights.allArrays)
        } catch {
            throw PrivacyFilterConfig.ModelError.fileUnreadable("model.safetensors")
        }
        return weights
    }

    /// The exact key set, derived from the config. Order is fixed so a missing key is deterministic.
    private static func tensorSpecs(_ config: PrivacyFilterConfig) -> [TensorSpec] {
        let hidden = config.hiddenSize
        let qDim = config.numHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim
        let inter = config.intermediateSize
        let experts = config.numExperts
        var specs: [TensorSpec] = []
        specs.append(.init("model.embed_tokens.weight", [config.vocabSize, hidden], .bfloat16))
        for index in 0 ..< config.numLayers {
            let p = "model.layers.\(index)."
            specs.append(.init(p + "input_layernorm.weight", [hidden], .bfloat16))
            specs.append(.init(p + "self_attn.q_proj.weight", [qDim, hidden], .bfloat16))
            specs.append(.init(p + "self_attn.q_proj.bias", [qDim], .bfloat16))
            specs.append(.init(p + "self_attn.k_proj.weight", [kvDim, hidden], .bfloat16))
            specs.append(.init(p + "self_attn.k_proj.bias", [kvDim], .bfloat16))
            specs.append(.init(p + "self_attn.v_proj.weight", [kvDim, hidden], .bfloat16))
            specs.append(.init(p + "self_attn.v_proj.bias", [kvDim], .bfloat16))
            specs.append(.init(p + "self_attn.o_proj.weight", [hidden, qDim], .bfloat16))
            specs.append(.init(p + "self_attn.o_proj.bias", [hidden], .bfloat16))
            // `_keep_in_fp32_modules_strict = ["sinks"]` in the reference.
            specs.append(.init(p + "self_attn.sinks", [config.numHeads], .float32))
            specs.append(.init(p + "post_attention_layernorm.weight", [hidden], .bfloat16))
            specs.append(.init(p + "mlp.router.weight", [experts, hidden], .bfloat16))
            specs.append(.init(p + "mlp.router.bias", [experts], .bfloat16))
            specs.append(.init(p + "mlp.experts.gate_up_proj", [experts, hidden, 2 * inter], .bfloat16))
            specs.append(.init(p + "mlp.experts.gate_up_proj_bias", [experts, 2 * inter], .bfloat16))
            specs.append(.init(p + "mlp.experts.down_proj", [experts, inter, hidden], .bfloat16))
            specs.append(.init(p + "mlp.experts.down_proj_bias", [experts, hidden], .bfloat16))
        }
        specs.append(.init("model.norm.weight", [hidden], .bfloat16))
        specs.append(.init("score.weight", [config.numLabels, hidden], .bfloat16))
        specs.append(.init("score.bias", [config.numLabels], .bfloat16))
        return specs
    }

    nonisolated private struct TensorSpec {
        let name: String
        let shape: [Int]
        let dtype: DType
        init(_ name: String, _ shape: [Int], _ dtype: DType) {
            self.name = name
            self.shape = shape
            self.dtype = dtype
        }
        var dtypeName: String { String(describing: dtype) }
        func dtypeName(of dtype: DType) -> String { String(describing: dtype) }
    }

    /// Every tensor, for the load-time evaluation.
    private var allArrays: [MLXArray] {
        var arrays: [MLXArray] = [embedTokens, norm, scoreWeight, scoreBias]
        for layer in layers {
            arrays.append(contentsOf: [
                layer.inputLayerNorm, layer.postAttentionLayerNorm,
                layer.qProjWeight, layer.qProjBias,
                layer.kProjWeight, layer.kProjBias,
                layer.vProjWeight, layer.vProjBias,
                layer.oProjWeight, layer.oProjBias, layer.sinks,
                layer.routerWeight, layer.routerBias,
                layer.gateUpProj, layer.gateUpProjBias,
                layer.downProj, layer.downProjBias,
            ])
        }
        return arrays
    }
}
