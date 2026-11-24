import Foundation

/// PF-5 — the Privacy Filter model's configuration, decoded from the checkpoint's `config.json`
/// (`OpenAIPrivacyFilterConfig` in transformers 5.12.0). The values are read, not hard-coded: the
/// loader and the network derive every shape from here, so a checkpoint with a different geometry
/// is refused at load time instead of silently running with the wrong shapes.
///
/// Everything is `nonisolated` (the project sets `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`); the
/// model types cross actors exactly like the other from-scratch ports (ModernBERT, RIFE, Laya).
nonisolated struct PrivacyFilterConfig: Sendable {

    /// One plain sentence per failure. Never quotes input text (decision 10); a tensor/config key
    /// name is not input, so naming the offending key is how the strict loader stays debuggable.
    nonisolated enum ModelError: Error, LocalizedError, Sendable, Equatable {
        case fileMissing(String)
        case fileUnreadable(String)
        case configKeyMissing(String)
        case configValueInvalid(String)
        case weightsTensorMissing(String)
        case weightsTensorUnused(String)
        case weightsTensorDuplicated(String)
        case weightsTensorShape(key: String, expected: [Int], got: [Int])
        case weightsTensorDType(key: String, expected: String, got: String)
        case modelMismatch(String)
        case inputInvalid(String)

        var errorDescription: String? {
            switch self {
            case .fileMissing(let name):
                return "the model folder is missing \(name)."
            case .fileUnreadable(let name):
                return "the model folder's \(name) could not be read."
            case .configKeyMissing(let key):
                return "the model's config.json is missing \(key)."
            case .configValueInvalid(let detail):
                return "the model's config.json is not valid: \(detail)."
            case .weightsTensorMissing(let key):
                return "the model's weights are missing the tensor \"\(key)\"."
            case .weightsTensorUnused(let key):
                return "the model's weights contain an unused tensor \"\(key)\"."
            case .weightsTensorDuplicated(let key):
                return "the model's weights contain the tensor \"\(key)\" more than once."
            case .weightsTensorShape(let key, let expected, let got):
                return "the tensor \"\(key)\" has shape \(got), expected \(expected)."
            case .weightsTensorDType(let key, let expected, let got):
                return "the tensor \"\(key)\" is \(got), expected \(expected)."
            case .modelMismatch(let detail):
                return "the model's weights do not match its config: \(detail)."
            case .inputInvalid(let detail):
                return "the input can't be run: \(detail)."
            }
        }
    }

    let hiddenSize: Int
    let numLayers: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let numExperts: Int
    let expertsPerToken: Int
    let vocabSize: Int
    let numLabels: Int
    let rmsNormEps: Float
    /// Inclusive half-band of the bidirectional attention mask: a query attends to a key when
    /// `abs(queryIndex - keyIndex) <= slidingWindow` (`transformers/masking_utils.py::
    /// sliding_window_bidirectional_overlay`).
    let slidingWindow: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let ropeFactor: Float
    let ropeBetaFast: Float
    let ropeBetaSlow: Float
    /// 33 in checkpoint order: `O` then `B/I/E/S-<category>` per category (`id2label`).
    let labelNames: [String]

    // Additive to the frozen contract (needed by `_compute_yarn_parameters`; the contract's field
    // list is what callers construct against and does not include these two).
    /// `rope_parameters.original_max_position_embeddings` (4096): the pretraining context the YaRN
    /// ramp is anchored on. Not the config's `max_position_embeddings` (131072).
    let originalMaxPositionEmbeddings: Int
    /// `rope_parameters.truncate` (false in this checkpoint): whether the correction range is
    /// floored/ceiled to integer dimensions.
    let ropeTruncate: Bool

    /// Read and validate `<directory>/config.json`.
    static func load(from directory: URL) throws -> PrivacyFilterConfig {
        let url = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.fileMissing("config.json")
        }
        guard let data = try? Data(contentsOf: url),
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw ModelError.fileUnreadable("config.json")
        }

        func integer(_ key: String) throws -> Int {
            guard let raw = root[key] else { throw ModelError.configKeyMissing(key) }
            guard let number = raw as? NSNumber else {
                throw ModelError.configValueInvalid("\(key) is not a number")
            }
            let value = number.intValue
            guard Double(value) == number.doubleValue else {
                throw ModelError.configValueInvalid("\(key) is not an integer")
            }
            return value
        }

        func float(_ dictionary: [String: Any], _ key: String) throws -> Float {
            guard let raw = dictionary[key] else { throw ModelError.configKeyMissing("rope_parameters.\(key)") }
            guard let number = raw as? NSNumber else {
                throw ModelError.configValueInvalid("rope_parameters.\(key) is not a number")
            }
            return number.floatValue
        }

        guard let rope = root["rope_parameters"] as? [String: Any] else {
            throw ModelError.configKeyMissing("rope_parameters")
        }
        guard let ropeType = rope["rope_type"] as? String else {
            throw ModelError.configKeyMissing("rope_parameters.rope_type")
        }
        guard ropeType == "yarn" else {
            throw ModelError.configValueInvalid("rope_type \"\(ropeType)\" is not YaRN")
        }

        let hiddenSize = try integer("hidden_size")
        let numLayers = try integer("num_hidden_layers")
        let numHeads = try integer("num_attention_heads")
        let numKVHeads = try integer("num_key_value_heads")
        let headDim = try integer("head_dim")
        let intermediateSize = try integer("intermediate_size")
        // `num_experts` is an alias of `num_local_experts` in `OpenAIPrivacyFilterConfig`.
        let numExperts: Int
        if root["num_local_experts"] != nil {
            numExperts = try integer("num_local_experts")
        } else {
            numExperts = try integer("num_experts")
        }
        let expertsPerToken = try integer("num_experts_per_tok")
        let vocabSize = try integer("vocab_size")
        let slidingWindow = try integer("sliding_window")
        let maxPositionEmbeddings = try integer("max_position_embeddings")

        guard let epsRaw = root["rms_norm_eps"] as? NSNumber else {
            throw ModelError.configKeyMissing("rms_norm_eps")
        }
        let rmsNormEps = epsRaw.floatValue
        guard rmsNormEps > 0 else {
            throw ModelError.configValueInvalid("rms_norm_eps must be positive")
        }

        guard let id2label = root["id2label"] as? [String: String], !id2label.isEmpty else {
            throw ModelError.configKeyMissing("id2label")
        }
        var labels = [String](repeating: "", count: id2label.count)
        for (key, label) in id2label {
            guard let index = Int(key), labels.indices.contains(index) else {
                throw ModelError.configValueInvalid("id2label has an out-of-range id \"\(key)\"")
            }
            labels[index] = label
        }
        guard labels.allSatisfy({ !$0.isEmpty }) else {
            throw ModelError.configValueInvalid("id2label has a gap")
        }

        let theta = try float(rope, "rope_theta")
        let factor = try float(rope, "factor")
        let betaFast = try float(rope, "beta_fast")
        let betaSlow = try float(rope, "beta_slow")
        guard let originalMax = (rope["original_max_position_embeddings"] as? NSNumber)?.intValue else {
            throw ModelError.configKeyMissing("rope_parameters.original_max_position_embeddings")
        }
        let truncate = (rope["truncate"] as? NSNumber)?.boolValue ?? true

        guard hiddenSize > 0, numLayers > 0, numHeads > 0, numKVHeads > 0, headDim > 0,
              intermediateSize > 0, numExperts > 0, vocabSize > 0, slidingWindow >= 0,
              maxPositionEmbeddings > 0, originalMax > 0 else {
            throw ModelError.configValueInvalid("a geometry value is not positive")
        }
        guard numHeads % numKVHeads == 0 else {
            throw ModelError.configValueInvalid(
                "num_attention_heads \(numHeads) is not a multiple of num_key_value_heads \(numKVHeads)")
        }
        guard expertsPerToken > 0, expertsPerToken <= numExperts else {
            throw ModelError.configValueInvalid(
                "num_experts_per_tok \(expertsPerToken) is outside 1...num_local_experts \(numExperts)")
        }
        guard headDim % 2 == 0 else {
            throw ModelError.configValueInvalid("head_dim \(headDim) must be even for interleaved RoPE")
        }
        guard theta > 0, factor >= 1, betaFast > 0, betaSlow > 0 else {
            throw ModelError.configValueInvalid("a RoPE parameter is out of range")
        }

        return PrivacyFilterConfig(
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            intermediateSize: intermediateSize,
            numExperts: numExperts,
            expertsPerToken: expertsPerToken,
            vocabSize: vocabSize,
            numLabels: labels.count,
            rmsNormEps: rmsNormEps,
            slidingWindow: slidingWindow,
            maxPositionEmbeddings: maxPositionEmbeddings,
            ropeTheta: theta,
            ropeFactor: factor,
            ropeBetaFast: betaFast,
            ropeBetaSlow: betaSlow,
            labelNames: labels,
            originalMaxPositionEmbeddings: originalMax,
            ropeTruncate: truncate)
    }
}
