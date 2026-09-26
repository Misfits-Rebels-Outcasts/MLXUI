import Foundation
import MLX
import MLXNN
import Tokenizers

/// Errors specific to loading/running Laya that don't fit `StageError`'s existing cases.
nonisolated enum LayaEngineError: Error, CustomStringConvertible {
    case incompleteCheckpoint(String)
    case invalidConfig(String)
    case missingSpecialToken(String)

    var description: String {
        switch self {
        case .incompleteCheckpoint(let id):
            return "'\(id)' is missing a required Laya file (model.safetensors, rl_agent_config.json, or encoder/config.json)."
        case .invalidConfig(let message):
            return "Laya config error: \(message)"
        case .missingSpecialToken(let name):
            return "Laya's tokenizer is missing a valid \(name)."
        }
    }
}

extension LayaQuestionType {
    /// `common.py::QTYPES` — `{"choice": 0, "score": 1, "noul": 2}`.
    var qtypeIndex: Int32 {
        switch self {
        case .choice: return 0
        case .score: return 1
        case .noul: return 2
        }
    }
}

/// `agent.py::Tokenizer` (via `tokenizer.py`), bridged onto this app's real HF tokenizer
/// loader (`Core/HFTokenizerLoader.swift`'s `AutoTokenizer`) instead of the Python package's
/// direct `tokenizers.Tokenizer` binding.
struct LayaTokenizer: LayaTokenizing {
    private let backend: any Tokenizers.Tokenizer
    let clsTokenID: Int
    let sepTokenID: Int
    let padTokenID: Int
    let maskTokenID: Int
    let maskToken: String

    static func load(modelDirectory: URL) async throws -> LayaTokenizer {
        let tokenizerDirectory = modelDirectory.appendingPathComponent("tokenizer")
        let backend = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        let configURL = tokenizerDirectory.appendingPathComponent("tokenizer_config.json")
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]

        func specialToken(_ name: String) throws -> (text: String, id: Int) {
            var value = json[name]
            if let dict = value as? [String: Any] { value = dict["content"] }
            guard let text = value as? String, let id = backend.convertTokenToId(text) else {
                throw LayaEngineError.missingSpecialToken(name)
            }
            return (text, id)
        }

        let cls = try specialToken("cls_token")
        let sep = try specialToken("sep_token")
        let pad = try specialToken("pad_token")
        let mask = try specialToken("mask_token")
        return LayaTokenizer(
            backend: backend, clsTokenID: cls.id, sepTokenID: sep.id, padTokenID: pad.id,
            maskTokenID: mask.id, maskToken: mask.text)
    }

    func encode(_ text: String) -> [Int] {
        backend.encode(text: text, addSpecialTokens: false)
    }
}

/// `rl_agent_config.json`'s decoded shape (`agent.py::Agent.__init__`'s `self.cfg`).
private struct LayaAgentConfigFile: Decodable {
    var headLayers = 2
    var maxLen = 512
    var headMaxLen = 192
    var actCosts: [String: Double] = [:]
    var temperature: [Float] = [1, 1, 1]
    var temperatureByOptions: [String: Float] = [:]

    enum CodingKeys: String, CodingKey {
        case headLayers = "head_layers"
        case maxLen = "max_len"
        case headMaxLen = "head_max_len"
        case actCosts = "act_costs"
        case temperature
        case temperatureByOptions = "temperature_by_options"
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        headLayers = (try? c.decodeIfPresent(Int.self, forKey: .headLayers)) ?? headLayers
        maxLen = (try? c.decodeIfPresent(Int.self, forKey: .maxLen)) ?? maxLen
        headMaxLen = (try? c.decodeIfPresent(Int.self, forKey: .headMaxLen)) ?? headMaxLen
        actCosts = (try? c.decodeIfPresent([String: Double].self, forKey: .actCosts)) ?? actCosts
        temperature = (try? c.decodeIfPresent([Float].self, forKey: .temperature)) ?? temperature
        temperatureByOptions =
            (try? c.decodeIfPresent([String: Float].self, forKey: .temperatureByOptions)) ?? temperatureByOptions
    }
}

/// `model.py::EncoderConfig.from_dict` — decodes `encoder/config.json`, validates it, and
/// resolves `layer_types`/`rope_parameters` (falling back to the global/local theta pair when
/// absent, as the embedder's own checkpoint has neither).
func loadModernBERTEncoderConfig(from url: URL) throws -> (config: ModernBERTEncoderConfig, maxPositionEmbeddings: Int) {
    guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
        throw LayaEngineError.invalidConfig("encoder/config.json is not a JSON object")
    }
    func intValue(_ key: String, _ fallback: Int) -> Int { (json[key] as? Int) ?? fallback }
    func floatValue(_ key: String, _ fallback: Float) -> Float {
        if let d = json[key] as? Double { return Float(d) }
        if let i = json[key] as? Int { return Float(i) }
        return fallback
    }

    let modelType = json["model_type"] as? String ?? "modernbert"
    guard modelType == "modernbert" else {
        throw LayaEngineError.invalidConfig("Unsupported encoder: \(modelType); expected modernbert")
    }
    let hiddenActivation = json["hidden_activation"] as? String ?? "gelu"
    guard hiddenActivation == "gelu" else {
        throw LayaEngineError.invalidConfig("Unsupported encoder activation: \(hiddenActivation)")
    }
    let hiddenSize = intValue("hidden_size", 768)
    let numHeads = intValue("num_attention_heads", 12)
    guard numHeads > 0, hiddenSize % numHeads == 0, (hiddenSize / numHeads) % 2 == 0 else {
        throw LayaEngineError.invalidConfig("ModernBERT requires an even, integral attention head dimension")
    }
    let numLayers = intValue("num_hidden_layers", 22)
    let globalEvery = intValue("global_attn_every_n_layers", 3)
    let resolvedLayerTypes = (json["layer_types"] as? [String]) ?? (0 ..< numLayers).map {
        $0 % globalEvery == 0 ? "full_attention" : "sliding_attention"
    }
    guard resolvedLayerTypes.count == numLayers,
          Set(resolvedLayerTypes).isSubset(of: ["full_attention", "sliding_attention"])
    else {
        throw LayaEngineError.invalidConfig("Invalid ModernBERT layer_types")
    }
    var ropeThetaByType: [String: Float] = [:]
    if let ropeParameters = json["rope_parameters"] as? [String: Any] {
        for kind in Set(resolvedLayerTypes) {
            guard let params = ropeParameters[kind] as? [String: Any] else { continue }
            let ropeType = params["rope_type"] as? String ?? "default"
            guard ropeType == "default" else {
                throw LayaEngineError.invalidConfig("Only default (unscaled) ModernBERT RoPE is supported")
            }
            if let theta = params["rope_theta"] as? Double { ropeThetaByType[kind] = Float(theta) }
            else if let theta = params["rope_theta"] as? Int { ropeThetaByType[kind] = Float(theta) }
        }
    }
    let config = ModernBERTEncoderConfig(
        vocabularySize: intValue("vocab_size", 50368),
        hiddenSize: hiddenSize,
        intermediateSize: intValue("intermediate_size", 1152),
        numHiddenLayers: numLayers,
        numAttentionHeads: numHeads,
        normEps: floatValue("norm_eps", 1e-5),
        normBias: (json["norm_bias"] as? Bool) ?? false,
        attentionBias: (json["attention_bias"] as? Bool) ?? false,
        mlpBias: (json["mlp_bias"] as? Bool) ?? false,
        localAttention: intValue("local_attention", 128),
        globalAttnEveryNLayers: globalEvery,
        globalRopeTheta: floatValue("global_rope_theta", 160_000),
        localRopeTheta: floatValue("local_rope_theta", 10_000),
        layerTypes: resolvedLayerTypes,
        ropeThetaByType: ropeThetaByType.isEmpty ? nil : ropeThetaByType)
    return (config, intValue("max_position_embeddings", 8192))
}

/// One decision answer. Mirrors `agent.py::Agent.system_one`'s per-question answer, widened
/// slightly for the Swift side: `probabilities`/`optionLabels` are always populated (Python's
/// public JSON omits `probabilities` for `.noul`; this port keeps the two raw probabilities
/// for `LayaRunView`'s bars, LY-5). `scoreLevels` holds `renderCriterion`-rendered display
/// text — Python's `legend` exposes the raw, possibly-structured criterion value instead.
nonisolated struct LayaAnswer: Sendable {
    var type: LayaQuestionType
    var confidence: Double
    var probabilities: [Double]
    var optionLabels: [String]
    var choiceLabel: String?
    var scoreValue: Double?
    var scoreLevels: [String] = []
    var noulProbability: Double?
    /// `LayaPrompt.buildSequence`'s `stateTruncated` — the input was cut to fit `maxLen`.
    var stateTruncated: Bool
}

/// `agent.py::collate_items`, minus `pad_to_multiple`/`max_length` (this port never chunks a
/// call across HF-style dynamic padding buckets — real callers pass a handful of questions).
private struct LayaBatch {
    let inputIds: MLXArray
    let attentionMask: MLXArray
    let markerPos: MLXArray
    let markerMask: MLXArray
    let qtype: MLXArray

    init(items: [(ids: [Int], markers: [Int], qtype: Int32)], padID: Int) {
        let n = items.count
        let length = items.map(\.ids.count).max() ?? 0
        let count = max(2, items.map(\.markers.count).max() ?? 0)
        var idsBuffer = [Int32](repeating: Int32(padID), count: n * length)
        var maskBuffer = [Int32](repeating: 0, count: n * length)
        var markerPosBuffer = [Int32](repeating: 0, count: n * count)
        var markerMaskBuffer = [Int32](repeating: 0, count: n * count)
        var qtypeBuffer = [Int32](repeating: 0, count: n)
        for (row, item) in items.enumerated() {
            for (col, id) in item.ids.enumerated() { idsBuffer[row * length + col] = Int32(id) }
            for col in 0 ..< item.ids.count { maskBuffer[row * length + col] = 1 }
            for (col, marker) in item.markers.enumerated() { markerPosBuffer[row * count + col] = Int32(marker) }
            for col in 0 ..< item.markers.count { markerMaskBuffer[row * count + col] = 1 }
            qtypeBuffer[row] = item.qtype
        }
        inputIds = MLXArray(idsBuffer, [n, length])
        attentionMask = MLXArray(maskBuffer, [n, length]).asType(.bool)
        markerPos = MLXArray(markerPosBuffer, [n, count])
        markerMask = MLXArray(markerMaskBuffer, [n, count]).asType(.bool)
        qtype = MLXArray(qtypeBuffer, [n])
    }
}

/// `laya_mlx.Agent` — loads a Laya checkpoint and answers typed questions about a piece of
/// text. `predict` mirrors `agent.py::Agent.system_one` (aliased there as `predict`).
final class LayaEngine {
    private let model: LayaDecisionModel
    private let tokenizer: LayaTokenizer
    private let maxLen: Int
    private let headMaxLen: Int
    private let temperature: [Float]
    private let temperatureByOptions: [String: Float]

    private init(
        model: LayaDecisionModel, tokenizer: LayaTokenizer, maxLen: Int, headMaxLen: Int,
        temperature: [Float], temperatureByOptions: [String: Float]
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.maxLen = maxLen
        self.headMaxLen = headMaxLen
        self.temperature = temperature
        self.temperatureByOptions = temperatureByOptions
    }

    /// `agent.py::Agent.__init__`: resolve the checkpoint, validate `rl_agent_config.json` +
    /// `encoder/config.json`, clamp calibration temperatures, load the tokenizer, build
    /// `LayaDecisionModel`, sanitize + load weights (strict), cast to fp16 (this catalog
    /// entry's shipped format).
    static func load(modelDirectory: URL) async throws -> LayaEngine {
        let agentConfigURL = modelDirectory.appendingPathComponent("rl_agent_config.json")
        let encoderConfigURL = modelDirectory.appendingPathComponent("encoder/config.json")
        let weightsURL = modelDirectory.appendingPathComponent("model.safetensors")
        let fm = FileManager.default
        guard fm.fileExists(atPath: agentConfigURL.path), fm.fileExists(atPath: encoderConfigURL.path),
              fm.fileExists(atPath: weightsURL.path)
        else {
            throw LayaEngineError.incompleteCheckpoint(modelDirectory.lastPathComponent)
        }

        let agentConfigData = try Data(contentsOf: agentConfigURL)
        let rawAgentJSON = try JSONSerialization.jsonObject(with: agentConfigData) as? [String: Any] ?? [:]
        guard rawAgentJSON["encoder"] != nil, rawAgentJSON["head_layers"] != nil else {
            throw LayaEngineError.invalidConfig("Laya config must specify encoder and head_layers")
        }
        let agentConfig = try JSONDecoder().decode(LayaAgentConfigFile.self, from: agentConfigData)
        let (encoderConfig, maxPositionEmbeddings) = try loadModernBERTEncoderConfig(from: encoderConfigURL)
        guard 4 < agentConfig.headMaxLen, agentConfig.headMaxLen < agentConfig.maxLen,
              agentConfig.maxLen <= maxPositionEmbeddings
        else {
            throw LayaEngineError.invalidConfig("Expected 4 < head_max_len < max_len <= max_position_embeddings")
        }
        guard agentConfig.temperature.count == 3,
              agentConfig.temperature.allSatisfy({ $0.isFinite && $0 > 0 }),
              agentConfig.temperatureByOptions.values.allSatisfy({ $0.isFinite && $0 > 0 })
        else {
            throw LayaEngineError.invalidConfig("Calibration temperatures must be finite and positive")
        }

        let clampedTemperature = agentConfig.temperature.map { LayaCalibration.clampTemperature($0) }
        let clampedByOptions = agentConfig.temperatureByOptions.mapValues { LayaCalibration.clampTemperature($0) }
        let anyClamped = clampedByOptions.contains { clampedByOptions[$0.key] != agentConfig.temperatureByOptions[$0.key] }
            || zip(agentConfig.temperature, clampedTemperature).contains { $0 != $1 }
        if anyClamped {
            // laya-mlx warns once per Agent() construction (`agent.py::Agent.__init__`); this
            // load happens once per engine load, the Swift equivalent moment.
            NSLog(
                "Laya: this checkpoint ships calibration temperatures outside [%.1f, %.1f], "
                    + "which would distort confidence; clamping. Treat confidence from the "
                    + "affected buckets as uncalibrated.",
                LayaCalibration.temperatureMin, LayaCalibration.temperatureMax)
        }

        let tokenizer = try await LayaTokenizer.load(modelDirectory: modelDirectory)
        let model = LayaDecisionModel(
            encoderConfig: encoderConfig, headLayers: agentConfig.headLayers,
            actCostCount: agentConfig.actCosts.count)
        var weights = try MLX.loadArrays(url: weightsURL)
        weights = layaSanitizeWeights(weights)
        weights = weights.mapValues { $0.asType(.float16) }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)

        return LayaEngine(
            model: model, tokenizer: tokenizer, maxLen: agentConfig.maxLen, headMaxLen: agentConfig.headMaxLen,
            temperature: clampedTemperature, temperatureByOptions: clampedByOptions)
    }

    /// `agent.py::Agent.system_one` (aliased `predict`). Answers every question about `state`
    /// in one batch, in the same order they were given.
    func predict(state: String, questions: [LayaQuestion]) throws -> [LayaAnswer] {
        guard !questions.isEmpty else { return [] }

        var items: [(ids: [Int], markers: [Int], qtype: Int32)] = []
        var stateTruncatedByRow: [Bool] = []
        for (index, question) in questions.enumerated() {
            let (ids, markers, truncated) = buildSequence(
                tokenizer: tokenizer, state: state, question: question, maxLen: maxLen, headMaxLen: headMaxLen)
            let expectedOptionCount = renderOptions(question).count
            guard markers.count == expectedOptionCount else {
                throw LayaPromptError.tooManyOptions(questionID: "\(index)", optionCount: expectedOptionCount)
            }
            items.append((ids, markers, question.type.qtypeIndex))
            stateTruncatedByRow.append(truncated)
        }

        let batch = LayaBatch(items: items, padID: tokenizer.padTokenID)
        let (logits, _) = model(
            inputIds: batch.inputIds, attentionMask: batch.attentionMask, markerPos: batch.markerPos,
            markerMask: batch.markerMask, qtype: batch.qtype)
        eval(logits)

        var answers: [LayaAnswer] = []
        for (row, question) in questions.enumerated() {
            let k = items[row].markers.count
            let rowLogits = logits[row].asArray(Float.self)
            let bucket = LayaCalibration.temperatureBucket(type: question.type, optionCount: k)
            let scale = temperatureByOptions[bucket] ?? temperature[Int(question.type.qtypeIndex)]
            let z = rowLogits.prefix(k).map { Double($0) / Double(scale) }
            let maxZ = z.max() ?? 0
            let expZ = z.map { exp($0 - maxZ) }
            let sumExp = expZ.reduce(0, +)
            let p = expZ.map { $0 / sumExp }
            let confidence = LayaCalibration.confidence(fromProbabilities: p, optionCount: k)

            var answer = LayaAnswer(
                type: question.type, confidence: confidence, probabilities: p,
                optionLabels: renderOptions(question), stateTruncated: stateTruncatedByRow[row])
            switch question.type {
            case .choice:
                var bestIndex = 0
                for i in 1 ..< p.count where p[i] > p[bestIndex] { bestIndex = i }
                answer.choiceLabel = question.choiceCriteria[bestIndex].label
            case .score:
                let weighted = p.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
                answer.scoreValue = weighted
                answer.scoreLevels = question.scoreCriteria.map(renderCriterion)
            case .noul:
                let pTrue = p.count > 1 ? p[1] : 0
                answer.noulProbability = pTrue
                answer.confidence = max(pTrue, 1 - pTrue)
            }
            answers.append(answer)
        }
        return answers
    }
}
