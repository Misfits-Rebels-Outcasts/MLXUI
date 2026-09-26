import Foundation

/// One cached, loaded `LayaEngine` per installed model directory — loading re-reads and
/// re-quantizes the checkpoint's weights, so a fresh `LayaEngine` per call would repeat that
/// work on every row of a flow (or every keystroke of a standalone Run). Shared with
/// `LayaRunView` (LY-5) so a flow row and the standalone Run surface don't each keep their own
/// copy of the same ~0.85 GB checkpoint in memory.
actor LayaEngineCache {
    static let shared = LayaEngineCache()
    private var engines: [String: LayaEngine] = [:]

    func engine(for directory: URL) async throws -> LayaEngine {
        let key = directory.path
        if let cached = engines[key] { return cached }
        let engine = try await LayaEngine.load(modelDirectory: directory)
        engines[key] = engine
        return engine
    }
}

/// `text → text` stage wrapping `LayaEngine.predict` for a single question. Real executor
/// wiring (LY-7, `RealExecutor.runDecider`) branches to `LayaEngine` directly rather than
/// through this generic `Media.text` boundary — deciders read the row's own criteria/tags,
/// not a caller-assembled JSON blob — so this Stage exists to satisfy the `ModelSDK` /
/// `ModelRegistry` contract (mirrors `RerankSDK`/`EmbeddingSDK`'s shape).
///
/// Request/response convention (internal to this Stage only, **not** a CAT Flow contract):
/// input is `{"type": "choice"|"score"|"noul", "instructions": <string|json>,
/// "criteria": <json>, "state": "<free text>"}`; output is the answer as JSON.
nonisolated struct LayaStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    private let ask: @Sendable (String) async throws -> String

    init(id: String, name: String, ask: @escaping @Sendable (String) async throws -> String) {
        self.id = id
        self.name = name
        self.ask = ask
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        guard case let .text(requestJSON) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.1)
        let result = try await ask(requestJSON)
        progress(1.0)
        return .text(result)
    }
}

private struct LayaStageRequest {
    let type: LayaQuestionType
    let instructions: LayaJSON
    let criteria: LayaJSON?
    let state: String

    init(json: [String: Any]) throws {
        guard let typeString = json["type"] as? String, let type = LayaQuestionType(rawValue: typeString) else {
            throw LayaPromptError.invalidCriteria("Missing or invalid 'type' (choice/score/noul)")
        }
        guard let state = json["state"] as? String else {
            throw LayaPromptError.invalidCriteria("Missing 'state'")
        }
        self.type = type
        self.state = state
        self.instructions = json["instructions"].map { LayaJSON(any: $0) } ?? .string("")
        self.criteria = json["criteria"].map { LayaJSON(any: $0) }
    }
}

extension LayaJSON {
    /// Bridges `JSONSerialization`'s `Any` into `LayaJSON`. Object key order is **not**
    /// preserved (`[String: Any]` is unordered) — acceptable only because this Stage's JSON
    /// boundary isn't the real production path; LY-7 builds `LayaQuestion` directly from a
    /// `.cat` row's declared tags, which preserves order, bypassing this bridge entirely.
    init(any value: Any) {
        if let s = value as? String { self = .string(s) }
        else if let b = value as? Bool { self = .bool(b) }
        else if let n = value as? NSNumber { self = .number(n.doubleValue) }
        else if let array = value as? [Any] { self = .array(array.map { LayaJSON(any: $0) }) }
        else if let object = value as? [String: Any] {
            self = .object(object.map { LayaJSONField(key: $0.key, value: LayaJSON(any: $0.value)) })
        } else {
            self = .null
        }
    }
}

/// `ModelSDK` for Laya's decision encoder. Claims every `.decision` entry backed by
/// `source == .mlx` — mirrors `RerankSDK`/`EmbeddingSDK`'s shape.
nonisolated struct LayaSDK: ModelSDK {
    let id = "laya"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let directory = ModelStore.shared.directory(forModelID: model.id)
        return LayaStage(id: model.id, name: model.displayName) { requestJSON in
            do {
                guard let data = requestJSON.data(using: .utf8),
                      let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                else {
                    throw LayaPromptError.invalidCriteria("Malformed Laya request")
                }
                let request = try LayaStageRequest(json: json)
                let definition = LayaQuestionDefinition(
                    type: request.type, instructions: request.instructions, criteria: request.criteria)
                let question = try definition.resolve()
                let engine = try await LayaEngineCache.shared.engine(for: directory)
                let answers = try engine.predict(state: request.state, questions: [question])
                guard let answer = answers.first else {
                    throw LayaPromptError.invalidCriteria("Laya produced no answer")
                }
                return try Self.encode(answer)
            } catch let error as StageError {
                throw error
            } catch {
                throw StageError.engineFailure(stage: "Laya", underlying: error)
            }
        }
    }

    private static func encode(_ answer: LayaAnswer) throws -> String {
        var json: [String: Any] = [
            "type": answer.type.rawValue,
            "confidence": answer.confidence,
            "probabilities": answer.probabilities,
            "optionLabels": answer.optionLabels,
            "stateTruncated": answer.stateTruncated,
        ]
        if let choiceLabel = answer.choiceLabel { json["choice"] = choiceLabel }
        if let scoreValue = answer.scoreValue {
            json["score"] = scoreValue
            json["scoreLevels"] = answer.scoreLevels
        }
        if let noulProbability = answer.noulProbability { json["noul"] = noulProbability }
        let data = try JSONSerialization.data(withJSONObject: json)
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}
