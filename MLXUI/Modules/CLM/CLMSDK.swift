import Foundation

/// The loaded `CLMEngine` per model directory. CLM is ~9 GB, the heaviest non-diffusion model a flow can load.
///
/// Gate F (`RSI/DelegateCLMBacklog.md` §CL-0) kept **at most one** loaded by replacing the held engine when a
/// different directory was asked for, because nothing else could say when to release one. **SC-2b:** the
/// entry now lives on the shared `MemoryBudget` (`EngineLeaseCache`) at its real size, with a lease per use and
/// held entries skipped by eviction, so the budget decides what is evicted and when — a 9 GB engine is pushed out
/// when something else needs the room, never while a row or a Run view is using it. (The one-engine rule is
/// subsumed: a second CLM directory is the same model family, installed once.)
actor CLMEngineCache {
    static let shared = CLMEngineCache()
    private let cache: EngineLeaseCache<CLMEngine>

    init(budget: MemoryBudget = .shared) {
        cache = EngineLeaseCache(budget: budget) { directory in try await CLMEngine.load(modelDirectory: directory) }
    }

    /// Unleased (kept for callers that only need the load to have happened).
    func engine(for directory: URL) async throws -> CLMEngine { try await cache.engine(for: directory) }

    /// The engine in use until `release()`.
    func lease(for directory: URL, holder: MemoryBudget.HolderKind? = nil) async throws -> EngineLeaseCache<CLMEngine>.Lease {
        try await cache.lease(for: directory, holder: holder)
    }
}

/// `text → text` stage wrapping `CLMEngine.answer` for a single question. Mirrors
/// `Modules/Laya/LayaSDK.swift`'s `LayaStage` — real executor wiring (CL-6) branches to
/// `CLMEngine` directly rather than through this generic `Media.text` boundary, so this Stage
/// exists only to satisfy the `ModelSDK`/`ModelRegistry` contract.
nonisolated struct CLMStage: PipelineStage {
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

/// The request shape `CLMStage` accepts — the same wire convention `LayaStage`'s (private)
/// `LayaStageRequest` uses: `{"type": "choice"|"score"|"noul", "instructions": <string|json>,
/// "criteria": <json>, "state": "<free text|json>"}`. Parsed with `CLMJSONParser`, not
/// `JSONSerialization` — the latter converts every number into an `NSNumber`, losing the
/// literal token `CLMJSON.numberText` needs (`RSI/DelegateCLMBacklog.md` CL-4a).
nonisolated private struct CLMStageRequest {
    let type: LayaQuestionType
    let instructions: CLMJSON
    let criteria: CLMJSON?
    let state: CLMJSON

    init(requestJSON: String) throws {
        guard case .object(let fields) = try CLMJSONParser.parse(requestJSON) else {
            throw LayaPromptError.invalidCriteria("CLM request must be a JSON object")
        }
        func field(_ key: String) -> CLMJSON? { fields.first { $0.key == key }?.value }

        guard case .string(let typeString)? = field("type"), let type = LayaQuestionType(rawValue: typeString) else {
            throw LayaPromptError.invalidCriteria("Missing or invalid 'type' (choice/score/noul)")
        }
        guard let state = field("state") else {
            throw LayaPromptError.invalidCriteria("Missing 'state'")
        }
        self.type = type
        self.state = state
        self.instructions = field("instructions") ?? .string("")
        self.criteria = field("criteria")
    }
}

/// `ModelSDK` for CLM's decision encoder. Claims `.decision` entries backed by `source == .mlx`
/// and `family == "CLM"` — the same `runnerKind`/`source` guard `LayaSDK` uses, discriminated
/// by family (CL-1 gate B) since both share `RunnerKind.decision`.
nonisolated struct CLMSDK: ModelSDK, StageBooksOwnWeights {
    let id = "clm"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx, model.family == "CLM" else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let directory = ModelStore.shared.directory(forModelID: model.id)
        return CLMStage(id: model.id, name: model.displayName) { requestJSON in
            do {
                let request = try CLMStageRequest(requestJSON: requestJSON)
                let lease = try await CLMEngineCache.shared.lease(for: directory, holder: .flow)
                defer { lease.release() }
                let results = try await lease.value.answer(
                    state: request.state,
                    questions: [(id: "q", type: request.type, instructions: request.instructions, criteria: request.criteria)])
                guard let answer = results.first?.answer else {
                    throw LayaPromptError.invalidCriteria("CLM produced no answer")
                }
                return try Self.encode(answer)
            } catch let error as StageError {
                throw error
            } catch {
                throw StageError.engineFailure(stage: "CLM", underlying: error)
            }
        }
    }

    /// Same wire shape `LayaSDK.encode` produces — both SDKs answer with the same `LayaAnswer`.
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
