import Foundation

/// The loaded `ClefEngine` per model directory, on the shared `MemoryBudget`.
///
/// **Mirrors `CLMEngineCache`** (the contract's wording: "released on the same trigger
/// `CLMEngineCache` uses"), which by SC-2b means the modern shape rather than CLM's original
/// one-engine rule: `EngineLeaseCache` books the engine at its real on-disk size
/// (`ModelLoadProfile.charge(kind: .decision, …)`), hands out a lease per use, and skips held entries
/// when evicting — so a ~6 GB Clef engine is pushed out when something else needs the room, and never
/// while a Run view or a flow row is using it. The backlog's older "at most one Clef engine resident"
/// is subsumed: a second Clef directory is the same family installed once.
actor ClefEngineCache {
    static let shared = ClefEngineCache()
    private let cache: EngineLeaseCache<ClefEngine>

    init(budget: MemoryBudget = .shared) {
        cache = EngineLeaseCache(budget: budget) { directory in
            try await ClefEngine.load(modelDirectory: directory)
        }
    }

    /// Unleased (for callers that only need the load to have happened).
    func engine(for directory: URL) async throws -> ClefEngine { try await cache.engine(for: directory) }

    /// What a Run window's load of `directory` adds to the ledger (0 when resident).
    func bytesNeeded(directory: URL) async -> Int64 { await cache.bytesNeeded(directory: directory) }

    /// The engine in use until `release()`.
    func lease(
        for directory: URL, holder: MemoryBudget.HolderKind? = nil
    ) async throws -> EngineLeaseCache<ClefEngine>.Lease {
        try await cache.lease(for: directory, holder: holder)
    }
}

/// `text → text` stage wrapping `ClefEngine.answer` for a single question. Mirrors `CLMStage` — the
/// real executor wiring (CF-7) branches to `ClefEngine` directly rather than across this generic
/// `Media.text` boundary, so the stage exists to satisfy the `ModelSDK`/`ModelRegistry` contract.
nonisolated struct ClefStage: PipelineStage {
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

/// The request shape `ClefStage` accepts — the same wire convention `LayaStage`/`CLMStage` use:
/// `{"type": "choice"|"score"|"noul", "instructions": <string|json>, "criteria": <json>,
/// "state": <free text|json>}`. Parsed with `ClefJSONParser`, never `JSONSerialization`: the latter
/// turns every number into an `NSNumber`, losing the int/float distinction `ClefJSON.render` needs.
nonisolated private struct ClefStageRequest {
    let type: ClefQuestionType
    let instructions: ClefJSON?
    let criteria: ClefJSON?
    let state: ClefJSON

    init(requestJSON: String) throws {
        guard case .object(let fields) = try ClefJSONParser.parse(requestJSON) else {
            throw ClefError.invalidQuestion("Clef request must be a JSON object")
        }
        func field(_ key: String) -> ClefJSON? { fields.first { $0.key == key }?.value }

        guard case .string(let typeText)? = field("type") else {
            throw ClefError.invalidQuestion("Missing or invalid 'type' (choice/score/noul)")
        }
        guard let state = field("state") else {
            throw ClefError.invalidQuestion("model and state are required")
        }
        self.type = try ClefQuestionType.parse(typeText, qid: "q")
        self.state = state
        self.instructions = field("instructions")
        self.criteria = field("criteria")
    }
}

/// `ModelSDK` for Clef Flash's decision runner. Claims `.decision` entries backed by `source == .mlx`
/// and `family == "Clef"` — the same `runnerKind`/`source` guard `LayaSDK`/`CLMSDK` use, discriminated
/// by family since all three share `RunnerKind.decision`.
nonisolated struct ClefSDK: ModelSDK, StageBooksOwnWeights {
    let id = "clef"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .decision, model.source == .mlx, model.family == "Clef" else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let directory = ModelStore.shared.directory(forModelID: model.id)
        return ClefStage(id: model.id, name: model.displayName) { requestJSON in
            do {
                let request = try ClefStageRequest(requestJSON: requestJSON)
                let lease = try await ClefEngineCache.shared.lease(for: directory, holder: .flow)
                defer { lease.release() }
                let results = try await lease.value.answer(
                    state: request.state,
                    questions: [(
                        id: "q", type: request.type,
                        instructions: request.instructions, criteria: request.criteria)])
                guard let answer = results.first?.answer else {
                    throw ClefError.invalidQuestion("Clef produced no answer")
                }
                return try Self.encode(answer)
            } catch let error as StageError {
                throw error
            } catch {
                throw StageError.engineFailure(stage: "Clef", underlying: error)
            }
        }
    }

    /// The same wire shape `LayaSDK.encode` and `CLMSDK.encode` produce — all three answer with a
    /// `LayaAnswer`.
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
