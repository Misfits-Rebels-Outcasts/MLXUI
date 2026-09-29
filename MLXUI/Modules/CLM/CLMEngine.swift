import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon

/// Ported from `RealityCat/CLM-v0.1-8B-MLX-8bit`'s vendored `clm_mlx/engine.py` — `Engine`.
/// An actor (not a plain class): it owns the loaded encoder container and heads, both
/// single-instance and not safe to call concurrently from two callers at once (the encoder
/// container already serializes internally, per `EmbedderModelContainer`'s own doc comment,
/// but the `(head, text)` cache below is this port's own state and needs the same
/// serialization).
///
/// Golden test: `MLXUITests/CLMEngineTests.swift` against `Fixtures/CLM/answers.json`.
actor CLMEngine {
    private struct CacheKey: Hashable {
        var which: CLMHeadPair.Which
        var text: String
    }

    private struct CachedVector {
        var vector: [Float]
        var truncated: Bool
    }

    private let container: EmbedderModelContainer
    private let heads: CLMHeadPair
    private let cacheSize: Int
    private var cache: [CacheKey: CachedVector] = [:]
    /// Oldest first — `engine.py`'s `OrderedDict` insertion/move-to-end order.
    private var cacheOrder: [CacheKey] = []
    private(set) var tokensSpent = 0

    private init(container: EmbedderModelContainer, heads: CLMHeadPair, cacheSize: Int) {
        self.container = container
        self.heads = heads
        self.cacheSize = cacheSize
    }

    /// `engine.py::Engine.__init__`. `modelDirectory` is the installed model's root
    /// (`encoder/` + `heads/` underneath it, per `RealityCat/CLM-v0.1-8B-MLX-8bit`'s layout).
    static func load(modelDirectory: URL, cacheSize: Int = 50_000) async throws -> CLMEngine {
        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: modelDirectory.appendingPathComponent("encoder"), using: HFTokenizerLoader())
        let heads = try CLMHeadPair(headsDirectory: modelDirectory.appendingPathComponent("heads"))
        return CLMEngine(container: container, heads: heads, cacheSize: cacheSize)
    }

    private func moveToEnd(_ key: CacheKey) {
        if let index = cacheOrder.firstIndex(of: key) {
            cacheOrder.remove(at: index)
        }
        cacheOrder.append(key)
    }

    /// `engine.py::Engine._vectors`. Embeds only the texts not already cached, evicts the
    /// oldest entries once over `cacheSize`, marks every requested text as recently used, and
    /// returns one `(vector, truncated)` per input text, in the given order (duplicates
    /// allowed — a caller may ask for the same text twice, e.g. as both a state and an action).
    private func vectors(for texts: [String], which: CLMHeadPair.Which) async throws -> [CachedVector] {
        var missing: [String] = []
        var seen = Set<String>()
        for text in texts where cache[CacheKey(which: which, text: text)] == nil {
            if seen.insert(text).inserted { missing.append(text) }
        }

        if !missing.isEmpty {
            let missingTexts = missing
            let localHeads = heads
            let (rows, tokenCount, truncatedFlags): ([[Float]], Int, [Bool]) =
                await container.perform { context in
                    let (embeddings, spent, truncated) = CLMEncoder.embed(texts: missingTexts, context: context)
                    let projected = localHeads.project(embeddings, which: which)
                    eval(projected)
                    let rows = (0 ..< projected.dim(0)).map { row in
                        projected[row ..< row + 1].squeezed(axis: 0).asArray(Float.self)
                    }
                    return (rows, spent, truncated)
                }
            tokensSpent += tokenCount
            for (index, text) in missing.enumerated() {
                let key = CacheKey(which: which, text: text)
                cache[key] = CachedVector(vector: rows[index], truncated: truncatedFlags[index])
                cacheOrder.append(key)
            }
            while cache.count > cacheSize, !cacheOrder.isEmpty {
                cache.removeValue(forKey: cacheOrder.removeFirst())
            }
        }

        for text in texts { moveToEnd(CacheKey(which: which, text: text)) }
        return texts.map { cache[CacheKey(which: which, text: $0)] ?? CachedVector(vector: [], truncated: false) }
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Double {
        var sum = 0.0
        for i in 0 ..< min(a.count, b.count) { sum += Double(a[i]) * Double(b[i]) }
        return sum
    }

    /// `engine.py::Engine.answer`. One `LayaAnswer` per question, in the given order.
    /// `temperature` must be in `(0, 100]`, matching upstream's own validation.
    func answer(
        state: CLMJSON,
        questions: [(id: String, type: LayaQuestionType, instructions: CLMJSON?, criteria: CLMJSON?)],
        temperature: Double = 1.0
    ) async throws -> [(id: String, answer: LayaAnswer)] {
        guard !questions.isEmpty else {
            throw LayaPromptError.invalidCriteria("questions must not be empty")
        }
        guard temperature > 0, temperature <= 100 else {
            throw LayaPromptError.invalidCriteria("temperature must be in (0, 100]")
        }

        let pairs = try CLMSchema.buildPairs(
            state: state,
            questions: questions.map { (id: $0.id, type: $0.type, instructions: $0.instructions, criteria: $0.criteria) })

        let stateVectors = try await vectors(for: pairs.map(\.stateText), which: .state)
        let actionVectors = try await vectors(for: pairs.flatMap(\.texts), which: .action)

        var results: [(id: String, answer: LayaAnswer)] = []
        var cursor = 0
        for (index, pair) in pairs.enumerated() {
            let optionCount = pair.texts.count
            let optionVectors = Array(actionVectors[cursor ..< cursor + optionCount])
            cursor += optionCount

            let stateVector = stateVectors[index]
            let logits = optionVectors.map { Double(heads.scale) * dot($0.vector, stateVector.vector) / temperature }
            let question = questions[index]
            let ans = try CLMSchema.answer(
                fromLogits: logits, type: question.type, criteria: question.criteria, keys: pair.keys,
                stateTruncated: stateVector.truncated)
            results.append((pair.id, ans))
        }
        return results
    }

    /// `engine.py::Engine.rank`. A `choice` over `candidates` keyed by their index, ordered by
    /// probability descending.
    func rank(
        state: CLMJSON, candidates: [String], instructions: CLMJSON? = nil, temperature: Double = 1.0
    ) async throws -> [(rank: Int, candidate: String, probability: Double)] {
        let criteria = CLMJSON.object(
            candidates.enumerated().map { CLMJSONField(key: String($0.offset), value: .string($0.element)) })
        let results = try await answer(
            state: state, questions: [(id: "rank", type: .choice, instructions: instructions, criteria: criteria)],
            temperature: temperature)
        guard let rankAnswer = results.first?.answer else {
            throw LayaPromptError.invalidCriteria("rank produced no answer")
        }
        let ordered = zip(rankAnswer.optionLabels, rankAnswer.probabilities).sorted { $0.1 > $1.1 }
        return ordered.enumerated().map { offset, pair in
            (rank: offset + 1, candidate: candidates[Int(pair.0) ?? 0], probability: pair.1)
        }
    }
}
