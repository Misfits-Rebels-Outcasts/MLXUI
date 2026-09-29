import Foundation

/// Ported from `Contrastive-LM/CLM`'s vendored `clm_mlx/schema.py` (shipped inside
/// `RealityCat/CLM-v0.1-8B-MLX-8bit`, itself vendored unchanged from
/// `Contrastive-LM/CLM/src/clm/schema.py` @ `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` — that
/// file's own header comment: "Do not edit: the MLX port must build exactly the same state
/// and candidate texts as upstream"). Built over `CLMJSON` (`Modules/CLM/CLMJSON.swift`) and
/// the existing `LayaQuestionType`/`LayaPromptError`/`LayaAnswer`
/// (`Modules/Laya/LayaPrompt.swift`) — CLM's schema is a state + typed question -> label +
/// probabilities shape, same as Laya's, but its `number` case must keep the literal JSON
/// token (`RSI/DelegateCLMBacklog.md` CL-4a, "Numbers: corrected 2026-09-28"), which
/// `LayaJSON.number(Double)` can't — see `CLMJSON`'s header comment.
///
/// Golden test: `MLXUITests/CLMSchemaTests.swift` against `Fixtures/CLM/schema.json`
/// (`Fixtures/CLM/PROVENANCE.md` records how it was generated) — byte-for-byte on `toText`/
/// `stateText`/`candidates`.
nonisolated enum CLMSchema {
    /// `schema.py::NOUL_KEYS = ("false", "true")`.
    static let noulKeys = ["false", "true"]

    /// `schema.py::to_text`. Renders a state/description that may be a string, object, or
    /// array as plain text: objects become `key: value` fields (a blank line between
    /// top-level fields, a newline for nested ones; key order preserved), arrays become one
    /// `- item` line per element. The heads are trained on prose, not JSON.
    static func toText(_ x: CLMJSON?, indent: Int = 0) -> String {
        guard let x else { return "" }
        switch x {
        case .null: return ""
        case .string(let s): return s
        case .bool(let b): return b ? "true" : "false"
        case .number(let token): return CLMJSON.numberText(token)
        case .object(let fields):
            let pad = String(repeating: " ", count: indent)
            let parts = fields.map { field -> String in
                if isNonEmptyContainer(field.value) {
                    return "\(pad)\(field.key):\n\(toText(field.value, indent: indent + 2))"
                }
                return "\(pad)\(field.key): \(toText(field.value))"
            }
            return parts.joined(separator: indent == 0 ? "\n\n" : "\n")
        case .array(let items):
            let pad = String(repeating: " ", count: indent)
            let parts = items.map { item -> String in
                if isNonEmptyContainer(item) {
                    return "\(pad)-\n\(toText(item, indent: indent + 2))"
                }
                return "\(pad)- \(toText(item))"
            }
            return parts.joined(separator: "\n")
        }
    }

    private static func isNonEmptyContainer(_ x: CLMJSON) -> Bool {
        switch x {
        case .object(let fields): return !fields.isEmpty
        case .array(let items): return !items.isEmpty
        default: return false
        }
    }

    /// `schema.py::state_text`. Context first, question last — the layout the heads were
    /// trained on.
    static func stateText(state: CLMJSON, instructions: CLMJSON?) -> String {
        let s = toText(state).trimmingCharacters(in: .whitespacesAndNewlines)
        let i = toText(instructions).trimmingCharacters(in: .whitespacesAndNewlines)
        if !s.isEmpty, !i.isEmpty { return "\(s)\n\n\(i)" }
        return s.isEmpty ? i : s
    }

    private static func isEmptyValue(_ value: CLMJSON?) -> Bool {
        guard let value else { return true }
        if case .string("") = value { return true }
        return false
    }

    /// `schema.py::candidates`. -> (option keys in answer order, candidate text per option).
    /// The action head embeds the option's own text: its description when one is given, else
    /// the key. Nothing is prefixed, so a candidate reaches the encoder exactly as the caller
    /// wrote it.
    static func candidates(type: LayaQuestionType, instructions: CLMJSON?, criteria: CLMJSON?)
        throws -> (keys: [String], texts: [String])
    {
        let ins = toText(instructions).trimmingCharacters(in: .whitespacesAndNewlines)
        switch type {
        case .choice:
            guard case .object(let fields) = criteria, !fields.isEmpty else {
                throw LayaPromptError.invalidCriteria("choice question needs a non-empty 'criteria' object")
            }
            let keys = fields.map(\.key)
            let texts = fields.map { field in isEmptyValue(field.value) ? field.key : toText(field.value) }
            return (keys, texts)
        case .score:
            guard case .array(let items) = criteria, items.count >= 2 else {
                throw LayaPromptError.invalidCriteria("score question needs 'criteria' as an ordered list of >= 2 levels")
            }
            let keys = (0 ..< items.count).map(String.init)
            return (keys, items.map { toText($0) })
        case .noul:
            var crit: [String: CLMJSON] = [:]
            if case .object(let fields) = criteria {
                for field in fields { crit[field.key] = field.value }
            }
            let texts = noulKeys.map { key -> String in
                var description = crit[key]
                if isEmptyValue(description) {
                    description = ins.isEmpty
                        ? .string(key)
                        : .string(key == "true" ? "Yes. This is true: \(ins)" : "No. This is false: \(ins)")
                }
                return "\(key): \(toText(description))"
            }
            return (noulKeys, texts)
        }
    }

    /// `schema.py::build_pairs`. `(state_text, option_keys, candidate_texts)` per question —
    /// the state head sees `context + question`: the state is the context, the question's
    /// instructions are appended after a blank line. Callers put the question in
    /// `instructions`, never repeated inside the state.
    static func buildPairs(
        state: CLMJSON,
        questions: [(id: String, type: LayaQuestionType, instructions: CLMJSON?, criteria: CLMJSON?)]
    ) throws -> [(id: String, stateText: String, keys: [String], texts: [String])] {
        try questions.map { entry in
            let text = stateText(state: state, instructions: entry.instructions)
            let (keys, texts) = try candidates(type: entry.type, instructions: entry.instructions, criteria: entry.criteria)
            return (entry.id, text, keys, texts)
        }
    }

    /// `schema.py::softmax`.
    static func softmax(_ logits: [Double]) -> [Double] {
        let m = logits.max() ?? 0
        let e = logits.map { exp($0 - m) }
        let z = e.reduce(0, +)
        return e.map { $0 / z }
    }

    /// `schema.py::confidence`. Top probability minus the mean of the rest, clamped to
    /// [0, 1]. Uniform across all three question types — CLM's real `engine.py::Engine.answer`
    /// calls `schema.answer_from_logits` directly with no per-type override, unlike Laya's own
    /// `LayaEngine.predict`, which overrides `.noul` confidence to `max(pTrue, 1 - pTrue)` as
    /// an app-specific choice on top of a *different* Python source (`agent.py`). Do not copy
    /// that override here.
    static func confidence(_ probs: [Double]) -> Double {
        guard probs.count >= 2 else { return 1.0 }
        var bestIndex = 0
        for i in 1 ..< probs.count where probs[i] > probs[bestIndex] { bestIndex = i }
        let rest = probs.enumerated().filter { $0.offset != bestIndex }.map(\.element)
        let restMean = rest.isEmpty ? 0 : rest.reduce(0, +) / Double(rest.count)
        return max(0, min(1, probs[bestIndex] - restMean))
    }

    /// `schema.py::answer_from_logits` -> `LayaAnswer`. `keys`/`criteria` must be the same
    /// ones `candidates(...)` returned/received for this question — `score`'s legend and
    /// `choice`'s label both read from them.
    static func answer(
        fromLogits logits: [Double], type: LayaQuestionType, criteria: CLMJSON?, keys: [String],
        stateTruncated: Bool
    ) throws -> LayaAnswer {
        try answer(
            fromProbabilities: softmax(logits), type: type, criteria: criteria, keys: keys,
            stateTruncated: stateTruncated)
    }

    /// `schema.py::answer_from_probs` -> `LayaAnswer`.
    static func answer(
        fromProbabilities probs: [Double], type: LayaQuestionType, criteria: CLMJSON?, keys: [String],
        stateTruncated: Bool
    ) throws -> LayaAnswer {
        var result = LayaAnswer(
            type: type, confidence: confidence(probs), probabilities: probs, optionLabels: keys,
            stateTruncated: stateTruncated)
        switch type {
        case .noul:
            let trueIndex = keys.firstIndex(of: "true") ?? min(1, probs.count - 1)
            result.noulProbability = probs.indices.contains(trueIndex) ? probs[trueIndex] : 0
        case .choice:
            var bestIndex = 0
            for i in 1 ..< probs.count where probs[i] > probs[bestIndex] { bestIndex = i }
            result.choiceLabel = keys[bestIndex]
        case .score:
            guard case .array(let levels) = criteria else {
                throw LayaPromptError.invalidCriteria("score answer needs the original criteria levels")
            }
            result.scoreValue = probs.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
            result.scoreLevels = levels.map { toText($0) }
        }
        return result
    }
}
