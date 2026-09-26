import Foundation

/// A JSON-ish value for question instructions/criteria — Laya's `instructions`/criterion
/// descriptions may be plain strings or small structured values
/// (`laya_mlx/common.py::render_criterion`, `laya_mlx/agent.py::Agent._to_internal`'s
/// non-string-`instructions` branch). Scoped to what real callers use: CAT Flow deciders only
/// ever pass string tags/instructions (LY-7); the richer shapes exist because `laya-mlx`'s own
/// parity corpus (`benchmarks/common.py::parity_cases`, the "structured" case) exercises them
/// and the LY-4 hard gate covers all 63 of its questions.
indirect enum LayaJSON: Equatable, Sendable {
    case string(String)
    case bool(Bool)
    case number(Double)
    case null
    case array([LayaJSON])
    case object([LayaJSONField])

    /// Mirrors Python's `json.dumps(value, ensure_ascii:, separators=(", ", ": "))` — the
    /// default separators, used explicitly by `render_criterion` (`ensure_ascii=False`) and
    /// implicitly by `_to_internal`'s non-string-`instructions` branch (`ensure_ascii=True`,
    /// Python's own default).
    func jsonDumps(ensureASCII: Bool) -> String {
        switch self {
        case .string(let s): return Self.quoted(s, ensureASCII: ensureASCII)
        case .bool(let b): return b ? "true" : "false"
        case .number(let n):
            if n == n.rounded(), abs(n) < 1e15 { return String(Int64(n)) }
            return String(n)
        case .null: return "null"
        case .array(let items):
            return "[" + items.map { $0.jsonDumps(ensureASCII: ensureASCII) }.joined(separator: ", ") + "]"
        case .object(let fields):
            let body = fields
                .map { "\(Self.quoted($0.key, ensureASCII: ensureASCII)): \($0.value.jsonDumps(ensureASCII: ensureASCII))" }
                .joined(separator: ", ")
            return "{" + body + "}"
        }
    }

    private static func quoted(_ s: String, ensureASCII: Bool) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 || (ensureASCII && scalar.value > 0x7E) {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
        return out
    }
}

/// An ordered `key: value` pair — Python dicts preserve insertion order, which `json.dumps`
/// output order (and `dict.fromkeys` choice-label order) depends on.
struct LayaJSONField: Equatable, Sendable {
    let key: String
    let value: LayaJSON
}

nonisolated enum LayaQuestionType: String, Sendable {
    case choice, score, noul
}

nonisolated enum LayaPromptError: Error, CustomStringConvertible, Equatable {
    case invalidCriteria(String)
    case tooManyOptions(questionID: String, optionCount: Int)

    var description: String {
        switch self {
        case .invalidCriteria(let message): return message
        case .tooManyOptions(let id, let count):
            return "Question '\(id)' has too many options (\(count)) for the token budget"
        }
    }
}

/// `false`/`""` mean "no description" for a criterion value; `0` and `false` (as a *criterion*
/// value, not its own text) are legitimate, per `common.py::render_options`'s own comment.
private func isEmptyDescription(_ value: LayaJSON?) -> Bool {
    guard let value else { return true }
    if case .string("") = value { return true }
    return false
}

/// `common.py::render_criterion`. Strings pass through; anything structured becomes compact
/// JSON so a rubric reads as JSON rather than a Python repr.
func renderCriterion(_ value: LayaJSON) -> String {
    if case .string(let s) = value { return s }
    return value.jsonDumps(ensureASCII: false)
}

/// The internal, already-validated question — `laya_mlx/agent.py::Agent._to_internal`'s
/// output shape (`{"t": kind, "ins": instructions, "crit": criteria}`), with `instructions`
/// already resolved to a string.
nonisolated struct LayaQuestion: Sendable {
    var type: LayaQuestionType
    var instructions: String
    /// `.choice` only: ordered (label, description) pairs — `dict.fromkeys`-preserved order.
    var choiceCriteria: [(label: String, description: LayaJSON?)] = []
    /// `.score` only: ordered level descriptions.
    var scoreCriteria: [LayaJSON] = []
    /// `.noul` only.
    var noulFalseDescription: LayaJSON?
    var noulTrueDescription: LayaJSON?
}

/// The raw, possibly-structured question a caller builds — mirrors upstream's dynamically
/// typed input before `_to_internal`'s validation runs. `resolve()` is that validation, ported.
nonisolated struct LayaQuestionDefinition: Sendable {
    var type: LayaQuestionType
    var instructions: LayaJSON
    var criteria: LayaJSON?

    /// `agent.py::Agent._to_internal`, minus the `qdef.get("type")` dict-shape check (the
    /// caller already supplies a typed `LayaQuestionType`).
    func resolve() throws -> LayaQuestion {
        let instructionsText: String
        if case .string(let s) = instructions {
            instructionsText = s
        } else {
            // `if not isinstance(instructions, str): instructions = json.dumps(instructions)`
            // — Python's own default `ensure_ascii=True`, not `render_criterion`'s False.
            instructionsText = instructions.jsonDumps(ensureASCII: true)
        }

        switch type {
        case .choice:
            var pairs: [(label: String, description: LayaJSON?)] = []
            switch criteria {
            case .object(let fields):
                pairs = fields.map { (label: $0.key, description: isEmptyDescription($0.value) ? nil : $0.value) }
            case .array(let items):
                // `dict.fromkeys(criteria)` — each label maps to no description.
                var seen = Set<String>()
                for item in items {
                    guard case .string(let label) = item else {
                        throw LayaPromptError.invalidCriteria("Choice labels must be strings")
                    }
                    guard seen.insert(label).inserted else {
                        throw LayaPromptError.invalidCriteria("Choice labels must be unique")
                    }
                    pairs.append((label: label, description: nil))
                }
            default:
                throw LayaPromptError.invalidCriteria("Choice criteria must be a nonempty dictionary or list")
            }
            guard !pairs.isEmpty else {
                throw LayaPromptError.invalidCriteria("Choice criteria must be a nonempty dictionary or list")
            }
            return LayaQuestion(type: .choice, instructions: instructionsText, choiceCriteria: pairs)

        case .score:
            guard case .array(let items) = criteria, !items.isEmpty else {
                throw LayaPromptError.invalidCriteria("Score criteria must be a nonempty list")
            }
            return LayaQuestion(type: .score, instructions: instructionsText, scoreCriteria: items)

        case .noul:
            var falseDescription: LayaJSON?
            var trueDescription: LayaJSON?
            if case .object(let fields) = criteria {
                falseDescription = fields.first { $0.key == "false" }?.value
                trueDescription = fields.first { $0.key == "true" }?.value
            }
            return LayaQuestion(
                type: .noul, instructions: instructionsText,
                noulFalseDescription: falseDescription, noulTrueDescription: trueDescription)
        }
    }
}

/// `common.py::render_options`. Renders option texts in label-index order; `.noul` is always
/// `[false, true]`.
func renderOptions(_ q: LayaQuestion) -> [String] {
    switch q.type {
    case .choice:
        return q.choiceCriteria.map { pair in
            guard let description = pair.description else { return pair.label }
            return "\(pair.label): \(renderCriterion(description))"
        }
    case .score:
        return q.scoreCriteria.enumerated().map { index, criterion in
            "level \(index): \(renderCriterion(criterion))"
        }
    case .noul:
        let falseText = isEmptyDescription(q.noulFalseDescription)
            ? "no, the statement does not hold" : renderCriterion(q.noulFalseDescription!)
        let trueText = isEmptyDescription(q.noulTrueDescription)
            ? "yes, the statement holds" : renderCriterion(q.noulTrueDescription!)
        return ["false: " + falseText, "true: " + trueText]
    }
}

/// The minimal tokenizer surface `LayaPrompt` needs. Real backing is `LayaTokenizer`
/// (`LayaEngine.swift`); test backing is a fixed-vocabulary fake (`LayaPromptTests`).
protocol LayaTokenizing {
    var clsTokenID: Int { get }
    var sepTokenID: Int { get }
    var maskTokenID: Int { get }
    var maskToken: String { get }
    /// `tok(text, add_special_tokens=False)["input_ids"]` — every call site in
    /// `laya_mlx/common.py` passes `add_special_tokens=False`.
    func encode(_ text: String) -> [Int]
}

/// `common.py::build_prefix`. Builds the question-only prefix, before state tokens and final
/// truncation: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP]`.
func buildPrefix(
    tokenizer: LayaTokenizing, question: LayaQuestion, headMaxLen: Int = 192,
    optionOrder: [Int]? = nil
) -> (ids: [Int], markers: [Int]) {
    let opts = renderOptions(question)
    let order = optionOrder ?? Array(0 ..< opts.count)
    let maskToken = tokenizer.maskToken
    let instructions = question.instructions.replacingOccurrences(of: maskToken, with: " ")
    var headIDs = tokenizer.encode("\(question.type.rawValue) question: \(instructions)")
    var optIDs: [[Int]] = order.map { index in
        let text = " " + opts[index].replacingOccurrences(of: maskToken, with: " ")
        return [tokenizer.maskTokenID] + Array(tokenizer.encode(text).prefix(48))
    }
    var optBudget = headMaxLen - optIDs.reduce(0) { $0 + $1.count }
    if optBudget < 16 {
        let per = max(4, (headMaxLen - 16) / max(1, optIDs.count))
        optIDs = optIDs.map { Array($0.prefix(per)) }
        optBudget = headMaxLen - optIDs.reduce(0) { $0 + $1.count }
    }
    headIDs = Array(headIDs.prefix(max(8, optBudget)))
    var ids = [tokenizer.clsTokenID] + headIDs + [tokenizer.sepTokenID]
    var markers: [Int] = []
    for option in optIDs {
        markers.append(ids.count)
        ids.append(contentsOf: option)
    }
    ids.append(tokenizer.sepTokenID)
    return (ids, markers)
}

/// `common.py::build_sequence`. Format: `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK]
/// opt1 … [SEP] state [SEP]`. `truncateLeft` mirrors the Python parameter for completeness,
/// but no real caller sets it `true` (`Agent.prepare` never passes it) — Python's own
/// `st[-room:]` has a `room == 0` quirk (`arr[-0:]` returns the *whole* array, not empty) that
/// is therefore unreachable in practice and not replicated here.
func buildSequence(
    tokenizer: LayaTokenizing, state: String, question: LayaQuestion, maxLen: Int = 512,
    headMaxLen: Int = 192, optionOrder: [Int]? = nil, truncateLeft: Bool = false
) -> (ids: [Int], markers: [Int], stateTruncated: Bool) {
    let (prefixIDs, markers) = buildPrefix(
        tokenizer: tokenizer, question: question, headMaxLen: headMaxLen, optionOrder: optionOrder)
    let room = max(0, maxLen - prefixIDs.count - 1)
    let stateText = state.replacingOccurrences(of: tokenizer.maskToken, with: " ")
    let stateIDs = tokenizer.encode(stateText)
    let stateTruncated = stateIDs.count > room
    let keptStateIDs = truncateLeft ? Array(stateIDs.suffix(room)) : Array(stateIDs.prefix(room))
    var ids = prefixIDs + keptStateIDs + [tokenizer.sepTokenID]
    ids = Array(ids.prefix(maxLen))
    let clippedMarkers = markers.filter { $0 < maxLen }
    return (ids, clippedMarkers, stateTruncated)
}
