import Foundation
import MLXLMCommon

/// Clef Flash's prompt builder — a from-reference port of `clef_mlx.py`, the loader shipped in
/// `mlx-community/clef-flash-4bit` @ `b5adbe2c3a95200d4bf04a0727c8ec457157d5e1` (Apache-2.0),
/// fetched to `RSI/notes/clef/hf/` (rule 18). The port map is `RSI/notes/clef/port-map.md` §1.
///
/// Every type below names the reference function it ports:
/// - `SYSTEM_PROMPT` `:29–32` → ``ClefPrompt/systemPrompt``
/// - `render` `:67–70` → ``ClefJSON/render(_:)``
/// - `question_options` `:73–84` → ``ClefPrompt/questionOptions(_:)``
/// - `_tokens` `:103–104` → ``ClefTokenizer/encode(_:)`` (per fragment, `add_special_tokens=False`)
/// - `EncodedQuestion` / `EncodedRecord` `:87–100` → ``ClefEncodedQuestion`` / ``ClefEncodedRecord``
/// - `encode_record` `:179–254` → ``ClefPrompt/encodeRecord(tokenizer:record:maxLength:maxStateTokens:truncate:)``
/// - `ContextTooLong` `:40` → ``ClefError``
/// - `QUESTION_TYPES` `:33` → ``ClefQuestionType/index``
///
/// Out of scope by gate A1 (text only): `_encode_media` `:131–176`, so `media_ids` is always empty
/// and the span offset never includes media tokens (port map §1.6).
/// `stateTruncated` is **app-side**, not a port (CF-A0 #3, readme §Corrections): the reference
/// truncates silently and `EncodedRecord` has no such field.

// MARK: - Errors

/// `clef_mlx.py::ContextTooLong` (`:40`) plus the validation failures `systemone` raises
/// (`:495–504`). The messages are the reference's, verbatim, so a run log can be diffed against
/// the Python line for line.
nonisolated enum ClefError: Error, Equatable, CustomStringConvertible {
    /// `f"schema requires {fixed} tokens before state; maximum is {max_length}"`
    case schemaTooLong(fixed: Int, maxLength: Int)
    /// `f"request needs {fixed + len(state)} tokens; maximum is {max_length}"`
    case requestTooLong(needed: Int, maxLength: Int)
    /// Python's `QUESTION_TYPES[type]` raises `KeyError` for anything else.
    case unknownQuestionType(String)
    /// `f"{qid}: type must be noul, choice, or score"`, `f"{qid}: criteria must not be empty"`, …
    case invalidQuestion(String)
    /// `"at least one question is required"`
    case emptyQuestions

    var description: String {
        switch self {
        case .schemaTooLong(let fixed, let maxLength):
            return "schema requires \(fixed) tokens before state; maximum is \(maxLength)"
        case .requestTooLong(let needed, let maxLength):
            return "request needs \(needed) tokens; maximum is \(maxLength)"
        case .unknownQuestionType(let type):
            return "\(type): type must be noul, choice, or score"
        case .invalidQuestion(let message):
            return message
        case .emptyQuestions:
            return "at least one question is required"
        }
    }
}

// MARK: - JSON (`render`, `:67–70`)

/// The value type `render` serialises. Deliberately **not** `CLMJSON` (backlog T5): Clef's state
/// is arbitrary JSON handed to it by a caller, so this type carries its own ordered parser, and
/// its numbers keep Python's `int`/`float` distinction — which `json.dumps` renders differently
/// (`1` vs `1.0`).
nonisolated indirect enum ClefJSON: Equatable, Sendable {
    case string(String)
    case int(Int)
    /// An integer literal outside `Int64`. Python's `int` is arbitrary precision and
    /// `json.dumps` prints it verbatim, so keep the digits rather than approximating.
    case bigInt(String)
    case double(Double)
    case bool(Bool)
    case null
    case array([ClefJSON])
    /// Document order, like Python's `dict`; ``render(_:)`` sorts (`sort_keys=True`).
    case object([ClefJSONField])

    /// Python truthiness, for `question.get("instructions") or str(question_id)` (`:199`).
    var isPyFalsy: Bool {
        switch self {
        case .null: return true
        case .bool(let b): return !b
        case .int(let i): return i == 0
        case .bigInt(let t): return t.allSatisfy { $0 == "0" || $0 == "-" }
        case .double(let d): return d == 0
        case .string(let s): return s.isEmpty
        case .array(let a): return a.isEmpty
        case .object(let o): return o.isEmpty
        }
    }
}

nonisolated struct ClefJSONField: Equatable, Sendable {
    let key: String
    let value: ClefJSON
}

extension ClefJSON {
    /// `render` (`:67–70`): a Python `str` comes back **as-is** — a plain-string state is not
    /// quoted — and anything else goes through `json.dumps`.
    nonisolated static func render(_ value: ClefJSON) -> String {
        if case .string(let text) = value { return text }
        return jsonText(value)
    }

    /// `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)` — the form
    /// nested values always take, where strings *are* quoted.
    nonisolated static func jsonText(_ value: ClefJSON) -> String {
        switch value {
        case .string(let s):
            return quoted(s)
        case .int(let i):
            return String(i)
        case .bigInt(let text):
            return text
        case .double(let d):
            return pythonFloat(d)
        case .bool(let b):
            return b ? "true" : "false"
        case .null:
            return "null"
        case .array(let items):
            return "[" + items.map(jsonText).joined(separator: ",") + "]"
        case .object(let fields):
            let sorted = fields.sorted { codePointLess($0.key, $1.key) }
            return "{" + sorted.map { quoted($0.key) + ":" + jsonText($0.value) }.joined(separator: ",") + "}"
        }
    }

    /// CPython's `repr(float)`, which is what `json.dumps` emits.
    ///
    /// Swift's `Double.description` is shortest-round-trip **and** byte-identical to `repr()` on
    /// every value it was checked against — including the notation thresholds (`1e15` fixed,
    /// `1e16` exponential, `1e-4`/`1e-5`, `-0.0`, `5e-324`, `1.7976931348623157e+308`) — so the
    /// only work here is Python's spellings for the non-finite values (`json.dumps` writes them
    /// with `allow_nan=True`). `ClefPromptTests.jsonNumberRenderingMatchesPython` pins the table.
    nonisolated static func pythonFloat(_ value: Double) -> String {
        if value.isNaN { return "NaN" }
        if value == .infinity { return "Infinity" }
        if value == -.infinity { return "-Infinity" }
        return value.description
    }

    /// Python sorts `str` by **code point**. Swift's `String.<` uses Unicode canonical ordering,
    /// which disagrees for e.g. `"A"` vs `"a"` vs `"Å"` — and `sort_keys=True` is part of the
    /// prompt's bytes.
    nonisolated static func codePointLess(_ a: String, _ b: String) -> Bool {
        var left = a.unicodeScalars.makeIterator()
        var right = b.unicodeScalars.makeIterator()
        while true {
            switch (left.next(), right.next()) {
            case (nil, nil): return false
            case (nil, _): return true
            case (_, nil): return false
            case (.some(let x), .some(let y)):
                if x.value != y.value { return x.value < y.value }
            }
        }
    }

    /// `json.dumps(..., ensure_ascii=False)`'s escaping: `"` and `\`, the five short control
    /// escapes, and every other code unit below `0x20` as lowercase `\u00xx`. Non-ASCII stays
    /// literal.
    nonisolated static func quoted(_ s: String) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}

// MARK: - Ordered JSON parser

/// The parser behind ``ClefJSON``. Ordered on purpose: the prompt emits the schema **in the
/// request's question order** (`encode_record` `:191`), so a `[String: Any]` decode would lose
/// the FIELD numbering (port map §1.4).
nonisolated enum ClefJSONError: Error, Equatable, CustomStringConvertible {
    case unexpectedCharacter(Character, at: Int)
    case unexpectedEnd
    case invalidEscape(at: Int)
    case invalidNumber(String)
    case trailingContent(at: Int)

    var description: String {
        switch self {
        case .unexpectedCharacter(let c, let i): return "unexpected character '\(c)' at \(i)"
        case .unexpectedEnd: return "unexpected end of JSON"
        case .invalidEscape(let i): return "invalid escape at \(i)"
        case .invalidNumber(let text): return "invalid number '\(text)'"
        case .trailingContent(let i): return "trailing content at \(i)"
        }
    }
}

nonisolated enum ClefJSONParser {
    static func parse(_ text: String) throws -> ClefJSON {
        var chars = Array(text)
        var index = 0
        skipWhitespace(&chars, &index)
        let value = try parseValue(&chars, &index)
        skipWhitespace(&chars, &index)
        guard index == chars.count else { throw ClefJSONError.trailingContent(at: index) }
        return value
    }

    private static func skipWhitespace(_ chars: inout [Character], _ index: inout Int) {
        while index < chars.count, chars[index] == " " || chars[index] == "\n" || chars[index] == "\r"
            || chars[index] == "\t"
        {
            index += 1
        }
    }

    private static func parseValue(_ chars: inout [Character], _ index: inout Int) throws -> ClefJSON {
        guard index < chars.count else { throw ClefJSONError.unexpectedEnd }
        switch chars[index] {
        case "{": return try parseObject(&chars, &index)
        case "[": return try parseArray(&chars, &index)
        case "\"": return .string(try parseString(&chars, &index))
        case "t": try expect("true", &chars, &index); return .bool(true)
        case "f": try expect("false", &chars, &index); return .bool(false)
        case "n": try expect("null", &chars, &index); return .null
        case "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9":
            return try parseNumber(&chars, &index)
        default:
            throw ClefJSONError.unexpectedCharacter(chars[index], at: index)
        }
    }

    private static func expect(_ literal: String, _ chars: inout [Character], _ index: inout Int) throws {
        for character in literal {
            guard index < chars.count, chars[index] == character else {
                throw ClefJSONError.unexpectedEnd
            }
            index += 1
        }
    }

    private static func parseObject(_ chars: inout [Character], _ index: inout Int) throws -> ClefJSON {
        index += 1  // "{"
        var fields: [ClefJSONField] = []
        skipWhitespace(&chars, &index)
        if index < chars.count, chars[index] == "}" { index += 1; return .object(fields) }
        while true {
            skipWhitespace(&chars, &index)
            guard index < chars.count, chars[index] == "\"" else {
                throw ClefJSONError.unexpectedCharacter(index < chars.count ? chars[index] : " ", at: index)
            }
            let key = try parseString(&chars, &index)
            skipWhitespace(&chars, &index)
            guard index < chars.count, chars[index] == ":" else {
                throw ClefJSONError.unexpectedCharacter(index < chars.count ? chars[index] : " ", at: index)
            }
            index += 1
            skipWhitespace(&chars, &index)
            fields.append(ClefJSONField(key: key, value: try parseValue(&chars, &index)))
            skipWhitespace(&chars, &index)
            guard index < chars.count else { throw ClefJSONError.unexpectedEnd }
            if chars[index] == "," { index += 1; continue }
            if chars[index] == "}" { index += 1; return .object(fields) }
            throw ClefJSONError.unexpectedCharacter(chars[index], at: index)
        }
    }

    private static func parseArray(_ chars: inout [Character], _ index: inout Int) throws -> ClefJSON {
        index += 1  // "["
        var items: [ClefJSON] = []
        skipWhitespace(&chars, &index)
        if index < chars.count, chars[index] == "]" { index += 1; return .array(items) }
        while true {
            skipWhitespace(&chars, &index)
            items.append(try parseValue(&chars, &index))
            skipWhitespace(&chars, &index)
            guard index < chars.count else { throw ClefJSONError.unexpectedEnd }
            if chars[index] == "," { index += 1; continue }
            if chars[index] == "]" { index += 1; return .array(items) }
            throw ClefJSONError.unexpectedCharacter(chars[index], at: index)
        }
    }

    private static func parseString(_ chars: inout [Character], _ index: inout Int) throws -> String {
        index += 1  // opening quote
        var out = String.UnicodeScalarView()
        while index < chars.count {
            let character = chars[index]
            if character == "\"" { index += 1; return String(out) }
            if character == "\\" {
                index += 1
                guard index < chars.count else { throw ClefJSONError.unexpectedEnd }
                switch chars[index] {
                case "\"": out.append("\""); index += 1
                case "\\": out.append("\\"); index += 1
                case "/": out.append("/"); index += 1
                case "b": out.append(Unicode.Scalar(0x08)); index += 1
                case "f": out.append(Unicode.Scalar(0x0C)); index += 1
                case "n": out.append("\n"); index += 1
                case "r": out.append("\r"); index += 1
                case "t": out.append("\t"); index += 1
                case "u":
                    let start = index - 1
                    index += 1
                    guard index + 4 <= chars.count,
                          let code = UInt32(String(chars[index ..< index + 4]), radix: 16),
                          let scalar = Unicode.Scalar(code)
                    else { throw ClefJSONError.invalidEscape(at: start) }
                    out.append(scalar)
                    index += 4
                default:
                    throw ClefJSONError.invalidEscape(at: index)
                }
                continue
            }
            out.append(contentsOf: character.unicodeScalars)
            index += 1
        }
        throw ClefJSONError.unexpectedEnd
    }

    /// Python's `json.loads` keeps `int` and `float` apart: `1` → `int`, `1.0`/`1e2` → `float`.
    private static func parseNumber(_ chars: inout [Character], _ index: inout Int) throws -> ClefJSON {
        let start = index
        var isFloat = false
        while index < chars.count {
            let character = chars[index]
            if character.isNumber {
                index += 1
            } else if character == "-" || character == "+" {
                index += 1
            } else if character == "." || character == "e" || character == "E" {
                isFloat = true
                index += 1
            } else {
                break
            }
        }
        let text = String(chars[start ..< index])
        if isFloat {
            guard let value = Double(text) else { throw ClefJSONError.invalidNumber(text) }
            return .double(value)
        }
        if let value = Int(text) { return .int(value) }
        guard text.contains(where: \.isNumber) else { throw ClefJSONError.invalidNumber(text) }
        return .bigInt(text)
    }
}

// MARK: - Requests

/// `QUESTION_TYPES` (`:33`) and `systemone`'s type validation (`:501–502`).
nonisolated enum ClefQuestionType: String, Sendable, CaseIterable {
    case noul
    case choice
    case score

    var index: Int {
        switch self {
        case .noul: return 0
        case .choice: return 1
        case .score: return 2
        }
    }

    static func parse(_ raw: String, qid: String) throws -> ClefQuestionType {
        guard let type = ClefQuestionType(rawValue: raw) else {
            throw ClefError.invalidQuestion("\(qid): type must be noul, choice, or score")
        }
        return type
    }
}

/// One entry of a request's `questions` object (`:191–223`).
nonisolated struct ClefQuestion: Equatable, Sendable {
    let id: String
    let type: ClefQuestionType
    /// `question.get("instructions")` — `nil` when absent, and `render`d when present.
    let instructions: ClefJSON?
    /// `choice`: an object of option-id → description; `score`: an array of descriptions;
    /// `noul`: optional `{"true": …, "false": …}` overrides. `nil`/`null` for a bare `choice` or
    /// `score` is what `systemone` refuses (`:503`).
    let criteria: ClefJSON?
}

/// The `{state, questions}` half of a Jev/SystemOne request body (`:493–517`).
nonisolated struct ClefRequest: Equatable, Sendable {
    let state: ClefJSON
    /// **Request order** — Python dict order, which is the FIELD numbering (`:191`).
    let questions: [ClefQuestion]

    /// The `questions` object's document order is the port's FIELD order, so this cannot go
    /// through `[String: Any]`.
    init(json: ClefJSON) throws {
        guard case .object(let root) = json else {
            throw ClefError.invalidQuestion("request body must be a JSON object")
        }
        guard let state = root.first(where: { $0.key == "state" })?.value else {
            throw ClefError.invalidQuestion("model and state are required")
        }
        guard case .object(let questionFields)? = root.first(where: { $0.key == "questions" })?.value else {
            throw ClefError.emptyQuestions
        }
        var questions: [ClefQuestion] = []
        for field in questionFields {
            guard case .object(let body) = field.value else {
                throw ClefError.invalidQuestion("\(field.key): type must be noul, choice, or score")
            }
            let rawType = body.first(where: { $0.key == "type" })?.value
            guard case .string(let typeText)? = rawType else {
                throw ClefError.invalidQuestion("\(field.key): type must be noul, choice, or score")
            }
            let type = try ClefQuestionType.parse(typeText, qid: field.key)
            let instructions = body.first(where: { $0.key == "instructions" })?.value
            let criteria = body.first(where: { $0.key == "criteria" })?.value
            questions.append(
                ClefQuestion(id: field.key, type: type, instructions: instructions, criteria: criteria))
        }
        guard !questions.isEmpty else { throw ClefError.emptyQuestions }
        self.state = state
        self.questions = questions
    }

    init(jsonText: String) throws {
        try self.init(json: ClefJSONParser.parse(jsonText))
    }
}

// MARK: - Encoding (`encode_record`, `:179–254`)

/// `EncodedQuestion` (`:87–93`).
nonisolated struct ClefEncodedQuestion: Equatable, Sendable {
    let questionID: String
    /// `QUESTION_TYPES[type]` — the head's `type_embedding` index.
    let type: Int
    let questionSpan: Range<Int>
    let optionSpans: [Range<Int>]
    /// Option ids in **encode order** (`choice`: id-sorted; `score`: list order).
    let optionIDs: [String]
}

/// `EncodedRecord` (`:96–100`), plus the app-side truncation numbers.
nonisolated struct ClefEncodedRecord: Equatable, Sendable {
    let inputIDs: [Int]
    let questions: [ClefEncodedQuestion]
    let stateTokensKept: Int
    let stateTokensTotal: Int

    /// CF-A0 #3 / readme §Corrections: the reference truncates silently, so this flag is the
    /// app's (`kept < total`), carried through to `ClefAnswer.stateTruncated`.
    var stateTruncated: Bool { stateTokensKept < stateTokensTotal }
}

/// `_tokens(tokenizer, text)` (`:103–104`): `tokenizer(text, add_special_tokens=False).input_ids`,
/// called **once per fragment** with the results concatenated (port map §1.2, readme T4). A whole
/// prompt encoded in one call would move every BPE boundary and every span.
nonisolated protocol ClefTokenizer: Sendable {
    func encode(_ text: String) -> [Int]
}

/// The app's tokenizer (`Core/HFTokenizerLoader.swift`, swift-transformers) behind the seam above.
/// The reference unwraps mlx-lm's `TokenizerWrapper` to the HF fast tokenizer (`:439–440`).
nonisolated struct ClefMLXTokenizer: ClefTokenizer, @unchecked Sendable {
    let upstream: any MLXLMCommon.Tokenizer

    func encode(_ text: String) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: false)
    }
}

/// `encode_record` (`:179–254`).
nonisolated enum ClefPrompt {
    /// `SYSTEM_PROMPT` (`:29–32`) — verbatim, one line, no trailing space.
    static let systemPrompt = """
        Read the complete state and schema. Decide every field jointly. Each answer must be \
        exactly one of that field's allowed options.
        """

    /// The user turn that opens the state (`:226`).
    static let statePrefix = "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n<|im_start|>user\nSTATE:\n"

    /// The assistant turn that closes the prompt (`:228–231`).
    static let assistantSuffix =
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"

    /// The schema preamble (`:189`).
    static let schemaHeader = "\n\nSCHEMA FIELDS:\n"

    /// `question_options`'s built-in `noul` descriptions (`:77–78`).
    static let noulTrueDescription = "The proposition is true or the answer is yes."
    static let noulFalseDescription = "The proposition is false or the answer is no."

    /// Default `max_length` (`:182`, `:493`) — gate F1.
    static let defaultMaxLength = 16_384

    /// `question_options` (`:73–84`): `(option_id, description)` pairs in **encode order**.
    ///
    /// - `noul` is always `true` then `false`; the request may override either description and any
    ///   other criteria key is merged and then ignored (`:80–81`).
    /// - `choice` is sorted by option id — Python's `str` sort, i.e. code point (port map §1.4).
    /// - `score` keeps list order with ids `"0"`, `"1"`, ….
    /// - A JSON `null` description is Python's `None`, which omits the `description` key entirely;
    ///   an **empty string** is kept (`:208`).
    static func questionOptions(_ question: ClefQuestion) throws -> [(id: String, description: ClefJSON?)] {
        switch question.type {
        case .noul:
            var trueText = ClefJSON.string(noulTrueDescription)
            var falseText = ClefJSON.string(noulFalseDescription)
            if case .object(let overrides)? = question.criteria {
                for field in overrides {
                    if field.key == "true" { trueText = field.value }
                    if field.key == "false" { falseText = field.value }
                }
            }
            return [("true", trueText), ("false", falseText)]

        case .choice:
            guard case .object(let fields)? = question.criteria, !fields.isEmpty else {
                throw ClefError.invalidQuestion("\(question.id): criteria must not be empty")
            }
            return fields
                .map { (id: $0.key, description: $0.value == .null ? nil : $0.value) }
                .sorted { ClefJSON.codePointLess($0.id, $1.id) }

        case .score:
            guard case .array(let items)? = question.criteria, !items.isEmpty else {
                throw ClefError.invalidQuestion("\(question.id): criteria must not be empty")
            }
            return items.enumerated().map { (id: String($0.offset), description: $0.element == .null ? nil : $0.element) }
        }
    }

    /// The `INSTRUCTION:` value (`:199`): `question.get("instructions") or str(question_id)`.
    static func instruction(for question: ClefQuestion) -> ClefJSON {
        if let instructions = question.instructions, !instructions.isPyFalsy { return instructions }
        return .string(question.id)
    }

    /// `{"option_id": …}` plus `{"description": …}` when there is one (`:206–211`).
    static func optionSemantics(id: String, description: ClefJSON?) -> ClefJSON {
        var fields = [ClefJSONField(key: "option_id", value: .string(id))]
        if let description { fields.append(ClefJSONField(key: "description", value: description)) }
        return .object(fields)
    }

    /// `encode_record`. `maxStateTokens` is the reference's optional early cap (`:235–236`);
    /// `truncate: false` makes an over-long request throw instead of trimming (`:240–241`).
    static func encodeRecord(
        tokenizer: some ClefTokenizer,
        record: ClefRequest,
        maxLength: Int = ClefPrompt.defaultMaxLength,
        maxStateTokens: Int? = nil,
        truncate: Bool = true
    ) throws -> ClefEncodedRecord {
        let encode: (String) -> [Int] = { tokenizer.encode($0) }

        // 1. The schema, in request order. Spans are relative to `schema_ids` for now.
        var schemaIDs = encode(schemaHeader)
        var questions: [ClefEncodedQuestion] = []
        for (offset, question) in record.questions.enumerated() {
            schemaIDs += encode(
                "\nFIELD \(offset + 1)\nID: \(question.id)\nTYPE: \(question.type.rawValue)\nINSTRUCTION: ")
            let questionStart = schemaIDs.count
            schemaIDs += encode(ClefJSON.render(instruction(for: question)))
            let questionEnd = schemaIDs.count

            schemaIDs += encode("\nALLOWED OPTIONS:\n")
            var optionSpans: [Range<Int>] = []
            var optionIDs: [String] = []
            for (index, option) in try questionOptions(question).enumerated() {
                schemaIDs += encode("OPTION \(index + 1): ")
                let optionStart = schemaIDs.count
                schemaIDs += encode(ClefJSON.render(optionSemantics(id: option.id, description: option.description)))
                optionSpans.append(optionStart ..< schemaIDs.count)
                optionIDs.append(option.id)
                schemaIDs += encode("\n")
            }
            schemaIDs += encode("END FIELD\n")

            questions.append(
                ClefEncodedQuestion(
                    questionID: question.id, type: question.type.index,
                    questionSpan: questionStart ..< questionEnd, optionSpans: optionSpans,
                    optionIDs: optionIDs))
        }

        // 2. Prefix, state, suffix.
        let prefixIDs = encode(statePrefix)
        let suffixIDs = encode(assistantSuffix)
        var stateIDs = encode(ClefJSON.render(record.state))
        let stateTokensTotal = stateIDs.count
        if let maxStateTokens { stateIDs = Array(stateIDs.prefix(maxStateTokens)) }

        // 3. The fixed part decides the state's budget (`:237–242`).
        let fixed = prefixIDs.count + schemaIDs.count + suffixIDs.count
        if fixed > maxLength {
            throw ClefError.schemaTooLong(fixed: fixed, maxLength: maxLength)
        }
        if !truncate, fixed + stateIDs.count > maxLength {
            throw ClefError.requestTooLong(needed: fixed + stateIDs.count, maxLength: maxLength)
        }
        stateIDs = Array(stateIDs.prefix(maxLength - fixed))
        let stateTokensKept = stateIDs.count

        // 4. Shift the spans past prefix + kept state (`:243–253`).
        let offset = prefixIDs.count + stateTokensKept
        let shifted = questions.map { question in
            ClefEncodedQuestion(
                questionID: question.questionID, type: question.type,
                questionSpan: (question.questionSpan.lowerBound + offset) ..< (question.questionSpan.upperBound + offset),
                optionSpans: question.optionSpans.map { ($0.lowerBound + offset) ..< ($0.upperBound + offset) },
                optionIDs: question.optionIDs)
        }

        return ClefEncodedRecord(
            inputIDs: prefixIDs + stateIDs + schemaIDs + suffixIDs, questions: shifted,
            stateTokensKept: stateTokensKept, stateTokensTotal: stateTokensTotal)
    }
}
