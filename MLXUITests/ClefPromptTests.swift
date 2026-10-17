import Foundation
import Testing

@testable import MLXUI

/// CF-5a's pure half, always runnable: the prompt builder (``ClefPrompt``, porting
/// `clef_mlx.py::encode_record` `:179–254`) and the JSON `render` it depends on (`:67–70`).
///
/// No model and no HF tokenizer are needed here — the reference's fragment *boundaries* are driven
/// with stand-in tokenizers, and the expected strings are CPython's own output for the same values
/// (`json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True)`). The ids the real
/// tokenizer produces are the gated golden in `ClefGoldenTests`.

/// One token per `encode` call, numbered in call order: `input_ids` *is* the fragment list, so a
/// coalesced whole-prompt encode cannot satisfy the assertions (readme T4, port map §1.2).
private final class CountingTokenizer: ClefTokenizer, @unchecked Sendable {
    private(set) var pieces: [String] = []

    func encode(_ text: String) -> [Int] {
        pieces.append(text)
        return [pieces.count - 1]
    }
}

/// One token per Unicode scalar, so a span's width is its character count.
private struct ScalarTokenizer: ClefTokenizer {
    func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
}

struct ClefPromptTests {
    private func request(_ json: String) throws -> ClefRequest {
        try ClefRequest(jsonText: json)
    }

    // MARK: - Fragments

    /// The reference encodes **one fragment at a time** (`:189–231`) and concatenates; this test
    /// pins every fragment's text and the two orders that matter: the *call* order (the schema is
    /// built first, then prefix/suffix/state) and the *assembly* order (`prefix + state + schema +
    /// suffix`, `:254`).
    @Test func promptFragmentsAreTheReferencesLines() throws {
        let tokenizer = CountingTokenizer()
        let record = try ClefPrompt.encodeRecord(
            tokenizer: tokenizer,
            record: try request(
                """
                {"state": "hello", "questions": {
                  "team": {"type": "choice", "instructions": "Which?",
                           "criteria": {"b": "B", "a": ""}},
                  "urgent": {"type": "noul"}}}
                """))

        let prefixIndex = try #require(tokenizer.pieces.firstIndex { $0.hasPrefix("<|im_start|>system") })
        let suffixIndex = try #require(tokenizer.pieces.firstIndex { $0.hasPrefix("\n<|im_end|>") })
        let stateIndex = try #require(tokenizer.pieces.firstIndex(of: "hello"))
        #expect(
            tokenizer.pieces[prefixIndex]
                == "<|im_start|>system\n\(ClefPrompt.systemPrompt)<|im_end|>\n<|im_start|>user\nSTATE:\n")
        #expect(tokenizer.pieces[suffixIndex] == ClefPrompt.assistantSuffix)
        // `render` returns a Python `str` as-is, so a plain-text state is not quoted.
        #expect(tokenizer.pieces[stateIndex] == "hello")

        let fixed = [prefixIndex, suffixIndex, stateIndex]
        let schemaIndices = tokenizer.pieces.indices.filter { !fixed.contains($0) }
        #expect(schemaIndices.count == tokenizer.pieces.count - 3)
        #expect(
            schemaIndices.map { tokenizer.pieces[$0] }
                == [
                    ClefPrompt.schemaHeader,
                    "\nFIELD 1\nID: team\nTYPE: choice\nINSTRUCTION: ",
                    "Which?",
                    "\nALLOWED OPTIONS:\n",
                    "OPTION 1: ", #"{"description":"","option_id":"a"}"#, "\n",
                    "OPTION 2: ", #"{"description":"B","option_id":"b"}"#, "\n",
                    "END FIELD\n",
                    "\nFIELD 2\nID: urgent\nTYPE: noul\nINSTRUCTION: ",
                    "urgent",
                    "\nALLOWED OPTIONS:\n",
                    "OPTION 1: ",
                    #"{"description":"The proposition is true or the answer is yes.","option_id":"true"}"#,
                    "\n",
                    "OPTION 2: ",
                    #"{"description":"The proposition is false or the answer is no.","option_id":"false"}"#,
                    "\n",
                    "END FIELD\n",
                ])

        // The assembled prompt is prefix + state + schema + suffix, and the ids are those fragments'
        // tokens in that order.
        #expect(record.inputIDs == [prefixIndex, stateIndex] + schemaIndices + [suffixIndex])
        #expect(record.stateTruncated == false)
        #expect(record.stateTokensTotal == 1)
    }

    /// `choice` options are encoded **id-sorted** (`:83`) and the spans/ids follow that order;
    /// the answer reports the criteria' insertion order instead (`:524`) — `ClefAnswerTests` holds
    /// that half. Both orders must survive the port.
    @Test func choiceEncodesInIdOrderAndRecordsThoseSpans() throws {
        let tokenizer = CountingTokenizer()
        let record = try ClefPrompt.encodeRecord(
            tokenizer: tokenizer,
            record: try request(
                #"{"state": "s", "questions": {"q": {"type": "choice", "criteria": {"zeta": "Z", "alpha": "A", "mid": "M"}}}}"#))

        #expect(record.questions.count == 1)
        #expect(record.questions[0].optionIDs == ["alpha", "mid", "zeta"])
        // With one token per fragment, the option spans are their fragment indices offset past the
        // prefix (1 token) and the state (1 token).
        #expect(record.questions[0].optionSpans.count == 3)
        for span in record.questions[0].optionSpans {
            #expect(span.count == 1)
            #expect(span.lowerBound >= 2)
        }
        #expect(record.questions[0].optionSpans[0].lowerBound < record.questions[0].optionSpans[1].lowerBound)
        #expect(record.questions[0].optionSpans[1].lowerBound < record.questions[0].optionSpans[2].lowerBound)
    }

    /// `question.get("instructions") or str(question_id)` (`:199`) — an absent *or falsy*
    /// instruction falls back to the question id.
    @Test func instructionsFallBackToTheQuestionIDWhenAbsentOrFalsy() throws {
        let tokenizer = CountingTokenizer()
        _ = try ClefPrompt.encodeRecord(
            tokenizer: tokenizer,
            record: try request(
                """
                {"state": "s", "questions": {
                  "a": {"type": "noul"},
                  "b": {"type": "noul", "instructions": ""},
                  "c": {"type": "noul", "instructions": null},
                  "d": {"type": "noul", "instructions": 0},
                  "e": {"type": "noul", "instructions": "real words"}}}
                """))
        let instructions = tokenizer.pieces.indices.compactMap { index -> String? in
            tokenizer.pieces[index].hasSuffix("INSTRUCTION: ") ? tokenizer.pieces[index + 1] : nil
        }
        // `e` carries a real instruction, so its piece is that text rather than the id.
        #expect(instructions == ["a", "b", "c", "d", "real words"])
    }

    /// An empty description is **kept** (`:208` checks `is not None`) while a JSON `null` — Python's
    /// `None` — omits the key. CF-7 may therefore keep sending `{tag: ""}`.
    @Test func emptyDescriptionIsKeptAndBareOptionOmitsTheKey() throws {
        let tokenizer = CountingTokenizer()
        _ = try ClefPrompt.encodeRecord(
            tokenizer: tokenizer,
            record: try request(
                #"{"state": "s", "questions": {"q": {"type": "choice", "criteria": {"empty": "", "bare": null, "real": "words"}}}}"#))

        #expect(tokenizer.pieces.contains(#"{"description":"","option_id":"empty"}"#))
        #expect(tokenizer.pieces.contains(#"{"option_id":"bare"}"#))
        #expect(tokenizer.pieces.contains(#"{"description":"words","option_id":"real"}"#))
    }

    // MARK: - Truncation

    /// `fixed = prefix + schema + suffix`; the state absorbs the whole remainder and is cut **from
    /// the end** (`:237–242`), and every span shifts by `prefix + kept` (`:243–253`).
    @Test func stateTruncationKeepsTheHeadOfTheStateAndShiftsTheSpans() throws {
        let json = #"{"state": "abcdefghij", "questions": {"n": {"type": "noul"}}}"#
        let full = try ClefPrompt.encodeRecord(
            tokenizer: ScalarTokenizer(), record: try request(json), maxLength: 100_000)
        // Four tokens short of the full prompt: the schema stays whole and four state tokens go.
        let maxLength = full.inputIDs.count - 4
        let truncated = try ClefPrompt.encodeRecord(
            tokenizer: ScalarTokenizer(), record: try request(json), maxLength: maxLength)

        let prefixCount = ClefPrompt.statePrefix.unicodeScalars.count
        #expect(full.stateTokensTotal == 10)
        #expect(full.stateTruncated == false)
        #expect(truncated.stateTokensTotal == 10)
        #expect(truncated.stateTokensKept == 6)
        #expect(truncated.stateTruncated)
        // The whole prompt fits the budget exactly when the state is the thing that gets cut.
        #expect(truncated.inputIDs.count == maxLength)

        // The kept state is the *head* of the state…
        let keptIDs = Array(truncated.inputIDs[prefixCount ..< prefixCount + truncated.stateTokensKept])
        let expected = Array("abcdefghij".unicodeScalars.prefix(truncated.stateTokensKept)).map { Int($0.value) }
        #expect(keptIDs == expected)

        // …and everything after it is the schema + suffix, unchanged, just nearer the front.
        let tail = Array(full.inputIDs[(prefixCount + 10)...])
        #expect(Array(truncated.inputIDs[(prefixCount + truncated.stateTokensKept)...]) == tail)

        // Spans move by exactly the number of dropped state tokens.
        let dropped = 4
        #expect(
            truncated.questions[0].questionSpan.lowerBound
                == full.questions[0].questionSpan.lowerBound - dropped)
        #expect(
            truncated.questions[0].optionSpans[0].lowerBound
                == full.questions[0].optionSpans[0].lowerBound - dropped)
    }

    /// `max_state_tokens` caps the state *before* the budget is computed (`:235–236`), and the
    /// reference's two `ContextTooLong` messages are carried verbatim (`:239`, `:241`).
    @Test func aSchemaThatAloneExceedsTheBudgetThrowsTheReferencesMessage() throws {
        let json = #"{"state": "state", "questions": {"n": {"type": "noul"}}}"#
        let error = try #require(throws: ClefError.self) {
            try ClefPrompt.encodeRecord(tokenizer: ScalarTokenizer(), record: try request(json), maxLength: 10)
        }
        // 10 tokens cannot hold 24 fragments' worth of schema (1 token per scalar), so the message
        // must be the reference's, naming the fixed size and the budget.
        guard case .schemaTooLong(let fixed, let maxLength) = error else {
            Issue.record("expected .schemaTooLong, got \(error)")
            return
        }
        #expect(maxLength == 10)
        #expect(error.description == "schema requires \(fixed) tokens before state; maximum is 10")
    }

    @Test func overLongStateWithTruncateFalseThrowsTheReferencesMessage() throws {
        let json = #"{"state": "abcdefghij", "questions": {"n": {"type": "noul"}}}"#
        let full = try ClefPrompt.encodeRecord(
            tokenizer: ScalarTokenizer(), record: try request(json), maxLength: 100_000)
        let maxLength = full.inputIDs.count - 4
        let error = try #require(throws: ClefError.self) {
            try ClefPrompt.encodeRecord(
                tokenizer: ScalarTokenizer(), record: try request(json), maxLength: maxLength,
                truncate: false)
        }
        guard case .requestTooLong(let needed, let maxLength) = error else {
            Issue.record("expected .requestTooLong, got \(error)")
            return
        }
        // Nothing was capped, so the request needs the whole untruncated prompt.
        #expect(needed == full.inputIDs.count)
        #expect(error.description == "request needs \(needed) tokens; maximum is \(maxLength)")
    }

    @Test func maxStateTokensCapsTheStateBeforeTheBudget() throws {
        let json = #"{"state": "abcdefghij", "questions": {"n": {"type": "noul"}}}"#
        let capped = try ClefPrompt.encodeRecord(
            tokenizer: ScalarTokenizer(), record: try request(json), maxLength: 100_000, maxStateTokens: 4)
        #expect(capped.stateTokensKept == 4)
        #expect(capped.stateTokensTotal == 10)
        #expect(capped.stateTruncated)
    }

    // MARK: - Request decoding

    @Test func requestDecodingPreservesQuestionOrder() throws {
        let request = try request(
            """
            {"state": {"b": 1, "a": 2}, "questions": {
              "zeta": {"type": "noul"}, "alpha": {"type": "score", "criteria": ["x", "y"]},
              "mid": {"type": "choice", "criteria": {"k": null}}}}
            """)
        #expect(request.questions.map(\.id) == ["zeta", "alpha", "mid"])
        #expect(request.questions.map(\.type) == [.noul, .score, .choice])
        // `sort_keys` sorts the *rendered* state; the decoded order is the document's.
        #expect(ClefJSON.render(request.state) == #"{"a":2,"b":1}"#)
    }

    @Test func unknownQuestionTypeIsRefusedWithTheReferencesMessage() throws {
        let error = try #require(throws: ClefError.self) {
            try request(#"{"state": "s", "questions": {"q": {"type": "rank"}}}"#)
        }
        #expect(error == .invalidQuestion("q: type must be noul, choice, or score"))
        #expect(error.description == "q: type must be noul, choice, or score")
    }

    @Test func aRequestWithoutQuestionsIsRefused() throws {
        let error = try #require(throws: ClefError.self) {
            try request(#"{"state": "s", "questions": {}}"#)
        }
        #expect(error == .emptyQuestions)
        #expect(error.description == "at least one question is required")
    }

    @Test func choiceAndScoreNeedCriteria() throws {
        #expect(throws: ClefError.invalidQuestion("q: criteria must not be empty")) {
            try ClefPrompt.questionOptions(
                ClefQuestion(id: "q", type: .choice, instructions: nil, criteria: .object([])))
        }
        #expect(throws: ClefError.invalidQuestion("q: criteria must not be empty")) {
            try ClefPrompt.questionOptions(
                ClefQuestion(id: "q", type: .score, instructions: nil, criteria: .array([])))
        }
    }

    /// `noul` merges the request's overrides over the built-in descriptions and then iterates only
    /// `("true", "false")` (`:76–81`) — an extra key is ignored, not an option.
    @Test func noulUsesBuiltInDescriptionsAndIgnoresExtraCriteriaKeys() throws {
        let options = try ClefPrompt.questionOptions(
            ClefQuestion(
                id: "q", type: .noul, instructions: nil,
                criteria: .object([
                    ClefJSONField(key: "true", value: .string("yes indeed")),
                    ClefJSONField(key: "maybe", value: .string("ignored")),
                ])))
        #expect(options.map(\.id) == ["true", "false"])
        #expect(options[0].description == .string("yes indeed"))
        #expect(options[1].description == .string(ClefPrompt.noulFalseDescription))
    }

    // MARK: - `render`

    /// CPython's `json.dumps` for floats (`repr`), pinned value by value. Swift's
    /// `Double.description` is shortest-round-trip and matches `repr` on all of these — including
    /// the fixed/exponential thresholds — so only the non-finite spellings need help.
    @Test func jsonNumberRenderingMatchesPython() {
        let doubles: [(Double, String)] = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (1250.0, "1250.0"),
            (0.30000000000000004, "0.30000000000000004"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (0.0001, "0.0001"),
            (1e-5, "1e-05"),
            (2.5e-7, "2.5e-07"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (2.0, "2.0"),
            (Double.leastNonzeroMagnitude, "5e-324"),
            (Double.greatestFiniteMagnitude, "1.7976931348623157e+308"),
            (123456789.123, "123456789.123"),
            (.nan, "NaN"),
            (.infinity, "Infinity"),
            (-.infinity, "-Infinity"),
        ]
        for (value, expected) in doubles {
            #expect(ClefJSON.render(.double(value)) == expected, "\(value) → \(ClefJSON.render(.double(value)))")
        }
        // Integers are not floats: Python keeps them apart, and so must the port.
        #expect(ClefJSON.render(.int(1)) == "1")
        #expect(ClefJSON.render(.int(-7)) == "-7")
        #expect(ClefJSON.render(.bigInt("123456789012345678901234567890")) == "123456789012345678901234567890")
    }

    @Test func jsonStringEscapingMatchesPython() {
        let cases: [(String, String)] = [
            ("plain", #""plain""#),
            (#"quote"inside"#, #""quote\"inside""#),
            (#"back\slash"#, #""back\\slash""#),
            ("tab\there", #""tab\there""#),
            ("line\nbreak", #""line\nbreak""#),
            ("café", #""café""#),
            ("日本", #""日本""#),
            ("\u{01}\u{1F}", #""\u0001\u001f""#),
        ]
        for (value, expected) in cases {
            #expect(ClefJSON.jsonText(.string(value)) == expected)
        }
    }

    /// `sort_keys=True` sorts by **code point** (`"A" < "a" < "z" < "Å" < "é" < "日本"`), which is
    /// not Swift's canonical `String` ordering.
    @Test func jsonKeysSortByCodePoint() {
        let object = ClefJSON.object([
            ClefJSONField(key: "z", value: .int(1)),
            ClefJSONField(key: "Å", value: .int(2)),
            ClefJSONField(key: "a", value: .int(3)),
            ClefJSONField(key: "é", value: .int(4)),
            ClefJSONField(key: "A", value: .int(5)),
            ClefJSONField(key: "日本", value: .int(6)),
        ])
        #expect(ClefJSON.render(object) == #"{"A":5,"a":3,"z":1,"Å":2,"é":4,"日本":6}"#)
    }

    @Test func nonAsciiNestedStateRendersLikePython() throws {
        let text = #"{"café": {"Ångström": 1.0, "日本": ["a", "b"], "z": -0.5}, "A": 2, "Zebra": null}"#
        #expect(
            ClefJSON.render(try ClefJSONParser.parse(text))
                == #"{"A":2,"Zebra":null,"café":{"z":-0.5,"Ångström":1.0,"日本":["a","b"]}}"#)
    }

    @Test func parsingKeepsIntAndFloatApart() throws {
        let value = try ClefJSONParser.parse(#"{"n": 1, "f": 1.0, "e": 1e2, "neg": -0, "big": 123456789012345678901234567890}"#)
        #expect(ClefJSON.render(value) == #"{"big":123456789012345678901234567890,"e":100.0,"f":1.0,"n":1,"neg":0}"#)
    }

    /// Round-tripping a state through the parser must not change its bytes: the golden's `state` is
    /// parsed, rendered and tokenized, and the reference's `json.loads`/`json.dumps` pair does the
    /// same normalization.
    @Test func stateRoundTripsThroughParseAndRender() throws {
        let text = #"{"b": [1, 2.5, {"c": true}], "a": "x"}"#
        let parsed = try ClefJSONParser.parse(text)
        let reparsed = try ClefJSONParser.parse(ClefJSON.jsonText(parsed))
        // `ClefJSON`'s equality is document-order sensitive on purpose (a `questions` object's order
        // *is* the FIELD order), so the round trip is compared on the canonical rendering.
        #expect(ClefJSON.jsonText(reparsed) == ClefJSON.jsonText(parsed))
    }

    @Test func aPlainStringStateIsUsedRaw() throws {
        let request = try request(#"{"state": "Checkout is failing.\nSecond line.", "questions": {"q": {"type": "noul"}}}"#)
        // Python's `render` returns a `str` unchanged — no quotes, no newline escaping.
        #expect(ClefJSON.render(request.state) == "Checkout is failing.\nSecond line.")
    }
}
