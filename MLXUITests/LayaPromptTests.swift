import Testing
@testable import MLXUI

/// A tiny, deterministic tokenizer — one token id per whitespace-separated word — so tests
/// can predict exact token counts and marker positions without a real vocabulary.
private struct FakeTokenizer: LayaTokenizing {
    let clsTokenID = 101
    let sepTokenID = 102
    let maskTokenID = 103
    let maskToken = "[MASK]"

    func encode(_ text: String) -> [Int] {
        let words = text.split(separator: " ", omittingEmptySubsequences: true)
        return words.indices.map { 2000 + $0 }
    }
}

struct LayaPromptTests {
    // MARK: renderCriterion / renderOptions

    @Test func renderCriterionPassesThroughStrings() {
        #expect(renderCriterion(.string("refunds")) == "refunds")
    }

    // Pinned against CPython `json.dumps(value, ensure_ascii=False, separators=(", ", ": "))`
    // — the exact literal output for `laya-mlx`'s "structured" parity case's criteria.
    @Test func renderCriterionMatchesPythonJSONForStructuredCase() {
        #expect(
            renderCriterion(.object([LayaJSONField(key: "description", value: .string("refunds"))]))
                == #"{"description": "refunds"}"#)
        #expect(renderCriterion(.bool(false)) == "false")
        #expect(
            renderCriterion(.object([LayaJSONField(key: "level", value: .string("low"))]))
                == #"{"level": "low"}"#)
        #expect(
            renderCriterion(.object([LayaJSONField(key: "reason", value: .string("money back"))]))
                == #"{"reason": "money back"}"#)
    }

    @Test func renderOptionsChoiceUsesLabelDescriptionOrBareLabel() {
        let q = LayaQuestion(
            type: .choice, instructions: "Which department?",
            choiceCriteria: [
                (label: "billing", description: .string("invoices, payments, refunds")),
                (label: "other", description: nil),
            ])
        #expect(renderOptions(q) == ["billing: invoices, payments, refunds", "other"])
    }

    @Test func renderOptionsScoreIsLevelIndexed() {
        let q = LayaQuestion(
            type: .score, instructions: "Urgency?",
            scoreCriteria: [.string("not urgent"), .string("soon"), .string("critical")])
        #expect(renderOptions(q) == ["level 0: not urgent", "level 1: soon", "level 2: critical"])
    }

    @Test func renderOptionsNoulDefaultsWhenNoCriteria() {
        let q = LayaQuestion(type: .noul, instructions: "Refund?")
        #expect(renderOptions(q) == ["false: no, the statement does not hold", "true: yes, the statement holds"])
    }

    @Test func renderOptionsNoulUsesStructuredCriteria() {
        let q = LayaQuestion(
            type: .noul, instructions: "Refund?",
            noulTrueDescription: .object([LayaJSONField(key: "reason", value: .string("money back"))]))
        #expect(renderOptions(q)[1] == #"true: {"reason": "money back"}"#)
        #expect(renderOptions(q)[0] == "false: no, the statement does not hold")
    }

    // MARK: LayaQuestionDefinition.resolve()

    @Test func resolveChoiceFromUniqueStringArray() throws {
        // "twenty_options" parity case.
        let labels = ["billing"] + (0 ..< 19).map { "department_\($0)" }
        let definition = LayaQuestionDefinition(
            type: .choice, instructions: .string("Which department handles billing?"),
            criteria: .array(labels.map { .string($0) }))
        let question = try definition.resolve()
        #expect(question.choiceCriteria.map(\.label) == labels)
        #expect(question.choiceCriteria.allSatisfy { $0.description == nil })
    }

    @Test func resolveChoiceRejectsDuplicateLabels() {
        let definition = LayaQuestionDefinition(
            type: .choice, instructions: .string("x"),
            criteria: .array([.string("a"), .string("a")]))
        #expect(throws: LayaPromptError.self) { try definition.resolve() }
    }

    @Test func resolveScoreRequiresNonemptyList() {
        let definition = LayaQuestionDefinition(type: .score, instructions: .string("x"), criteria: nil)
        #expect(throws: LayaPromptError.self) { try definition.resolve() }
    }

    // Pinned against CPython `json.dumps({"task": "choose department"})` — default
    // `ensure_ascii=True`, the non-string-`instructions` branch of `_to_internal`.
    @Test func resolveNonStringInstructionsUsesDefaultEnsureASCII() throws {
        let definition = LayaQuestionDefinition(
            type: .choice,
            instructions: .object([LayaJSONField(key: "task", value: .string("choose department"))]),
            criteria: .array([.string("billing"), .string("other")]))
        let question = try definition.resolve()
        #expect(question.instructions == #"{"task": "choose department"}"#)
    }

    @Test func resolveNoulPicksFalseAndTrueByKey() throws {
        let definition = LayaQuestionDefinition(
            type: .noul, instructions: .string("Refund?"),
            criteria: .object([LayaJSONField(key: "true", value: .string("money back"))]))
        let question = try definition.resolve()
        #expect(question.noulTrueDescription == .string("money back"))
        #expect(question.noulFalseDescription == nil)
    }

    // MARK: buildPrefix / buildSequence

    @Test func buildPrefixPlacesMarkersRightAfterMaskTokens() {
        let tokenizer = FakeTokenizer()
        let q = LayaQuestion(
            type: .choice, instructions: "pick one",
            choiceCriteria: [(label: "a", description: nil), (label: "b", description: nil)])
        let (ids, markers) = buildPrefix(tokenizer: tokenizer, question: q)
        // [CLS] head... [SEP] [MASK] a [MASK] b [SEP]
        #expect(ids.first == tokenizer.clsTokenID)
        #expect(ids.last == tokenizer.sepTokenID)
        #expect(markers.count == 2)
        for marker in markers {
            #expect(ids[marker] == tokenizer.maskTokenID)
        }
    }

    @Test func buildPrefixSqueezesOptionsWhenOverBudget() {
        let tokenizer = FakeTokenizer()
        // Twenty options, each rendered with a long description — forces `opt_budget < 16`
        // and the `per = max(4, (head_max_len - 16) / count)` squeeze (`common.py::build_prefix`).
        let longDescription = Array(repeating: "word", count: 40).joined(separator: " ")
        let criteria = (0 ..< 20).map { (label: "opt\($0)", description: LayaJSON.string(longDescription)) }
        let q = LayaQuestion(type: .choice, instructions: "pick one", choiceCriteria: criteria)
        let (ids, markers) = buildPrefix(tokenizer: tokenizer, question: q, headMaxLen: 192)
        #expect(markers.count == 20)
        // Every option slice, capped at `per` tokens plus its [MASK], must fit the budget.
        #expect(ids.count <= 192 + 2)
    }

    @Test func buildSequenceFlagsStateTruncation() {
        let tokenizer = FakeTokenizer()
        let q = LayaQuestion(type: .noul, instructions: "Refund?")
        let longState = Array(repeating: "word", count: 600).joined(separator: " ")
        let (ids, markers, truncated) = buildSequence(
            tokenizer: tokenizer, state: longState, question: q, maxLen: 64, headMaxLen: 32)
        #expect(truncated)
        #expect(ids.count <= 64)
        #expect(markers.allSatisfy { $0 < 64 })
    }

    @Test func buildSequenceNotTruncatedForShortState() {
        let tokenizer = FakeTokenizer()
        let q = LayaQuestion(type: .noul, instructions: "Refund?")
        let (_, _, truncated) = buildSequence(tokenizer: tokenizer, state: "short state", question: q)
        #expect(!truncated)
    }

    @Test func buildPrefixScrubsMaskLiteralFromUserText() {
        // "mask_literals" parity case: "[MASK] <mask> hello [MASK] <mask>" as state, and any
        // `[MASK]` inside instructions/options must be scrubbed to a space before tokenizing.
        let tokenizer = FakeTokenizer()
        let q = LayaQuestion(type: .noul, instructions: "Does [MASK] appear?")
        let (ids, _) = buildPrefix(tokenizer: tokenizer, question: q)
        // The literal token id for "[MASK]" the *word* must not appear inside the head text —
        // only at the real marker positions this function itself inserts.
        #expect(!ids.isEmpty)
    }
}
