import Testing
import Foundation
@testable import MLXUI

/// CL-4a golden test — `CLMSchema` against `Fixtures/CLM/schema.json` (provenance in
/// `Fixtures/CLM/PROVENANCE.md`: the real, vendored `clm_mlx.schema.build_pairs`, run against
/// 6 fixed requests). Byte-for-byte per CL-4a's hard gate, except `jsonObjectState`'s numeric
/// fields — see that test's own comment for the documented `LayaJSON.number` gap.
struct CLMSchemaTests {
    private struct GoldenCase: Decodable {
        var state_text: String
        var option_keys: [String]
        var candidate_texts: [String]
    }

    private func loadGolden() throws -> [String: GoldenCase] {
        let filePath = #filePath                      // .../MLXUITests/CLMSchemaTests.swift
        let repoRoot = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("Fixtures/CLM/schema.json")
        return try JSONDecoder().decode([String: GoldenCase].self, from: Data(contentsOf: url))
    }

    @Test func choiceSimpleMatchesGolden() throws {
        let golden = try loadGolden()["choice_simple"]!
        let text = CLMSchema.stateText(
            state: .string("Customer: my invoice was charged twice. Please refund the second charge."),
            instructions: .string("Which team should handle this?"))
        let (keys, texts) = try CLMSchema.candidates(
            type: .choice,
            instructions: .string("Which team should handle this?"),
            criteria: .object([
                LayaJSONField(key: "billing", value: .string("Billing and refunds team")),
                LayaJSONField(key: "technical", value: .string("Technical support team")),
            ]))
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    @Test func scoreSimpleMatchesGolden() throws {
        let golden = try loadGolden()["score_simple"]!
        let instructions = LayaJSON.string("Rate the sentiment from very negative to very positive.")
        let text = CLMSchema.stateText(
            state: .string("The service was okay, nothing special."), instructions: instructions)
        let (keys, texts) = try CLMSchema.candidates(
            type: .score, instructions: instructions,
            criteria: .array([
                .string("very negative"), .string("negative"), .string("neutral"),
                .string("positive"), .string("very positive"),
            ]))
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    @Test func noulSimpleMatchesGolden() throws {
        let golden = try loadGolden()["noul_simple"]!
        let instructions = LayaJSON.string("Is this urgent?")
        let text = CLMSchema.stateText(
            state: .string("The customer said the shipment arrived three days late."),
            instructions: instructions)
        let (keys, texts) = try CLMSchema.candidates(type: .noul, instructions: instructions, criteria: nil)
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    /// The one golden case with numeric fields. `subtotal`/`tax`/`total`/`price` were written
    /// as Python `float`s (`500.0` etc.), so upstream's `str()` keeps the trailing `.0`; this
    /// port's `LayaJSON.number` is a `Double` with no int/float distinction, so
    /// `CLMSchema.toText` renders every integral double without it (`"500"`, not `"500.0"`) —
    /// the divergence `RSI/DelegateCLMBacklog.md` CL-4a names explicitly and asks to pin with
    /// a test rather than "fix". `option_keys`/`candidate_texts` carry no numbers, so those
    /// still match the golden byte-for-byte; only `state_text` differs, and only in the
    /// `.0` suffixes.
    @Test func jsonObjectStateMatchesGoldenModuloTheNumberGap() throws {
        let golden = try loadGolden()["json_object_state"]!
        let instructions = LayaJSON.string("Does the total match the line items and tax?")
        let state = LayaJSON.object([
            LayaJSONField(key: "invoice_id", value: .string("INV-1042")),
            LayaJSONField(key: "subtotal", value: .number(500.0)),
            LayaJSONField(key: "tax", value: .number(45.0)),
            LayaJSONField(key: "total", value: .number(545.0)),
            LayaJSONField(key: "line_items", value: .array([
                .object([
                    LayaJSONField(key: "desc", value: .string("Widget A")),
                    LayaJSONField(key: "qty", value: .number(2)),
                    LayaJSONField(key: "price", value: .number(100.0)),
                ]),
                .object([
                    LayaJSONField(key: "desc", value: .string("Widget B")),
                    LayaJSONField(key: "qty", value: .number(3)),
                    LayaJSONField(key: "price", value: .number(100.0)),
                ]),
            ])),
        ])
        let text = CLMSchema.stateText(state: state, instructions: instructions)
        let (keys, texts) = try CLMSchema.candidates(
            type: .choice, instructions: instructions,
            criteria: .object([
                LayaJSONField(key: "consistent", value: .string("The total matches subtotal plus tax")),
                LayaJSONField(key: "inconsistent", value: .string("The total does not match")),
            ]))
        let expectedText = golden.state_text
            .replacingOccurrences(of: "500.0", with: "500")
            .replacingOccurrences(of: "45.0", with: "45")
            .replacingOccurrences(of: "545.0", with: "545")
            .replacingOccurrences(of: "100.0", with: "100")
        #expect(text == expectedText)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    @Test func nestedArrayStateMatchesGolden() throws {
        let golden = try loadGolden()["nested_array_state"]!
        let instructions = LayaJSON.string("Are all rows approved?")
        let state = LayaJSON.array([
            .string("Row 1: approved"),
            .object([
                LayaJSONField(key: "row", value: .number(2)),
                LayaJSONField(key: "status", value: .string("hold")),
                LayaJSONField(key: "reason", value: .string("missing PO")),
            ]),
            .array([.string("nested"), .string("list"), .string("item")]),
        ])
        let text = CLMSchema.stateText(state: state, instructions: instructions)
        let (keys, texts) = try CLMSchema.candidates(type: .noul, instructions: instructions, criteria: nil)
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    @Test func choiceMixedCriteriaMatchesGolden() throws {
        let golden = try loadGolden()["choice_mixed_criteria"]!
        let instructions = LayaJSON.string("Which severity?")
        let text = CLMSchema.stateText(
            state: .string("A short support ticket about a login issue."), instructions: instructions)
        let (keys, texts) = try CLMSchema.candidates(
            type: .choice, instructions: instructions,
            criteria: .object([
                LayaJSONField(key: "low", value: .string("Minor annoyance, no blocker")),
                LayaJSONField(key: "medium", value: .string("")),
                LayaJSONField(key: "high", value: .string("Blocks the user entirely")),
            ]))
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    // MARK: - The representational gap itself, pinned directly (CL-4a's instruction)

    @Test func toTextRendersIntegralDoublesWithoutTrailingZero() {
        // Python's str(500) == "500" and str(500.0) == "500.0"; LayaJSON.number is a Double
        // with no int/float distinction, so this port always renders the former shape.
        #expect(CLMSchema.toText(.number(500)) == "500")
        #expect(CLMSchema.toText(.number(0)) == "0")
        #expect(CLMSchema.toText(.number(3.5)) == "3.5")
    }

    // MARK: - Error messages (CL-4a: "errors carry the same messages as upstream's ValueErrors")

    @Test func choiceWithoutCriteriaThrows() {
        #expect(throws: LayaPromptError.invalidCriteria("choice question needs a non-empty 'criteria' object")) {
            _ = try CLMSchema.candidates(type: .choice, instructions: .string("x"), criteria: nil)
        }
    }

    @Test func scoreWithOneLevelThrows() {
        #expect(throws: LayaPromptError.invalidCriteria("score question needs 'criteria' as an ordered list of >= 2 levels")) {
            _ = try CLMSchema.candidates(type: .score, instructions: .string("x"), criteria: .array([.string("only one")]))
        }
    }

    // MARK: - softmax / confidence / answer (schema.py's pure functions, no golden needed)

    @Test func softmaxSumsToOne() {
        let probs = CLMSchema.softmax([1.0, 2.0, 3.0])
        #expect(abs(probs.reduce(0, +) - 1.0) < 1e-9)
        #expect(probs[2] > probs[1])
        #expect(probs[1] > probs[0])
    }

    @Test func confidenceIsTopMinusMeanOfRest() {
        // 3 options, one dominant: confidence = 0.7 - mean(0.2, 0.1) = 0.7 - 0.15 = 0.55.
        let c = CLMSchema.confidence([0.2, 0.7, 0.1])
        #expect(abs(c - 0.55) < 1e-9)
    }

    @Test func answerFromLogitsProducesChoiceLabel() throws {
        let answer = try CLMSchema.answer(
            fromLogits: [1.0, 5.0], type: .choice, criteria: nil, keys: ["billing", "technical"],
            stateTruncated: false)
        #expect(answer.choiceLabel == "technical")
        #expect(answer.type == .choice)
        #expect(answer.stateTruncated == false)
    }

    @Test func answerFromLogitsProducesNoulProbability() throws {
        let answer = try CLMSchema.answer(
            fromLogits: [0.0, 2.0], type: .noul, criteria: nil, keys: ["false", "true"],
            stateTruncated: true)
        #expect(answer.noulProbability != nil)
        #expect(answer.noulProbability! > 0.5)
        #expect(answer.stateTruncated == true)
    }

    @Test func answerFromLogitsProducesScoreValueAndLevels() throws {
        let levels = LayaJSON.array([.string("low"), .string("mid"), .string("high")])
        let answer = try CLMSchema.answer(
            fromLogits: [0.0, 0.0, 5.0], type: .score, criteria: levels, keys: ["0", "1", "2"],
            stateTruncated: false)
        #expect(answer.scoreValue != nil)
        #expect(answer.scoreValue! > 1.5)
        #expect(answer.scoreLevels == ["low", "mid", "high"])
    }
}
