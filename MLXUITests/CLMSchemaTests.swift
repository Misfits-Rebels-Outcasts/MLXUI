import Testing
import Foundation
@testable import MLXUI

/// CL-4a golden test — `CLMSchema` against `Fixtures/CLM/schema.json` (provenance in
/// `Fixtures/CLM/PROVENANCE.md`: the real, vendored `clm_mlx.schema.build_pairs`, run against
/// 6 fixed requests). Byte-for-byte per CL-4a's hard gate, including `jsonObjectState`'s
/// numeric fields — `CLMJSON.number` keeps the literal JSON token (int/float classified like
/// Python's `json` module), so there's no representational gap left to work around
/// (`RSI/DelegateCLMBacklog.md` CL-4a, "Numbers: corrected 2026-09-28").
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
                CLMJSONField(key: "billing", value: .string("Billing and refunds team")),
                CLMJSONField(key: "technical", value: .string("Technical support team")),
            ]))
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    @Test func scoreSimpleMatchesGolden() throws {
        let golden = try loadGolden()["score_simple"]!
        let instructions = CLMJSON.string("Rate the sentiment from very negative to very positive.")
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
        let instructions = CLMJSON.string("Is this urgent?")
        let text = CLMSchema.stateText(
            state: .string("The customer said the shipment arrived three days late."),
            instructions: instructions)
        let (keys, texts) = try CLMSchema.candidates(type: .noul, instructions: instructions, criteria: nil)
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    /// The one golden case with numeric fields — `subtotal`/`tax`/`total`/`price` are Python
    /// `float`s (`500.0` etc.), `qty` is a plain `int`. `CLMJSON.number` keeps the literal
    /// token, so this now matches byte-for-byte with no workaround (contrast the CL-4b journal,
    /// which had to strip `.0` suffixes here before this fix).
    @Test func jsonObjectStateMatchesGoldenByteForByte() throws {
        let golden = try loadGolden()["json_object_state"]!
        let instructions = CLMJSON.string("Does the total match the line items and tax?")
        let state = CLMJSON.object([
            CLMJSONField(key: "invoice_id", value: .string("INV-1042")),
            CLMJSONField(key: "subtotal", value: .number("500.0")),
            CLMJSONField(key: "tax", value: .number("45.0")),
            CLMJSONField(key: "total", value: .number("545.0")),
            CLMJSONField(key: "line_items", value: .array([
                .object([
                    CLMJSONField(key: "desc", value: .string("Widget A")),
                    CLMJSONField(key: "qty", value: .number("2")),
                    CLMJSONField(key: "price", value: .number("100.0")),
                ]),
                .object([
                    CLMJSONField(key: "desc", value: .string("Widget B")),
                    CLMJSONField(key: "qty", value: .number("3")),
                    CLMJSONField(key: "price", value: .number("100.0")),
                ]),
            ])),
        ])
        let text = CLMSchema.stateText(state: state, instructions: instructions)
        let (keys, texts) = try CLMSchema.candidates(
            type: .choice, instructions: instructions,
            criteria: .object([
                CLMJSONField(key: "consistent", value: .string("The total matches subtotal plus tax")),
                CLMJSONField(key: "inconsistent", value: .string("The total does not match")),
            ]))
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    @Test func nestedArrayStateMatchesGolden() throws {
        let golden = try loadGolden()["nested_array_state"]!
        let instructions = CLMJSON.string("Are all rows approved?")
        let state = CLMJSON.array([
            .string("Row 1: approved"),
            .object([
                CLMJSONField(key: "row", value: .number("2")),
                CLMJSONField(key: "status", value: .string("hold")),
                CLMJSONField(key: "reason", value: .string("missing PO")),
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
        let instructions = CLMJSON.string("Which severity?")
        let text = CLMSchema.stateText(
            state: .string("A short support ticket about a login issue."), instructions: instructions)
        let (keys, texts) = try CLMSchema.candidates(
            type: .choice, instructions: instructions,
            criteria: .object([
                CLMJSONField(key: "low", value: .string("Minor annoyance, no blocker")),
                CLMJSONField(key: "medium", value: .string("")),
                CLMJSONField(key: "high", value: .string("Blocks the user entirely")),
            ]))
        #expect(text == golden.state_text)
        #expect(keys == golden.option_keys)
        #expect(texts == golden.candidate_texts)
    }

    // MARK: - Number rendering (CL-4a "Numbers: corrected 2026-09-28")

    /// `Fixtures/CLM/numbers.json` — 13 literal JSON tokens run through the real
    /// `clm_mlx.schema.to_text(json.loads(literal))`, generated on-Mac (`Fixtures/CLM/
    /// PROVENANCE.md`, "`numbers.json`"). Byte-for-byte against `CLMSchema.toText`.
    private struct NumberGoldenCase: Decodable {
        var literal: String
        var to_text: String
    }

    @Test func toTextRendersNumbersLikePythonStr() throws {
        let filePath = #filePath
        let repoRoot = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("Fixtures/CLM/numbers.json")
        let golden = try JSONDecoder().decode([String: NumberGoldenCase].self, from: Data(contentsOf: url))
        #expect(golden.count >= 10)

        for (name, testCase) in golden {
            let parsed = try CLMJSONParser.parse(testCase.literal)
            switch parsed {
            case .number, .bool:
                #expect(CLMSchema.toText(parsed) == testCase.to_text, "\(name): literal '\(testCase.literal)'")
            default:
                Issue.record("\(name): literal '\(testCase.literal)' did not parse as a number or bool")
            }
        }
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
        let levels = CLMJSON.array([.string("low"), .string("mid"), .string("high")])
        let answer = try CLMSchema.answer(
            fromLogits: [0.0, 0.0, 5.0], type: .score, criteria: levels, keys: ["0", "1", "2"],
            stateTruncated: false)
        #expect(answer.scoreValue != nil)
        #expect(answer.scoreValue! > 1.5)
        #expect(answer.scoreLevels == ["low", "mid", "high"])
    }
}
