import Foundation
import Testing

@testable import MLXUI

/// `ClefAnswer` — the port of `clef_mlx.py::systemone_answer` (`:520–539`). The three things that
/// differ from CLM 8B are the whole point (readme T8, port map §4), so each has a test here:
/// `confidence` is the **max probability** (not top-minus-mean), `choice` reports the criteria'
/// **insertion order** with a first-inserted tie-break (not the id-sorted encode order), and every
/// reported number is Python's `round(x, 4)`.
struct ClefAnswerTests {
    private func choiceQuestion(_ criteria: [(String, ClefJSON?)]) -> ClefQuestion {
        ClefQuestion(
            id: "q", type: .choice, instructions: nil,
            criteria: .object(
                criteria.map { ClefJSONField(key: $0.0, value: $0.1 ?? .null) }))
    }

    private func scoreQuestion(_ levels: [ClefJSON]) -> ClefQuestion {
        ClefQuestion(id: "q", type: .score, instructions: nil, criteria: .array(levels))
    }

    private var noulQuestion: ClefQuestion {
        ClefQuestion(id: "q", type: .noul, instructions: nil, criteria: nil)
    }

    // MARK: - choice

    /// `max(options, key=probabilities.__getitem__)` (`:525`) over `question["criteria"]` in
    /// insertion order — so an exact tie goes to the option inserted **first**, which is neither
    /// the alphabetically first id nor the prompt's encode order.
    @Test func choiceReportsInsertionOrderAndBreaksTiesTowardTheFirstInserted() throws {
        let question = choiceQuestion([("zeta", .string("Z")), ("alpha", .string("A")), ("mid", .string("M"))])
        let answer = try ClefAnswer.answer(
            question: question,
            probabilities: ["zeta": 0.5, "alpha": 0.5, "mid": 0.5],
            stateTruncated: false)

        #expect(answer.optionLabels == ["zeta", "alpha", "mid"])   // insertion, not sorted
        #expect(answer.choiceLabel == "zeta")                     // first maximum
        #expect(answer.type == .choice)
        #expect(answer.probabilities == [0.5, 0.5, 0.5])
    }

    /// The reference's `confidence` is `round(probabilities[choice], 4)` (`:529`). CLM's rule
    /// (top − mean of the rest) would read 0.85 here; this must read 0.9 (T8).
    @Test func confidenceIsTheMaxProbabilityNotTopMinusMean() throws {
        let answer = try ClefAnswer.answer(
            question: choiceQuestion([("a", .string("A")), ("b", .string("B")), ("c", .string("C"))]),
            probabilities: ["a": 0.9, "b": 0.05, "c": 0.05],
            stateTruncated: false)
        #expect(answer.confidence == 0.9)
        #expect(answer.choiceLabel == "a")
    }

    @Test func choiceProbabilitiesAreRoundedAndAlignedWithTheReportedOrder() throws {
        let answer = try ClefAnswer.answer(
            question: choiceQuestion([("b", .string("B")), ("a", .string("A"))]),
            probabilities: ["b": 0.123456, "a": 0.876544],
            stateTruncated: false)
        #expect(answer.optionLabels == ["b", "a"])
        #expect(answer.probabilities == [0.1235, 0.8765])
        #expect(answer.choiceLabel == "a")
        #expect(answer.confidence == 0.8765)
    }

    @Test func aNullableDescriptionIsNotReportedAndAnEmptyOneIs() throws {
        // `question_options` skips `None` but keeps `""` (`:208`); the answer doesn't carry
        // descriptions at all, so this pins that `reportedOptions` keeps both ids.
        let answer = try ClefAnswer.answer(
            question: choiceQuestion([("empty", .string("")), ("bare", nil)]),
            probabilities: ["empty": 1.0, "bare": 0.0],
            stateTruncated: false)
        #expect(answer.optionLabels == ["empty", "bare"])
        #expect(answer.choiceLabel == "empty")
    }

    // MARK: - score

    /// `score = round(sum(i * p_i), 4)` with levels `"0"…`, `confidence = round(max p, 4)`, and
    /// `legend` holding the declared words (`:532–538`).
    @Test func scoreIsTheExpectedLevelWithTheDeclaredWordsAsTheLegend() throws {
        let answer = try ClefAnswer.answer(
            question: scoreQuestion([.string("calm"), .string("annoyed"), .string("furious")]),
            probabilities: ["0": 0.2, "1": 0.3, "2": 0.5],
            stateTruncated: false)

        #expect(answer.type == .score)
        #expect(answer.optionLabels == ["0", "1", "2"])
        #expect(answer.scoreValue == 1.3)
        #expect(answer.confidence == 0.5)
        #expect(answer.scoreLevels == ["calm", "annoyed", "furious"])
        #expect(answer.choiceLabel == nil)
        #expect(answer.noulProbability == nil)
    }

    @Test func scoreValueIsRoundedLikePython() throws {
        let answer = try ClefAnswer.answer(
            question: scoreQuestion([.string("a"), .string("b")]),
            probabilities: ["0": 0.12345, "1": 0.87655],
            stateTruncated: false)
        // 0 * 0.12345 + 1 * 0.87655 = 0.87655, whose nearest double is just above the tie:
        // Python gives 0.8766 and so must this port.
        #expect(answer.scoreValue == 0.8766)
    }

    @Test func aStructuredLevelRendersAsItsText() throws {
        let answer = try ClefAnswer.answer(
            question: scoreQuestion([.object([ClefJSONField(key: "level", value: .string("low"))]), .string("high")]),
            probabilities: ["0": 0.4, "1": 0.6],
            stateTruncated: false)
        #expect(answer.scoreLevels == [#"{"level":"low"}"#, "high"])
    }

    // MARK: - noul

    /// `{"type": "noul", "noul": round(probabilities["true"], 4)}` (`:522`) — the false side is not
    /// reported. `confidence` has no reference value here and is filled with the larger
    /// probability (flagged in the source).
    @Test func noulReportsOnlyTrueAndFillsConfidence() throws {
        let answer = try ClefAnswer.answer(
            question: noulQuestion, probabilities: ["true": 0.7, "false": 0.3], stateTruncated: false)
        #expect(answer.type == .noul)
        #expect(answer.optionLabels == ["true", "false"])
        #expect(answer.noulProbability == 0.7)
        #expect(answer.confidence == 0.7)
        #expect(answer.choiceLabel == nil)
        #expect(answer.scoreValue == nil)
    }

    @Test func aFalseLeaningNoulKeepsTheTrueProbability() throws {
        let answer = try ClefAnswer.answer(
            question: noulQuestion, probabilities: ["true": 0.12, "false": 0.88], stateTruncated: false)
        #expect(answer.noulProbability == 0.12)
        #expect(answer.confidence == 0.88)
    }

    // MARK: - round4 and the flag

    /// Python's `round(x, 4)`, value by value — half-even on the exact double, which
    /// `(x * 1e4).rounded() / 1e4` does not always reproduce.
    @Test func round4MatchesPython() {
        let cases: [(Double, Double)] = [
            (0.5, 0.5),
            (0.125, 0.125),
            (0.12345, 0.1235),
            (0.00005, 0.0001),
            (2.675, 2.675),
            (1.00005, 1.0001),
            (0.99995, 1.0),
            (1.0 / 3.0, 0.3333),
            (0.30000000000000004, 0.3),
            (0.000025, 0.0),
            (0.0, 0.0),
            (-0.12345, -0.1235),
            (1e20, 1e20),
        ]
        for (value, expected) in cases {
            #expect(ClefAnswer.round4(value) == expected, "round4(\(value)) == \(ClefAnswer.round4(value))")
        }
    }

    @Test func stateTruncatedIsCarriedThrough() throws {
        let answer = try ClefAnswer.answer(
            question: noulQuestion, probabilities: ["true": 0.5, "false": 0.5], stateTruncated: true)
        #expect(answer.stateTruncated)
    }
}
