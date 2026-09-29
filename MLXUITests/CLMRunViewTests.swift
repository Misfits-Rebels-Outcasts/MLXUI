import Testing
@testable import MLXUI

/// CL-5-FIX-1 — `clmRunCriteria` (`Views/Run/CLMRunView.swift`): the pure screen-state →
/// `CLMJSON` builder, tested directly rather than through the view. Feeds its output into
/// `CLMSchema.candidates` (the same call `CLMEngine.answer` makes) to assert it does not
/// throw for the screen's real default states — smoke row 99 failed exactly because this
/// builder produced a shape `CLMSchema.candidates` rejects (`.array` of bare labels for
/// choice, where it requires a non-empty `.object`).
struct CLMRunViewTests {
    @Test func defaultChoiceStateDoesNotThrow() throws {
        let criteria = try clmRunCriteria(
            type: .choice, choiceLabels: ["billing", "technical", "sales"], scoreLevels: [])
        _ = try CLMSchema.candidates(type: .choice, instructions: .string("Who should handle this?"), criteria: criteria)
    }

    @Test func defaultScoreStateDoesNotThrow() throws {
        let criteria = try clmRunCriteria(
            type: .score, choiceLabels: [], scoreLevels: ["not urgent", "soon", "critical"])
        _ = try CLMSchema.candidates(type: .score, instructions: .string("How urgent?"), criteria: criteria)
    }

    @Test func yesNoDoesNotThrow() throws {
        let criteria = try clmRunCriteria(type: .noul, choiceLabels: [], scoreLevels: [])
        #expect(criteria == nil)
        _ = try CLMSchema.candidates(type: .noul, instructions: .string("Is this urgent?"), criteria: criteria)
    }

    @Test func choiceWithABlankRowDoesNotThrow() throws {
        let criteria = try clmRunCriteria(
            type: .choice, choiceLabels: ["billing", "", "technical"], scoreLevels: [])
        let (keys, _) = try CLMSchema.candidates(type: .choice, instructions: .string("Who?"), criteria: criteria)
        #expect(keys == ["billing", "technical"])
    }

    @Test func duplicateChoiceLabelsThrows() {
        #expect(throws: CLMRunQuestionError.duplicateChoiceLabels) {
            _ = try clmRunCriteria(type: .choice, choiceLabels: ["billing", "billing"], scoreLevels: [])
        }
    }
}
