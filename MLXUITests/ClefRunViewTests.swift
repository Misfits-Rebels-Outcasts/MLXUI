import Testing

@testable import MLXUI

/// CF-6 — `clefRunCriteria` (`Views/Run/ClefRunView.swift`): the pure screen-state → `ClefJSON`
/// builder, tested directly rather than through the view. It feeds the same call
/// `ClefEngine.answer` makes, so the shapes are what the engine sees: **choice** an object
/// `{label: ""}` (the CL-5-FIX-1 lesson — a list loses the option ids the answer is keyed by),
/// **score** an ordered array, **noul** nothing.
struct ClefRunViewTests {
    @Test func defaultChoiceStateIsAnObjectWithEmptyDescriptions() throws {
        let criteria = try clefRunCriteria(
            type: .choice, choiceLabels: ["billing", "technical", "sales"], scoreLevels: [])
        // The reference's own shape: ordered keys, `""` descriptions kept (not dropped).
        #expect(criteria == .object([
            ClefJSONField(key: "billing", value: .string("")),
            ClefJSONField(key: "technical", value: .string("")),
            ClefJSONField(key: "sales", value: .string("")),
        ]))
        // …and the prompt builder accepts it, keeping every option.
        let options = try ClefPrompt.questionOptions(
            ClefQuestion(id: "q", type: .choice, instructions: nil, criteria: criteria))
        #expect(options.map(\.id) == ["billing", "sales", "technical"])   // id-sorted for the prompt
        #expect(options.map(\.description) == [.string(""), .string(""), .string("")])
    }

    @Test func defaultScoreStateIsAnOrderedArray() throws {
        let criteria = try clefRunCriteria(
            type: .score, choiceLabels: [], scoreLevels: ["not urgent", "soon", "critical"])
        #expect(criteria == .array([.string("not urgent"), .string("soon"), .string("critical")]))
        let options = try ClefPrompt.questionOptions(
            ClefQuestion(id: "q", type: .score, instructions: nil, criteria: criteria))
        #expect(options.map(\.id) == ["0", "1", "2"])
    }

    @Test func yesNoSendsNoCriteria() throws {
        #expect(try clefRunCriteria(type: .noul, choiceLabels: [], scoreLevels: []) == nil)
    }

    @Test func blankChoiceRowsAreDroppedBeforeTheEngineSeesThem() throws {
        let criteria = try clefRunCriteria(
            type: .choice, choiceLabels: ["billing", "  ", "technical"], scoreLevels: [])
        #expect(criteria == .object([
            ClefJSONField(key: "billing", value: .string("")),
            ClefJSONField(key: "technical", value: .string("")),
        ]))
    }

    @Test func blankScoreLevelsBecomeAPlaceholder() throws {
        let criteria = try clefRunCriteria(type: .score, choiceLabels: [], scoreLevels: ["calm", "", "furious"])
        #expect(criteria == .array([.string("calm"), .string("—"), .string("furious")]))
    }

    @Test func duplicateChoiceLabelsThrow() {
        #expect(throws: ClefRunQuestionError.duplicateChoiceLabels) {
            _ = try clefRunCriteria(type: .choice, choiceLabels: ["billing", "billing"], scoreLevels: [])
        }
    }
}
