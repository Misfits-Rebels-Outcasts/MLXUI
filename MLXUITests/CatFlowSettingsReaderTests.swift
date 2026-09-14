import Testing
import Foundation
@testable import MLXUI

/// Phase SP, SP-3 (`RSI/DelegateFixItBacklog.md`; `SPEC_QUESTIONS.md` Q226 in `catflow-mlx`,
/// resolved "absent", owner ruling 2026-09-14) — `FlowSettings` now reads a valueless `key=`
/// (nothing at all after the `=`) the same way `FlowValidator.parseSettingsKV`'s `reKV`
/// already does: **absent**, not present with an empty value. This is a deliberate parity
/// divergence from `catflow-mlx`'s `tools/_settings.py::Settings`, which has no such check,
/// until the reference adopts the same ruling — see `FlowSettings.swift`'s `init`.
struct CatFlowSettingsReaderTests {

    // MARK: - The core reading rule

    @Test func aBareValuelessKeyReadsAsAbsent() {
        let s = FlowSettings("timeout=")
        #expect(s.value(for: "timeout") == nil)
        #expect(s.has("timeout") == false)
        #expect(s.multi(for: "timeout") == [])
        #expect(s.orderedKeys.isEmpty)
    }

    @Test func aBareValuelessKeyAmongOthersReadsAsAbsentWithoutDisturbingTheRest() {
        let s = FlowSettings("default=approve; timeout=")
        #expect(s.value(for: "default") == "approve")
        #expect(s.value(for: "timeout") == nil)
        #expect(s.orderedKeys == ["default"])
    }

    @Test func aQuotedEmptyValueIsUnaffectedAndStillReadsAsPresent() {
        // `key=""` has two characters after the `=` (the quote marks) — not the bare,
        // nothing-at-all case Q226 is about. Both `FlowSettings` and `FlowValidator.reKV`
        // already agreed on this one; SP-3 must not change it.
        let s = FlowSettings(#"timeout="""#)
        #expect(s.value(for: "timeout") == "")
        #expect(s.has("timeout"))
    }

    @Test func anOrdinaryValueIsUnaffected() {
        let s = FlowSettings("timeout=10m")
        #expect(s.value(for: "timeout") == "10m")
    }

    @Test func aRepeatedKeyWhereOneOccurrenceIsValuelessDropsOnlyThatOccurrence() {
        // `multi` is every occurrence in source order; a valueless one contributes nothing.
        let s = FlowSettings("x=1; x=; x=3")
        #expect(s.multi(for: "x") == ["1", "3"])
        #expect(s.value(for: "x") == "3")   // last *present* occurrence
    }

    // MARK: - `waitPolicyDescription` — the user-visible symptom SP-3 closes

    @Test func aRowWithAValuelessTimeoutNoLongerShowsAFakePolicy() {
        // The exact text a hand-edited or pre-SP-2 `.cat` might still contain.
        let description = FlowEditorModel.waitPolicyDescription(settings: "default=approve; timeout=")
        #expect(!description.contains("Gives up after"))
        #expect(description.contains("No waiting policy is set"))
    }

    @Test func aRowWithARealTimeoutStillShowsTheRealPolicy() {
        let description = FlowEditorModel.waitPolicyDescription(settings: "default=approve; timeout=10m")
        #expect(description == "Gives up after 10m and proceeds as \"approve\".")
    }

    // MARK: - Both parsers now agree

    @Test func flowSettingsAndTheValidatorNowAgreeOnAValuelessTimeout() throws {
        // The exact shape `CatFlowHumanRowsTests` already exercises as a running flow, with
        // `timeout=1s` reduced to a bare `timeout=`.
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Ask Human    (1)  "Send this reply?" ; default=edit; timeout=
           -> { edit: 3 }
        3. Save Text    out.md
        """
        let flow = try CatParser.parseForValidation(text)
        let found = FlowValidator.checkFlow(flow)
        #expect(found.contains { $0.code == "E501" })

        // The row that raised E501 also reads (via FlowSettings) as having no timeout —
        // the exact disagreement Q226 closed. Before SP-3 this would have been `""`.
        let row = try #require(flow.rows.first { $0.task == "Ask Human" })
        #expect(FlowSettings(row.settings).value(for: "timeout") == nil)
    }
}
