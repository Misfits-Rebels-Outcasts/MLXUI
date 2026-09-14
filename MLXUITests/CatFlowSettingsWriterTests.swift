import Testing
import Foundation
@testable import MLXUI

/// Phase SP, SP-2 (`RSI/DelegateFixItBacklog.md`; `SPEC_QUESTIONS.md` Q226 in `catflow-mlx`,
/// open) — `FlowSettingsEditor.replace(key:value:)` now treats an empty or whitespace-only
/// value the same as `nil` (a removal), so no writer in this app can author a valueless
/// `key=` token. This is the cheaper of Phase SP's two fixes and is correct regardless of how
/// Q226 is eventually answered: it only changes what a fresh edit *writes*, never how an
/// existing file's text is *read* (`FlowSettings`/`FlowValidator.parseSettingsKV` both
/// untouched — see `CatFlowHumanPolicyTests
/// .switchingToGiveUpAfterWithAnEmptyDurationStillRaisesE501` for the human-row symptom this
/// closes).
struct CatFlowSettingsWriterTests {

    @Test func anEmptyValueOnANewKeyWritesNothing() {
        #expect(FlowSettingsEditor.replace(key: "timeout", value: "", in: "") == "")
        #expect(FlowSettingsEditor.replace(key: "timeout", value: "", in: nil) == "")
    }

    @Test func aWhitespaceOnlyValueOnANewKeyWritesNothing() {
        #expect(FlowSettingsEditor.replace(key: "timeout", value: "   ", in: "") == "")
    }

    @Test func anEmptyValueOnAnExistingKeyRemovesTheTokenEntirely() {
        // `removingToken` (the same path a `nil` removal already takes — see
        // `removalWithNilAndRemovalWithEmptyStringAgreeOnTheResult` below) strips only one
        // adjacent separator, so a trailing `;` survives when the removed token was last;
        // pre-existing `FlowSettingsEditor` behavior, unrelated to SP-2. The point here is the
        // token itself is gone.
        let result = FlowSettingsEditor.replace(key: "timeout", value: "", in: "default=approve; timeout=10m")
        #expect(result == "default=approve;")
        #expect(result?.contains("timeout") == false)
    }

    @Test func aWhitespaceOnlyValueOnAnExistingKeyRemovesTheTokenEntirely() {
        let result = FlowSettingsEditor.replace(key: "timeout", value: "  ", in: "default=approve; timeout=10m")
        #expect(result == "default=approve;")
    }

    @Test func anOrdinaryNonEmptyValueStillWritesNormally() {
        #expect(FlowSettingsEditor.replace(key: "timeout", value: "10m", in: "") == "timeout=10m")
        #expect(FlowSettingsEditor.replace(key: "timeout", value: "10m", in: "default=approve") == "default=approve; timeout=10m")
    }

    @Test func removalWithNilAndRemovalWithEmptyStringAgreeOnTheResult() {
        let base = "default=approve; timeout=10m"
        #expect(FlowSettingsEditor.replace(key: "timeout", value: nil, in: base)
                == FlowSettingsEditor.replace(key: "timeout", value: "", in: base))
    }

    /// SP-2's own exit test, stated in the backlog: the picker's exact call for a blank
    /// duration field leaves no `timeout` token at all.
    @Test @MainActor func setHumanWaitPolicyWithAnEmptyTimeoutLeavesNoTimeoutToken() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("catflow-sp2-\(UUID().uuidString)")
        let r = Row(id: UUID(), task: "Ask Human", settings: "wait=forever", tags: ["approve", "edit"])
        let model = FlowEditorModel(name: "SP-2", document: FlowDocument(version: "0.8", rows: [r]),
                                    workspace: FlowWorkspace(root: root))
        model.setHumanWaitPolicy(.giveUpAfter(timeout: "", default: "approve"), for: r.id)
        let settings = model.row(withID: r.id)?.settings ?? ""
        #expect(!settings.contains("timeout"))
        #expect(settings.contains("default=approve"))
    }
}
