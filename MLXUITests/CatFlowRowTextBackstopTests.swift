import Testing
import Foundation
@testable import MLXUI

/// RT-5 (`RSI/DelegateRowTextBacklog.md`) — the no-dead-ends backstop: `FlowEditorModel
/// .setRowText` writes a row's raw settings string outright, but refuses a commit that makes
/// things worse — either the whole document stops parsing, or the row gains a validator issue
/// it didn't already have — rolling the mutation back before returning, inside the same
/// `commitChange` call so a refusal never reaches the undo stack. The View half
/// (`rowTextDisclosure`'s commit-on-blur wiring, "left dirty" display) isn't independently
/// testable without instantiating the View (no precedent for that in this suite); this covers
/// the model layer it calls into, the same split RT-1..RT-4 used.
struct CatFlowRowTextBackstopTests {

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-rt5-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "RT5", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String?, model: String? = nil, settings: String? = nil,
                     refs: [Ref] = []) -> Row {
        Row(id: UUID(), task: task, blockKind: nil, model: model, settings: settings,
            refs: refs, children: [], clause: nil)
    }

    // MARK: - No-op

    @Test func settingSameTextIsANoOpAndPushesNoUndo() throws {
        let r = row("Transcribe", model: "Whisper Large v3", settings: "lang=en")
        let model = try editor([r])
        let undoCountBefore = model.undoStack.count
        let refusal = model.setRowText("lang=en", for: r.id)
        #expect(refusal == nil)
        #expect(model.row(withID: r.id)?.settings == "lang=en")
        #expect(model.undoStack.count == undoCountBefore)
    }

    // MARK: - An accepted edit commits and is undoable, like any other editor here

    @Test func acceptedEditCommitsAndIsUndoable() throws {
        let r = row("Transcribe", model: "Whisper Large v3", settings: "lang=en; timestamps=true")
        let model = try editor([r])
        let undoCountBefore = model.undoStack.count

        let refusal = model.setRowText("lang=de; timestamps=true", for: r.id)
        #expect(refusal == nil)
        #expect(model.row(withID: r.id)?.settings == "lang=de; timestamps=true")
        #expect(model.undoStack.count == undoCountBefore + 1)

        model.undo()
        #expect(model.row(withID: r.id)?.settings == "lang=en; timestamps=true")
    }

    /// This is the invariant RT-5 exists for: even a `Template` row (whole-string, RT-1) or a
    /// task with no dedicated Properties control at all (`Embed`, still deferred per RT-3) can
    /// be edited through this box.
    @Test func worksOnATaskWithNoDedicatedPropertiesControl() throws {
        let r = row("Embed", model: "BGE-M3", settings: "\"old query\"")
        let model = try editor([r])
        #expect(model.setRowText("\"a brand new query\"", for: r.id) == nil)
        #expect(model.row(withID: r.id)?.settings == "\"a brand new query\"")
    }

    // MARK: - A commit that introduces a new validator issue is refused, rolled back, off the undo stack

    @Test func editIntroducingANewPlaceholderErrorIsRefused() throws {
        let src1 = row("Read Text", settings: "a.txt")
        let src2 = row("Read Text", settings: "b.txt")
        let tmpl = row("Template", settings: "\"{1} {2}\"", refs: [.rowRef(src1.id), .rowRef(src2.id)])
        let model = try editor([src1, src2, tmpl])
        #expect(!model.issues(for: tmpl.id).contains { $0.code == "E207" })
        let undoCountBefore = model.undoStack.count
        let original = model.row(withID: tmpl.id)?.settings

        let refusal = model.setRowText("\"{1} {9}\"", for: tmpl.id)

        #expect(refusal != nil)
        #expect(refusal?.contains("position") == true)   // E207's own template mentions positions
        // Rolled back: the row's actual data is untouched, and no undo entry was pushed.
        #expect(model.row(withID: tmpl.id)?.settings == original)
        #expect(model.undoStack.count == undoCountBefore)
        #expect(!model.issues(for: tmpl.id).contains { $0.code == "E207" })
    }

    /// Edit back to something the row already tolerated before RT-5 ever touched it — a plain
    /// regression check that a *good* edit right after a *refused* one still goes through.
    @Test func aValidEditAfterARefusalStillCommits() throws {
        let src1 = row("Read Text", settings: "a.txt")
        let src2 = row("Read Text", settings: "b.txt")
        let tmpl = row("Template", settings: "\"{1} {2}\"", refs: [.rowRef(src1.id), .rowRef(src2.id)])
        let model = try editor([src1, src2, tmpl])

        #expect(model.setRowText("\"{1} {9}\"", for: tmpl.id) != nil)   // refused
        #expect(model.setRowText("\"{2} {1}\"", for: tmpl.id) == nil)   // valid -> accepted
        #expect(model.row(withID: tmpl.id)?.settings == "\"{2} {1}\"")
    }

    // MARK: - A commit that makes the whole document unparseable is refused

    @Test func editThatMakesTheDocumentUnparseableIsRefused() throws {
        let r = row("Read Text", settings: "in.txt")
        let model = try editor([r])
        #expect(model.canSave)
        let undoCountBefore = model.undoStack.count

        let refusal = model.setRowText("\"unterminated", for: r.id)

        #expect(refusal != nil)
        #expect(model.row(withID: r.id)?.settings == "in.txt")
        #expect(model.undoStack.count == undoCountBefore)
        #expect(model.canSave)   // the document as a whole is still fine
    }

    /// If the document was *already* unparseable before this edit (a pre-existing problem
    /// unrelated to this row), `setRowText` must not refuse an otherwise-fine edit just
    /// because the document couldn't parse to begin with — only a *newly* introduced parse
    /// failure is refused.
    @Test func doesNotRefuseWhenTheDocumentWasAlreadyUnparseableBeforeThisEdit() throws {
        let broken = row("Read Text", settings: "\"already unterminated")
        let other = row("Save Text", settings: "out.txt")
        let model = try editor([broken, other])
        #expect(!model.canSave)   // already broken, independent of anything this test does

        // Editing the *other*, unrelated row to something perfectly valid must still commit —
        // the pre-existing brokenness elsewhere isn't this edit's fault.
        let refusal = model.setRowText("output.md", for: other.id)
        #expect(refusal == nil)
        #expect(model.row(withID: other.id)?.settings == "output.md")
    }
}
