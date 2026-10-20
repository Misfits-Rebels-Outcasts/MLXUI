import Testing
import Foundation
@testable import MLXUI

/// RP-4 (`RSI/DelegateRowPortsBacklog.md`, owner 2026-10-20: "Just quietly add the blank line.")
/// Picking a single `(n)` that isn't the row above, while the row above would also fit, sets
/// `chainBreak` in the same undo step, so FlowValidator's ambiguous-auto-chain lint (Python
/// parity, untouched) never has a reason to fire.
struct CatFlowReferenceBreakTests {

    @MainActor private func editor(_ text: String) throws -> FlowEditorModel {
        FlowEditorModel(name: "rp4", document: try CatParser.parse(text),
                        workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
    }

    private func lint(_ e: FlowEditorModel) throws -> Bool {
        try FlowValidator.checkFlow(CatParser.parseForValidation(e.catText)).contains { $0.code == "ambiguous-auto-chain" }
    }

    private let flow = "mlxflow 0.8\n1. Read Text   notes.txt\n2. Summarize   \"a\"\n3. Summarize   \"b\""

    @Test @MainActor func pickingANonAdjacentRowWritesABlankLineAndTheLintStaysQuiet() throws {
        // Control: the same shape without the blank line is the lint's exact trigger.
        let unbroken = try editor("mlxflow 0.8\n1. Read Text   notes.txt\n2. Summarize   \"a\"\n3. Summarize   (1)   \"b\"")
        #expect(try lint(unbroken))
        let e = try editor(flow)
        let r = e.document.rows
        e.setReference(to: r[0].id, for: r[2].id)
        #expect(e.row(withID: r[2].id)?.chainBreak == true)
        #expect(e.catText.contains("\n\n3. Summarize"), "\(e.catText)")
        #expect(try !lint(e))
        #expect(try CatSerializer.serialize(CatParser.parse(e.catText)) == e.catText)
    }

    @Test @MainActor func switchingBackToTheRowAboveRemovesItAndUndoRestoresBoth() throws {
        let e = try editor(flow)
        let r = e.document.rows
        let original = e.catText
        e.setReference(to: r[0].id, for: r[2].id)
        let withBreak = e.catText
        e.setReference(to: r[1].id, for: r[2].id)
        #expect(e.row(withID: r[2].id)?.chainBreak == false)
        #expect(try !lint(e))
        e.undo()
        #expect(e.catText == withBreak)
        e.undo()
        #expect(e.catText == original)
        #expect(e.row(withID: r[2].id)?.refs.isEmpty == true && e.row(withID: r[2].id)?.chainBreak == false)
    }

    @Test @MainActor func bundlesAndAnyKindAreUntouched() throws {
        let t = try editor("mlxflow 0.8\n1. Read Text   a.txt\n2. Read Text   b.txt\n3. Template   (1,2)   \"{1}{2}\"")
        let r = t.document.rows
        #expect(!r[2].chainBreak)
        t.setReference(to: r[1].id, slot: 2, for: r[2].id)
        #expect(t.row(withID: r[2].id)?.chainBreak == false)        // two refs: left as is

        let any = try editor("mlxflow 0.8\n1. Read Text   a.txt\n2. Summarize   \"a\"\n3. Range   1..3")
        any.setReference(to: any.document.rows[0].id, for: any.document.rows[2].id)
        #expect(any.row(withID: any.document.rows[2].id)?.chainBreak == false)   // anyKind
    }

    @Test @MainActor func aRowWhoseAboveDoesNotFitGetsNoBreak() throws {
        let e = try editor("mlxflow 0.8\n1. Read Text   a.txt\n2. Read Image   p.png\n3. Summarize   \"b\"")
        let r = e.document.rows
        e.setReference(to: r[0].id, for: r[2].id)
        #expect(e.row(withID: r[2].id)?.chainBreak == false)
    }

    @Test @MainActor func openingAFileNeverChangesIt() throws {
        let e = try editor(flow)
        let before = e.catText
        _ = e.validInputs(for: e.document.rows[2].id)
        #expect(e.catText == before && e.undoStack.isEmpty)
    }

    @Test @MainActor func openingAnyGalleryFlowNeverRecordsAChange() throws {
        for meta in GalleryLoader.loadMetadata() {
            let fid = (meta.filename as NSString).deletingPathExtension
            guard let doc = try? GalleryLoader.loadDocument(flowID: fid) else { continue }
            let e = FlowEditorModel(name: fid, document: doc, workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
            #expect(e.undoStack.isEmpty && e.catText == CatSerializer.serialize(doc), "\(fid)")
        }
    }
}
