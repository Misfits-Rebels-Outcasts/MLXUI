import Testing
import Foundation
@testable import MLXUI

/// RP-3 (`RSI/DelegateRowPortsBacklog.md`, owner ruling 2026-10-20, "go with your cheaper
/// version") — a fed Generate row ignores its typed prompt; the Step tab says so and offers
/// "Use my prompt instead". These pin the model side (`generatePromptSource`,
/// `useMyPromptInstead`, `startsNewChainNote`); the SwiftUI is a thin reader of them.
struct CatFlowGeneratePromptSourceTests {

    @MainActor private func editor(_ text: String) throws -> FlowEditorModel {
        FlowEditorModel(name: "rp3", document: try CatParser.parse(text),
                        workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
    }

    private func chain(_ gen: String, blank: Bool = false) -> String {
        """
        mlxflow 0.8
        1. Read Text            notes.txt
        2. Summarize            "Two sentences"
        \(blank ? "\n" : "")3. \(gen)   "my own prompt"
        """
    }

    private static let generators = ["Generate Image", "Generate Sound", "Generate Video"]

    // MARK: - The note

    @Test @MainActor func theNoteShowsForAllThreeGeneratorsUnderATextRow() throws {
        for g in Self.generators {
            let e = try editor(chain(g))
            let src = try #require(e.generatePromptSource(for: e.document.rows[2].id), "\(g)")
            #expect(src.kind == .rowAbove)
            #expect(src.note == "Not used — the row above (2 · Summarize) gives this row its prompt.")
        }
    }

    @Test @MainActor func theNoteShowsForATextRef() throws {
        for g in Self.generators {
            let e = try editor("""
            mlxflow 0.8
            1. Read Text            notes.txt
            2. Read Image           photo.png
            3. \(g)   (1)   "my own prompt"
            """)
            let src = try #require(e.generatePromptSource(for: e.document.rows[2].id), "\(g)")
            #expect(src.kind == .reference)
            #expect(src.note == "Not used — row 1 · Read Text gives this row its prompt.")
        }
    }

    @Test @MainActor func noNoteAtRowOneAfterABreakUnderNonTextOrWithAnEmptyPrompt() throws {
        let first = try editor("mlxflow 0.8\n1. Generate Sound   \"calm\"")
        #expect(first.generatePromptSource(for: first.document.rows[0].id) == nil)

        let broken = try editor(chain("Generate Sound", blank: true))
        #expect(broken.generatePromptSource(for: broken.document.rows[2].id) == nil)

        let image = try editor("""
        mlxflow 0.8
        1. Read Image           photo.png
        2. Generate Sound   "my own prompt"
        """)
        #expect(image.generatePromptSource(for: image.document.rows[1].id) == nil)

        let empty = try editor(chain("Generate Sound"))
        empty.setInstruction(nil, for: empty.document.rows[2].id)
        #expect(empty.generatePromptSource(for: empty.document.rows[2].id) == nil)

        let other = try editor("""
        mlxflow 0.8
        1. Read Text            notes.txt
        2. Summarize            "Two sentences"
        """)
        #expect(other.generatePromptSource(for: other.document.rows[1].id) == nil)
    }

    // MARK: - The button

    @Test @MainActor func autoChainButtonAddsABlankLineKeepsThePromptAndUndoes() throws {
        let e = try editor(chain("Generate Sound"))
        let id = e.document.rows[2].id
        let original = e.catText
        e.useMyPromptInstead(for: id)

        let text = e.catText
        #expect(text.contains("\n\n3. Generate Sound"), "\(text)")
        #expect(text.contains("\"my own prompt\""))
        #expect(e.generatePromptSource(for: id) == nil)
        #expect(try CatSerializer.serialize(CatParser.parse(text)) == text)

        e.undo()
        #expect(e.catText == original)
        #expect(e.generatePromptSource(for: id) != nil)
    }

    @Test @MainActor func refButtonRemovesTheRefAndSetsTheBreakWhenTheRowAboveMakesText() throws {
        let e = try editor("""
        mlxflow 0.8
        1. Read Text            notes.txt
        2. Summarize   (1)      "Two sentences"
        3. Generate Image   (1)   "my own prompt"
        """)
        let id = e.document.rows[2].id
        let original = e.catText
        e.useMyPromptInstead(for: id)
        let row = try #require(e.row(withID: id))
        #expect(row.refs.isEmpty && row.chainBreak)
        #expect(e.generatePromptSource(for: id) == nil)
        e.undo()
        #expect(e.catText == original)
        #expect(e.row(withID: id)?.refs.isEmpty == false)
    }

    @Test @MainActor func refButtonLeavesNoBreakWhenTheRowAboveIsNotText() throws {
        let e = try editor("""
        mlxflow 0.8
        1. Read Text            notes.txt
        2. Read Image           photo.png
        3. Generate Image   (1)   "my own prompt"
        """)
        let id = e.document.rows[2].id
        e.useMyPromptInstead(for: id)
        let row = try #require(e.row(withID: id))
        #expect(row.refs.isEmpty && !row.chainBreak)
    }

    @Test @MainActor func buttonWithAnEmptyPromptFillsTheRP2Default() throws {
        for g in Self.generators {
            let e = try editor(chain(g))
            let id = e.document.rows[2].id
            e.setInstruction(nil, for: id)
            e.useMyPromptInstead(for: id)
            let row = try #require(e.row(withID: id))
            #expect(row.chainBreak)
            #expect(row.settings == FlowEditorModel.defaultSettings(forTask: g), "\(g)")
            e.undo()
            #expect(e.row(withID: id)?.chainBreak == false)
        }
    }

    // MARK: - Opening never rewrites

    @Test @MainActor func openingAFedRowLeavesTheTextUnchanged() throws {
        let text = chain("Generate Video") + "\n"
        let e = try editor(text)
        let before = e.catText
        let doc = e.document
        _ = e.generatePromptSource(for: e.document.rows[2].id)
        _ = e.startsNewChainNote(for: e.document.rows[2].id)
        #expect(e.catText == before)
        #expect(e.document == doc)
        #expect(e.undoStack.isEmpty)
    }

    // MARK: - "Rows still run"

    @Test @MainActor func theStillRunLineAppearsOnlyWhileTheRowAboveFeedsNothing() throws {
        let e = try editor(chain("Generate Sound"))
        let id = e.document.rows[2].id
        #expect(e.startsNewChainNote(for: id) == nil)               // still fed: no switch yet
        e.useMyPromptInstead(for: id)
        #expect(e.startsNewChainNote(for: id) == "Starts a new chain. Rows 1–2 still run but don't feed this row.")

        // Something else uses row 2's output → it still feeds something → no line.
        let used = try editor("""
        mlxflow 0.8
        1. Read Text            notes.txt
        2. Summarize            "Two sentences"

        3. Generate Sound   "my own prompt"
        4. Save Text   (2)   out.md
        """)
        #expect(used.startsNewChainNote(for: used.document.rows[2].id) == nil)

        // Nothing above the break to keep running.
        let top = try editor("mlxflow 0.8\n1. Generate Sound   \"calm\"")
        #expect(top.startsNewChainNote(for: top.document.rows[0].id) == nil)
    }
}
