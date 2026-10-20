import Testing
import Foundation
@testable import MLXUI

/// RP-5 (`RSI/DelegateRowPortsBacklog.md`, owner 2026-10-20) — "Use my prompt" is a checkbox on
/// the three Generate rows, derived from the file every time (no stored flag), on by default
/// (a blank line on add when the row above would feed it); the Input picker is disabled while it
/// is ticked.
struct CatFlowUseMyPromptCheckboxTests {

    @MainActor private func editor(_ text: String) throws -> FlowEditorModel {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
            .domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let e = FlowEditorModel(name: "rp5", document: try CatParser.parse(text),
                                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        e.modelCatalog = catalog
        e.claimableModelIDs = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        return e
    }

    private func gen(_ task: String = "Generate Sound", blank: Bool = false) -> String {
        "mlxflow 0.8\n1. Read Text   notes.txt\n2. Summarize   \"Two sentences\"\n\(blank ? "\n" : "")3. \(task)   \"my own prompt\""
    }

    // MARK: - Derived state

    @Test @MainActor func theStateIsDerivedFromTheFile() throws {
        let row1 = try editor("mlxflow 0.8\n1. Generate Sound   \"calm\"")
        #expect(row1.usesOwnPrompt(for: row1.document.rows[0].id) == true)
        let broken = try editor(gen(blank: true))
        #expect(broken.usesOwnPrompt(for: broken.document.rows[2].id) == true)
        let auto = try editor(gen())
        #expect(auto.usesOwnPrompt(for: auto.document.rows[2].id) == false)
        let ref = try editor("mlxflow 0.8\n1. Read Text   notes.txt\n2. Read Image   p.png\n3. Generate Image   (1)   \"x\"")
        #expect(ref.usesOwnPrompt(for: ref.document.rows[2].id) == false)
        let nonText = try editor("mlxflow 0.8\n1. Read Image   p.png\n2. Generate Image   \"x\"")
        #expect(nonText.usesOwnPrompt(for: nonText.document.rows[1].id) == true)
        #expect(auto.usesOwnPrompt(for: auto.document.rows[1].id) == nil)      // not a Generate row
        // Fed with an empty prompt is still unticked (the note, not the state, needs a prompt).
        auto.setInstruction(nil, for: auto.document.rows[2].id)
        #expect(auto.usesOwnPrompt(for: auto.document.rows[2].id) == false)
    }

    // MARK: - On by default

    @Test @MainActor func addingUnderATextRowArrivesTickedWithABlankLine() throws {
        for g in ["Generate Image", "Generate Sound", "Generate Video"] {
            let e = try editor("mlxflow 0.8\n1. Read Text   notes.txt\n2. Summarize   \"Two sentences\"")
            e.selectedRowID = e.document.rows.last?.id
            e.add(task: g)
            let row = try #require(e.document.rows.last)
            #expect(row.task == g && row.chainBreak, "\(g)")
            #expect(e.usesOwnPrompt(for: row.id) == true)
            #expect(e.catText.contains("\n\n3. \(g)"), "\(e.catText)")
            #expect(row.settings == FlowEditorModel.defaultSettings(forTask: g))
            e.undo()
            #expect(e.document.rows.count == 2)
        }
        // Under a non-text row, or as row 1, there is nothing to cut: no blank line.
        let img = try editor("mlxflow 0.8\n1. Read Image   p.png")
        img.selectedRowID = img.document.rows.last?.id
        img.add(task: "Generate Image")
        #expect(img.document.rows.last?.chainBreak == false)
        let empty = try editor("mlxflow 0.8")
        empty.add(task: "Generate Sound")
        #expect(empty.document.rows.first?.chainBreak == false)
    }

    // MARK: - Unticking

    @Test @MainActor func unticksToTheRowAboveKeepsThePromptAndRetickRestoresIt() throws {
        let e = try editor(gen(blank: true))
        let id = e.document.rows[2].id
        let ticked = e.catText
        e.setUsesOwnPrompt(false, for: id)
        let row = try #require(e.row(withID: id))
        #expect(row.refs.isEmpty && !row.chainBreak)                         // plain auto-chain
        #expect(e.usesOwnPrompt(for: id) == false)
        #expect(e.catText.contains("\"my own prompt\""))                     // prompt kept
        #expect(e.autoChainInputLabel(for: id) == "row above · 2 Summarize")
        #expect(e.generatePromptSource(for: id)?.note == "Not used — the row above (2 · Summarize) gives this row its prompt.")
        e.setUsesOwnPrompt(true, for: id)
        #expect(e.catText == ticked)                                         // re-tick restores the file
        #expect(e.startsNewChainNote(for: id) == "Starts a new chain. Rows 1–2 still run but don't feed this row.")
        e.undo(); #expect(e.usesOwnPrompt(for: id) == false)                 // one undo per action
        e.undo(); #expect(e.catText == ticked)
    }

    @Test @MainActor func untickFallsBackToTheNearestEarlierTextRow() throws {
        let e = try editor("mlxflow 0.8\n1. Read Text   notes.txt\n2. Read Image   p.png\n\n3. Generate Image   \"x\"")
        let id = e.document.rows[2].id
        #expect(e.usesOwnPrompt(for: id) == true && e.canUseUpstreamPrompt(for: id))
        e.setUsesOwnPrompt(false, for: id)
        #expect(e.row(withID: id)?.refs == [.rowRef(e.document.rows[0].id)])
        #expect(e.generatePromptSource(for: id)?.note == "Not used — row 1 · Read Text gives this row its prompt.")
        e.undo()
        #expect(e.row(withID: id)?.refs.isEmpty == true)
    }

    @Test @MainActor func withNoEarlierTextRowTheBoxStaysTickedAndCannotBeUnticked() throws {
        let e = try editor("mlxflow 0.8\n1. Read Image   p.png\n2. Generate Image   \"x\"")
        let id = e.document.rows[1].id
        #expect(!e.canUseUpstreamPrompt(for: id))
        e.setUsesOwnPrompt(false, for: id)
        #expect(e.usesOwnPrompt(for: id) == true && e.undoStack.isEmpty)
        let first = try editor("mlxflow 0.8\n1. Generate Image   \"x\"")
        #expect(!first.canUseUpstreamPrompt(for: first.document.rows[0].id))
    }

    // MARK: - The picker, and opening

    @Test @MainActor func theInputPickerIsOffExactlyWhileTicked() throws {
        // The inspector disables the picker on `usesOwnPrompt == true` and nothing else.
        let e = try editor(gen(blank: true))
        let id = e.document.rows[2].id
        #expect(e.usesOwnPrompt(for: id) == true)
        e.setUsesOwnPrompt(false, for: id)
        #expect(e.usesOwnPrompt(for: id) == false)
    }

    @Test @MainActor func openingAFedRowShowsItUntickedAndRewritesNothing() throws {
        let e = try editor(gen("Generate Video"))
        let before = e.catText, doc = e.document
        let id = e.document.rows[2].id
        #expect(e.usesOwnPrompt(for: id) == false)
        _ = e.generatePromptSource(for: id); _ = e.autoChainInputLabel(for: id); _ = e.canUseUpstreamPrompt(for: id)
        #expect(e.catText == before && e.document == doc && e.undoStack.isEmpty)
    }
}
