import Testing
import Foundation
@testable import MLXUI

/// RT-2 (`RSI/DelegateRowTextBacklog.md`) — `Ask Human`/`Human Input` get their question back:
/// an `.instructionBox`-shaped control, labelled "What should the person be asked?", written
/// with the existing `setInstruction` (first quoted token) — these rows also carry
/// `timeout=`/`default=`/`wait=`, which the splice already leaves alone. `tags:` isn't part of
/// `settings` at all (`CatParser` lifts it into `Row.tags` before the editor ever sees it), so
/// the round-trip tests go through a real save → reparse, not a hand-built `Row`.
struct CatFlowHumanQuestionEditorTests {

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-rt2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "RT2", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String?, settings: String? = nil) -> Row {
        Row(id: UUID(), task: task, blockKind: nil, model: nil, settings: settings,
            refs: [], children: [], clause: nil)
    }

    // MARK: - Real corpus rows: timeout=/default=/tags: survive an edit byte-identically

    @Test func editingAskHumanQuestionPreservesEverythingElseOn38ReplyApproval() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "38-ReplyApproval")
        let askHuman = doc.rows[3]
        #expect(askHuman.task == "Ask Human")
        let original = askHuman.settings ?? ""
        let originalTags = askHuman.tags

        let model = try editor(doc.rows)
        // A no-op edit (rewriting the same question back) is byte-identical (QR9).
        model.setInstruction(FlowInterpreter.rowPromptText(askHuman), for: askHuman.id)
        #expect(model.row(withID: askHuman.id)?.settings == original)

        // A real edit changes only the quoted span — everything else (timeout=, default=,
        // whatever else the line carries) survives byte-identically around it.
        model.setInstruction("Send this reply to the customer?", for: askHuman.id)
        let edited = model.row(withID: askHuman.id)?.settings ?? ""
        let originalQuoted = try #require(FlowSettingsEditor.firstQuotedToken(in: original)).token
        let editedQuoted = try #require(FlowSettingsEditor.firstQuotedToken(in: edited)).token
        #expect(edited == original.replacingOccurrences(of: originalQuoted, with: editedQuoted))
        #expect(model.row(withID: askHuman.id)?.tags == originalTags)
        #expect(FlowInterpreter.rowPromptText(model.row(withID: askHuman.id)!) == "Send this reply to the customer?")

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        let reparsedRow = reparsed.rows[3]
        #expect(reparsedRow.settings == edited)
        #expect(reparsedRow.tags == originalTags)
        #expect(FlowInterpreter.rowPromptText(reparsedRow) == "Send this reply to the customer?")
    }

    @Test func editingHumanInputQuestionPreservesEverythingElseOn18DocChat() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "18-DocChat")
        let humanInput = doc.rows[1]
        #expect(humanInput.task == "Human Input")
        let original = humanInput.settings ?? ""
        #expect(FlowSettings(original).value(for: "wait") == "forever")
        // `firstBare()` is the quoted question itself here (it's bare too — no `key=` label);
        // `ctx+` is the *second* bare token, only reachable via the full list.
        #expect(FlowSettings(original).testBare.contains("ctx+"))

        let model = try editor(doc.rows)
        model.setInstruction(FlowInterpreter.rowPromptText(humanInput), for: humanInput.id)
        #expect(model.row(withID: humanInput.id)?.settings == original)

        model.setInstruction("What do you want to know?", for: humanInput.id)
        let edited = model.row(withID: humanInput.id)?.settings ?? ""
        let originalQuoted = try #require(FlowSettingsEditor.firstQuotedToken(in: original)).token
        let editedQuoted = try #require(FlowSettingsEditor.firstQuotedToken(in: edited)).token
        #expect(edited == original.replacingOccurrences(of: originalQuoted, with: editedQuoted))
        #expect(FlowSettings(edited).value(for: "wait") == "forever")
        #expect(FlowSettings(edited).testBare.contains("ctx+"))
        #expect(FlowInterpreter.rowPromptText(model.row(withID: humanInput.id)!) == "What do you want to know?")

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        let reparsedRow = reparsed.rows[1]
        #expect(reparsedRow.settings == edited)
        #expect(FlowInterpreter.rowPromptText(reparsedRow) == "What do you want to know?")
    }

    // MARK: - `rowPromptText` and `firstQuotedToken` must agree, corpus-wide

    /// Fact 23b: `rowPromptText` (what the human actually sees at run time) and
    /// `firstQuotedToken` + `unquote` (what the Properties box reads and writes) are two
    /// different readers of the same settings string. If they ever disagreed on a real row,
    /// the box would be editing a token the runtime doesn't display — this is the regression
    /// gate for that.
    @Test func rowPromptTextAgreesWithTheEditorsReadingOnEveryHumanRowInTheCorpus() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
        let dirs = [
            root.appendingPathComponent("MLXUI/Resources/Gallery"),
            root.appendingPathComponent("MLXUI/Resources/Workspaces"),
        ]
        var checked = 0
        for dir in dirs {
            for file in try FileManager.default.contentsOfDirectory(atPath: dir.path)
                .filter({ $0.hasSuffix(".cat") }).sorted() {
                let text = try String(contentsOf: dir.appendingPathComponent(file), encoding: .utf8)
                let doc = try CatParser.parse(text)
                for r in allRows(doc.rows) where r.task == "Ask Human" || r.task == "Human Input" {
                    let runtime = FlowInterpreter.rowPromptText(r)
                    let editorReading = FlowSettingsEditor.firstQuotedToken(in: r.settings ?? "")
                        .map { FlowSettings.unquote($0.token) } ?? ""
                    #expect(runtime == editorReading,
                            "\(file): rowPromptText '\(runtime)' != editor reading '\(editorReading)' for settings '\(r.settings ?? "")'")
                    checked += 1
                }
            }
        }
        #expect(checked >= 5, "expected several human rows in the bundled corpus (got \(checked))")
    }

    /// The one input that actually separates the two readers: an unquoted `;`/`·` ends
    /// `rowPromptText`'s scan, but one *inside* quotes doesn't — and `firstQuotedToken` never
    /// treats it specially either way. No corpus row happens to carry one today (the check
    /// above covers what exists); this constructs the case so it's covered regardless.
    @Test func aQuestionContainingASemicolonRoundTripsIdenticallyThroughBothReaders() throws {
        let settings = "\"Ready to ship; ok?\"; timeout=1h; default=no"
        let r = row("Ask Human", settings: settings)
        let runtime = FlowInterpreter.rowPromptText(r)
        let editorReading = FlowSettingsEditor.firstQuotedToken(in: settings)
            .map { FlowSettings.unquote($0.token) } ?? ""
        #expect(runtime == "Ready to ship; ok?")
        #expect(editorReading == "Ready to ship; ok?")
        #expect(runtime == editorReading)

        // And it survives an edit through the box's own writer.
        let model = try editor([r])
        model.setInstruction("Still ready to ship; ok?", for: r.id)
        let edited = model.row(withID: r.id)?.settings
        #expect(edited == "\"Still ready to ship; ok?\"; timeout=1h; default=no")
        #expect(FlowInterpreter.rowPromptText(model.row(withID: r.id)!) == "Still ready to ship; ok?")
    }

    private func allRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + allRows($0.children) }
    }
}
