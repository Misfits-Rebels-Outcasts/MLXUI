import Testing
import Foundation
@testable import MLXUI

/// RT-1 (`RSI/DelegateRowTextBacklog.md`) — the `Template` row Pattern editor: `FlowEditorModel
/// .setPattern` → `FlowSettingsEditor.replaceWholeSettings`, writing the row's **entire**
/// settings string (fact 11 — the runtime reads `TextTools.unquoteWhole(settings)`, not a
/// single spliced token), plus the E207 placeholder check that already reaches the inspector
/// (fact 13) staying live as the box is edited.
struct CatFlowTemplateEditorTests {

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-rt1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "RT1", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String?, settings: String? = nil, refs: [Ref] = []) -> Row {
        Row(id: UUID(), task: task, blockKind: nil, model: nil, settings: settings,
            refs: refs, children: [], clause: nil)
    }

    // MARK: - QR9 on the real gallery row (the owner's own report)

    /// The exit criterion's own row: `73-WebSummaryLinks.cat`'s row 3.2. A no-op edit through
    /// `setPattern` must leave the settings string byte-identical, on the live document and
    /// after a save → reparse round trip.
    @Test func noOpPatternEditIsByteIdenticalOn73WebSummaryLinks() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "73-WebSummaryLinks")
        let templateRow = doc.rows[2].children[1]
        #expect(templateRow.task == "Template")
        let original = templateRow.settings
        let pattern = TextTools.unquoteWhole(original ?? "")
        #expect(pattern.contains("PAGE_TEXT"))   // sanity: this is really row 3.2's pattern

        let model = try editor(doc.rows)
        model.setPattern(pattern, for: templateRow.id)
        #expect(model.row(withID: templateRow.id)?.settings == original)

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        #expect(reparsed.rows[2].children[1].settings == original)
    }

    // MARK: - Escaping round-trips

    @Test func newlinesInThePatternBecomeEscapedOnDiskAndBack() throws {
        let r1 = row("Template", settings: "\"old\"")
        let model = try editor([r1])
        let pattern = "Line one\nLine two\n\n{1}"
        model.setPattern(pattern, for: r1.id)
        let onDisk = model.row(withID: r1.id)?.settings
        #expect(onDisk == "\"Line one\\nLine two\\n\\n{1}\"")
        #expect(TextTools.unquoteWhole(onDisk ?? "") == pattern)

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        #expect(TextTools.unquoteWhole(reparsed.rows[0].settings ?? "") == pattern)
    }

    @Test func embeddedQuoteRoundTrips() throws {
        let r1 = row("Template", settings: "\"old\"")
        let model = try editor([r1])
        let pattern = #"Say "hello" to {1}"#
        model.setPattern(pattern, for: r1.id)
        #expect(TextTools.unquoteWhole(model.row(withID: r1.id)?.settings ?? "") == pattern)

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        #expect(TextTools.unquoteWhole(reparsed.rows[0].settings ?? "") == pattern)
    }

    /// The single-quote / angle-bracket case the real 73-WebSummaryLinks pattern actually
    /// carries (hazard 2 — `quote`/`unquote` must stay exact inverses on these).
    @Test func singleQuotesAndAngleBracketsRoundTrip() throws {
        let r1 = row("Template", settings: "\"old\"")
        let model = try editor([r1])
        let pattern = "<a href='{1}'>Read the story</a>"
        model.setPattern(pattern, for: r1.id)
        #expect(TextTools.unquoteWhole(model.row(withID: r1.id)?.settings ?? "") == pattern)

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        #expect(TextTools.unquoteWhole(reparsed.rows[0].settings ?? "") == pattern)
    }

    // MARK: - Clearing

    @Test func clearingThePatternRemovesTheSettingsString() throws {
        let r1 = row("Template", settings: "\"old\"")
        let model = try editor([r1])
        model.setPattern("", for: r1.id)
        #expect(model.row(withID: r1.id)?.settings == nil)
        model.setPattern(nil, for: r1.id)
        #expect(model.row(withID: r1.id)?.settings == nil)
    }

    // MARK: - E207 stays live as the box is edited (fact 13)

    @Test func placeholderBeyondInputCountRaisesE207() throws {
        let src1 = row("Read Text", settings: "a.txt")
        let src2 = row("Read Text", settings: "b.txt")
        let tmpl = row("Template", settings: "\"{1} {2}\"", refs: [.rowRef(src1.id), .rowRef(src2.id)])
        let model = try editor([src1, src2, tmpl])
        #expect(!model.issues(for: tmpl.id).contains { $0.code == "E207" })

        model.setPattern("{1} {9}", for: tmpl.id)
        #expect(model.issues(for: tmpl.id).contains { $0.code == "E207" })
        #expect(model.warning(for: tmpl.id) != nil)

        // Editing back to a valid placeholder clears it in the same model instance — no
        // extra cache-clear call, exactly what "reaches the pane live" means.
        model.setPattern("{1} {2}", for: tmpl.id)
        #expect(!model.issues(for: tmpl.id).contains { $0.code == "E207" })
    }

    // MARK: - The serializer still wraps at 74 after a long edit (fact 14)

    @Test func serializerStillWrapsLongPatternsAt74() throws {
        let r1 = row("Template", settings: "\"short\"")
        let model = try editor([r1])
        let longPattern = "This is a very long template pattern that keeps going and going " +
            "with {1} placeholders and {input} content until it is well past seventy four " +
            "characters wide, which the serializer must wrap onto continuation lines " +
            "indented to the settings column rather than emit as one long unbroken line."
        model.setPattern(longPattern, for: r1.id)

        let text = model.catText
        #expect(text.contains(longPattern.prefix(20)))   // sanity: the edit actually landed
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            #expect(line.unicodeScalars.count <= 74, "line exceeds wrap width 74: '\(line)'")
        }
        // Re-parses clean and round-trips to the same wrapped text (fmt idempotence).
        let reparsed = try CatParser.parse(text)
        #expect(TextTools.unquoteWhole(reparsed.rows[0].settings ?? "") == longPattern)
        #expect(CatSerializer.serialize(reparsed) == text)
    }

    // MARK: - RT-1's `isInsideEach` (the `{index}`/`{item}` chip gate)

    @Test func isInsideEachFindsNestingAtAnyDepth() throws {
        let inner = row("Template", settings: "\"{1}\"")
        let each = Row(id: UUID(), task: nil, blockKind: .each, model: nil, settings: nil,
                       refs: [], children: [inner], clause: nil)
        let outer = row("Template", settings: "\"{1}\"")
        let model = try editor([outer, each])
        #expect(model.isInsideEach(inner.id) == true)
        #expect(model.isInsideEach(outer.id) == false)
    }
}
