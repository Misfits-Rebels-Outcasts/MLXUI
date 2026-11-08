import Testing
import Foundation
@testable import MLXUI

/// SF — a new `Save *` row arrives with a default path named for what it saves; editor-only
/// soft warnings when the name contradicts the writer. No validator or grammar change.
struct CatFlowSaveDefaultNameTests {

    private func editor(_ rows: [Row] = []) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-sf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "SF", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String, settings: String? = nil) -> Row {
        Row(id: UUID(), task: task, blockKind: nil, model: nil, settings: settings,
            refs: [], children: [], clause: nil)
    }

    private func path(_ m: FlowEditorModel, _ id: UUID) -> String? {
        FlowSettings(m.row(withID: id)?.settings).pathValue()
    }

    @Test(arguments: [
        ("Save Text", "output.txt"), ("Save Image", "image.png"), ("Save Video", "video.mp4"),
        ("Save Audio", "audio.wav"), ("Save Images", "images"), ("Save Context", "context.ctx"),
    ])
    func seedsDefaultAndOneUndoRemovesRowAndPath(task: String, expected: String) throws {
        let m = try editor()
        m.add(task: task)
        let id = try #require(m.selectedRowID)
        #expect(m.row(withID: id)?.settings == expected)
        m.undo()
        #expect(m.document.rows.isEmpty)
    }

    @Test func childRowIsSeededToo() throws {
        let block = Row(id: UUID(), task: nil, blockKind: .each, blockName: "x", children: [row("Template", settings: "\"a\"")])
        let m = try editor([block])
        m.addChild(task: "Save Text", into: block.id)
        let id = try #require(m.selectedRowID)
        #expect(path(m, id) == "output.txt")
    }

    @Test func upstreamExtensionWins() throws {
        let m = try editor([row("Read Image", settings: "photo.jpg")])
        m.selectedRowID = m.document.rows[0].id
        m.add(task: "Save Image")
        let id = try #require(m.selectedRowID)
        #expect(path(m, id) == "image.jpg")
        #expect(m.warning(for: id) == nil)
    }

    @Test func collisionsGetSuffix() throws {
        let m = try editor()
        m.add(task: "Save Image")
        m.add(task: "Save Image")
        m.add(task: "Save Image")
        let paths = m.document.rows.compactMap { FlowSettings($0.settings).pathValue() }
        #expect(paths == ["image.png", "image-2.png", "image-3.png"])
        m.add(task: "Save Images")
        m.add(task: "Save Images")
        #expect(path(m, m.document.rows[4].id) == "images-2")
    }

    @Test func roundTripsByteStable() throws {
        let m = try editor()
        m.add(task: "Save Text")
        m.add(task: "Save Audio")
        try m.save()
        let text = try String(contentsOf: #require(m.savedURL), encoding: .utf8)
        #expect(text.contains("Save Text") && text.contains("output.txt") && !text.contains("path="))
        let reparsed = try CatParser.parse(text)
        #expect(reparsed.rows.map(\.settings) == ["output.txt", "audio.wav"])
        #expect(CatSerializer.serialize(reparsed) == text)
    }

    @Test func warnings() throws {
        let m = try editor([row("Read Video", settings: "clip.mov"), row("Save Video", settings: "out.mp4"),
                            row("Save Audio", settings: "a.mp3")])
        let v = m.document.rows[1].id, a = m.document.rows[2].id
        #expect(m.warning(for: v)?.contains("copies a .mov unchanged — name it .mov.") == true)
        #expect(m.warning(for: a)?.contains("Save Audio always writes WAV — name the file .wav.") == true)
        m.setPath("out.mov", for: v)
        m.setPath("a.wav", for: a)
        #expect(m.warning(for: v) == nil)
        #expect(m.warning(for: a) == nil)
    }

    @Test func existingRowsKeepTheirPaths() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "38-ReplyApproval")
        let before = doc.rows.map(\.settings)
        let m = try editor(doc.rows)
        #expect(m.document.rows.map(\.settings) == before)
    }
}
