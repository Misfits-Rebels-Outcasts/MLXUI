import Foundation
import Testing
@testable import MLXUI

/// ER-FIX-4 -- the inspector's viewer choice (logic level; the view itself is owner-checked).
struct InspectorViewerTests {
    private func item(_ kind: Kind, path: String? = nil) -> Item {
        Item(kind: kind, value: nil, path: path.map { URL(fileURLWithPath: $0) }, sourceText: nil)
    }

    @Test func aPDFFileGetsQuickLook() {
        #expect(InspectorViewer.choose(for: item(.file, path: "/tmp/x/label.PDF")) == .quickLook(URL(fileURLWithPath: "/tmp/x/label.PDF")))
    }

    @Test func otherFilesAndFolderStillHaveNoViewer() {
        #expect(InspectorViewer.choose(for: item(.file, path: "/tmp/x/a.zip")) == .none)
        #expect(InspectorViewer.choose(for: item(.file)) == .none)
        #expect(InspectorViewer.choose(for: item(.folder, path: "/tmp/x")) == .none)
        #expect(InspectorViewer.choose(for: item(.video, path: "/tmp/x/a.mov")) == .none)
    }

    @Test func existingKindsKeepTheirViewers() {
        #expect(InspectorViewer.choose(for: item(.audio)) == .audio)
        #expect(InspectorViewer.choose(for: item(.text)) == .text)
        #expect(InspectorViewer.choose(for: item(.status)) == .status)
        #expect(InspectorViewer.choose(for: item(.image)) == .image)
    }

    @Test func aTableIsShownAsTableToTextMarkdown() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("iv-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let table = try TableTool.tableItem(columns: ["speaker", "start"], rows: [["Speaker 1", 0.5], ["Speaker 2", 3]], in: dir)
        #expect(InspectorViewer.choose(for: table) == .tableText)
        let text = try #require(InspectorViewer.tableText(for: table))
        let expected = try #require(try TableTool.tableToText(settings: nil, from: table).items.first?.value)
        #expect(text == expected)
        #expect(text == "| speaker | start |\n| --- | --- |\n| Speaker 1 | 0.5 |\n| Speaker 2 | 3 |")
    }

    @Test func anUnreadableTableHasNoText() {
        #expect(InspectorViewer.tableText(for: item(.table, path: "/nonexistent/t.table.json")) == nil)
        #expect(InspectorViewer.choose(for: item(.table)) == .none)
    }
}
