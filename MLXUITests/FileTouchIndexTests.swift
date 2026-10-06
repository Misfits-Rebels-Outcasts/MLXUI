import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// CW-5 — `FileTouchIndex` (C14): which files a chat touched, derived from the `ToolActivity` cards
/// it already persists. Pure, so it needs no runner and no model. Never `AgentAuditLog` — that log
/// is in-memory only, so it would be empty after a relaunch, which is exactly when a user looks.
struct FileTouchIndexTests {

    private func tool(
        _ name: String,
        path: String?,
        status: ToolActivity.Status = .finished,
        result: String? = "ok"
    ) -> TranscriptItem {
        var arguments: [String: JSONValue] = [:]
        if let path { arguments["path"] = .string(path) }
        return .tool(ToolActivity(
            name: name, arguments: arguments, status: status, result: result))
    }

    private func user(_ text: String) -> TranscriptItem {
        .message(ChatMessage(role: .user, text: text))
    }

    // MARK: - each tool → its kind

    @Test func eachToolMapsToItsKind() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "/a/read.txt"),
            tool("list_directory", path: "/a/dir"),
            tool("search_files", path: "/a/searched"),
            tool("write_file", path: "/a/written.txt", result: "Wrote 12 bytes to /a/written.txt."),
            tool("edit_file", path: "/a/edited.txt", result: "Edited /a/edited.txt"),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/read.txt")?.kind == .read)
        #expect(evaluation.entry(for: "/a/dir")?.kind == .read)
        #expect(evaluation.entry(for: "/a/searched")?.kind == .read)
        #expect(evaluation.entry(for: "/a/written.txt")?.kind == .created)
        #expect(evaluation.entry(for: "/a/edited.txt")?.kind == .edited)
        #expect(evaluation.reads.count == 3)
        #expect(evaluation.changes.count == 2)
    }

    /// A write the tools refused answers with "Error …"; the row is Declined, not New.
    @Test func aFailedWriteIsDeclined() {
        let items: [TranscriptItem] = [
            tool("write_file", path: "/a/nope.txt",
                 result: "Error: access to /a/nope.txt is not permitted. This chat can use: …"),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/nope.txt")?.kind == .declined)
        #expect(evaluation.declined.count == 1)
        #expect(evaluation.changes.isEmpty)
    }

    // MARK: - unfinished and denied

    @Test func unfinishedToolsAreIgnored() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "/a/running.txt", status: .running, result: nil),
            tool("read_file", path: "/a/done.txt"),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/running.txt") == nil)
        #expect(evaluation.entry(for: "/a/done.txt") != nil)
        #expect(evaluation.entries.count == 1)
    }

    @Test func aDeniedEditIsDeclined() {
        let items: [TranscriptItem] = [
            tool("edit_file", path: "/a/secret.txt", status: .denied, result: nil),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/secret.txt")?.kind == .declined)
        #expect(evaluation.declined.count == 1)
    }

    @Test func aFailedReadIsStillARead() {
        // A read that failed still tells the user what the chat reached for.
        let items: [TranscriptItem] = [
            tool("read_file", path: "/a/missing.txt", status: .failed("no such file"), result: nil),
        ]
        #expect(FileTouchIndex.entries(items: items).entry(for: "/a/missing.txt")?.kind == .read)
    }

    // MARK: - paths

    @Test func aRelativePathResolvesAgainstTheWorkingFolder() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "notes/todo.md"),
        ]
        let evaluation = FileTouchIndex.entries(
            items: items, workingFolder: "/Users/me/workfolder")
        #expect(evaluation.entries.first?.path == "/Users/me/workfolder/notes/todo.md")
    }

    @Test func anAbsolutePathIsKeptAndStandardised() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "/Users/me/a/../b/file.txt"),
        ]
        let evaluation = FileTouchIndex.entries(items: items, workingFolder: "/ignored")
        #expect(evaluation.entries.first?.path == "/Users/me/b/file.txt")
    }

    @Test func aTildePathExpands() {
        let items: [TranscriptItem] = [tool("read_file", path: "~/Documents/report.md")]
        let evaluation = FileTouchIndex.entries(items: items)
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        #expect(evaluation.entries.first?.path == "\(home)/Documents/report.md")
    }

    /// With no working folder a relative argument is kept as written rather than guessed at.
    @Test func aRelativePathWithoutAWorkingFolderIsKept() {
        let items: [TranscriptItem] = [tool("read_file", path: "notes/todo.md")]
        #expect(FileTouchIndex.entries(items: items).entries.first?.path == "notes/todo.md")
    }

    // MARK: - grouping and ordering

    /// The case the contract calls out: read then edited is **one** row, as edited, with both turns.
    @Test func aPathReadThenEditedIsOneEditedRowWithBothActivities() throws {
        let items: [TranscriptItem] = [
            user("fix it"),
            tool("read_file", path: "/a/report.md", result: "contents"),
            tool("edit_file", path: "/a/report.md", result: "Edited /a/report.md: replaced 1 occurrence"),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entries.count == 1)
        let entry = try #require(evaluation.entry(for: "/a/report.md"))
        #expect(entry.kind == .edited)
        #expect(entry.activities.count == 2)
        #expect(entry.activities.map(\.name) == ["read_file", "edit_file"])
        #expect(evaluation.changes.count == 1)
        #expect(evaluation.reads.isEmpty)
    }

    /// Two edits to one file: the row stays one, and the diff list is newest first (CW-6).
    @Test func editsOfOnePathKeepTheNewestFirst() {
        let items: [TranscriptItem] = [
            tool("edit_file", path: "/a/f.txt", result: "Edited /a/f.txt: first"),
            tool("edit_file", path: "/a/f.txt", result: "Edited /a/f.txt: second"),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        let edits = evaluation.editActivities(for: "/a/f.txt")
        #expect(edits.count == 2)
        #expect(edits.first?.result?.contains("second") == true)
        #expect(edits.last?.result?.contains("first") == true)
    }

    @Test func orderingIsFirstTouchOrder() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "/a/one.txt"),
            tool("write_file", path: "/a/two.txt", result: "Wrote 1 bytes to /a/two.txt."),
            tool("read_file", path: "/a/three.txt"),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entries.map(\.path) == ["/a/one.txt", "/a/two.txt", "/a/three.txt"])
    }

    @Test func aToolWithoutAPathIsIgnored() {
        let items: [TranscriptItem] = [
            tool("list_directory", path: nil),
            tool("calculator", path: "/a/not-a-file-tool"),
            tool("read_file", path: "/a/real.txt"),
        ]
        #expect(FileTouchIndex.entries(items: items).entries.map(\.path) == ["/a/real.txt"])
    }

    @Test func anEmptyTranscriptHasNothing() {
        #expect(FileTouchIndex.entries(items: []).entries.isEmpty)
    }

    // MARK: - the row's own labels

    @Test func kindLabelsMatchTheSectionTitles() {
        #expect(FileTouchIndex.Kind.created.label == "New")
        #expect(FileTouchIndex.Kind.edited.label == "Edited")
        #expect(FileTouchIndex.Kind.read.label == "Read")
        #expect(FileTouchIndex.Kind.declined.label == "Declined")
        #expect(FileTouchIndex.Kind.created.isChange)
        #expect(!FileTouchIndex.Kind.read.isChange)
        #expect(!FileTouchIndex.Kind.declined.isChange)
    }

    @Test func entryNameIsTheLastPathComponent() {
        let items: [TranscriptItem] = [tool("read_file", path: "/Users/me/Invoices 2026/rates.csv")]
        let entry = FileTouchIndex.entries(items: items).entries.first
        #expect(entry?.name == "rates.csv")
    }
}
