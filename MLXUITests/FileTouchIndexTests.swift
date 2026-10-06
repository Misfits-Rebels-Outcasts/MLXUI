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

    /// A write the tools refused answers with "Error …"; the row is Not done, not New.
    @Test func aFailedWriteIsNotDone() {
        let items: [TranscriptItem] = [
            tool("write_file", path: "/a/nope.txt",
                 result: "Error: access to /a/nope.txt is not permitted. This chat can use: …"),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/nope.txt")?.kind == .failed)
        #expect(evaluation.notDone.count == 1)
        #expect(evaluation.changes.isEmpty)
    }

    // MARK: - K2: a returned "Error…" is Not done, whatever the tool

    /// The C5 refusal case: the model asked for a file in an unattached folder and got the refusal
    /// text back with status `.finished`. The row must be Not done, not Read.
    @Test func aRefusedReadIsFailedNotRead() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "/Users/me/grant-b/secret.txt",
                 result: "Error: access to /Users/me/grant-b/secret.txt is not permitted."
                     + " This chat can use: grant-a (/Users/me/grant-a)."),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/Users/me/grant-b/secret.txt")?.kind == .failed)
        #expect(evaluation.reads.isEmpty)
        #expect(evaluation.notDone.count == 1)
    }

    /// A failed edit must not offer a diff of a change that never happened.
    @Test func aFailedEditIsFailedAndHasNoDiff() {
        let items: [TranscriptItem] = [
            tool("edit_file", path: "/a/report.md",
                 result: "Error: old_string not found. Copy the exact text from read_file."),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/report.md")?.kind == .failed)
        #expect(evaluation.changes.isEmpty)
        #expect(evaluation.editActivities(for: "/a/report.md").isEmpty)
    }

    /// A successful edit still offers its diff.
    @Test func aSuccessfulEditStillHasADiff() {
        let items: [TranscriptItem] = [
            tool("edit_file", path: "/a/report.md", result: "Edited /a/report.md: 1 occurrence"),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/report.md")?.kind == .edited)
        #expect(evaluation.editActivities(for: "/a/report.md").count == 1)
    }

    /// `hasPrefix`, not `contains`: the *path* may contain the word.
    @Test func aSuccessfulWriteToAnErrorNamedFileIsNew() {
        let items: [TranscriptItem] = [
            tool("write_file", path: "/a/Error-log.txt",
                 result: "Wrote 12 bytes to /a/Error-log.txt."),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/Error-log.txt")?.kind == .created)
        #expect(evaluation.notDone.isEmpty)
    }

    /// Read, then a refused read of the same path: the latest kind wins and both are kept.
    @Test func theLatestKindWinsAndBothActivitiesAreKept() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "/a/f.txt", result: "contents"),
            tool("read_file", path: "/a/f.txt", result: "Error: access is not permitted."),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        let entry = evaluation.entry(for: "/a/f.txt")
        #expect(entry?.kind == .failed)
        #expect(entry?.activities.count == 2)
        #expect(evaluation.reads.isEmpty)
    }

    /// A dispatch-level failure (no result string at all) is Not done too.
    @Test func aFailedStatusWithNoResultIsFailed() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "/a/f.txt", status: .failed("timed out"), result: nil),
        ]
        #expect(FileTouchIndex.entries(items: items).entry(for: "/a/f.txt")?.kind == .failed)
    }

    @Test func theNotDoneLabelAndResultRule() {
        #expect(FileTouchIndex.Kind.failed.label == "Not done")
        #expect(FileTouchIndex.Kind.declined.label == "Declined")
        #expect(!FileTouchIndex.Kind.failed.hasResult)
        #expect(!FileTouchIndex.Kind.declined.hasResult)
        #expect(FileTouchIndex.Kind.read.hasResult)
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
        #expect(evaluation.notDone.count == 1)
    }

    /// A read that failed still gets a row — the user wants to know what the chat reached for —
    /// but under "Not done", never as a Read (K2).
    @Test func aFailedReadGetsANotDoneRow() {
        let items: [TranscriptItem] = [
            tool("read_file", path: "/a/missing.txt", status: .failed("no such file"), result: nil),
        ]
        let evaluation = FileTouchIndex.entries(items: items)
        #expect(evaluation.entry(for: "/a/missing.txt")?.kind == .failed)
        #expect(evaluation.reads.isEmpty)
        #expect(evaluation.notDone.count == 1)
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
