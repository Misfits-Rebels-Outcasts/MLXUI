import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// EF-3 (D10) — the pure formatter behind the approval sheet. The sheet's own layout is verified
/// by build + smoke (rows EF-S1/S4); this covers exactly what it will render.
struct ApprovalPreviewTests {

    private func titles(_ sections: [ApprovalPreview.Section]) -> [String] {
        sections.map(\.title)
    }

    @Test func editFileShowsPathReplaceAndWith() {
        let sections = ApprovalPreview.sections(toolName: "edit_file", arguments: [
            "path": .string("/tmp/notes.txt"),
            "old_string": .string("beta"),
            "new_string": .string("BETA"),
        ])
        #expect(titles(sections) == ["Path", "Replace", "With"])
        #expect(sections.map(\.style) == [.plain, .removed, .added])
        #expect(sections[0].body == "/tmp/notes.txt")
        #expect(sections[1].body == "beta")
        #expect(sections[2].body == "BETA")
    }

    @Test func editFileNamesTheAllOccurrencesModeOnlyWhenItIsOn() {
        let on = ApprovalPreview.sections(toolName: "edit_file", arguments: [
            "path": .string("/tmp/notes.txt"),
            "old_string": .string("x"),
            "new_string": .string("y"),
            "replace_all": .bool(true),
        ])
        #expect(titles(on) == ["Path", "Replace", "With", "All occurrences"])

        let off = ApprovalPreview.sections(toolName: "edit_file", arguments: [
            "path": .string("/tmp/notes.txt"),
            "old_string": .string("x"),
            "new_string": .string("y"),
            "replace_all": .bool(false),
        ])
        #expect(titles(off) == ["Path", "Replace", "With"])
    }

    @Test func writeFileShowsThePathAndTheWholeContents() {
        let contents = "line one\nline two\n"
        let sections = ApprovalPreview.sections(toolName: "write_file", arguments: [
            "path": .string("/tmp/new.txt"),
            "contents": .string(contents),
        ])
        #expect(titles(sections) == ["Path", "Contents"])
        #expect(sections[0].style == .plain)
        #expect(sections[1].style == .code)
        #expect(sections[1].body == contents)
    }

    /// The regression guard for the AG1 gap: the sheet used to show only the tool name, so a user
    /// approving `run_shell` never saw the command they were approving.
    @Test func runShellShowsTheCommandVerbatim() {
        let command = "rm -rf '~/tmp dir' && echo \"done\""
        let sections = ApprovalPreview.sections(toolName: "run_shell", arguments: [
            "command": .string(command),
            "working_directory": .string("/Users/me/project"),
            "timeout_seconds": .int(30),
        ])
        #expect(titles(sections) == ["Command", "Working directory", "Timeout"])
        #expect(sections[0].body == command)
        #expect(sections[0].style == .code)
        #expect(sections[1].body == "/Users/me/project")
        #expect(sections[2].body == "30 seconds")
    }

    @Test func runShellOmitsTheOptionalRowsWhenTheyAreAbsent() {
        let sections = ApprovalPreview.sections(toolName: "run_shell", arguments: [
            "command": .string("git status --short"),
        ])
        #expect(titles(sections) == ["Command"])
    }

    @Test func anUnknownToolRendersEveryArgumentSortedAsCompactJSON() {
        let sections = ApprovalPreview.sections(toolName: "some_future_tool", arguments: [
            "zeta": .int(2),
            "alpha": .string("a"),
            "nested": .object(["b": .bool(true), "a": .array([.int(1)])]),
        ])
        #expect(titles(sections) == ["alpha", "nested", "zeta"])
        #expect(sections.allSatisfy { $0.style == .code })
        #expect(sections[0].body == "\"a\"")
        #expect(sections[1].body == "{\"a\":[1],\"b\":true}")
        #expect(sections[2].body == "2")
    }

    @Test func noArgumentsProduceNoSections() {
        #expect(ApprovalPreview.sections(toolName: "edit_file", arguments: [:]).isEmpty)
        #expect(ApprovalPreview.sections(toolName: "some_future_tool", arguments: [:]).isEmpty)
        #expect(ApprovalPreview.sections(toolName: "run_shell", arguments: [:]).isEmpty)
    }

    @Test func aBodyOverSixtyFourKilobytesIsClippedWithANotice() {
        let characters = 70_000
        let long = String(repeating: "a", count: characters)
        let sections = ApprovalPreview.sections(toolName: "write_file", arguments: [
            "path": .string("/tmp/big.txt"),
            "contents": .string(long),
        ])
        guard let contents = sections.first(where: { $0.title == "Contents" }) else {
            Issue.record("no Contents section: \(titles(sections))")
            return
        }
        // 70,000 characters minus the 65,536 that fit in 64 KB.
        let dropped = characters - ApprovalPreview.maxBodyBytes
        #expect(contents.body.contains("(\(dropped) more characters not shown)"))
        #expect(contents.body.utf8.count <= ApprovalPreview.maxBodyBytes + 64)
        #expect(contents.body.hasPrefix(String(repeating: "a", count: 1_000)))
    }

    @Test func clippingNeverSplitsACharacter() {
        // Two-byte characters: the kept prefix must end on a scalar boundary, with no U+FFFD.
        let long = String(repeating: "é", count: 40_000)          // 80,000 bytes
        let clipped = ApprovalPreview.clipped(
            ApprovalPreview.Section(title: "T", body: long, style: .code))
        #expect(clipped.body.contains("more characters not shown"))
        #expect(!clipped.body.unicodeScalars.contains("\u{FFFD}"))
        #expect(clipped.body.utf8.count <= ApprovalPreview.maxBodyBytes + 64)
    }
}
