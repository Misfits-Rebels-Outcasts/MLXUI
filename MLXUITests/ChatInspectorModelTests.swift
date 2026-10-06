import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// CW-4 — `ChatInspectorModel`: the C15 meter, the per-reply tool-call count and the Context
/// badge. Pure arithmetic, so the view stays dumb and none of this needs a model or a window.
struct ChatInspectorModelTests {

    // MARK: - the meter (C15)

    @Test func meterEstimatesTokensAtOnePerFourBytes() {
        #expect(ChatInspectorModel.estimatedTokens(of: "") == 0)
        #expect(ChatInspectorModel.estimatedTokens(of: "abc") == 1)
        #expect(ChatInspectorModel.estimatedTokens(of: String(repeating: "x", count: 4_000)) == 1_000)
    }

    @Test func meterFractionAndAmberThreshold() {
        let low = ChatInspectorModel.meter(estimatedTokens: 1_000, window: 8_192)
        #expect(low.fraction > 0.12 && low.fraction < 0.13)
        #expect(!low.isAmber)

        let atThreshold = ChatInspectorModel.meter(estimatedTokens: 800, window: 1_000)
        #expect(atThreshold.fraction == 0.8)
        #expect(atThreshold.isAmber)                       // amber AT 80%, not only above it

        #expect(!ChatInspectorModel.meter(estimatedTokens: 799, window: 1_000).isAmber)
    }

    @Test func meterLabelUsesTheEstimateConventionAndGrouping() {
        let meter = ChatInspectorModel.meter(estimatedTokens: 12_300, window: 40_960)
        #expect(meter.label == "~12,300 / 40,960 (30%)")
    }

    @Test func meterNeverDividesByZero() {
        let meter = ChatInspectorModel.meter(estimatedTokens: 10, window: 0)
        #expect(meter.fraction == 0)
        #expect(!meter.isAmber)
    }

    // MARK: - K3: the bounded preview read

    private func markdown(_ bytes: Int) -> String { String(repeating: "x", count: bytes) }

    @Test func aOneMegabyteTextFileIsClippedAndTheReadIsBounded() throws {
        let url = URL(fileURLWithPath: "/tmp/big.txt")
        var requested = 0
        let read = ChatFilePreview.read(path: url.path) { _, limit in
            requested = limit
            return ChatFilePreview.FileRead(
                data: Data(repeating: 0x78, count: min(limit, 1_048_576)),
                fileSize: 1_048_576, error: nil)
        }
        #expect(requested == ChatFilePreview.readLimit)          // never more than cap + 1
        #expect(requested <= ChatFilePreview.textCap + 1)

        let content = ChatFilePreview.content(
            for: url, presentation: .text, read: read)
        guard case .truncatedText(let text, let shown, let total) = content else {
            Issue.record("expected truncatedText, got \(content)")
            return
        }
        #expect(shown <= ChatFilePreview.textCap)
        #expect(shown == text.utf8.count)
        #expect(total == 1_048_576)
        #expect(text.count == ChatFilePreview.textCap)
    }

    /// Markdown is clipped like text — CW-6 rendered it unclipped.
    @Test func aOneMegabyteMarkdownFileIsClippedWithTheSameNotice() {
        let url = URL(fileURLWithPath: "/tmp/big.md")
        let read = ChatFilePreview.FileRead(
            data: Data(repeating: 0x78, count: ChatFilePreview.readLimit),
            fileSize: 1_048_576, error: nil)
        let content = ChatFilePreview.content(for: url, presentation: .text, read: read)
        guard case .truncatedMarkdown(let text, let shown, let total) = content else {
            Issue.record("expected truncatedMarkdown, got \(content)")
            return
        }
        #expect(text.count == ChatFilePreview.textCap)
        #expect(shown == ChatFilePreview.textCap)
        #expect(total == 1_048_576)
        #expect(ChatFilePreview.truncationNotice(shownBytes: shown, totalBytes: total)
            .contains("of"))
    }

    /// The clip backs off to a UTF-8 boundary: a 4-byte character straddling the cap is dropped
    /// whole rather than turned into invalid text (which used to fall back to Quick Look).
    @Test func theClipBacksOffToAUTF8Boundary() throws {
        let character = "😀".data(using: .utf8) ?? Data()          // 4 bytes
        #expect(character.count == 4)
        // 65_535 ASCII bytes, then an emoji straddling the cap: 2 bytes inside, 2 outside.
        var data = Data(repeating: 0x78, count: ChatFilePreview.textCap - 2)
        data.append(character)
        data.append(Data(repeating: 0x79, count: 100))
        #expect(data.count > ChatFilePreview.textCap)

        let clipped = ChatFilePreview.clip(data)
        #expect(clipped.count <= ChatFilePreview.textCap)
        // The clip is valid UTF-8 and no partial emoji survived.
        #expect(String(data: clipped, encoding: .utf8) != nil)
        #expect(!String(data: clipped, encoding: .utf8)!.contains("😀"))
    }

    @Test func theClipLeavesAShortFileAlone() {
        let data = Data(repeating: 0x61, count: 10)
        #expect(ChatFilePreview.clip(data) == data)
    }

    @Test func aReadErrorBecomesUnavailable() {
        let read = ChatFilePreview.FileRead(
            data: Data(), fileSize: 0, error: "This file could not be read.")
        let content = ChatFilePreview.content(
            for: URL(fileURLWithPath: "/a/x.txt"), presentation: .text, read: read)
        #expect(content == .unavailable("This file could not be read."))
    }

    @Test func anImageAndAPDFAreNotReadAtAll() {
        let empty = ChatFilePreview.FileRead(data: Data(), fileSize: 0, error: nil)
        #expect(ChatFilePreview.content(
            for: URL(fileURLWithPath: "/a/x.png"), presentation: .image, read: empty) == .image)
        #expect(ChatFilePreview.content(
            for: URL(fileURLWithPath: "/a/x.pdf"), presentation: .pdf, read: empty) == .quickLook)
    }

    // MARK: - tool calls

    @Test func toolCallsCountOnlyTheCurrentReply() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "first")),
            .tool(ToolActivity(name: "read_file", arguments: [:], status: .finished, result: "a")),
            .message(ChatMessage(role: .assistant, text: "done")),
            .message(ChatMessage(role: .user, text: "second")),
            .tool(ToolActivity(name: "read_file", arguments: [:], status: .finished, result: "b")),
            .tool(ToolActivity(name: "write_file", arguments: [:], status: .finished, result: "c")),
        ]
        #expect(ChatInspectorModel.toolCallsSinceLastUser(items) == 2)
    }

    @Test func toolCallsAreZeroBeforeAnyToolInTheReply() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "only a prompt")),
        ]
        #expect(ChatInspectorModel.toolCallsSinceLastUser(items) == 0)
    }

    /// A tool card before any user message still counts: the transcript can open with one.
    @Test func toolCallsCountWhenTheTranscriptOpensWithATool() {
        let items: [TranscriptItem] = [
            .tool(ToolActivity(name: "read_file", arguments: [:], status: .finished, result: "a")),
            .message(ChatMessage(role: .assistant, text: "done")),
        ]
        #expect(ChatInspectorModel.toolCallsSinceLastUser(items) == 1)
    }

    @Test func toolCallLabelIsUsedOverLimit() {
        #expect(ChatInspectorModel.toolCallLabel(used: 3, limit: 8) == "3 / 8")
        #expect(ChatInspectorModel.toolCallLabel(used: 0, limit: 8) == "0 / 8")
    }

    // MARK: - Context badges

    @Test func accessBadgeFollowsTheWriteTools() {
        let readOnly: Set<String> = ["read_file", "list_directory"]
        let writable: Set<String> = ["read_file", "write_file"]
        #expect(!ChatInspectorModel.canWrite(enabledToolNames: readOnly))
        #expect(ChatInspectorModel.accessBadge(canWrite: false) == "read only")
        #expect(ChatInspectorModel.canWrite(enabledToolNames: writable))
        #expect(ChatInspectorModel.accessBadge(canWrite: true) == "read + write")
        // edit_file is enough on its own.
        #expect(ChatInspectorModel.canWrite(enabledToolNames: ["edit_file"]))
    }

    // MARK: - The review pane (CW-6)

    private func edit(_ old: String, _ new: String, result: String) -> ToolActivity {
        ToolActivity(
            name: "edit_file",
            arguments: ["path": .string("/a/report.md"),
                        "old_string": .string(old),
                        "new_string": .string(new)],
            status: .finished, result: result)
    }

    /// The edits arrive newest-first, so the specs keep that order and number the calls from the
    /// newest. Each spec is one `edit_file` rendered through the approval sheet's own formatter.
    @Test func editSpecsKeepTheNewestFirstAndUseTheSheetsFormatter() {
        let newest = edit("second line", "SECOND LINE", result: "Edited /a/report.md: second")
        let older = edit("first line", "FIRST LINE", result: "Edited /a/report.md: first")

        let specs = ChatInspectorModel.editSpecs([newest, older])
        #expect(specs.count == 2)
        #expect(specs.map(\.callNumber) == [1, 2])
        #expect(specs.map(\.id) == [newest.id, older.id])
        // The sheet's own sections for that call, verbatim.
        #expect(specs[0].sections == ApprovalPreview.sections(
            toolName: "edit_file", arguments: newest.arguments))
        #expect(specs[0].sections.contains { $0.title == "Replace" && $0.style == .removed })
        #expect(specs[0].sections.contains { $0.title == "With" && $0.style == .added })
    }

    @Test func editSpecsAreEmptyWithoutEdits() {
        #expect(ChatInspectorModel.editSpecs([]).isEmpty)
    }

    /// CW-6's marker: it exists exactly while the runner is blocked, and names the tool and path.
    @Test func theWaitingLabelNamesTheToolAndPath() {
        let waiting = ChatInspectorModel.waiting(toolName: "edit_file", path: "/a/report.md")
        #expect(waiting.label == "Waiting for your approval: edit_file /a/report.md")

        let withoutPath = ChatInspectorModel.waiting(toolName: "run_shell", path: nil)
        #expect(withoutPath.label == "Waiting for your approval: run_shell")
    }

    /// The row appears and disappears with `pendingApproval` (the closure means the sheet's own
    /// type cannot be compared, so the snapshot is what carries the state into the derivation).
    @Test func awaitingMarkerTracksTheRunnersPendingApproval() {
        let runner = ModelRunner()
        func marker() -> ChatInspectorModel.Waiting? {
            guard let pending = runner.pendingApproval else { return nil }
            return ChatInspectorModel.waiting(
                toolName: pending.toolName, path: pending.arguments.string("path"))
        }
        #expect(marker() == nil)

        runner.pendingApproval = PendingApproval(
            toolName: "edit_file",
            toolDescription: "Edits a file.",
            arguments: ["path": .string("/a/report.md")],
            respond: { _ in })
        #expect(marker()?.label == "Waiting for your approval: edit_file /a/report.md")

        runner.pendingApproval = nil
        #expect(marker() == nil)
    }

    @Test func toolsOnLabelCountsBothSides() {
        #expect(ChatInspectorModel.toolsOnLabel(enabled: 4, available: 12) == "4 of 12 tools on")
    }
}
