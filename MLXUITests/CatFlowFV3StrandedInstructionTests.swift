import Testing
import Foundation
@testable import MLXUI

/// FV-3 (`RSI/DelegateFrameViewBacklog.md`) — the warning for the five framed tasks whose
/// frame text never reads `{settings}` at all (`Answer`, `Revise`, `Verify`, `Think`) or reads
/// only a keyed setting (`Translate` — `{settings.to}`), so the quoted instruction the
/// Properties tab still invites never reaches the model. Covers the frame-derived condition
/// (never a hardcoded task list — a frame's wording is Part C's to change), the §7 copy, the
/// `Translate` Add-menu fix, and the exit criterion against the owner's own bundled example
/// (`31-ResearchBrief.cat` row 5).
struct CatFlowFV3StrandedInstructionTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    // MARK: - The condition is derived from the frame text, not a task list

    @Test func exactlyFiveFramedTasksLackABareSettingsPlaceholder() throws {
        let frameTasks = TaskCatalog.allTasks().filter { $0.refKind == .frame }
        #expect(frameTasks.count == 16)

        var lacking: Set<String> = []
        for desc in frameTasks {
            let text = try FrameRenderer.loadFrame(named: FramePreview.frameFileName(refName: desc.refName))
            if !text.contains("{settings}") { lacking.insert(desc.name) }
        }
        // §0's own list: Answer/Revise/Verify/Think have no {settings} placeholder at all;
        // Translate reads only the keyed {settings.to}, so the bare token is absent too.
        #expect(lacking == ["Answer", "Revise", "Verify", "Think", "Translate"])
    }

    // MARK: - §7 copy

    @Test func strandedInstructionMessageUsesTheSpecificAnswerCopy() {
        #expect(FlowRowInspectorView.strandedInstructionMessage(task: "Answer") ==
                "Answer takes no instruction — its question arrives with the input, so the text in quotes never reaches the model.")
    }

    @Test func strandedInstructionMessageUsesTheSpecificTranslateCopy() {
        #expect(FlowRowInspectorView.strandedInstructionMessage(task: "Translate") ==
                "Translate reads to= for the target language — the text in quotes isn't used.")
    }

    @Test func strandedInstructionMessageUsesTheGenericCopyForEverythingElse() {
        for task in ["Revise", "Verify", "Think"] {
            #expect(FlowRowInspectorView.strandedInstructionMessage(task: task) ==
                    "\(task) doesn't use the text in quotes — its frame never reads it, so the model never sees it.")
        }
    }

    // MARK: - Translate's Add-menu fix

    @Test func translateOffersToInTheAddSettingMenu() {
        #expect(FlowRowInspectorView.knownSettingKeys(for: "Translate") == ["to"])
    }

    // MARK: - Exit criterion: an Answer row carrying quoted text says so, in the pane

    @Test func answerRowCarryingQuotedTextTriggersTheWarningChain() throws {
        // 31-ResearchBrief.cat row 5: `5. Answer   Qwen3 8B; "cite sources by number"` — §0's
        // own named example of dead text no model has ever seen. Reconstructs
        // `strandedInstructionWarning`'s condition chain from its non-private parts, since the
        // View method itself isn't reachable from a test (this codebase's own convention —
        // see journal 2026-291).
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/31-ResearchBrief.cat")
        let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
        let row = try #require(doc.rows.first { $0.task == "Answer" })
        #expect(row.settings == "\"cite sources by number\"")

        let desc = try #require(TaskCatalog.get("Answer"))
        #expect(desc.refKind == .frame)

        let quoted = try #require(FlowSettingsEditor.firstQuotedToken(in: row.settings ?? ""))
        let instruction = FlowSettings.unquote(quoted.token)
        #expect(instruction == "cite sources by number")

        let frameText = try FrameRenderer.loadFrame(named: FramePreview.frameFileName(refName: desc.refName))
        #expect(!frameText.contains("{settings}"), "Answer's frame must not read {settings} for this warning to fire")

        let message = FlowRowInspectorView.strandedInstructionMessage(task: "Answer")
        #expect(message == "Answer takes no instruction — its question arrives with the input, so the text in quotes never reaches the model.")
    }

    // MARK: - §5 Q3: the three bundled flows keep their dead text, untouched

    @Test func theThreeNamedBundledFlowsStillCarryTheirDeadAnswerText() throws {
        let cases: [(file: String, instruction: String)] = [
            ("31-ResearchBrief.cat", "cite sources by number"),
            ("44-FrontierEscalate.cat", "cite sources"),
            ("45-RouterDesk.cat", "acknowledge; summarize for a human"),
        ]
        for (file, expectedInstruction) in cases {
            let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(file)")
            let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
            // 44-FrontierEscalate.cat has two `Answer` rows (row 5, no settings; row 8, the
            // named one) — find the one carrying quoted text, not just the first `Answer`.
            let answerRow = try #require(doc.rows.first { $0.task == "Answer" && $0.settings != nil },
                                         "expected an Answer row with settings in \(file)")
            let quoted = try #require(FlowSettingsEditor.firstQuotedToken(in: answerRow.settings ?? ""))
            #expect(FlowSettings.unquote(quoted.token) == expectedInstruction,
                    "\(file)'s Answer row's quoted instruction changed — §5 Q3 says leave it")
        }
    }
}
