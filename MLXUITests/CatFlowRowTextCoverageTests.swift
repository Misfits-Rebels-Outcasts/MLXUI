import Testing
import Foundation
@testable import MLXUI

/// RT-0 (`RSI/DelegateRowTextBacklog.md`) — the audit turned into a gate.
///
/// A row's settings string can carry a **quoted bare token** — author-written prose with no
/// `key=` label. §0 scopes the count to exactly that: `firstBareQuotedToken(_:)` below only
/// counts a row whose `FlowSettings.firstBare()` token (i.e. genuinely bare — the parser already
/// routed any `key="value"` pair, quotes included, into `pairs`, never `bare`) was *also*
/// written as a quoted span in the source. That excludes two things the naive "any bare token"
/// or "any quoted span" reading would wrongly sweep in: unquoted single-word/path/literal bare
/// tokens (`FlowSettings.firstBare()` alone is wider — voice ids, watched-folder paths, `Range`
/// literals, `Store Index` names, and one stray `Web Search` row whose settings tail is a
/// shape-signature comment, `31-ResearchBrief.cat:4`'s `text -> [url]`, none of them prose), and
/// a quoted `key="value"` pair like `Join Text`'s `separator="\n"` (`FlowSettingsEditor
/// .firstQuotedToken` alone is wider — it matches any quoted span, key-labelled or not; those
/// are §0's Tier-2 rows, already reachable as a labelled Settings field, just poorly presented —
/// RT-4's job, not RT-0's count). Today the Properties tab is blind to most quoted bare tokens —
/// the owner's report was a `Template` row whose entire meaning (its pattern) has no control at
/// all (`RSI/DelegateRowTextBacklog.md` §0–§2). This walks the bundled gallery + workspace
/// corpus and checks, per row with a quoted bare token, whether *some* Properties-tab surface
/// reaches it: one of the four `promptControl` shapes (`FlowRowInspectorView.promptControl`),
/// the file/folder path picker (`hasPathSetting`), or a `Save *` row's filename field
/// (`task.hasPrefix("Save")`). `RSI/DelegateRowTextBacklog.md` §0 hand-counted **78** against
/// this tree; this walk — the same tree, mechanically — measures **80**, all 8 non-`Template`
/// task counts matching §0 exactly (`Ask Human` 5, `Compare` 1, `Embed` 3, `Generate Image` 1,
/// `Generate Sound` 1, `Human Input` 4, `Improvise` 1, `Web Search` 1) and `Template` at 63, not
/// 61 — two more genuine rows (inspected individually; both carry real authored pattern text,
/// neither a parse artifact nor a double-count). Pinned at the number this walk actually
/// measures, **80**, since a manual count across 71 files is exactly the kind of arithmetic a
/// mechanical audit is meant to correct — see the journal for this cycle. RT-1..RT-5 lower it;
/// no later change may raise it. Fixes nothing itself (RT-0's own rule).
struct CatFlowRowTextCoverageTests {

    private struct UncoveredRow {
        let flow: String
        let task: String
        let token: String
    }

    /// Mirrors `FlowRowInspectorView.promptControl` + `hasPathSetting` + the `Save *` filename
    /// field's `task.hasPrefix("Save")` gate — the three surfaces (plus, after RT-6, the
    /// raw-text backstop, not built yet) that can put a row's authored text in front of the
    /// user today. Task-level, not row-level: none of the three read anything but the task's
    /// own `TaskDescriptor`, except `engines.vlm.ocr`, whose actual control (instruction box vs.
    /// mode picker vs. nothing) depends on the row's *named model*. No bare-token `OCR` row
    /// exists in the bundled corpus today (verified: every `OCR` row's `settings` is `nil`), so
    /// this treats it as covered without resolving a model — narrowing that would need the same
    /// registry wiring `CatFlowR9Tests.loadedCatalogAndClaimable()` uses, for zero rows it would
    /// currently affect.
    private func hasAControlFor(task: String) -> Bool {
        if task.hasPrefix("Save") { return true }                            // saveFilenameField
        guard let desc = TaskCatalog.get(task) else { return false }
        switch desc.accepts {
        case .single(.file), .single(.folder): return true                  // chooseFileButton
        default: break
        }
        if desc.refName.hasPrefix("frames/") { return true }                // instructionBox
        if desc.refName == "engines.vlm.describe_image" { return true }     // instructionBox
        if desc.refName == "engines.llm.extract_structured" { return true } // schemaEditor (ES-UI-1)
        if desc.refName == "engines.vlm.ocr" { return true }                // model-dependent; see above
        return false
    }

    private func allRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + allRows($0.children) }
    }

    /// The row's first bare token, but only when it was *also* written as a quoted span in the
    /// source — see the type doc comment for why both halves are needed. Returns the unquoted
    /// text (matching what `FlowSettings.firstBare()` already returns).
    private func firstBareQuotedToken(_ settings: String?) -> String? {
        guard let bare = FlowSettings(settings).firstBare(),
              let quoted = FlowSettingsEditor.firstQuotedToken(in: settings ?? ""),
              FlowSettings.unquote(quoted.token) == bare
        else { return nil }
        return bare
    }

    /// Every row in the bundled gallery + workspace corpus — 71 `.cat` files, 598 numbered rows
    /// as of `RSI/DelegateRowTextBacklog.md` §0 (nested `<each>`/`<list>`/`<parallel>` children
    /// included, unnumbered here — the audit only needs the task and the settings string).
    private func corpusRows() throws -> [(flow: String, row: Row)] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
        let dirs = [
            root.appendingPathComponent("MLXUI/Resources/Gallery"),
            root.appendingPathComponent("MLXUI/Resources/Workspaces"),
        ]
        var out: [(String, Row)] = []
        for dir in dirs {
            for file in try FileManager.default.contentsOfDirectory(atPath: dir.path)
                .filter({ $0.hasSuffix(".cat") }).sorted() {
                let text = try String(contentsOf: dir.appendingPathComponent(file), encoding: .utf8)
                let doc = try CatParser.parse(text)
                for row in allRows(doc.rows) {
                    out.append((file, row))
                }
            }
        }
        return out
    }

    @Test func everyBareTokenRowHasAPropertiesControl() throws {
        let rows = try corpusRows()
        #expect(rows.count >= 500, "expected the bundled corpus to be substantial (got \(rows.count))")

        var uncovered: [UncoveredRow] = []
        for (flow, row) in rows {
            guard let task = row.task else { continue }
            guard let token = firstBareQuotedToken(row.settings) else { continue }
            if !hasAControlFor(task: task) {
                uncovered.append(UncoveredRow(flow: flow, task: task, token: token))
            }
        }

        var byTask: [String: Int] = [:]
        for u in uncovered { byTask[u.task, default: 0] += 1 }
        let summary = byTask.sorted { $0.key < $1.key }
            .map { "\($0.key): \($0.value)" }
            .joined(separator: ", ")

        // Regression guard first: 80 is what this walk measures against this tree today (see
        // the type doc comment — 2 more Template rows than RSI/DelegateRowTextBacklog.md §0's
        // hand count, both verified genuine). RT-1..RT-5 must only lower this, never raise it.
        #expect(uncovered.count <= 80,
                "regression: \(uncovered.count) rows now uncovered (ceiling 80) — \(summary)")
        // RT-0's own exit: expected RED today. A row whose authored text has no Properties-tab
        // control at all is the owner's report (Template row 3.2 of 73-WebSummaryLinks.cat, the
        // largest single case — 63 of the 80) generalized to the whole corpus.
        #expect(uncovered.isEmpty,
                "\(uncovered.count) rows carry authored text with no Properties-tab control — \(summary)")
    }
}
