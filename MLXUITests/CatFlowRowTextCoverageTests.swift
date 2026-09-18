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
/// the RT-0 tree; RT-0's mechanical walk measured **80** on that same tree instead — all 8
/// non-`Template` task counts matched §0 exactly, but `Template` was 63, not 61 (two more
/// genuine rows, inspected individually — real authored pattern text, no parse artifact or
/// double-count; see journal `2026-284`). RT-1 (journal `2026-285`) gave `Template` a Pattern
/// editor, so `hasAControlFor` below now covers it — the ceiling drops to **17**, exactly
/// 80 − 63. RT-2..RT-5 lower it further; no later change may raise it. This test itself fixes
/// nothing (RT-0's own rule) — it only ever reflects what the surfaces above already do.
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
        if desc.refName == "tools.text.template" { return true }           // patternEditor (RT-1)
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

        // Regression guard first: 17 is what this walk measures after RT-1 (see the type doc
        // comment). RT-2..RT-5 must only lower this, never raise it.
        #expect(uncovered.count <= 17,
                "regression: \(uncovered.count) rows now uncovered (ceiling 17) — \(summary)")
        // RT-0's own exit, still red until RT-5: a row whose authored text has no
        // Properties-tab control at all. RT-1 closed the owner's own report (Template); RT-2's
        // Ask Human/Human Input rows are next.
        #expect(uncovered.isEmpty,
                "\(uncovered.count) rows carry authored text with no Properties-tab control — \(summary)")
    }
}
