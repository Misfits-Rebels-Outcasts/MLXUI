import Testing
import Foundation
@testable import MLXUI

/// Phase HU — a human row's waiting policy is authorable from the inspector, not just a
/// hand-edit of the `.cat` text.
///
/// - **HU-1**: a freshly added `Ask Human`/`Human Input` row seeds `wait=forever`, so it never
///   shows **E501** the moment it's added.
/// - **HU-2**: `FlowEditorModel.setHumanWaitPolicy` writes `wait=forever` xor
///   `timeout=…`/`default=…` in one commit (mutual exclusion, one undo step);
///   `waitPolicyDescription` is the pure read the App Store build's read-only line uses, and it
///   never assumes `wait=forever` for a row that already declares something else (HU-3).
/// - **HU-3**: `knownSettingKeys` offers `wait`/`timeout`/`default` for both human tasks, so the
///   generic "Add a setting…" menu can repair an imported flow.
struct CatFlowHumanPolicyTests {

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-hu-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "HU", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String?, settings: String? = nil, tags: [String]? = nil,
                     clause: Clause? = nil) -> Row {
        Row(id: UUID(), task: task, settings: settings, clause: clause, tags: tags)
    }

    private func issues(for text: String) throws -> [FlowIssue] {
        let flow = try CatParser.parseForValidation(text)
        return FlowValidator.checkFlow(flow)
    }

    // MARK: - HU-1: seeding a fresh row

    @Test func defaultSettingsSeedsWaitForeverForBothHumanTasksOnly() {
        // HR-3: Human Input additionally seeds a quoted criterion, so its sheet is never
        // blank; Ask Human is left as `wait=forever` alone (recommended seed is Human
        // Input-only — RSI/DelegateHumanRowBacklog.md HR-3).
        #expect(FlowEditorModel.defaultSettings(forTask: "Ask Human") == "wait=forever")
        #expect(FlowEditorModel.defaultSettings(forTask: "Human Input") == "\"What should I use?\"; wait=forever")
        #expect(FlowEditorModel.defaultSettings(forTask: "Read Text") == nil)
        #expect(FlowEditorModel.defaultSettings(forTask: "Summarize") == nil)
    }

    @Test @MainActor func addingAHumanInputRowSeedsAQuestionAndWaitForeverAndNeedsNoWarning() throws {
        let model = try editor([row("Read Text", settings: "memo.txt")])
        model.selectedRowID = model.document.rows[0].id
        model.add(task: "Human Input")
        let newRow = try #require(model.document.rows.last)
        #expect(newRow.task == "Human Input")
        #expect(newRow.settings == "\"What should I use?\"; wait=forever")
        #expect(model.warning(for: newRow.id) == nil)
    }

    @Test @MainActor func addingAnAskHumanRowSeedsWaitForeverAndRaisesNoE501() throws {
        let model = try editor([row("Read Text", settings: "memo.txt")])
        model.selectedRowID = model.document.rows[0].id
        model.add(task: "Ask Human")
        let newRow = try #require(model.document.rows.last)
        #expect(newRow.task == "Ask Human")
        #expect(newRow.settings == "wait=forever")
        #expect(!model.issues(for: newRow.id).contains { $0.code == "E501" })
    }

    @Test @MainActor func addingAHumanChildRowIntoABlockAlsoSeedsAQuestionAndWaitForever() throws {
        let block = Row(id: UUID(), task: nil, blockKind: .each, blockName: "Each",
                        children: [Row(id: UUID(), task: "Template", settings: "\"\"")])
        let model = try editor([block])
        model.addChild(task: "Human Input", into: block.id)
        let child = try #require(model.document.rows.first?.children.last)
        #expect(child.task == "Human Input")
        #expect(child.settings == "\"What should I use?\"; wait=forever")
    }

    @Test func freshlySeededHumanInputRowValidatesWithNoE501() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Human Input  (1)  wait=forever
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E501" })
    }

    // MARK: - HU-2: the policy setter (mutual exclusion, one commit)

    @Test @MainActor func settingWaitForeverClearsTimeoutAndDefaultInOneCommit() throws {
        let r = row("Ask Human", settings: "timeout=1h; default=approve", tags: ["approve", "edit"])
        let model = try editor([r])
        let before = model.undoStack.count
        model.setHumanWaitPolicy(.waitForever, for: r.id)
        #expect(model.row(withID: r.id)?.settings == "wait=forever")
        #expect(model.undoStack.count == before + 1)
    }

    @Test @MainActor func settingGiveUpAfterClearsWaitInOneCommit() throws {
        let r = row("Ask Human", settings: "wait=forever", tags: ["approve", "edit"])
        let model = try editor([r])
        let before = model.undoStack.count
        model.setHumanWaitPolicy(.giveUpAfter(timeout: "2h", default: "approve"), for: r.id)
        let settings = FlowSettings(model.row(withID: r.id)?.settings)
        #expect(settings.value(for: "wait") == nil)
        #expect(settings.value(for: "timeout") == "2h")
        #expect(settings.value(for: "default") == "approve")
        #expect(model.undoStack.count == before + 1)
    }

    @Test @MainActor func bothPoliciesRoundTripThroughSerializeAndReparse() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let r3 = row("Save Text", settings: "out.md")
        let r2 = row("Ask Human", settings: "wait=forever", tags: ["approve", "edit"],
                     clause: .decide(edges: [ClauseEdge(tag: "approve", target: .row(number: 3)),
                                             ClauseEdge(tag: "edit", target: .row(number: 3))]))
        let model = try editor([r1, r2, r3])

        model.setHumanWaitPolicy(.giveUpAfter(timeout: "2h", default: "approve"), for: r2.id)
        let once = model.catText
        let reparsed = try CatParser.parse(once)
        #expect(CatSerializer.serialize(reparsed) == once)   // fmt idempotence
        let afterGiveUp = FlowSettings(reparsed.rows[1].settings)
        #expect(afterGiveUp.value(for: "timeout") == "2h")
        #expect(afterGiveUp.value(for: "default") == "approve")
        #expect(afterGiveUp.value(for: "wait") == nil)

        model.setHumanWaitPolicy(.waitForever, for: r2.id)
        let twice = model.catText
        let reparsedAgain = try CatParser.parse(twice)
        let afterWaitForever = FlowSettings(reparsedAgain.rows[1].settings)
        #expect(afterWaitForever.value(for: "wait") == "forever")
        #expect(afterWaitForever.value(for: "timeout") == nil)
        #expect(afterWaitForever.value(for: "default") == nil)
    }

    // MARK: - E502 still catches an out-of-tags default (regression guard, not new behavior)

    @Test func askHumanDefaultOutsideDeclaredTagsRaisesE502() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Ask Human    (1)  "Approve?" ; timeout=1h; default=nope ; tags: approve, edit
           -> { approve: 3 | edit: 3 }
        3. Save Text    out.md
        """
        let found = try issues(for: text)
        #expect(found.contains { $0.code == "E502" })
    }

    @Test func askHumanDefaultInsideDeclaredTagsRaisesNoE502() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Ask Human    (1)  "Approve?" ; timeout=1h; default=edit ; tags: approve, edit
           -> { approve: 3 | edit: 3 }
        3. Save Text    out.md
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E502" })
        #expect(!found.contains { $0.code == "E501" })
    }

    // MARK: - HU-FOLLOWUP-1 / SP-2: an empty `timeout=` still raises E501 — the picker no
    // longer writes it at all.
    //
    // Originally (HU-FOLLOWUP-1) this test pinned a *disagreement*: `checkHumanRows`
    // (`FlowValidator.swift:1717`) only tests `kv["timeout"] != nil`, and `FlowValidator.reKV`
    // (`:340`, `^([\w-]+)\s*=\s*(.+)$`) via `parseSettingsKV` reads a valueless `timeout=` as
    // absent (`(.+)` requires ≥1 character after `=`) — so E501 fired — while the editor's own
    // `FlowSettings` read the *same* text as `timeout` present with an empty string. Two
    // readers of one row disagreeing about whether it has a waiting policy is exactly the bug
    // Phase FIX/SP exists to close (`RSI/DelegateFixItBacklog.md`, MS-FOLLOWUP-2 →
    // `SPEC_QUESTIONS.md` Q226 in `catflow-mlx`, open).
    //
    // SP-2 fixes the cheaper half regardless of Q226's eventual answer: `FlowSettingsEditor
    // .replace(key:value:)` now treats an empty/whitespace value as a removal, so
    // `setHumanWaitPolicy(.giveUpAfter(timeout: "", …))` (what the Direct picker calls when
    // its duration field is left blank) never writes `timeout=` at all — `FlowSettings` and
    // the validator now **agree**: the key is genuinely absent, not present-but-empty. This
    // test still pins the same thing it always did (E501 fires), by a route that no longer
    // depends on `reKV`/`FlowSettings` disagreeing — so a future widening of `reKV` to `(.*)`
    // (Q226's other possible answer) still can't silently produce a human row that passes
    // every check and waits for nothing.
    @Test @MainActor func switchingToGiveUpAfterWithAnEmptyDurationStillRaisesE501() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let r3 = row("Save Text", settings: "out.md")
        let r2 = row("Ask Human", settings: "wait=forever", tags: ["approve", "edit"],
                     clause: .decide(edges: [ClauseEdge(tag: "approve", target: .row(number: 3)),
                                             ClauseEdge(tag: "edit", target: .row(number: 3))]))
        let model = try editor([r1, r2, r3])
        model.setHumanWaitPolicy(.giveUpAfter(timeout: "", default: "approve"), for: r2.id)
        let rawSettings = model.row(withID: r2.id)?.settings ?? ""
        // SP-2: no `timeout` token is written at all — not even a bare `timeout=`.
        #expect(!rawSettings.contains("timeout"))
        let settings = FlowSettings(rawSettings)
        #expect(settings.value(for: "timeout") == nil)
        #expect(settings.value(for: "wait") == nil)   // still cleared, per the mutual exclusion

        // The row still needs a policy — `default=approve` alone isn't one — so E501 still
        // fires, now because *both* parsers agree the key is absent, not because they disagree.
        let found = try issues(for: model.catText)
        #expect(found.contains { $0.code == "E501" })
    }

    // MARK: - HU-3: the repair path

    @Test func knownSettingKeysOffersWaitTimeoutDefaultForBothHumanTasks() {
        #expect(FlowRowInspectorView.knownSettingKeys(for: "Ask Human") == ["wait", "timeout", "default"])
        #expect(FlowRowInspectorView.knownSettingKeys(for: "Human Input") == ["wait", "timeout", "default"])
    }

    // MARK: - HU-2/HU-3: the App Store read-only line never assumes wait=forever

    @Test func waitPolicyDescriptionReflectsWaitForever() {
        #expect(FlowEditorModel.waitPolicyDescription(settings: "wait=forever") == "Waits until you answer.")
    }

    @Test func waitPolicyDescriptionReflectsAnExistingTimeoutDefaultWithoutAssumingWaitForever() {
        let description = FlowEditorModel.waitPolicyDescription(settings: "timeout=2h; default=edit")
        #expect(description.contains("2h"))
        #expect(description.contains("edit"))
        #expect(description != "Waits until you answer.")
    }

    @Test func waitPolicyDescriptionNamesTheGapWhenNeitherIsSet() {
        let description = FlowEditorModel.waitPolicyDescription(settings: nil)
        #expect(description != "Waits until you answer.")
        #expect(description.contains("wait=forever"))
    }

    // MARK: - HU-3: an existing timeout=/default= flow keeps validating and isn't rewritten

    @Test func existingTimeoutDefaultFlowValidatesCleanAndReadingItsPolicyDoesNotMutateIt() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Ask Human    (1)  "Send this reply?" ; timeout=2h; default=edit ; tags: edit
           -> { edit: 3 }
        3. Save Text    out.md
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E501" })
        #expect(!found.contains { $0.code == "E502" })

        let doc = try CatParser.parse(text)
        let before = CatSerializer.serialize(doc)
        let description = FlowEditorModel.waitPolicyDescription(settings: doc.rows[1].settings)
        #expect(description.contains("2h"))
        #expect(description.contains("edit"))
        // Reading the policy for display is a pure read — the row's own bytes are untouched.
        #expect(CatSerializer.serialize(doc) == before)
    }

    // MARK: - HR-3: the prompt sheet is never blank

    @Test func promptTextPassesThroughATrimmedNonEmptyPrompt() {
        #expect(FlowHumanPromptView.promptText(prompt: "Enter URL:", task: "Human Input") == "Enter URL:")
        #expect(FlowHumanPromptView.promptText(prompt: "  Enter URL:  ", task: "Human Input") == "Enter URL:")
    }

    @Test func promptTextFallsBackByTaskWhenTheCriterionIsEmpty() {
        #expect(FlowHumanPromptView.promptText(prompt: "", task: "Human Input") == "This step needs your input:")
        #expect(FlowHumanPromptView.promptText(prompt: "   ", task: "Human Input") == "This step needs your input:")
        #expect(FlowHumanPromptView.promptText(prompt: "", task: "Ask Human") == "This step needs your decision:")
    }

    // MARK: - HR-4: the prompt sheet prefills the incoming default (SPEC-Q233)

    @Test func effectivePrefillPassesThroughAFittingDefault() {
        #expect(FlowHumanPromptView.effectivePrefill(
            defaultText: "https://news.ycombinator.com", task: "Human Input")
            == "https://news.ycombinator.com")
    }

    @Test func effectivePrefillIsNilWithNoDefaultText() {
        #expect(FlowHumanPromptView.effectivePrefill(defaultText: nil, task: "Human Input") == nil)
    }

    @Test func effectivePrefillIsNilForAskHumanEvenWithADefault() {
        // Ask Human answers with tag buttons, not a text field — there is nothing to prefill.
        #expect(FlowHumanPromptView.effectivePrefill(defaultText: "approve", task: "Ask Human") == nil)
    }

    @Test func effectivePrefillIsNilPastTheDisplayCap() {
        let tooLong = String(repeating: "a", count: 501)
        #expect(FlowHumanPromptView.effectivePrefill(defaultText: tooLong, task: "Human Input") == nil)
        let fits = String(repeating: "a", count: 500)
        #expect(FlowHumanPromptView.effectivePrefill(defaultText: fits, task: "Human Input") == fits)
    }

    @Test func effectivePrefillIsNilForAnEmptyDefaultText() {
        #expect(FlowHumanPromptView.effectivePrefill(defaultText: "", task: "Human Input") == nil)
    }
}
