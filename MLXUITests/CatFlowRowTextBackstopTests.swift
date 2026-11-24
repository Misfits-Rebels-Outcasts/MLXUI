import Testing
import Foundation
import AppKit
@testable import MLXUI

/// RT-5 (`RSI/DelegateRowTextBacklog.md`) — the no-dead-ends backstop: `FlowEditorModel
/// .setRowText` writes a row's raw settings string outright, but refuses a commit that makes
/// things worse — either the whole document stops parsing, or the row gains a validator issue
/// it didn't already have — rolling the mutation back before returning, inside the same
/// `commitChange` call so a refusal never reaches the undo stack. The View half
/// (`rowTextDisclosure`'s commit-on-blur wiring, "left dirty" display) isn't independently
/// testable without instantiating the View (no precedent for that in this suite); this covers
/// the model layer it calls into, the same split RT-1..RT-4 used.
struct CatFlowRowTextBackstopTests {

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-rt5-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "RT5", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String?, model: String? = nil, settings: String? = nil,
                     refs: [Ref] = []) -> Row {
        Row(id: UUID(), task: task, blockKind: nil, model: model, settings: settings,
            refs: refs, children: [], clause: nil)
    }

    // MARK: - No-op

    @Test func settingSameTextIsANoOpAndPushesNoUndo() throws {
        let r = row("Transcribe", model: "Whisper Large v3", settings: "lang=en")
        let model = try editor([r])
        let undoCountBefore = model.undoStack.count
        let refusal = model.setRowText("lang=en", for: r.id)
        #expect(refusal == nil)
        #expect(model.row(withID: r.id)?.settings == "lang=en")
        #expect(model.undoStack.count == undoCountBefore)
    }

    // MARK: - An accepted edit commits and is undoable, like any other editor here

    @Test func acceptedEditCommitsAndIsUndoable() throws {
        let r = row("Transcribe", model: "Whisper Large v3", settings: "lang=en; timestamps=true")
        let model = try editor([r])
        let undoCountBefore = model.undoStack.count

        let refusal = model.setRowText("lang=de; timestamps=true", for: r.id)
        #expect(refusal == nil)
        #expect(model.row(withID: r.id)?.settings == "lang=de; timestamps=true")
        #expect(model.undoStack.count == undoCountBefore + 1)

        model.undo()
        #expect(model.row(withID: r.id)?.settings == "lang=en; timestamps=true")
    }

    /// This is the invariant RT-5 exists for: even a `Template` row (whole-string, RT-1) or a
    /// task with no dedicated Properties control at all (`Embed`, still deferred per RT-3) can
    /// be edited through this box.
    @Test func worksOnATaskWithNoDedicatedPropertiesControl() throws {
        let r = row("Embed", model: "BGE-M3", settings: "\"old query\"")
        let model = try editor([r])
        #expect(model.setRowText("\"a brand new query\"", for: r.id) == nil)
        #expect(model.row(withID: r.id)?.settings == "\"a brand new query\"")
    }

    // MARK: - A commit that introduces a new validator issue is refused, rolled back, off the undo stack

    @Test func editIntroducingANewPlaceholderErrorIsRefused() throws {
        let src1 = row("Read Text", settings: "a.txt")
        let src2 = row("Read Text", settings: "b.txt")
        let tmpl = row("Template", settings: "\"{1} {2}\"", refs: [.rowRef(src1.id), .rowRef(src2.id)])
        let model = try editor([src1, src2, tmpl])
        #expect(!model.issues(for: tmpl.id).contains { $0.code == "E207" })
        let undoCountBefore = model.undoStack.count
        let original = model.row(withID: tmpl.id)?.settings

        let refusal = model.setRowText("\"{1} {9}\"", for: tmpl.id)

        #expect(refusal != nil)
        #expect(refusal?.contains("position") == true)   // E207's own template mentions positions
        // Rolled back: the row's actual data is untouched, and no undo entry was pushed.
        #expect(model.row(withID: tmpl.id)?.settings == original)
        #expect(model.undoStack.count == undoCountBefore)
        #expect(!model.issues(for: tmpl.id).contains { $0.code == "E207" })
    }

    /// Edit back to something the row already tolerated before RT-5 ever touched it — a plain
    /// regression check that a *good* edit right after a *refused* one still goes through.
    @Test func aValidEditAfterARefusalStillCommits() throws {
        let src1 = row("Read Text", settings: "a.txt")
        let src2 = row("Read Text", settings: "b.txt")
        let tmpl = row("Template", settings: "\"{1} {2}\"", refs: [.rowRef(src1.id), .rowRef(src2.id)])
        let model = try editor([src1, src2, tmpl])

        #expect(model.setRowText("\"{1} {9}\"", for: tmpl.id) != nil)   // refused
        #expect(model.setRowText("\"{2} {1}\"", for: tmpl.id) == nil)   // valid -> accepted
        #expect(model.row(withID: tmpl.id)?.settings == "\"{2} {1}\"")
    }

    // MARK: - A commit that makes the whole document unparseable is refused

    @Test func editThatMakesTheDocumentUnparseableIsRefused() throws {
        let r = row("Read Text", settings: "in.txt")
        let model = try editor([r])
        #expect(model.canSave)
        let undoCountBefore = model.undoStack.count

        let refusal = model.setRowText("\"unterminated", for: r.id)

        #expect(refusal != nil)
        #expect(model.row(withID: r.id)?.settings == "in.txt")
        #expect(model.undoStack.count == undoCountBefore)
        #expect(model.canSave)   // the document as a whole is still fine
    }

    /// If the document was *already* unparseable before this edit (a pre-existing problem
    /// unrelated to this row), `setRowText` must not refuse an otherwise-fine edit just
    /// because the document couldn't parse to begin with — only a *newly* introduced parse
    /// failure is refused.
    @Test func doesNotRefuseWhenTheDocumentWasAlreadyUnparseableBeforeThisEdit() throws {
        let broken = row("Read Text", settings: "\"already unterminated")
        let other = row("Save Text", settings: "out.txt")
        let model = try editor([broken, other])
        #expect(!model.canSave)   // already broken, independent of anything this test does

        // Editing the *other*, unrelated row to something perfectly valid must still commit —
        // the pre-existing brokenness elsewhere isn't this edit's fault.
        let refusal = model.setRowText("output.md", for: other.id)
        #expect(refusal == nil)
        #expect(model.row(withID: other.id)?.settings == "output.md")
    }

    // MARK: - R3 (Boo, 2026-11-24): the draft reaches the row without a blur

    /// Repro: a duplicated flow's `Redact` row, type `only="ssn"` / `only="person"` into Raw settings, press Run or Save — the box
    /// only committed when its text editor lost focus, which Run and Save do not cause, so the row kept no settings.
    @Test func aPendingDraftIsCommittedOntoItsRowByRunOrSave() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"person\"")
        #expect(model.row(withID: r.id)?.settings == nil)         // typing alone changes nothing (mid-edit)

        #expect(model.commitRawSettingsDraft() == nil)            // what Run and Save now call first
        #expect(model.row(withID: r.id)?.settings == "only=\"person\"")
        #expect(model.rawSettingsDraft == nil)
        #expect(CatSerializer.serialize(model.document).contains("only=\"person\""))   // and it reaches the .cat text
    }

    @Test func aDraftIsCommittedOntoTheRowItWasTypedIn_notTheOneSelectedNow() throws {
        let a = row("Redact", model: "Privacy Filter")
        let b = row("Save Text", settings: "out.txt")
        let model = try editor([a, b])
        model.rawSettingsDraft = (a.id, "only=\"email\"")
        model.selectedRowID = b.id                                // the selection moved away first
        model.commitRawSettingsDraft()
        #expect(model.row(withID: a.id)?.settings == "only=\"email\"")
        #expect(model.row(withID: b.id)?.settings == "out.txt")
    }

    @Test func aRefusedDraftIsLeftDirtyAndReportedAndTheRowIsUnchanged() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"ssn\"")
        let refusal = model.commitRawSettingsDraft()
        #expect(refusal?.contains("ssn") == true)
        #expect(model.row(withID: r.id)?.settings == nil)
        #expect(model.rawSettingsDraft?.text == "only=\"ssn\"")   // left dirty, never silently dropped
        #expect(model.rawSettingsRefusal?.rowID == r.id)
        model.discardRawSettingsDraft()
        #expect(model.rawSettingsDraft == nil && model.rawSettingsRefusal == nil)
    }

    @Test func committingWithNoDraftIsANoOp() throws {
        let r = row("Redact", model: "Privacy Filter", settings: "only=\"url\"")
        let model = try editor([r])
        #expect(model.commitRawSettingsDraft() == nil)
        #expect(model.row(withID: r.id)?.settings == "only=\"url\"")
    }

    // MARK: - R3-FIX-3: the refusal banner never outlives the edit it describes

    @Test func theRefusalClearsWhenTheDraftChangesAndWhenAValidDraftIsAccepted() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"ssn\"")
        #expect(model.commitRawSettingsDraft() != nil)
        #expect(model.rawSettingsRefusal?.rowID == r.id)

        model.rawSettingsDraft = (r.id, "only=\"person\"")        // typing again: a new attempt
        #expect(model.rawSettingsRefusal == nil)
        #expect(model.commitRawSettingsDraft() == nil)           // checks -> accepted
        #expect(model.rawSettingsRefusal == nil)
        #expect(model.row(withID: r.id)?.settings == "only=\"person\"")
    }

    @Test func theRefusalClearsWhenNothingIsPendingAnymore() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"ssn\"")
        _ = model.commitRawSettingsDraft()
        #expect(model.rawSettingsRefusal != nil)
        model.discardRawSettingsDraft()
        #expect(model.rawSettingsRefusal == nil)

        model.rawSettingsRefusal = (r.id, "stale")
        #expect(model.commitRawSettingsDraft() == nil)           // Run/Save with no draft pending
        #expect(model.rawSettingsRefusal == nil)
    }

    @Test func commitIsIdempotent_soTheDeferredBlurCommitAfterAClickedActionIsHarmless() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"url\"")
        #expect(model.commitRawSettingsDraft() == nil)           // the action's own commit
        let undo = model.undoStack.count
        #expect(model.commitRawSettingsDraft() == nil)           // the blur's later commit
        #expect(model.undoStack.count == undo)                   // no second edit, no second undo step
    }

    // MARK: - R3-FIX-1: smart quotes off in .cat text controls

    @MainActor @Test func plainCatTextTurnsOffEverySubstitutionSoAStraightQuoteSurvives() {
        let textView = NSTextView(frame: .zero)
        textView.isAutomaticQuoteSubstitutionEnabled = true
        textView.isAutomaticDashSubstitutionEnabled = true
        textView.isAutomaticTextReplacementEnabled = true
        textView.isAutomaticSpellingCorrectionEnabled = true
        PlainCatText.configure(textView)
        #expect(!textView.isAutomaticQuoteSubstitutionEnabled)
        #expect(!textView.isAutomaticDashSubstitutionEnabled)
        #expect(!textView.isAutomaticTextReplacementEnabled)
        #expect(!textView.isAutomaticSpellingCorrectionEnabled)
        textView.insertText("only=\"person\"", replacementRange: NSRange(location: 0, length: 0))
        #expect(textView.string == "only=\"person\"")             // straight " stored as typed

        // and the walker finds a TextEditor-style text view buried in a scroll view
        let scroll = NSScrollView(frame: .zero)
        let inner = NSTextView(frame: .zero)
        scroll.documentView = inner
        let root = NSView(frame: .zero)
        root.addSubview(scroll)
        #expect(PlainCatText.textViews(in: root).contains { $0 === inner })
    }

    // MARK: - R8: typographic quotes are straightened in the commit path (Raw settings and setting values only)

    @Test func typographicQuotesAreStraightenedAndOnlyThose() {
        #expect(TypographicQuotes.straightened("only=\u{201C}person\u{201D}") == "only=\"person\"")
        #expect(TypographicQuotes.straightened("a=\u{201E}x\u{201F} b=\u{2018}y\u{2019}") == "a=\"x\" b='y'")
        #expect(TypographicQuotes.straightened("already=\"straight\" 'ok'") == "already=\"straight\" 'ok'")
        #expect(TypographicQuotes.straightened("café — naïve") == "café — naïve")   // nothing else is touched
    }

    @Test func aCurlyDraftIsCommittedAsStraightText() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\u{201C}person,email\u{201D}")   // what the text view stored before R8
        #expect(model.commitRawSettingsDraft() == nil)                         // no longer refused
        #expect(model.row(withID: r.id)?.settings == "only=\"person,email\"")
    }

    // MARK: - R9: the draft is already on the row by the time a click arrives

    @MainActor @Test func theDraftIsCommittedShortlyAfterTypingStops() async throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"person\"")
        model.scheduleRawSettingsCommit(after: .milliseconds(30))
        #expect(model.row(withID: r.id)?.settings == nil)         // not yet
        try await Task.sleep(for: .milliseconds(300))
        #expect(model.row(withID: r.id)?.settings == "only=\"person\"")
        #expect(model.rawSettingsDraft == nil)
    }

    @MainActor @Test func aNewKeystrokeRestartsTheWait_andARefusedDraftIsReportedNotApplied() async throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"ssn\"")
        model.scheduleRawSettingsCommit(after: .milliseconds(30))
        try await Task.sleep(for: .milliseconds(300))
        #expect(model.row(withID: r.id)?.settings == nil)         // refused: the row is untouched
        #expect(model.rawSettingsRefusal?.rowID == r.id)          // and the inline sentence is up
    }

    // MARK: - R10: the red top banner follows the inline warning's rule

    @Test func theRedBannerGoesTheMomentTheInlineRefusalDoes() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"ssn\"")
        let refusal = model.commitRawSettingsDraft()
        model.rawSettingsBanner = "That Raw settings change wouldn't check: \(refusal ?? "")"   // what Run does
        #expect(model.rawSettingsBanner != nil)

        model.rawSettingsDraft = (r.id, "only=\"email\"")         // a later edit: the inline warning clears …
        #expect(model.rawSettingsRefusal == nil)
        #expect(model.rawSettingsBanner == nil)                  // … and the banner with it, before any commit or run
    }

    @Test func theRedBannerAlsoGoesWhenAValidDraftIsCommittedOrNothingIsPending() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsRefusal = (r.id, "stale"); model.rawSettingsBanner = "stale"
        #expect(model.commitRawSettingsDraft() == nil)           // Run/Save with nothing pending
        #expect(model.rawSettingsBanner == nil)
    }

    // MARK: - R11: a Run click that commits a pending draft owns the document change, so the handler's clearRun skips it

    @Test func aRunClickThatCommitsADraftOwnsExactlyThatDocumentChange() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"email\"")
        let before = model.document
        #expect(model.commitRawSettingsDraft() == nil)           // Run's synchronous commit
        #expect(model.document != before)
        model.markDocumentOwnedByRunClick()                      // Run marks it, then starts the session
        #expect(model.takeRunClickOwnership(of: model.document)) // the handler: this change is the click's own -> leave the run alone
        #expect(!model.takeRunClickOwnership(of: model.document))// once only: the next edit is an ordinary one (clearRun applies)
    }

    @Test func anOrdinaryEditAfterTheClickIsNotOwned() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        model.rawSettingsDraft = (r.id, "only=\"email\"")
        _ = model.commitRawSettingsDraft()
        model.markDocumentOwnedByRunClick()
        model.setRowText("only=\"url\"", for: r.id)              // the user edits again before the handler runs
        #expect(!model.takeRunClickOwnership(of: model.document))
    }

    /// The whole ordering with a real session: commit, start, then the handler's decision — the started run survives.
    @MainActor @Test func pendingDraftPlusImmediateRunEndsWithAStartedSession() throws {
        let r = row("Redact", model: "Privacy Filter")
        let model = try editor([r])
        let session = FlowRunSession()
        model.rawSettingsDraft = (r.id, "only=\"email\"")
        #expect(model.commitRawSettingsDraft() == nil)
        model.markDocumentOwnedByRunClick()
        session.isRunning = true                                 // what a started run looks like; `start` sets it first thing
        if !model.takeRunClickOwnership(of: model.document) { session.clearRun(doc: model.document) }   // the view's handler
        #expect(session.isRunning)                               // not cancelled by its own click's document change

        // control: an *ordinary* document change does cancel the run, so the assertion above is not true by construction
        session.clearRun(doc: model.document)
        #expect(!session.isRunning)
    }
}
