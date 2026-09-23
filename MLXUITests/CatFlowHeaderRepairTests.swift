import Testing
import Foundation
@testable import MLXUI

/// Phase FIX, FIX-1 (`RSI/DelegateFixItBacklog.md`) — the one header-flag repair serving
/// E103/E109/E118/E120/E604. Every one of those five messages ends "(fmt will do this for
/// you.)"; this app has no `fmt`, so the fix is a single-flag header edit through
/// `FlowHeaderRepair` + `FlowEditorModel.applyHeaderRepair`.
struct CatFlowHeaderRepairTests {

    private func editor(_ document: FlowDocument) -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-fixit-\(UUID().uuidString)")
        return FlowEditorModel(name: "FIX-1", document: document, workspace: FlowWorkspace(root: root))
    }

    // MARK: - The table

    @Test func flagForCodeMapsAllFiveFlagErrors() {
        #expect(FlowHeaderRepair.flag(forCode: "E103") == .network)
        #expect(FlowHeaderRepair.flag(forCode: "E109") == .improvise)
        #expect(FlowHeaderRepair.flag(forCode: "E118") == .code)
        #expect(FlowHeaderRepair.flag(forCode: "E120") == .offdevice)
        #expect(FlowHeaderRepair.flag(forCode: "E604") == .events)
        #expect(FlowHeaderRepair.flag(forCode: "E999") == nil)
    }

    // MARK: - `FlowHeaderRepair.apply`

    @Test func applyIsIdempotent() {
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [])
        let once = FlowHeaderRepair.apply(.offdevice, to: doc)
        let twice = FlowHeaderRepair.apply(.offdevice, to: once)
        #expect(once == twice)
        #expect(CatSerializer.serialize(once) == CatSerializer.serialize(twice))
    }

    @Test func applyPreservesTheHeaderKeywordsSpelling() {
        let catflowDoc = FlowDocument(version: "0.8", headerKeyword: "catflow", rows: [])
        let mlxflowDoc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [])
        let repairedCat = FlowHeaderRepair.apply(.offdevice, to: catflowDoc)
        let repairedMlx = FlowHeaderRepair.apply(.offdevice, to: mlxflowDoc)
        #expect(CatSerializer.serialize(repairedCat).hasPrefix("cat"))
        #expect(CatSerializer.serialize(repairedMlx).hasPrefix("mlx"))
    }

    @Test func applyPreservesExistingFlagsOrderAndAppendsTheNewFlag() {
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [],
                               flags: [.network], flagsOrder: [.network])
        let repaired = FlowHeaderRepair.apply(.offdevice, to: doc)
        #expect(repaired.flagsOrder == [.network, .offdevice])
        #expect(repaired.flags == [.network, .offdevice])
    }

    // MARK: - `FlowHeaderRepair.remove` (FH-4)

    @Test func removeIsIdempotent() {
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [],
                               flags: [.offdevice], flagsOrder: [.offdevice])
        let once = FlowHeaderRepair.remove(.offdevice, from: doc)
        let twice = FlowHeaderRepair.remove(.offdevice, from: once)
        #expect(once == twice)
        #expect(CatSerializer.serialize(once) == CatSerializer.serialize(twice))
    }

    @Test func removeOnAnUndeclaredFlagReturnsTheDocumentUnchanged() {
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [],
                               flags: [.network], flagsOrder: [.network])
        let removed = FlowHeaderRepair.remove(.offdevice, from: doc)
        #expect(removed == doc)
    }

    @Test func removeDropsFromBothFlagsAndFlagsOrder() {
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [],
                               flags: [.network, .offdevice], flagsOrder: [.network, .offdevice])
        let removed = FlowHeaderRepair.remove(.network, from: doc)
        #expect(removed.flags == [.offdevice])
        #expect(removed.flagsOrder == [.offdevice])
    }

    /// FH-4's own byte-identical round-trip golden: `apply` then `remove` returns exactly the
    /// original bytes, and a lone `remove` changes only line one.
    @Test func applyThenRemoveRoundTripsByteIdentically() throws {
        let text = """
        mlxflow 0.8; network
        1. Read Text    memo.txt
        2. Save Text    out.md
        """
        let doc = try CatParser.parse(text)
        let original = CatSerializer.serialize(doc)

        let repaired = FlowHeaderRepair.apply(.offdevice, to: doc)
        let roundTripped = FlowHeaderRepair.remove(.offdevice, from: repaired)
        #expect(CatSerializer.serialize(roundTripped) == original)

        let removedOnly = FlowHeaderRepair.remove(.network, from: doc)
        let originalLines = original.split(separator: "\n", omittingEmptySubsequences: false)
        let removedLines = CatSerializer.serialize(removedOnly).split(separator: "\n", omittingEmptySubsequences: false)
        #expect(originalLines.count == removedLines.count)
        #expect(originalLines[0] != removedLines[0])
        #expect(originalLines.dropFirst().elementsEqual(removedLines.dropFirst()))
    }

    // MARK: - Round trip: repair -> serialize -> reparse -> checkFlow

    @Test func e120RoundTripClearsTheIssueAndConfinesTheByteDiffToLineOne() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Answer       (1)  claude-sonnet @ anthropic
        3. Save Text    out.md

        models:
          claude-sonnet @ anthropic = anthropic/claude-sonnet-4
        """
        let before = try CatParser.parseForValidation(text)
        #expect(FlowValidator.checkFlow(before).contains { $0.code == "E120" })

        let doc = try CatParser.parse(text)
        let repaired = FlowHeaderRepair.apply(.offdevice, to: doc)
        let originalLines = CatSerializer.serialize(doc).split(separator: "\n", omittingEmptySubsequences: false)
        let repairedLines = CatSerializer.serialize(repaired).split(separator: "\n", omittingEmptySubsequences: false)
        #expect(originalLines.count == repairedLines.count)
        #expect(originalLines[0] != repairedLines[0])
        #expect(originalLines.dropFirst().elementsEqual(repairedLines.dropFirst()))

        let after = try CatParser.parseForValidation(CatSerializer.serialize(repaired))
        #expect(!FlowValidator.checkFlow(after).contains { $0.code == "E120" })
    }

    @Test func e604RoundTripClearsTheIssueAndConfinesTheByteDiffToLineOne() throws {
        let text = """
        mlxflow 0.8
        1. On File      inbox/
        2. Save Text    out.md
        """
        let before = try CatParser.parseForValidation(text)
        #expect(FlowValidator.checkFlow(before).contains { $0.code == "E604" })

        let doc = try CatParser.parse(text)
        let repaired = FlowHeaderRepair.apply(.events, to: doc)
        let originalLines = CatSerializer.serialize(doc).split(separator: "\n", omittingEmptySubsequences: false)
        let repairedLines = CatSerializer.serialize(repaired).split(separator: "\n", omittingEmptySubsequences: false)
        #expect(originalLines[0] != repairedLines[0])
        #expect(originalLines.dropFirst().elementsEqual(repairedLines.dropFirst()))

        let after = try CatParser.parseForValidation(CatSerializer.serialize(repaired))
        #expect(!FlowValidator.checkFlow(after).contains { $0.code == "E604" })
    }

    // MARK: - `FlowEditorModel.headerRepair` / `applyHeaderRepair`

    @Test @MainActor func headerRepairOffersOffdeviceForARemoteRowAndUndoRestoresTheOriginalBytes() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Answer       (1)  claude-sonnet @ anthropic
        3. Save Text    out.md

        models:
          claude-sonnet @ anthropic = anthropic/claude-sonnet-4
        """
        let doc = try CatParser.parse(text)
        let model = editor(doc)
        let answerRow = try #require(model.document.rows.first { $0.task == "Answer" })
        #expect(model.headerRepair(for: answerRow.id) == .offdevice)

        let before = model.catText
        model.applyHeaderRepair(.offdevice)
        #expect(model.catText != before)
        #expect(model.document.flags.contains(.offdevice))
        #expect(model.headerRepair(for: answerRow.id) == nil, "the issue is gone, so there's nothing left to repair")

        model.undo()
        #expect(model.catText == before)
        #expect(!model.document.flags.contains(.offdevice))
    }

    /// The mirror of the test above, for the Flow tab's untick path (FH-4) rather than the
    /// row-level one-click repair: starting from a flow that already declares `offdevice` and
    /// genuinely needs it, `removeHeaderFlag` — the exact function `FlowInspectorPane`'s
    /// checkbox calls when unticked — must actually block Save, not just change the flag.
    /// Written on direct request to confirm this end-to-end, after `FH-4-FIX-1` found the
    /// opposite gap for `network` (a status that claimed enforcement nothing backed).
    @Test @MainActor func removingOffdeviceFromARemoteRowFlowBlocksSaveAndUndoRestoresIt() throws {
        let text = """
        mlxflow 0.8; offdevice
        1. Read Text    memo.txt
        2. Answer       (1)  claude-sonnet @ anthropic
        3. Save Text    out.md

        models:
          claude-sonnet @ anthropic = anthropic/claude-sonnet-4
        """
        let doc = try CatParser.parse(text)
        let model = editor(doc)
        #expect(model.canSave, "the flow is valid as declared — nothing should block Save yet")

        let (workspace, cleanup) = tempFlagInventoryWorkspace()
        defer { cleanup() }
        let beforeStatus = FlowFlagInventory.status(of: .offdevice, in: model.document,
                                                    workspace: workspace, flowID: "flow-1")
        guard case .requiredAndDeclared(let rowNumber, let task) = beforeStatus else {
            Issue.record("expected .requiredAndDeclared before removal, got \(beforeStatus)")
            return
        }
        #expect(rowNumber == "2")
        #expect(task == "Answer")

        let before = model.catText
        model.removeHeaderFlag(.offdevice)
        #expect(!model.document.flags.contains(.offdevice))
        #expect(model.catText != before)

        // The actual gate a Flow-tab untick must trip: Save is blocked, with a real message.
        #expect(!model.canSave, "unticking a genuinely-needed offdevice must block Save")
        let reason = try #require(model.saveBlockReason)
        #expect(reason.contains("offdevice"))

        let afterStatus = FlowFlagInventory.status(of: .offdevice, in: model.document,
                                                   workspace: workspace, flowID: "flow-1")
        guard case .requiredButMissing(let code, let message) = afterStatus else {
            Issue.record("expected .requiredButMissing after removal, got \(afterStatus)")
            return
        }
        #expect(code == "E120")
        #expect(message == reason, "the Flow tab's reason line and the real Save-block reason must be the same sentence")

        model.undo()
        #expect(model.catText == before)
        #expect(model.document.flags.contains(.offdevice))
        #expect(model.canSave)
    }

    private func tempFlagInventoryWorkspace() -> (workspace: FlowWorkspace, cleanup: () -> Void) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-flaginventory-\(UUID().uuidString)")
        return (FlowWorkspace(root: root), { try? FileManager.default.removeItem(at: root) })
    }

    /// Under `APPSTORE_BUILD` (the MLXUI test host — `CatFlowCapabilityGateTests` pins this),
    /// `code`/`improvise` are in `CapabilityGate.appStoreRefusedFlags`: offering "add `code`"
    /// there would produce a flow that same build then refuses to run. `offdevice`/`events`
    /// are not in that set, so their repairs are offered in every edition.
    @Test @MainActor func underAppStoreBuildNoActionForCodeOrImproviseButOneForOffdeviceAndEvents() throws {
        #expect(CapabilityGate.isAppStoreBuild,
                "the MLXUI test host builds with Config/AppStore.xcconfig, so APPSTORE_BUILD must be defined")

        let improviseText = """
        mlxflow 0.8
        1. Read Text   memo.txt
        2. Improvise   workdir=scratch; max_actions=5; timeout=30s
        """
        let improviseDoc = try CatParser.parse(improviseText)
        let improviseModel = editor(improviseDoc)
        let improviseRow = try #require(improviseModel.document.rows.first { $0.task == "Improvise" })
        #expect(improviseModel.headerRepair(for: improviseRow.id) == nil)

        let transformsText = """
        mlxflow 0.8
        1. Tidy
        transforms:
          Tidy  text -> text
            run: script.sh
        """
        let transformsDoc = try CatParser.parse(transformsText)
        let transformsModel = editor(transformsDoc)
        let tidyRow = try #require(transformsModel.document.rows.first)
        #expect(transformsModel.headerRepair(for: tidyRow.id) == nil)

        let offdeviceText = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Answer       (1)  claude-sonnet @ anthropic
        3. Save Text    out.md

        models:
          claude-sonnet @ anthropic = anthropic/claude-sonnet-4
        """
        let offdeviceDoc = try CatParser.parse(offdeviceText)
        let offdeviceModel = editor(offdeviceDoc)
        let answerRow = try #require(offdeviceModel.document.rows.first { $0.task == "Answer" })
        #expect(offdeviceModel.headerRepair(for: answerRow.id) == .offdevice)

        let eventsText = """
        mlxflow 0.8
        1. On File      inbox/
        2. Save Text    out.md
        """
        let eventsDoc = try CatParser.parse(eventsText)
        let eventsModel = editor(eventsDoc)
        let triggerRow = try #require(eventsModel.document.rows.first)
        #expect(eventsModel.headerRepair(for: triggerRow.id) == .events)
    }

    // MARK: - Rule 11 tripwire

    /// Rule 11 (`DelegateFixItBacklog.md`): never reword an error string to make a fix
    /// unnecessary. All five templates promise "fmt will do this for you" — this test fails
    /// if that promise is ever softened instead of built.
    @Test func allFiveFlagMessagesStillPromiseTheFixRule11Tripwire() throws {
        for code in FlowHeaderRepair.flagForCode.keys {
            let spec = try #require(ErrorCatalog.catalogV08[code])
            #expect(spec.template.hasSuffix("(fmt will do this for you.)"),
                    "rule 11: \(code)'s template must stay a promise this fix keeps, not one edited away")
        }
    }
}
