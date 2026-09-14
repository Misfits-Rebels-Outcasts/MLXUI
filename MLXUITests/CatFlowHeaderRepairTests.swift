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
