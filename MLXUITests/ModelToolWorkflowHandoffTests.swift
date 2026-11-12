import Testing
import Foundation
@testable import MLXUI

/// MT5-2: when "Save as workflow" is on, what the hand-off makes on disk, and that the editor opens on a flow it will let
/// the user save (the editor's own validator, against the installed registry).
@MainActor @Suite struct ModelToolWorkflowHandoffTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func temp() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mt5-handoff-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func goldenFlow(_ name: String) throws -> ModelToolFlow {
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("Fixtures/ModelTools/\(name).trace.json"))
        return ModelToolFlowBuilder.build(try JSONDecoder().decode(ModelToolTrace.self, from: data))
    }

    private func assetsFolder(with names: [String]) throws -> URL {
        let dir = try temp()
        for n in names { try Data("data-\(n)".utf8).write(to: dir.appendingPathComponent(n)) }
        return dir
    }

    // MARK: Button state

    @Test func theButtonStateSaysWhyItIsOff() throws {
        let receipt = try goldenFlow("receipt-ocr-answer-speak")
        let empty = ModelToolFlowBuilder.build(ModelToolTrace(events: [], assets: []))
        typealias H = ModelToolWorkflowHandoff
        #expect(H.availability(flow: receipt, isRunning: false, workflowsAvailable: true) == H.Availability(enabled: true, reason: nil))
        #expect(H.availability(flow: receipt, isRunning: true, workflowsAvailable: true).reason == "Wait for the reply to finish.")
        #expect(H.availability(flow: empty, isRunning: false, workflowsAvailable: true).enabled == false)
        #expect(H.availability(flow: empty, isRunning: false, workflowsAvailable: true).reason?.hasPrefix("Nothing to save") == true)
        #expect(H.availability(flow: receipt, isRunning: false, workflowsAvailable: false).enabled == false)
        // Plain Q&A with no tool and no text attachment is an empty flow by design.
        let plain = ModelToolFlowBuilder.build(ModelToolTrace(
            events: [.init(kind: .user, text: "hi"), .init(kind: .assistant, text: "hello", modelID: "apple-intelligence")], assets: []))
        #expect(plain.isEmpty)
    }

    @Test func theWorkflowNameIsTheChatTitleOrADefault() {
        #expect(ModelToolWorkflowHandoff.workflowName(chatTitle: "Receipt total") == "Receipt total")
        #expect(ModelToolWorkflowHandoff.workflowName(chatTitle: "New Chat") == "Chat workflow")
        #expect(ModelToolWorkflowHandoff.workflowName(chatTitle: "  ") == "Chat workflow")
        #expect(ModelToolWorkflowHandoff.workflowName(chatTitle: nil) == "Chat workflow")
        #expect(ModelToolWorkflowHandoff.workflowName(chatTitle: String(repeating: "x", count: 200)).count == 60)
    }

    // MARK: Hand-off

    @Test func prepareMakesAFlowFolderWithTheCopiesAndAnUnsavedTarget() throws {
        let flow = try goldenFlow("receipt-ocr-answer-speak")
        let assets = try assetsFolder(with: ["a1.jpg"]); defer { try? FileManager.default.removeItem(at: assets) }
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let workspace = FlowWorkspace(root: root)

        let target = try ModelToolWorkflowHandoff.prepare(flow: flow, chatTitle: "Receipt", assetsDirectory: assets, workspace: workspace)
        let dir = workspace.directory(for: target.flowID)
        #expect(UUID(uuidString: target.flowID) != nil)
        #expect(try Data(contentsOf: dir.appendingPathComponent("receipt.jpg")) == Data("data-a1.jpg".utf8))
        #expect(target.document == flow.document)
        #expect(target.savedText == nil)
        #expect(target.fileURL == nil)
        #expect(target.workspace == nil)
        #expect(target.name == "Receipt")
        #expect(target.leftOut == flow.banner.map(\.line))
        #expect(target.removeFolderIfUnsaved)
        // Nothing is written as a flow until the user saves.
        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        #expect(files == ["receipt.jpg"])
        // The chat's own copy is left alone.
        #expect(FileManager.default.fileExists(atPath: assets.appendingPathComponent("a1.jpg").path))
    }

    @Test func aFlowWithNoAttachmentsStillGetsAFolder() throws {
        let flow = try goldenFlow("text-translate-speak")
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let target = try ModelToolWorkflowHandoff.prepare(flow: flow, chatTitle: nil, assetsDirectory: root, workspace: FlowWorkspace(root: root))
        #expect(FileManager.default.fileExists(atPath: FlowWorkspace(root: root).directory(for: target.flowID).path))
        #expect(target.name == "Chat workflow")
    }

    @Test func aMissingSourceFailsPlainlyAndLeavesNoFolder() throws {
        let flow = try goldenFlow("receipt-ocr-answer-speak")
        let assets = try assetsFolder(with: []); defer { try? FileManager.default.removeItem(at: assets) }
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: ModelToolWorkflowHandoff.Failure.missingSource("receipt.jpg")) {
            try ModelToolWorkflowHandoff.prepare(flow: flow, chatTitle: nil, assetsDirectory: assets, workspace: FlowWorkspace(root: root))
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
        #expect(ModelToolWorkflowHandoff.Failure.missingSource("receipt.jpg").description.contains("receipt.jpg"))
    }

    @Test func aFailedCopyRemovesTheHalfBuiltFolder() throws {
        var flow = try goldenFlow("receipt-ocr-answer-speak")
        // Two copies to one name: the second cannot land.
        flow.copies = [ModelToolFlow.Copy(handle: "a1", source: "a1.jpg", destination: "x.jpg"),
                       ModelToolFlow.Copy(handle: "a1", source: "a1.jpg", destination: "x.jpg")]
        let assets = try assetsFolder(with: ["a1.jpg"]); defer { try? FileManager.default.removeItem(at: assets) }
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: ModelToolWorkflowHandoff.Failure.copyFailed("x.jpg")) {
            try ModelToolWorkflowHandoff.prepare(flow: flow, chatTitle: nil, assetsDirectory: assets, workspace: FlowWorkspace(root: root))
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
    }

    @Test func anEmptyFlowIsRefusedBeforeAnythingIsMade() throws {
        let empty = ModelToolFlowBuilder.build(ModelToolTrace(events: [], assets: []))
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: ModelToolWorkflowHandoff.Failure.notSavable) {
            try ModelToolWorkflowHandoff.prepare(flow: empty, chatTitle: nil, assetsDirectory: root, workspace: FlowWorkspace(root: root))
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
    }

    // MARK: Closing the editor without saving

    @Test func closingUnsavedRemovesTheFolderButNeverAFlowOrAnOpenEditor() throws {
        let flow = try goldenFlow("receipt-ocr-answer-speak")
        let assets = try assetsFolder(with: ["a1.jpg"]); defer { try? FileManager.default.removeItem(at: assets) }
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let workspace = FlowWorkspace(root: root)
        let target = try ModelToolWorkflowHandoff.prepare(flow: flow, chatTitle: nil, assetsDirectory: assets, workspace: workspace)
        let dir = workspace.directory(for: target.flowID)

        // Something pushed over the editor, or a saved flow: kept.
        #expect(!ModelToolWorkflowHandoff.discardIfUnsaved(flowID: target.flowID, workspace: workspace, saved: false, stillOpen: true))
        #expect(!ModelToolWorkflowHandoff.discardIfUnsaved(flowID: target.flowID, workspace: workspace, saved: true, stillOpen: false))
        #expect(FileManager.default.fileExists(atPath: dir.path))
        // A folder that has a flow file in it is never removed.
        try "mlxflow 0.8\n".write(to: dir.appendingPathComponent("x.cat"), atomically: true, encoding: .utf8)
        #expect(!ModelToolWorkflowHandoff.discardIfUnsaved(flowID: target.flowID, workspace: workspace, saved: false, stillOpen: false))
        try FileManager.default.removeItem(at: dir.appendingPathComponent("x.cat"))
        // Closed, never saved: gone.
        #expect(ModelToolWorkflowHandoff.discardIfUnsaved(flowID: target.flowID, workspace: workspace, saved: false, stillOpen: false))
        #expect(!FileManager.default.fileExists(atPath: dir.path))
    }

    // MARK: The editor opens on it and gates Save

    private func editor(for flow: ModelToolFlow) throws -> FlowEditorModel {
        let root = try temp()
        return FlowEditorModel(name: "Chat workflow", document: flow.document, workspace: FlowWorkspace(root: root),
                               sampleSourceDir: nil, savedText: nil)
    }

    @Test(arguments: ["receipt-ocr-answer-speak", "meeting-audio-transcribe-answer", "text-translate-speak"])
    func theEditorOpensEachGoldenDirtyAndSaveIsEnabled(name: String) throws {
        let flow = try goldenFlow(name)
        let model = try editor(for: flow)
        #expect(model.isDirty)
        #expect(model.saveBlockReason == nil, "\(model.saveBlockReason ?? "")")
        #expect(model.canSave)
        #expect(model.catText == flow.text)
    }

    @Test func savingWritesTheFlowIntoItsFolderBesideTheCopiedFiles() throws {
        let flow = try goldenFlow("receipt-ocr-answer-speak")
        let assets = try assetsFolder(with: ["a1.jpg"]); defer { try? FileManager.default.removeItem(at: assets) }
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let workspace = FlowWorkspace(root: root)
        let target = try ModelToolWorkflowHandoff.prepare(flow: flow, chatTitle: "Receipt", assetsDirectory: assets, workspace: workspace)
        let model = FlowEditorModel(name: target.name, flowID: target.flowID, document: target.document, workspace: workspace,
                                    sampleSourceDir: nil, savedText: target.savedText)
        try model.save()
        let files = try FileManager.default.contentsOfDirectory(atPath: workspace.directory(for: target.flowID).path).sorted()
        #expect(files == ["Receipt.cat", "receipt.jpg"])
        #expect(!model.isDirty)
        // Saved: the discard hook must keep it.
        #expect(!ModelToolWorkflowHandoff.discardIfUnsaved(flowID: target.flowID, workspace: workspace, saved: model.savedURL != nil, stillOpen: false))
    }

    @Test func aPinnedUnbridgedModelValidatesInTheEditor() throws {
        let trace = ModelToolTrace(
            events: [.init(kind: .user, text: "say it: hello"), .init(kind: .tool, text: "hello", tool: "speak", output: "a1")],
            assets: [.init(handle: "a1", kind: "audio", name: "Speech", fileName: "a1.wav", rowText: "1. Speak   Qwen3-TTS 1.7B", model: "Qwen3-TTS 1.7B")])
        let flow = ModelToolFlowBuilder.build(
            trace, catalog: ModelToolFlowCatalog(hfModelIDs: ["Qwen3-TTS 1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"]))
        #expect(!flow.document.models.isEmpty)
        let model = try editor(for: flow)
        #expect(model.canSave, "\(model.saveBlockReason ?? "")")
    }

    @Test func anInvalidDocumentCannotBeSaved() throws {
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [
            Row(task: "Speak", model: "No Such Voice 9Z", settings: nil),
        ])
        let model = FlowEditorModel(name: "Bad", document: doc, workspace: FlowWorkspace(root: try temp()), sampleSourceDir: nil)
        #expect(!model.canSave)
        #expect(model.saveBlockReason != nil)
        #expect(throws: (any Error).self) { try model.save() }
    }
}
