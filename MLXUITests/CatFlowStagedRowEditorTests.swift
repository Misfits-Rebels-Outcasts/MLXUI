import Testing
import Foundation
@testable import MLXUI

/// ER-FIX-2: the editor (Duplicate & Edit) must not mark a staged row "can't complete". The run path already serves `.staged`
/// (`FlowRunner.rowClassRefusal` is nil for it; `TaskAvailability` is `.available`); the editor's own hand list was never told.
struct CatFlowStagedRowEditorTests {

    private func editor(for doc: FlowDocument) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("er-fix2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "Copy", document: doc, workspace: FlowWorkspace(root: root))
    }

    private func warnings(_ model: FlowEditorModel) -> [String] {
        model.document.rows.compactMap { model.warning(for: $0.id) }
    }

    @Test func duplicateAndEditOfShippingLabelMarksNoRowUnrunnable() throws {
        let model = try editor(for: try GalleryLoader.loadDocument(flowID: "19-ShippingLabel"))
        let found = warnings(model)
        #expect(!found.contains { $0.contains("can't complete") }, "\(found)")
    }

    @Test func stageSendAndStagePostAreNotMarkedEither() throws {
        let doc = try CatParser.parse("mlxflow 0.8\n1. Template   \"hello\"\n2. Stage Send   outbox\n3. Stage Post   threads\n")
        let found = warnings(try editor(for: doc))
        #expect(!found.contains { $0.contains("can't complete") }, "\(found)")
    }

    @Test func theEditorFollowsTheRunnersOwnRuleForStagedRows() throws {
        // The check is derived from the runner's own rule, not a second list: these are the verdicts it reads.
        #expect(FlowRunner.rowClassRefusal("Stage Print") == nil)
        #expect(FlowRunner.rowClassRefusal("Stage Send") == nil)
        #expect(FlowRunner.rowClassRefusal("No Such Task") == .unknownTask)
    }
}
