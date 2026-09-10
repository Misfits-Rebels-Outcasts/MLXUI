import Testing
import Foundation
@testable import MLXUI

/// DA-9 (`RSI/DelegateDeciderBacklog.md`, journal `2026-246`) — SPEC-Q214.
///
/// A `.model`-class task that is model-optional (`TaskModels.isModelOptional`) must read the
/// same way at **all three sites that independently gate a model-less row**:
///
///  1. the editor's per-row warning (`FlowEditorModel.warning`) — DA-6
///  2. `FlowPreflight.run` / `FlowRunnability.refusalReason` — DA-9
///  3. `RealExecutor` — the branch that runs ahead of `resolveModel` — DA-6
///
/// Each of the three was found broken by a person hitting it, never by a test. This pins them
/// agreeing, driven off `TaskModels.modelOptionalTaskNames`, so a future model-optional task
/// exercises all three at once — add its fixture below or this test fails.
struct CatFlowModelOptionalConsistencyTests {

    /// A model-less row that legitimately runs, per task: the settings and the input asset the
    /// task's no-model fast path accepts. A `nil` here (missing entry) fails `everyModelOptionalTaskHasAFixture`.
    private func fixture(for task: String) -> (settings: String, input: Asset)? {
        switch task {
        case "Text to Table":
            // The gallery-30 shape: a `Table to Text format=csv` sibling wrote this CSV; the
            // model-less `Text to Table` row inverts it via `TableTool.parseDelimitedTable`.
            return ("expected=\"merchant, total\"",
                    Asset(items: [Item(kind: .text, value: "merchant,total\nCafe,12\nBooks,30",
                                       path: nil, sourceText: nil)]))
        default:
            return nil
        }
    }

    @Test func everyModelOptionalTaskHasAFixture() {
        for task in TaskModels.modelOptionalTaskNames {
            #expect(fixture(for: task) != nil,
                    "\(task) is model-optional but has no fixture — DA-9's cross-site test can't cover it")
        }
    }

    @MainActor @Test func editorPreflightAndExecutorAgreeForAModellessRow() async throws {
        let catalogURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: catalogURL))
            .domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))

        for task in TaskModels.modelOptionalTaskNames {
            let fx = try #require(fixture(for: task), "no fixture for \(task)")

            // Row 2 (after a leading source row) names this task with NO model.
            let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [
                Row(id: UUID(), task: "Read Text", settings: "in.txt"),
                Row(id: UUID(), task: task, model: nil, settings: fx.settings, refs: []),
            ])
            let rowID = doc.rows[1].id

            // 1. Editor: no "needs a model" warning on the model-less row.
            let editor = FlowEditorModel(name: "da9-\(task)", document: doc,
                                         workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
            editor.modelCatalog = catalog
            editor.claimableModelIDs = claimable
            #expect(editor.warning(for: rowID)?.contains("needs a model") != true,
                    "\(task): editor warns 'needs a model' on a model-optional row")

            // 2. Preflight: the flow is not blocked, and there is no refusal sentence.
            let preflight = FlowPreflight.run(doc, catalog: catalog,
                                              installedModelIDs: [], totalRAMGB: 1_000_000)
            #expect(!preflight.isBlocked, "\(task): FlowPreflight blocks a model-optional row")
            #expect(FlowPreflight.blockedReason(preflight, totalRAMGB: 1_000_000) == nil,
                    "\(task): FlowPreflight has a refusal sentence for a model-optional row")

            // 3. Executor: runs the row without throwing, without reaching the model stage
            //    (which is wired to throw, proving the model-less branch handled it).
            let blob = FileManager.default.temporaryDirectory
                .appendingPathComponent("da9-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: blob, withIntermediateDirectories: true)
            let executor = RealExecutor(
                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory
                    .appendingPathComponent("da9-ws-\(UUID().uuidString)")),
                flowID: "da9",
                blobDirectory: blob,
                makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "none", kind: .llm) },
                installedModelIDs: [],
                catalog: catalog)
            let out = try await executor.execute(
                path: "2", row: doc.rows[1], inputs: [fx.input],
                transcript: nil, context: nil, usedFlowContent: nil)
            #expect(!out.items.isEmpty, "\(task): executor produced nothing for a model-optional row")
        }
    }
}
