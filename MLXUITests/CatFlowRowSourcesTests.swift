import Testing
import Foundation
@testable import MLXUI

/// RP-1 (`RSI/DelegateRowPortsBacklog.md`, owner ruling G1, 2026-10-19) — `RowSources` is the
/// one table behind the empty-flow picker and the row-1 warning, and it must not claim a row
/// can start a flow when `RealExecutor` would throw for want of an input.
struct CatFlowRowSourcesTests {

    private func loadCatalog() throws -> [ModelEntry] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
            .domains.flatMap { $0.allModels }
    }

    /// Settings that stand in for the row's own value. RP-2 moves these to
    /// `FlowEditorModel.defaultSettings(forTask:)` / `SampleSeed`.
    private func ownSettings(for entry: RowSources.Entry) -> String? {
        if let seed = SampleSeed.seedValue(for: entry.task) { return seed }
        switch entry.task {
        case "Generate Image", "Generate Sound", "Generate Video": return "\"a prompt\""
        case "Human Input": return "\"What should I use?\"; wait=forever"
        case "Template": return "\"hello\""
        case "Range": return "1..3"
        case "Web Fetch", "HTTP Get", "Fetch Feed", "Download File":
            return "url=\"http://127.0.0.1:1/\""       // refused at once; never leaves the Mac
        case "Web Search": return "query=\"mlx\""
        default: return "missing.bin"                  // Read Video / Index / Context
        }
    }

    // MARK: - The table

    @Test func theTableListsExactlyTheRulingsRows() {
        let offered = Set(RowSources.entries.filter(\.offeredOnEmptyFlow).map(\.task))
        let reads = Set(SampleSeed.samples.keys)
        #expect(offered == reads.union(["Generate Image", "Generate Sound", "Generate Video",
                                        "Human Input", "Template", "Range"]))
        for web in ["Web Fetch", "HTTP Get", "Fetch Feed", "Download File", "Web Search"] {
            #expect(RowSources.contains(web) && !RowSources.isOfferedOnEmptyFlow(web))
        }
        #expect(Set(RowSources.entries.map(\.task)).count == RowSources.entries.count)
        #expect(!RowSources.contains("Generate"))      // G1b pending
    }

    @Test func everyTableTaskIsInTheCatalog() {
        for e in RowSources.entries { #expect(TaskCatalog.get(e.task) != nil, "\(e.task)") }
    }

    // MARK: - The empty-flow list

    @Test func theEmptyFlowListHasTheNewRowsAndNoneOfTheRemovals() {
        let starts = Set(FlowEditorModel.startingNodes().map(\.name))
        for t in ["Generate Image", "Generate Sound", "Generate Video", "Range", "Template",
                  "Human Input", "Read Text"] { #expect(starts.contains(t), "\(t)") }
        // G1 (2026-10-19): these can't run with nothing in.
        for t in ["Save Text", "Save Audio", "Save Image", "Save Video",
                  "Classify", "Gate", "Score", "Judge", "Think",
                  "Load Checkpoint", "Init Latent",
                  "Web Fetch", "HTTP Get", "Fetch Feed", "Download File", "Web Search",
                  "Read Video", "Read Index", "Read Context", "Generate", "Decide"] {
            #expect(!starts.contains(t), "\(t)")
        }
    }

    // MARK: - The warning

    @MainActor private func fixture(_ task: String, settings: String?) throws -> (FlowEditorModel, Row) {
        let catalog = try loadCatalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        let model = TaskCatalog.get(task)?.taskClass == .model
            ? FlowEditorModel.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable)
            : nil
        let row = Row(id: UUID(), task: task, model: model, settings: settings, refs: [])
        let editor = FlowEditorModel(
            name: "rp1",
            document: FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [row]),
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        editor.modelCatalog = catalog
        editor.claimableModelIDs = claimable
        return (editor, row)
    }

    @Test @MainActor func aSourceRowAtRowOneHasNoWarning() throws {
        for (task, settings) in [("Generate Sound", "\"calm\""),
                                 ("Web Fetch", "url=\"https://example.com\""),
                                 ("Read Index", "kb.index")] {
            let (editor, row) = try fixture(task, settings: settings)
            #expect(editor.warning(for: row.id) == nil, "\(task)")
        }
    }

    @Test @MainActor func aSummarizeRowAtRowOneStillSaysNothingFeedsIt() throws {
        let (editor, row) = try fixture("Summarize", settings: nil)
        #expect(editor.warning(for: row.id)?.contains("needs an input — nothing feeds it") == true)
    }

    // MARK: - Consistency: the table must not claim what the runtime can't do

    @MainActor @Test func everyListedRowRunsWithNoInput() async throws {
        let catalog = try loadCatalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))

        for entry in RowSources.entries {
            let task = entry.task
            let settings = ownSettings(for: entry)
            let model = TaskCatalog.get(task)?.taskClass == .model
                ? FlowEditorModel.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable)
                : nil
            let row = Row(id: UUID(), task: task, model: model, settings: settings, refs: [])
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("rp1-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let workspace = FlowWorkspace(root: root)
            // A seeded Read row gets its sample, exactly as `add(task:)` copies it in.
            if let assets = SampleSeed.seedAssets(for: task),
               let sourceDir = Bundle.main.resourceURL {
                try? workspace.prepare(flowID: "rp1", sourceDir: sourceDir, bundledAssets: assets)
            }

            if task == "Web Search" {
                // The executor path would use a key from the Keychain and the network.
                #expect((try? WebSearchTool.resolveQuery(inputs: [], settings: FlowSettings(settings))) != nil,
                        "Web Search can't resolve its own query")
                continue
            }

            let executor = RealExecutor(
                workspace: workspace, flowID: "rp1", blobDirectory: root,
                makeModelStage: { _, _ in RowSourcesStubStage() },
                installedModelIDs: Set(catalog.map(\.id)),
                catalog: catalog)
            do {
                _ = try await executor.execute(path: "1", row: row, inputs: [],
                                               transcript: nil, context: nil, usedFlowContent: nil)
            } catch let error as FlowError {
                // Anything but "you gave me no input": a refused connection or a missing file
                // is the row's own value being unusable, not the row wanting an input.
                switch error {
                case .badInputCardinality, .missingInlineValue:
                    // A Read row's missing sample (the test host has no bundle copy) reports as
                    // a missing file, not here; reaching this case means the input path failed.
                    Issue.record("\(task) threw for want of an input: \(error)")
                default: break
                }
            } catch {
                // Non-FlowError (URLError, StageError…) means it got past the input path.
            }
        }
    }
}

/// A no-op stage: the consistency test exercises the executor's input path, not a model.
private nonisolated struct RowSourcesStubStage: PipelineStage {
    let id = "stub.rp1"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        progress(1.0)
        return input
    }
}
