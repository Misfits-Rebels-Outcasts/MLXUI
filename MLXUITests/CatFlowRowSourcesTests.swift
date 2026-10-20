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

    /// The settings the row arrives with from + Add step (RP-2): a `SampleSeed` sample or
    /// `FlowEditorModel.defaultSettings`. Rows the table offers no default for (the web rows,
    /// and Read Video / Index / Context) get a hand-written value: the consistency test is
    /// about the executor's input path, so they need *a* value, not the editor's.
    private func ownSettings(for entry: RowSources.Entry) -> String? {
        if let seed = SampleSeed.seedValue(for: entry.task) { return seed }
        if let dflt = FlowEditorModel.defaultSettings(forTask: entry.task) { return dflt }
        switch entry.task {
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
                  "Read Video", "Read Index", "Read Context", "Store Query", "Store Read",
                  "Generate", "Decide"] {
            #expect(!starts.contains(t), "\(t)")
        }
    }

    // MARK: - The warning

    @MainActor private func fixture(_ task: String, settings: String?, empty: Bool = false) throws -> (FlowEditorModel, Row) {
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
            document: FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: empty ? [] : [row]),
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        editor.modelCatalog = catalog
        editor.claimableModelIDs = claimable
        return (editor, row)
    }

    @Test @MainActor func aSourceRowAtRowOneHasNoWarning() throws {
        for (task, settings) in [("Generate Sound", "\"calm\""),
                                 ("Web Fetch", "url=\"https://example.com\""),
                                 ("Read Index", "kb.index"),
                                 // Owner ruling 2026-10-19: reads a file named in its own settings.
                                 ("Store Query", "orders.db; table=orders"),
                                 ("Store Read", "orders.db; table=orders")] {
            let (editor, row) = try fixture(task, settings: settings)
            #expect(editor.warning(for: row.id) == nil, "\(task)")
        }
    }

    @Test @MainActor func aSummarizeRowAtRowOneStillSaysNothingFeedsIt() throws {
        let (editor, row) = try fixture("Summarize", settings: nil)
        #expect(editor.warning(for: row.id)?.contains("needs an input — nothing feeds it") == true)
    }

    // MARK: - RP-2: defaults on add

    private static let defaults: [(task: String, settings: String)] = [
        ("Generate Image", "\"a cozy reading nook by a rainy window, warm light\""),
        ("Generate Sound", "\"calm ambient background music, soft piano, no drums\""),
        ("Generate Video", "\"waves rolling onto a sandy beach at sunset, slow camera pan\""),
        ("Template", "\"Type the text this row starts with.\""),
        ("Range", "1..3"),
    ]

    @Test @MainActor func addingEachTableTaskSeedsExactlyTheTablesDefaultAndUndoRemovesBoth() throws {
        for (task, expected) in Self.defaults {
            let (editor, _) = try fixture(task, settings: nil, empty: true)
            editor.add(task: task)
            #expect(editor.document.rows.count == 1)
            #expect(editor.document.rows.first?.settings == expected, "\(task)")
            editor.undo()
            #expect(editor.document.rows.isEmpty, "\(task) undo")
        }
        // Web rows get no default URL.
        for web in ["Web Fetch", "HTTP Get", "Fetch Feed", "Download File", "Web Search"] {
            #expect(FlowEditorModel.defaultSettings(forTask: web) == nil)
        }
    }

    @Test @MainActor func addingInsideABlockSeedsTheSameDefault() throws {
        let (editor, _) = try fixture("Range", settings: nil, empty: true)
        editor.insertBlock(kind: .each, name: "x", after: nil)
        let block = try #require(editor.document.rows.first)
        editor.addChild(task: "Range", into: block.id)
        let added = try #require(editor.document.rows.first?.children.last)
        #expect(added.task == "Range" && added.settings == "1..3")
    }

    @Test @MainActor func everyDefaultRowIsWarningFreeAtRowOne() throws {
        for (task, _) in Self.defaults {
            let (editor, _) = try fixture(task, settings: nil, empty: true)
            editor.add(task: task)
            let row = try #require(editor.document.rows.first)
            #expect(editor.warning(for: row.id) == nil, "\(task)")
            #expect(editor.saveBlockReason == nil, "\(task) blocks Save")
        }
    }

    // MARK: - RP-2: warnings that say what is missing

    @Test @MainActor func clearingAGeneratorsPromptShowsThePromptSentenceAndRefillingClearsIt() throws {
        let (editor, row) = try fixture("Generate Sound", settings: "\"calm\"")
        editor.setInstruction(nil, for: row.id)
        #expect(editor.warning(for: row.id)
                == "Row 1 needs a prompt — type one in the Step tab, or connect a row that makes text.")
        editor.setInstruction("calm", for: row.id)
        #expect(editor.warning(for: row.id) == nil)
    }

    @Test @MainActor func eachValueKindHasItsOwnSentence() {
        func sentence(_ task: String, _ settings: String?) -> String? {
            RowSources.missingValueSentence(for: Row(task: task, settings: settings), path: "3")
        }
        #expect(sentence("Read Text", nil)
                == "Row 3 needs a file — choose one in the Step tab, or connect a row that makes one.")
        #expect(sentence("Template", "\"\"") == "Row 3 needs text — type it in the Step tab.")
        #expect(sentence("Range", nil) == "Row 3 needs a range — e.g. 1..3.")
        #expect(sentence("Web Fetch", nil) == "Row 3 needs a URL — add url= in the Step tab.")
        #expect(sentence("Web Search", nil) == "Row 3 needs a search — add query= in the Step tab.")
        #expect(sentence("Human Input", nil) == nil)
        #expect(sentence("Generate Image", "\"x\"") == nil)
    }

    @Test @MainActor func aGeneratorFedByAnUpstreamTextRowWithNoPromptShowsNoWarning() throws {
        let (editor, _) = try fixture("Template", settings: "\"a lighthouse\"")
        editor.add(task: "Generate Image")          // pins its seeded model, as + Add step does
        let gen = try #require(editor.document.rows.last)
        // RP-5 (owner 2026-10-20): a Generate row now arrives with "Use my prompt" ticked, i.e. a
        // blank line above it; this test is about a *fed* generator, so untick it first.
        editor.setUsesOwnPrompt(false, for: gen.id)
        editor.setInstruction(nil, for: gen.id)
        let cleared = try #require(editor.row(withID: gen.id))
        #expect(RowSources.missingValueSentence(for: cleared, path: "2") != nil)   // promptless, yet fed
        #expect(editor.warning(for: gen.id) == nil, "\(editor.warning(for: gen.id) ?? "")")
        #expect(FlowInputAdvisory.advisory(for: editor.document, scope: scope()) == nil)
    }

    // MARK: - RP-2: the pre-run advisory

    private func scope() -> FlowScope {
        FlowScope(identity: "rp2", workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
                  locationID: "rp2")
    }

    @Test func theAdvisoryNamesAPromptlessGeneratorAndOnlyTheFirst() {
        let a = Row(task: "Generate Image", settings: nil)
        let b = Row(task: "Generate Sound", settings: nil)
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [a, b])
        let advisory = FlowInputAdvisory.advisory(for: doc, scope: scope())
        #expect(advisory?.task == "Generate Image")
        #expect(advisory?.reason
                == "Row 1 needs a prompt — type one in the Step tab, or connect a row that makes text.")
    }

    @Test func theAdvisoryStaysNilWhenTheGeneratorHasAPromptOrIsFed() {
        let withPrompt = FlowDocument(version: "0.8", headerKeyword: "mlxflow",
                                      rows: [Row(task: "Generate Image", settings: "\"a cat\"")])
        #expect(FlowInputAdvisory.advisory(for: withPrompt, scope: scope()) == nil)
        let fed = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [
            Row(task: "Template", settings: "\"a cat\""), Row(task: "Generate Image", settings: nil)])
        #expect(FlowInputAdvisory.advisory(for: fed, scope: scope()) == nil)
    }

    // MARK: - RP-1-FIX-1: Range takes no upstream input

    @Test func onlyRangeTakesNoUpstreamInputToday() {
        #expect(RowSources.entries.filter(\.takesNoUpstreamInput).map(\.task) == ["Range"])
    }

    @Test func theAfterARowPickerNeverOffersRangeButTheOtherListsDo() {
        let outputs: [Shape?] = [.single(.text), .single(.audio), .single(.image),
                                 .listOf(.text), .sameAsInput, .anyKind, nil]
        for output in outputs {
            let names = FlowEditorModel.stepsAccepting(output).map(\.name)
            #expect(!names.contains("Range"), "\(String(describing: output))")
            #expect(!FlowEditorModel.stepsAccepting(output, includeAdvanced: false).isEmpty)
        }
        #expect(FlowEditorModel.startingNodes().map(\.name).contains("Range"))
        #expect(FlowEditorModel.stepsAccepting(.single(.text), includeAdvanced: true).map(\.name).contains("Range"))
    }

    @Test @MainActor func rangeAddedBelowRowsStartsANewChainAndRoundTrips() throws {
        let (editor, _) = try fixture("Template", settings: "\"hello\"")
        editor.add(task: "Range")
        #expect(editor.document.rows.count == 2)
        let added = try #require(editor.document.rows.last)
        #expect(added.task == "Range" && added.settings == "1..3" && added.chainBreak)

        let text = CatSerializer.serialize(editor.document)
        let lines = text.components(separatedBy: "\n")
        let at = try #require(lines.firstIndex { $0.contains("Range") })
        #expect(lines[at - 1].isEmpty, "no blank line above Range:\n\(text)")

        let reparsed = try CatParser.parse(text)
        #expect(reparsed.rows.last?.task == "Range")
        #expect(reparsed.rows.last?.chainBreak == true)
        #expect(CatSerializer.serialize(reparsed) == text)

        // One undo removes the row and its break together.
        editor.undo()
        #expect(editor.document.rows.count == 1)
    }

    @Test @MainActor func rangeAddedFirstOrOnAnEmptyFlowHasNoBreak() throws {
        let (editor, _) = try fixture("Range", settings: nil, empty: true)
        editor.add(task: "Range")
        #expect(editor.document.rows.first?.chainBreak == false)
        #expect(!CatSerializer.serialize(editor.document).contains("\n\n1."))
    }

    @Test @MainActor func galleryFlow66KeepsRangeAtRowOneUnchanged() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "66-SeedSweep")
        let first = try #require(doc.rows.first)
        #expect(first.task == "Range" && first.settings == "3..6" && !first.chainBreak)
        let editor = FlowEditorModel(name: "g66", document: doc,
                                     workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        #expect(editor.warning(for: first.id) == nil)
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
