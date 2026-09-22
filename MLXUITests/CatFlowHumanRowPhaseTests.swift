import Testing
import Foundation
@testable import MLXUI

/// HR-6 (`RSI/DelegateHumanRowBacklog.md`) — the regression bed for Phase HR (HR-1…HR-4):
/// `Human Input` as a first-class way to start a flow. Nothing in either repo ships a row-1
/// human row (checked: 73 gallery flows, 4 workspace flows, `catflow-mlx/conformance/` — every
/// existing `Human Input`/`Ask Human` sits at row 2 or later), so the row-1/picker/round-trip
/// facts below use synthetic documents deliberately, per the backlog's own "Rules for the
/// implementer." The shipped half — `74-WebPageSummary`, the one gallery flow this phase adds —
/// is opened and run for real, the same way `CatFlowWorkspaceAdvisoryTests` (WA-6) opens the
/// bundled workspace flows rather than only asserting against synthetics.
struct CatFlowHumanRowPhaseTests {

    // MARK: - Helpers

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-hr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "HR", document: FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String?, settings: String? = nil) -> Row {
        Row(id: UUID(), task: task, settings: settings)
    }

    @MainActor
    private func loadedCatalogAndClaimable() throws -> (catalog: [ModelEntry], claimable: Set<String>) {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
            .domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        return (catalog, claimable)
    }

    // MARK: - HR-1: row 1 `Human Input` under each waiting policy

    @Test func rowOneHumanInputUnderEachWaitingPolicy() throws {
        // Clean — wait=forever, the row sources itself from a person.
        let waitForever = row("Human Input", settings: "\"Enter URL:\"; wait=forever")
        #expect(try editor([waitForever]).warning(for: waitForever.id) == nil)

        // New sentence — a complete timeout=/default= pair has nothing to fall back on at row 1.
        let timeoutComplete = row("Human Input", settings: "\"Enter URL:\"; timeout=30s; default=unchanged")
        #expect(try editor([timeoutComplete]).warning(for: timeoutComplete.id) ==
                "Row 1 has nothing to fall back on — if nobody answers, this row produces nothing.")

        // Neither sentence — an incomplete or absent policy is a genuine E501, not a row-1 read.
        let noPolicy = row("Human Input", settings: "\"Enter URL:\"")
        let noPolicyWarning = try #require(try editor([noPolicy]).warning(for: noPolicy.id))
        #expect(noPolicyWarning.contains("doesn't say what happens if nobody answers"))
    }

    // MARK: - HR-2: the picker offers `Human Input`, never `Ask Human`

    @Test func humanInputJoinsTheStepPickerStartingNodesAskHumanDoesNot() throws {
        let starts = Set(FlowEditorModel.startingNodes().map(\.name))
        #expect(starts.contains("Human Input"))
        #expect(!starts.contains("Ask Human"))

        let model = try editor([])
        model.add(task: "Human Input")
        let r = try #require(model.document.rows.first)
        #expect(r.task == "Human Input")
        #expect(model.warning(for: r.id) == nil)
    }

    // MARK: - HR-3: the seeded question survives a save/reparse round trip

    @Test func theSeededQuestionSurvivesASaveAndReparseRoundTrip() throws {
        let model = try editor([])
        model.add(task: "Human Input")
        let text = model.catText

        let reparsed = try CatParser.parse(text)
        let r = try #require(reparsed.rows.first)
        #expect(r.task == "Human Input")
        #expect(FlowInterpreter.rowPromptText(r) == "What should I use?")

        // A second round trip is a true no-op — the seeded row is canonical, not just parseable.
        #expect(CatSerializer.serialize(reparsed) == text)
    }

    // MARK: - HR-4/HR-6: the shipped example opens clean and its prefill reaches ParkedInfo

    @MainActor
    @Test func theShippedWebPageSummaryFlowOpensCleanInTheEditor() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "74-WebPageSummary")
        #expect(doc.rows.map(\.task) == ["Template", "Human Input", "Web Fetch", "Summarize", "Save Text"])

        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let model = FlowEditorModel(name: "74-WebPageSummary", document: doc,
                                    workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory
                                        .appendingPathComponent("catflow-hr-\(UUID().uuidString)")))
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable

        for r in doc.rows {
            #expect(model.warning(for: r.id) == nil, "\(r.task ?? "?") shows a warning: \(model.warning(for: r.id) ?? "")")
        }
    }

    @Test func theShippedFlowsHumanInputRowPrefillsFromTheTemplateRowAbove() async throws {
        let doc = try GalleryLoader.loadDocument(flowID: "74-WebPageSummary")
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-hr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        // The real executor: `Template`'s literal pattern passthrough is what makes row 1's
        // output the exact default-URL text HR-4's prefill depends on.
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: base), flowID: "74-WebPageSummary",
            blobDirectory: base,
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [], catalog: [])
        let session = FlowRunSession()
        let context = FlowRunner.RunContext(flowID: "74-WebPageSummary", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: executor)
        let runner = FlowRunner()
        session.prepareInstall(FlowPreflight.run(doc, catalog: [], installedModelIDs: [],
                                                 totalRAMGB: 128, claimableModelIDs: []),
                               doc: doc)

        session.start(doc: doc, runner: runner, context: context)
        var parkedInfo: FlowRunSession.ParkedInfo?
        for _ in 0..<40 {
            if let parked = session.parked { parkedInfo = parked; break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let parked = try #require(parkedInfo)
        #expect(parked.prompt == "Enter a URL to summarize:")
        #expect(parked.defaultText == "https://news.ycombinator.com")
        session.cancel()
    }

    // MARK: - WR-2 (`RSI/DelegateWorkspaceRunBacklog.md`): "Stop the run" stops the run

    /// Root cause 2: by the time a row parks, the run task has already drained and
    /// `apply(.parked)` already set `isRunning = false` — `cancel()`'s three statements were
    /// all already in their target state, so the Stop button was inert by construction.
    /// `cancel()` now clears `parked` too, which is the only thing that changed.
    @Test func stoppingAParkedRunClearsParkedAndLeavesEarnedDotsIntact() async throws {
        let doc = try GalleryLoader.loadDocument(flowID: "74-WebPageSummary")
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-hr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: base), flowID: "74-WebPageSummary",
            blobDirectory: base,
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [], catalog: [])
        let session = FlowRunSession()
        let context = FlowRunner.RunContext(flowID: "74-WebPageSummary", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: executor)
        let runner = FlowRunner()
        session.prepareInstall(FlowPreflight.run(doc, catalog: [], installedModelIDs: [],
                                                 totalRAMGB: 128, claimableModelIDs: []),
                               doc: doc)

        session.start(doc: doc, runner: runner, context: context)
        var parkedInfo: FlowRunSession.ParkedInfo?
        for _ in 0..<40 {
            if let parked = session.parked { parkedInfo = parked; break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let parked = try #require(parkedInfo)
        let templateRow = try #require(doc.rows.first)
        // Row 1 (Template) finished before row 2 parked — the dot it earned.
        #expect(session.status(for: templateRow.id) == .succeeded)

        session.cancel()

        #expect(session.parked == nil)
        #expect(session.isRunning == false)
        // "dots aren't reset on cancel" — unchanged by this fix, confirmed still true.
        #expect(session.status(for: templateRow.id) == .succeeded)

        // A `timeout=` fallback that fires after Stop must not restart the flow: the guard
        // compares `self.parked` (now nil) against the captured `parked`, so it no-ops.
        session.fallbackTimeout(for: parked, doc: doc, runner: runner, context: context)
        #expect(session.isRunning == false)
        #expect(session.parked == nil)
    }

    // MARK: - HR-5: "Give this row a default value…" (Q3 ruled (c), 2026-09-20)

    @Test func addDefaultValueRowInsertsAnEmptyTemplateAboveAndSelectsIt() throws {
        let humanInput = row("Human Input", settings: "\"Enter URL:\"; timeout=30s; default=unchanged")
        let model = try editor([humanInput])
        model.addDefaultValueRow(above: humanInput.id)

        #expect(model.document.rows.count == 2)
        let template = try #require(model.document.rows.first)
        #expect(template.task == "Template")
        #expect(template.settings == "\"\"")
        #expect(model.document.rows[1].id == humanInput.id)
        // The target row's own settings are untouched — HR-5 doesn't add default=unchanged
        // itself, only the row above it.
        #expect(model.document.rows[1].settings == humanInput.settings)
        #expect(model.selectedRowID == template.id)
    }

    @Test func addDefaultValueRowIsByteIdenticalToHandTypingTheTwoRows() throws {
        let humanInput = row("Human Input", settings: "\"Enter URL:\"; timeout=30s; default=unchanged")
        let model = try editor([humanInput])
        model.addDefaultValueRow(above: humanInput.id)

        let handTyped = """
        mlxflow 0.8
        1. Template      ""
        2. Human Input   "Enter URL:"; timeout=30s; default=unchanged
        """
        let handTypedDoc = try CatParser.parse(handTyped)
        #expect(CatSerializer.serialize(handTypedDoc) == model.catText)
    }

    @Test func addDefaultValueRowWorksForABlockChild() throws {
        let humanInput = row("Human Input", settings: "\"Enter URL:\"; wait=forever")
        let block = Row(id: UUID(), task: nil, blockKind: .each, blockName: "each_group",
                        children: [humanInput])
        let model = try editor([block])
        model.addDefaultValueRow(above: humanInput.id)

        let children = try #require(model.document.rows.first?.children)
        #expect(children.count == 2)
        #expect(children[0].task == "Template")
        #expect(children[1].id == humanInput.id)
        #expect(model.selectedRowID == children[0].id)
    }
}
