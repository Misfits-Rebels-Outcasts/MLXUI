import Testing
import Foundation
@testable import MLXUI

/// WA-6 — the regression bed the six false-yellow-row reports never had
/// (`RSI/DelegateWorkspaceAdvisoryBacklog.md`). Opens **every** flow in **both** bundled
/// workspaces the way `WorkspaceListView.open` does — through `BundledWorkspaces.prepare`
/// into `workspaces/<id>/`, never `Resources/` directly (the flat `<id>--Name.cat` names
/// there are not what the editor opens) — and asserts `FlowEditorModel` shows no yellow
/// row and no structural issue anywhere. These six rows survived ~109 test files precisely
/// because nothing ever opened a shipped workspace flow in the editor.
struct CatFlowWorkspaceAdvisoryTests {

    /// The bundled catalog + the registry's own claim answer (CFM-R14-2) — so a model row's
    /// "needs a model" check has a real verdict instead of the vacuous empty-catalog one.
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

    private func flattened(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + flattened($0.children) }
    }

    @Test(arguments: BundledWorkspaces.all.map(\.id))
    @MainActor func everyFlowInABundledWorkspaceOpensCleanInTheEditor(workspaceID: String) throws {
        let meta = try #require(BundledWorkspaces.meta(id: workspaceID))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-wa6-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let root = base.appendingPathComponent("workspaces")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: root)
        try BundledWorkspaces.prepare(meta, workspace: ws)
        let listed = try #require(WorkspaceStore.scan(workspace: ws).first { $0.workspaceID == workspaceID })
        #expect(!listed.flows.isEmpty)
        let (catalog, claimable) = try loadedCatalogAndClaimable()

        for flow in listed.flows {
            let text = try String(contentsOf: flow.url, encoding: .utf8)
            let doc = try CatParser.parse(text)
            // Matches `WorkspaceListView.open`: a workspace flow's `flowID` **is** the
            // workspace id, `savedURL` seeded from the real file
            // (`FlowEditorModel.seedSavedURL`'s production case), `isSharedWorkspaceFolder`
            // true (a shared `workspaces/<id>/` folder, per the doc comment on that flag).
            let model = FlowEditorModel(name: flow.title, flowID: workspaceID, document: doc,
                                        workspace: ws, savedText: text,
                                        isSharedWorkspaceFolder: true, savedURL: flow.url)
            model.modelCatalog = catalog
            model.claimableModelIDs = claimable

            let structural = model.structuralIssues()
            #expect(structural.isEmpty, "\(flow.title): \(structural.map(\.message))")
            for r in flattened(model.document.rows) {
                let warning = model.warning(for: r.id)
                let label = r.task ?? r.blockKind?.rawValue ?? "?"
                #expect(warning == nil, "\(flow.title) row \(model.displayNumber(of: r.id) ?? "?") (\(label)): \(warning ?? "")")
            }
        }
    }
}
