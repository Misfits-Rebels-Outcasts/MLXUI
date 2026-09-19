import Testing
import Foundation
@testable import MLXUI

/// WA-4 — `mlxflow check` loads the curated registry unconditionally
/// (`catflow-mlx/src/catflow/cli/check.py:62`); the Swift app didn't (`AppState.swift`,
/// `FlowEditorModel.swift` both called `FlowValidator.checkFlow` with no registry), so a
/// display name that resolves fine at run time (`CatalogBridge`) still raised a false E104
/// in the validator, and leaked as E116 through a `uses:` call's nested check. Both live
/// call sites now pass `CuratedManifest.installedFlowRegistry()`.
///
/// This is the invariant that wiring should hold forever: every display name a bundled
/// Gallery or Workspace flow names resolves through the registry. A future flow naming an
/// unmanifested model fails **here**, in CI, instead of showing a false-or-true E104 to a
/// user who has no way to tell which it is.
///
/// **`knownUnresolvedDisplays`** — the WA-4 gallery audit (`RSI/journal`) found two display
/// names the curated manifest set (`Resources/CatFlow/models/`) has never covered: "Z-Image
/// Turbo (4-bit)" (60, 61, 66, 67, 69) and "Depth Pro" (62). Both raised E104 **before** this
/// wiring too (no registry at all is the same as one that can't resolve the name) — the
/// audit's own comparison confirmed wiring the registry removed 134 issues and added zero,
/// across every Gallery and Workspace flow. This is a pre-existing curated-manifest coverage
/// gap, not something WA-4 caused; it is filed as its own finding, unfixed, rather than
/// silenced here. Naming the two known gaps explicitly — instead of filtering every E104 —
/// keeps this test able to catch a **third**, genuinely new flow that names an unmanifested
/// model.
struct CatFlowRegistryInvariantTests {
    private static let knownUnresolvedDisplays: Set<String> = ["Z-Image Turbo (4-bit)", "Depth Pro"]

    @Test func everyBundledGalleryFlowResolvesItsModelsThroughTheRegistry() throws {
        let registry = CuratedManifest.installedFlowRegistry()
        var failures: [String] = []
        for meta in GalleryLoader.loadMetadata() {
            guard let text = try? GalleryLoader.rawCatText(flowID: meta.flowID),
                  let parsed = try? CatParser.parseForValidation(text) else {
                failures.append("\(meta.filename): could not load/parse")
                continue
            }
            let e104 = FlowValidator.checkFlow(parsed, registry: registry).filter { issue in
                issue.code == "E104" && !Self.knownUnresolvedDisplays.contains { issue.message.contains("\"\($0)\"") }
            }
            if !e104.isEmpty {
                failures.append("\(meta.filename): \(e104.map(\.message))")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test func everyBundledWorkspaceFlowResolvesItsModelsThroughTheRegistry() throws {
        let registry = CuratedManifest.installedFlowRegistry()
        var failures: [String] = []
        for meta in BundledWorkspaces.all {
            let base = FileManager.default.temporaryDirectory
                .appendingPathComponent("catflow-registry-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: base) }
            let root = base.appendingPathComponent("workspaces")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let ws = FlowWorkspace(root: root)
            try BundledWorkspaces.prepare(meta, workspace: ws)
            let listed = try #require(WorkspaceStore.scan(workspace: ws).first { $0.workspaceID == meta.id })
            for flow in listed.flows {
                let text = try String(contentsOf: flow.url, encoding: .utf8)
                guard let parsed = try? CatParser.parseForValidation(text) else {
                    failures.append("\(meta.id)/\(flow.title): could not parse")
                    continue
                }
                let issues = FlowValidator.checkFlow(parsed, registry: registry, workspace: ws, flowID: meta.id)
                let bad = issues.filter { $0.code == "E104" || $0.code == "E116" }
                if !bad.isEmpty {
                    failures.append("\(meta.id)/\(flow.title): \(bad.map(\.message))")
                }
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }
}
