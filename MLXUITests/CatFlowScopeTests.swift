import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R17-1: `FlowScope` splits `flowID`'s two jobs — **identity** (run seed,
/// `On Flow` matching, run records) and **location** (`workspace` + `locationID`, the key
/// `FlowWorkspace.resolve` and every tool land on). The gate is that a plain `flows/` flow
/// is byte-identical to pre-R17: same resolved directory, same run seed. A workspace flow
/// resolves into the shared workspace directory and derives its seed from its own text.
struct CatFlowScopeTests {

    // MARK: - A plain flow: byte-identical to pre-R17

    @Test func plainScopeIdentityAndLocationAreTheSameID() {
        let scope = FlowScope.plain("01-SpokenSummary")
        #expect(scope.identity == "01-SpokenSummary")
        #expect(scope.locationID == "01-SpokenSummary")
    }

    @Test func plainScopeResolvesIntoFlowsDirectory() {
        let dir = FlowScope.plain("01-SpokenSummary").directory
        // Exactly `<flows>/01-SpokenSummary`, the same URL the pre-R17 factory built via
        // `FlowWorkspace(root: ModelStore.shared.flowsDirectory).directory(for: flowID)`.
        #expect(dir == ModelStore.shared.flowsDirectory
            .appendingPathComponent("01-SpokenSummary", isDirectory: true))
        #expect(dir == FlowWorkspace.shared.directory(for: "01-SpokenSummary"))
        #expect(dir.deletingLastPathComponent().lastPathComponent == "flows")
    }

    @Test func plainScopeRunSeedMatchesStoredGolden() {
        // Stored, not re-derived: int.from_bytes(SHA-256(cat text)[:4], "big") of the
        // bundled `.cat` files — the number the pre-R17 `cachingContext` fed
        // `CachingExecutor` through `GalleryLoader.rawCatText(flowID:)`.
        #expect(FlowScope.plain("01-SpokenSummary").runSeed == 2_999_492_883)
        #expect(FlowScope.plain("16-IngestFolder").runSeed == 3_205_850_175)
    }

    @Test func plainScopeUnknownFlowKeepsZeroFallback() {
        // A flow whose text isn't in the bundle keeps the pre-R17 `?? 0` fallback exactly.
        #expect(FlowScope.plain("no-such-flow-\(UUID().uuidString)").runSeed == 0)
    }

    // MARK: - A workspace flow: identity ≠ location

    @Test func workspaceScopeResolvesIntoTheSharedWorkspaceDirectory() {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ws-\(UUID().uuidString)", isDirectory: true)
        let ws = FlowWorkspace(root: root)
        let ask = FlowScope(identity: "AskYourDocs", workspace: ws, locationID: "docs",
                            flowText: "catflow 0.8\n1. Read Index library.index\n")
        let index = FlowScope(identity: "IngestFolder", workspace: ws, locationID: "docs",
                              flowText: "catflow 0.8\n6. Store Index library.index\n")
        // Identity stays each flow's own id …
        #expect(ask.identity == "AskYourDocs")
        #expect(index.identity == "IngestFolder")
        // … while both resolve into the one shared directory, so a relative path like
        // `library.index` names the same file for both — the whole point of a workspace.
        #expect(ask.directory == root.appendingPathComponent("docs", isDirectory: true))
        #expect(index.directory == ask.directory)
    }

    @Test func workspaceScopeSeedComesFromItsOwnText() {
        let text = "catflow 0.8\n1. Read Audio other.m4a\n"
        let scope = FlowScope(identity: "w", workspace: FlowWorkspace(root: URL(fileURLWithPath: "/tmp")),
                              locationID: "shared", flowText: text)
        #expect(scope.runSeed == FlowSeed.runSeed(for: text))
        #expect(scope.runSeed != 0)   // did not fall through to an absent bundled file
    }

    // MARK: - canRun sees the scope

    @Test func canRunScopeOverloadForwardsLocationNotIdentity() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "16-IngestFolder")
        let ws = FlowWorkspace(root: ModelStore.shared.flowsDirectory)
        // The scope overload must key the E117 `uses:` resolution on `locationID` +
        // `workspace`, never on `identity` — identical verdict to the explicit form.
        let viaScope = FlowRunner.canRun(
            doc, scope: FlowScope(identity: "unrelated", workspace: ws, locationID: "16-IngestFolder"))
        let viaExplicit = FlowRunner.canRun(doc, flowID: "16-IngestFolder", workspace: ws)
        #expect(viaScope == viaExplicit)
    }
}
