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
        // `CachingExecutor` through `GalleryLoader.rawCatText(flowID:)`. CFM-R18-4 flipped
        // line one `catflow 0.8` -> `mlxflow 0.8`, so the hash of every bundled flow's
        // text moved with it; these are the re-derived values for the `mlxflow`-headed files.
        #expect(FlowScope.plain("01-SpokenSummary").runSeed == 3_161_680_488)
        #expect(FlowScope.plain("16-IngestFolder").runSeed == 2_432_159_665)
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
        // Stored, not re-derived: int.from_bytes(SHA-256(text)[:4], "big") of the string above.
        // A workspace scope seeds from the text it is handed, never from a bundled-file lookup
        // keyed on `identity` — `identity` is "w", which has no bundled `.cat`, so an identity
        // lookup would give the `?? 0` fallback the second expectation rules out.
        #expect(scope.runSeed == 3_747_261_887)
        #expect(scope.runSeed != 0)
    }

    // MARK: - canRun sees the scope

    @Test func canRunScopeOverloadForwardsLocationNotIdentity() throws {
        // A flow whose `uses:` target only resolves against `locationID` — so a wrong
        // `identity` and a wrong `locationID` give *different* verdicts, and the overload's
        // choice is observable (the old test used a `uses:`-free doc where it wasn't).
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-scope-\(UUID().uuidString)")
        let root = base.appendingPathComponent("workspaces")
        let dir = root.appendingPathComponent("kb", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let caller = "catflow 0.8\n1. Helper\n\nuses:\n  Helper = ./Helper.cat\n"
        try caller.write(to: dir.appendingPathComponent("Caller.cat"), atomically: true, encoding: .utf8)
        try "catflow 0.8\n1. Read Text a.txt\n2. Save Text b.md\n"
            .write(to: dir.appendingPathComponent("Helper.cat"), atomically: true, encoding: .utf8)
        let ws = FlowWorkspace(root: root)
        let doc = try CatParser.parse(caller)

        // `identity` is junk, `locationID` is the real workspace → `Helper.cat` resolves and
        // the flow is runnable.
        let viaScope = FlowRunner.canRun(
            doc, scope: FlowScope(identity: "unrelated", workspace: ws, locationID: "kb", flowText: caller))
        #expect(viaScope == .runnable)

        // Identical to the explicit form keyed on the same location (the scope overload's only
        // job is to build that `usesGraph` from `locationID` + `workspace` and forward it).
        let viaExplicit = FlowRunner.canRun(
            doc, flowID: "kb", workspace: ws,
            usesGraph: UsesResolver.resolve(doc, workspace: ws, flowID: "kb"))
        #expect(viaScope == viaExplicit)

        // Swap them: a real id in `identity`, junk in `locationID` → `Helper.cat` does not
        // resolve, so the `Helper` row is refused. Different verdict ⇒ the overload keys on
        // `locationID`, not `identity` (the old test's `uses:`-free doc couldn't see this).
        let viaWrongLocation = FlowRunner.canRun(
            doc, scope: FlowScope(identity: "kb", workspace: ws, locationID: "unrelated", flowText: caller))
        #expect(viaWrongLocation != viaScope)
        if case .runnable = viaWrongLocation {
            Issue.record("a used flow that can't resolve must not report .runnable")
        }
    }
}
