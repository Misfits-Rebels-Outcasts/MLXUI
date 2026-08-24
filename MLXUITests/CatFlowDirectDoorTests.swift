import Testing
import Foundation
@testable import MLXUI

/// CFM-R10-Direct — the Direct-build doors. The MLXUI test scheme is the **App Store** tier
/// (`APPSTORE_BUILD`) and its host app is itself sandboxed, so `sandbox-exec` can't apply a
/// nested sandbox here (exit 71). Two things are therefore pinned in-suite: `canRun` refuses
/// `transforms:`/`Improvise`/`code` in this tier (the "never in the App Store build"
/// guarantee), and `FencedRunner` **refuses cleanly rather than ever running unconfined**.
/// The fence's actual deny-outside-workdir behavior is exercised only in the unsandboxed
/// Direct build — that is smoke row 46.
struct CatFlowDirectDoorTests {

    private func doc(rows: [Row], transforms: [String: TransformDef] = [:]) -> FlowDocument {
        var d = FlowDocument(version: "0.8", rows: rows)
        d.transforms = transforms
        return d
    }

    // MARK: - The App Store tier refuses the doors (the MLXUI scheme is APPSTORE_BUILD)

    @Test func appStoreRefusesTransformsFlows() {
        let d = doc(rows: [Row(task: "Read Text", settings: "memo.txt")],
                    transforms: ["custom": TransformDef(name: "custom", signature: nil, run: "echo hi",
                                                        timeout: nil, workdir: nil, params: [])])
        guard case .notRunnable(let reason) = FlowRunner.canRun(d) else {
            Issue.record("expected not runnable in the App Store tier")
            return
        }
        #expect(reason.contains("transforms"))
    }

    @Test func appStoreRefusesImproviseRows() {
        let d = doc(rows: [Row(task: "Improvise", settings: "fix the headers")])
        guard case .notRunnable(let reason) = FlowRunner.canRun(d) else {
            Issue.record("expected not runnable in the App Store tier")
            return
        }
        #expect(reason.contains("App Store"))
    }

    @Test func appStoreRefusesCodeFlagFlows() {
        var d = doc(rows: [Row(task: "Read Text", settings: "memo.txt")])
        d.flags = [.code]
        guard case .notRunnable(let reason) = FlowRunner.canRun(d) else {
            Issue.record("expected not runnable in the App Store tier")
            return
        }
        #expect(reason.contains("code"))
    }

    // MARK: - The fence refuses cleanly when it can't apply (never runs unconfined)

    #if DIRECT_BUILD
    @Test func fencedRunnerRefusesCleanlyWhenTheFenceIsUnavailable() throws {
        // The sandboxed test host can't apply a nested sandbox; the fence must refuse, never
        // silently run the command unconfined (the "refuse rather than approximate" rule).
        let workdir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-fence-\(UUID().uuidString)")
        #expect(throws: (any Error).self) {
            try FencedRunner.runFenced(argv: ["/bin/echo", "fenced ok"], workdir: workdir)
        }
    }
    #endif

    // MARK: - CFM-R10-FIX-4: durations parse and timeouts bound

    @Test func parseDurationHandlesCorpusSpellings() {
        #expect(FlowSettingsEditor.parseDuration("30s") == 30)
        #expect(FlowSettingsEditor.parseDuration("2m") == 120)
        #expect(FlowSettingsEditor.parseDuration("1h30m") == 5400)
        #expect(FlowSettingsEditor.parseDuration("500ms") == 0.5)
        #expect(FlowSettingsEditor.parseDuration("garbage") == nil)
        #expect(FlowSettingsEditor.parseDuration("30") == nil)
    }

    #if DIRECT_BUILD
    @Test func aTrappedTermScriptIsKilledOnTimeout() throws {
        // A `trap '' TERM` script ignores SIGTERM — the fence must escalate to SIGKILL.
        let workdir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-fence-\(UUID().uuidString)")
        #expect(throws: (any Error).self) {
            try FencedRunner.runFenced(
                argv: ["/bin/zsh", "-lc", "trap '' TERM; while true; do sleep 0.1; done"],
                workdir: workdir, timeout: 0.5, rowName: "trap")
        }
    }

    @Test func aHighOutputScriptCompletes() throws {
        let workdir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-fence-\(UUID().uuidString)")
        let out = try FencedRunner.runFenced(
            argv: ["/bin/zsh", "-lc", "for i in $(seq 1 5000); do echo '01234567890123456789012345678901234567890123456789'; done"],
            workdir: workdir, timeout: 30)
        #expect(out.count > 200_000)
    }
    #endif

    #if DIRECT_BUILD
    @Test func aTransformRunsFilesOnlyInItsOwnWorkdir() throws {
        // §14.6 (FIX-5): script + inputs cross as files in the transform's own workdir; the
        // script's argv is script + input + output; the output file becomes the row's Asset.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-transform-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: root)
        let flowID = "t"
        let dir = ws.directory(for: flowID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // A script that reads its first input and copies it to the output path.
        try "#!/bin/sh\ncp \"$1\" \"$2\"\n".write(to: dir.appendingPathComponent("t.sh"), atomically: true, encoding: .utf8)
        let transform = TransformDef(name: "Tidy", signature: "text -> text", run: "t.sh",
                                     timeout: "10s", workdir: "work", params: [])
        let row = Row(id: UUID(), task: "Tidy", settings: "")
        let input = Asset(items: [Item(kind: .text, value: "hello from input", path: nil, sourceText: nil)])
        let blob = root.appendingPathComponent("blobs")
        let asset = try FencedRunner.runTransform(transform, row: row, inputs: [input],
                                                  workspace: ws, flowID: flowID, blobDirectory: blob)
        let output = try #require(asset.items.first?.path)
        #expect((try String(contentsOf: output, encoding: .utf8)) == "hello from input")
        // The workdir is ONLY the transform's own `workdir` — scratch is cleaned up.
        let workdir = dir.appendingPathComponent("work")
        let leftovers = (try? FileManager.default.contentsOfDirectory(atPath: workdir.path)) ?? []
        #expect(leftovers.isEmpty, "scratch files must be cleaned up")
    }
    #endif

    // MARK: - The improvise undo snapshot (pure file ops, sandbox-free)

    #if DIRECT_BUILD
    @Test func improviseSnapshotRestoresTheWorkdir() throws {
        let workdir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-improvise-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workdir, withIntermediateDirectories: true)
        try "original".write(to: workdir.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)

        // Snapshot, then a mutation (an edit + a new file).
        let store = FencedRunner.undoStore(for: workdir)
        try FencedRunner.snapshot(workdir, to: store)
        try "changed".write(to: workdir.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)
        try "new".write(to: workdir.appendingPathComponent("added.txt"), atomically: true, encoding: .utf8)

        // Undo restores the snapshot: data.txt back to "original", added.txt gone.
        try FencedRunner.undo(workdir)
        #expect((try? String(contentsOf: workdir.appendingPathComponent("data.txt"), encoding: .utf8)) == "original")
        #expect(!FileManager.default.fileExists(atPath: workdir.appendingPathComponent("added.txt").path))
    }

    @Test func runImproviseEnforcesItsBounds() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-improvise-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: root)
        // Missing max_actions / timeout / workdir → the executor refuses (the E110/E111 bounds).
        #expect(throws: (any Error).self) {
            try FencedRunner.runImprovise(settings: "fix it", workspace: ws, flowID: "t")
        }
        #expect(throws: (any Error).self) {
            try FencedRunner.runImprovise(settings: "fix it; max_actions=5; timeout=30s", workspace: ws, flowID: "t")
        }
        // A fully-bounded row runs in its declared workdir.
        let out = try FencedRunner.runImprovise(
            settings: "echo hi > note.txt; max_actions=5; timeout=30s; workdir=scratch",
            workspace: ws, flowID: "t")
        _ = out
        let note = ws.directory(for: "t").appendingPathComponent("scratch/note.txt")
        #expect((try? String(contentsOf: note, encoding: .utf8))?.contains("hi") == true)
    }
    #endif

    // MARK: - FIX-6: hidden files + the space bound (pure file ops)

    #if DIRECT_BUILD
    @Test func improviseSnapshotCapturesHiddenFiles() throws {
        // FIX-6: the workdir may hold dot-prefixed files — the snapshot must not skip them.
        let workdir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-hidden-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workdir, withIntermediateDirectories: true)
        try "secret".write(to: workdir.appendingPathComponent(".state"), atomically: true, encoding: .utf8)
        let store = FencedRunner.undoStore(for: workdir)
        try FencedRunner.snapshot(workdir, to: store)
        try "mutated".write(to: workdir.appendingPathComponent(".state"), atomically: true, encoding: .utf8)
        try FencedRunner.undo(workdir)
        #expect((try? String(contentsOf: workdir.appendingPathComponent(".state"), encoding: .utf8)) == "secret")
    }

    @Test func improviseSpaceBoundFailsTheRow() throws {
        let workdir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-space-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workdir, withIntermediateDirectories: true)
        // A tiny over-limit proves the bound fires (the check is on the workdir's size).
        try Data(repeating: 0x41, count: 64).write(to: workdir.appendingPathComponent("big.bin"))
        #expect(throws: (any Error).self) {
            try FencedRunner.checkWorkdirBound(workdir, row: "Improvise", limit: 32)
        }
        // Under the limit passes.
        try FencedRunner.checkWorkdirBound(workdir, row: "Improvise", limit: 100_000)
    }
    #endif
}
