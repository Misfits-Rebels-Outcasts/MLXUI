import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// CW-1 (and its CW-A fix) — the conversation's folder scope: the interim rule (C6), the two
/// `@TaskLocal` paths that carry it into the file tools, and the scoped tools end to end against a
/// temp directory. No model weights and no `~/Library` writes: each tool's `injected` scope stands
/// in for the live grant store, exactly as the backlog asks ("inject via the seam").
///
/// The sandbox mirrors production's shape — a container (what the app always owns) with the
/// attached folders **inside** it — so `withAccess`'s container fast path is what the end-to-end
/// tests exercise, and `roots` is only ever a narrowing gate (CW-1-FIX-1 F1).
struct FileScopeTests {

    // MARK: fixtures

    /// `<base>/container-lib/` with `notes/` (notes.txt, sub/), `other/` and `outside/` inside it,
    /// plus `elsewhere/` **outside** the container.
    private struct Sandbox {
        let base: URL
        let container: URL
        let root: URL
        let other: URL
        let outside: URL
        let elsewhere: URL
        let notes: URL
        let otherNotes: URL
        let sub: URL
    }

    private func makeSandbox() throws -> Sandbox {
        let fm = FileManager.default
        let base = fm.temporaryDirectory
            .appendingPathComponent("FileScopeTests-\(UUID().uuidString)", isDirectory: true)
        let container = base.appendingPathComponent("container-lib", isDirectory: true)
        let root = container.appendingPathComponent("notes", isDirectory: true)
        let other = container.appendingPathComponent("other", isDirectory: true)
        let outside = container.appendingPathComponent("outside", isDirectory: true)
        let elsewhere = base.appendingPathComponent("elsewhere", isDirectory: true)
        let sub = root.appendingPathComponent("sub", isDirectory: true)
        for dir in [root, other, outside, elsewhere, sub] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        let notes = root.appendingPathComponent("notes.txt")
        let otherNotes = other.appendingPathComponent("other.txt")
        try "root notes".write(to: notes, atomically: true, encoding: .utf8)
        try "other notes".write(to: otherNotes, atomically: true, encoding: .utf8)
        return Sandbox(
            base: base, container: container, root: root, other: other,
            outside: outside, elsewhere: elsewhere, notes: notes, otherNotes: otherNotes, sub: sub)
    }

    private func cleanup(_ s: Sandbox) {
        try? FileManager.default.removeItem(at: s.base)
    }

    /// `notes` is attached and is the working folder; `other` is attached but is not.
    private func scope(_ s: Sandbox) -> FileScope {
        FileScope(folders: [s.root.path, s.other.path], workingFolder: s.root.path)
    }

    private func sandboxRoots(_ s: Sandbox) -> [URL] {
        [s.root, s.other, s.container]
    }

    private func selection(_ s: Sandbox) -> ScopedFileTools.ScopeSelection {
        ScopedFileTools.ScopeSelection(container: s.container, roots: sandboxRoots(s))
    }

    private func grant(_ path: String) -> FolderGrant {
        FolderGrant(path: path, bookmark: Data())
    }

    // MARK: FileScope.interim (C6)

    @Test func interimWithNoGrantsHasNoFoldersOrWorkingFolder() {
        let scope = FileScope.interim(grants: [], edition: .appStore)
        #expect(scope.folders.isEmpty)
        #expect(scope.workingFolder == nil)
    }

    @Test func interimWithOneGrantMakesItTheWorkingFolder() {
        let scope = FileScope.interim(
            grants: [grant("/Users/me/workfolder")], edition: .appStore)
        #expect(scope.folders == ["/Users/me/workfolder"])
        #expect(scope.workingFolder == "/Users/me/workfolder")
    }

    @Test func interimWithTwoGrantsHasNoWorkingFolder() {
        let scope = FileScope.interim(
            grants: [grant("/Users/me/a"), grant("/Users/me/b")], edition: .appStore)
        #expect(scope.folders == ["/Users/me/a", "/Users/me/b"])
        #expect(scope.workingFolder == nil)
    }

    @Test func interimDirectCarriesNoFolders() {
        let scope = FileScope.interim(
            grants: [grant("/Users/me/workfolder")], edition: .direct)
        #expect(scope.folders.isEmpty)
        #expect(scope.workingFolder == nil)
    }

    // MARK: root selection from a scope

    /// Only the scope's folders become roots; the container is always there.
    @Test func scopedRootsKeepOnlyTheScopesFolders() {
        let kept = FolderGrants.rootURLs(
            grants: [grant("/tmp/a"), grant("/tmp/b")],
            scope: FileScope(folders: ["/tmp/a"]),
            container: URL(fileURLWithPath: "/container", isDirectory: true))
        #expect(kept.map(\.path) == ["/tmp/a", "/container"])
    }

    /// A grant path spelled with a trailing slash still names the same folder.
    @Test func scopedRootsComparePathsCanonically() {
        let kept = FolderGrants.rootURLs(
            grants: [grant("/tmp/a/")],
            scope: FileScope(folders: ["/tmp/a"]),
            container: URL(fileURLWithPath: "/container", isDirectory: true))
        #expect(kept.map(\.path) == ["/tmp/a", "/container"])
    }

    @Test func noFolderMessagePointsAtThePlusButton() {
        #expect(FileScope.noFolderMessage.contains("+"))
        #expect(FileScope.noFolderMessage.hasPrefix("No folder is attached"))
    }

    @Test func noPathMessageDoesNotClaimNothingIsAttached() {
        #expect(FileScope.noPathMessage.contains("give a path"))
        #expect(FileScope.noPathMessage.contains("no working folder"))
        #expect(!FileScope.noPathMessage.contains("No folder is attached"))
    }

    @Test func folderListNamesTheFolderAndItsPath() {
        #expect(FileScope.folderList([]) == "none")
        #expect(FileScope.folderList(["/Users/me/Invoices 2026"])
            == "Invoices 2026 (/Users/me/Invoices 2026)")
    }

    // MARK: the scoped tools, end to end, against a temp directory

    @Test func relativePathReadsInsideTheWorkingFolder() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let result = try await FileScope.$current.withValue(scope(s)) {
            try await ScopedReadFileTool(injected: selection(s))
                .execute(arguments: ["path": .string("notes.txt")])
        }
        #expect(result == "root notes")
    }

    /// A relative path is joined to the working folder and then judged: `../../elsewhere` escapes
    /// every root even though it never looked absolute.
    @Test func relativePathEscapingEveryRootIsRejected() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let result = try await FileScope.$current.withValue(scope(s)) {
            try await ScopedReadFileTool(injected: selection(s))
                .execute(arguments: ["path": .string("../../elsewhere/x.txt")])
        }
        #expect(result.contains("not permitted"))
        #expect(result.contains("This chat can use:"))
        #expect(result.contains("+ button"))
    }

    @Test func outOfScopePathIsRejectedNamingOnlyTheScopeFolders() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        // `elsewhere` sits outside the container, so it is outside every root too — the only way
        // to reach the C5 path (a container path is always reachable).
        let outside = s.elsewhere.appendingPathComponent("outside.txt")
        try "outside".write(to: outside, atomically: true, encoding: .utf8)
        let message = FileScope.$current.withValue(scope(s)) {
            ScopedFileTools.rejection(
                for: outside.path, container: s.container, roots: sandboxRoots(s))
        }
        #expect(message.contains("not permitted"))
        #expect(message.contains("This chat can use:"))
        #expect(message.contains("notes (\(s.root.path))"))
        #expect(message.contains("other (\(s.other.path))"))
        #expect(message.contains("+ button"))
    }

    /// F2: no path, no working folder, folders in scope ⇒ "give a path", never "nothing attached".
    @Test func missingPathWithFoldersButNoWorkingFolderAsksForAPath() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let noWorkingFolder = FileScope(folders: [s.root.path], workingFolder: nil)
        let message = FileScope.$current.withValue(noWorkingFolder) {
            ScopedFileTools.rejection(for: nil, container: s.container, roots: sandboxRoots(s))
        }
        #expect(message.hasPrefix("Error: give a path"))
        #expect(message.contains("no working folder"))
        #expect(message.contains("It can use: notes (\(s.root.path))"))
        #expect(!message.contains("No folder is attached"))
    }

    /// An empty scope is the only case that gets the "nothing is attached" line.
    @Test func missingPathWithAnEmptyScopeSaysNoFolderIsAttached() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let empty = FileScope(folders: [], workingFolder: nil)
        let message = FileScope.$current.withValue(empty) {
            ScopedFileTools.rejection(for: nil, container: s.container, roots: sandboxRoots(s))
        }
        #expect(message == FileScope.noFolderMessage)
    }

    @Test func listDirectoryWithoutAPathListsTheWorkingFolder() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let listing = try await FileScope.$current.withValue(scope(s)) {
            try await ScopedListDirectoryTool(injected: selection(s))
                .execute(arguments: [:])
        }
        #expect(listing == "notes.txt\nsub/")
    }

    @Test func searchWithoutAPathSearchesTheWorkingFolder() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        try "x".write(
            to: s.sub.appendingPathComponent("more-notes.md"), atomically: true, encoding: .utf8)
        let found = try await FileScope.$current.withValue(scope(s)) {
            try await ScopedSearchFilesTool(injected: selection(s))
                .execute(arguments: ["query": .string("notes")])
        }
        let names = found.split(separator: "\n").map(String.init).sorted()
        #expect(names == ["notes.txt", "sub/more-notes.md"])
    }

    /// With no scope bound, the tools behave exactly as before: every grant plus the container,
    /// and a temp-directory path is outside both.
    @Test func noScopeKeepsTodaysBehaviour() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        #expect(FileScope.current == nil)
        #expect(ScopedFileTools.access(s.notes.path) == nil)
        let message = ScopedFileTools.rejection(for: s.notes.path)
        #expect(message.contains("not permitted"))
        #expect(!message.contains("This chat can use:"))
        #expect(message.contains("Settings"))
    }

    // MARK: F1 — `roots` is a gate, never a way past the grant lookup

    private final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        func set() { lock.lock(); value = true; lock.unlock() }
        var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    }

    /// (a) A target outside `roots` is refused and the body never runs.
    @Test func rootsGateRefusesAPathOutsideThem() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let ran = Flag()
        let result = FolderGrants.withAccess(
            to: s.notes, grants: [], container: s.container, roots: [s.other]
        ) { _ in ran.set(); return "read" }
        #expect(result.hasPrefix("Error: no granted folder covers"))
        #expect(!ran.isSet)
    }

    /// (b) Under `roots` **and** inside the container: the fast path runs the body.
    @Test func rootsGateRunsTheBodyForAPathInsideTheContainer() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let result = FolderGrants.withAccess(
            to: s.notes, grants: [], container: s.container, roots: [s.container]
        ) { _ in "read" }
        #expect(result == "read")
    }

    /// (c) **The F1 regression guard.** Under `roots`, **outside** the container and with no
    /// grants, the covering-grant lookup must still run — so the call is refused. A `roots` bypass
    /// returns "read" here, which is the production defect CW-A found.
    @Test func rootsGateStillRequiresTheCoveringGrantOutsideTheContainer() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let ran = Flag()
        let result = FolderGrants.withAccess(
            to: s.notes, grants: [], container: s.other, roots: [s.root]
        ) { _ in ran.set(); return "read" }
        #expect(result == "Error: no granted folder covers \(s.notes.path).")
        #expect(!ran.isSet)
    }

    /// (d) The production selection's roots are the container plus the live grant store — never a
    /// sandbox path, and never nil.
    @Test func productionSelectionRootsAreTheContainerPlusGrants() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let production = ScopedFileTools.selection(nil)
        let roots: [URL] = production.roots
        let paths = roots.map(\.path)
        #expect(paths.contains(ModelStore.shared.baseDirectory.path))
        #expect(!paths.contains(s.root.path))
        #expect(!paths.contains(s.other.path))
        #expect(!paths.contains(s.container.path))
    }

    // MARK: @TaskLocal propagation through AgentSession's dispatch closure (the CW-1 guard)

    private actor Seen {
        private(set) var scopes: [FileScope?] = []
        func record(_ scope: FileScope?) { scopes.append(scope) }
    }

    /// Reads `FileScope.current` where a real tool reads it — inside `execute`.
    private nonisolated struct ScopeReader: AgentTool {
        let seen: Seen
        let name = "scope_reader"
        let toolDescription = "Reads the current file scope."
        let parameters: [ToolParameter] = []

        func execute(arguments: [String: JSONValue]) async throws -> String {
            await seen.record(FileScope.current)
            return "ok"
        }
    }

    /// The timed variant: `AgentSession.dispatch` runs the body inside a `withThrowingTaskGroup`
    /// child task, which is the path a `@TaskLocal` most easily loses.
    private nonisolated struct TimedScopeReader: AgentTool {
        let seen: Seen
        let name = "timed_scope_reader"
        let toolDescription = "Reads the current file scope, with a timeout."
        let parameters: [ToolParameter] = []
        var executionTimeout: TimeInterval? { 5 }

        func execute(arguments: [String: JSONValue]) async throws -> String {
            await seen.record(FileScope.current)
            return "ok"
        }
    }

    private func call(_ name: String) -> ToolCall {
        ToolCall(function: .init(name: name, arguments: [:]))
    }

    @Test func taskLocalScopeReachesAnUntimedTool() async {
        let seen = Seen()
        let bound = FileScope(folders: ["/tmp/untimed"], workingFolder: "/tmp/untimed")
        let result = await FileScope.$current.withValue(bound) {
            await AgentSession.dispatch(
                call("scope_reader"), registry: ToolRegistry([ScopeReader(seen: seen)]),
                budget: ToolBudget(limit: 4), approve: { _, _ in true }, onEvent: { _ in })
        }
        let observed = await seen.scopes
        #expect(result == .ok("ok"))
        #expect(observed == [bound])
    }

    @Test func taskLocalScopeReachesATimedTool() async {
        let seen = Seen()
        let bound = FileScope(folders: ["/tmp/timed"], workingFolder: "/tmp/timed")
        let result = await FileScope.$current.withValue(bound) {
            await AgentSession.dispatch(
                call("timed_scope_reader"), registry: ToolRegistry([TimedScopeReader(seen: seen)]),
                budget: ToolBudget(limit: 4), approve: { _, _ in true }, onEvent: { _ in })
        }
        let observed = await seen.scopes
        #expect(result == .ok("ok"))
        #expect(observed == [bound])
    }

    @Test func dispatchWithoutAScopeSeesNil() async {
        let seen = Seen()
        let result = await AgentSession.dispatch(
            call("scope_reader"), registry: ToolRegistry([ScopeReader(seen: seen)]),
            budget: ToolBudget(limit: 4), approve: { _, _ in true }, onEvent: { _ in })
        let observed = await seen.scopes
        #expect(result == .ok("ok"))
        #expect(observed == [nil])
    }
}
