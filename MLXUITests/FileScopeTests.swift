import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// CW-1 — the conversation's folder scope: the interim rule (C6), the two `@TaskLocal` paths that
/// carry it into the file tools, and the scoped tools end to end against a temp directory. No model
/// weights and no `~/Library` writes: each tool's `injected` scope stands in for the live grant
/// store, exactly as the backlog asks ("inject via the seam; don't create real bookmarks").
struct FileScopeTests {

    // MARK: fixtures

    /// root/ (with notes.txt and sub/) plus other/ and container/.
    private struct Sandbox {
        let base: URL
        let root: URL
        let other: URL
        let notes: URL
        let otherNotes: URL
        let sub: URL
        let container: URL
    }

    private func makeSandbox() throws -> Sandbox {
        let fm = FileManager.default
        let base = fm.temporaryDirectory
            .appendingPathComponent("FileScopeTests-\(UUID().uuidString)", isDirectory: true)
        let root = base.appendingPathComponent("root", isDirectory: true)
        let other = base.appendingPathComponent("other", isDirectory: true)
        let sub = root.appendingPathComponent("sub", isDirectory: true)
        let container = base.appendingPathComponent("container", isDirectory: true)
        for dir in [root, other, sub, container] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        let notes = root.appendingPathComponent("notes.txt")
        let otherNotes = other.appendingPathComponent("other.txt")
        try "root notes".write(to: notes, atomically: true, encoding: .utf8)
        try "other notes".write(to: otherNotes, atomically: true, encoding: .utf8)
        return Sandbox(
            base: base, root: root, other: other,
            notes: notes, otherNotes: otherNotes, sub: sub, container: container)
    }

    private func cleanup(_ s: Sandbox) {
        try? FileManager.default.removeItem(at: s.base)
    }

    /// root is attached and is the working folder; other is attached but is not.
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

    /// A relative path is joined to the working folder, then judged: `../../outside.txt` escapes
    /// every root even though it never looked absolute.
    @Test func relativePathEscapingEveryRootIsRejected() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let outside = s.base.appendingPathComponent("outside.txt")
        try "outside".write(to: outside, atomically: true, encoding: .utf8)
        let result = try await FileScope.$current.withValue(scope(s)) {
            try await ScopedReadFileTool(injected: selection(s))
                .execute(arguments: ["path": .string("../../outside.txt")])
        }
        #expect(result.contains("not permitted"))
        #expect(result.contains("This chat can use:"))
        #expect(result.contains("+ button"))
    }

    @Test func outOfScopePathIsRejectedNamingOnlyTheScopeFolders() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let outside = s.base.appendingPathComponent("outside.txt")
        try "outside".write(to: outside, atomically: true, encoding: .utf8)
        let message = FileScope.$current.withValue(scope(s)) {
            ScopedFileTools.rejection(
                for: outside.path, container: s.container, roots: sandboxRoots(s))
        }
        #expect(message.contains("not permitted"))
        #expect(message.contains("This chat can use:"))
        #expect(message.contains(s.root.path))
        #expect(message.contains(s.other.path))
        #expect(message.contains("+ button"))
    }

    /// No working folder ⇒ no base for a relative path, so the model is told how to get one.
    @Test func missingPathWithoutAWorkingFolderExplainsThePlusButton() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let noWorkingFolder = FileScope(folders: [s.root.path], workingFolder: nil)
        let message = FileScope.$current.withValue(noWorkingFolder) {
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

    /// Explicit `roots` **is** the containment test: a path under one of them runs without a
    /// security scope, and a path outside all of them is refused before any I/O.
    @Test func withAccessHonoursExplicitRoots() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let inside = FolderGrants.withAccess(
            to: s.notes, grants: [], container: s.container, roots: [s.root]
        ) { _ in "read" }
        #expect(inside == "read")

        let outside = FolderGrants.withAccess(
            to: s.notes, grants: [], container: s.container, roots: [s.other]
        ) { _ in "read" }
        #expect(outside.hasPrefix("Error:"))
    }

    /// A scope with no folders (Direct) plus no grants: the container is the only root, so a temp
    /// path is refused even though a scope is bound.
    @Test func emptyFoldersScopeDoesNotWidenRoots() async throws {
        let s = try makeSandbox()
        defer { cleanup(s) }
        let direct = FileScope(folders: [])
        let message = FileScope.$current.withValue(direct) {
            ScopedFileTools.rejection(for: s.notes.path, container: s.container)
        }
        #expect(message == FileScope.noFolderMessage)
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
