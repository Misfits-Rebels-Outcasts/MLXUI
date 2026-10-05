import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG5 — scoped file tools + `FolderGrants`. Pure parts (store round-trip, covering-grant
/// matcher, search walk + formatting, tool specs) are tested here; security-scoped bookmark
/// resolution and NSOpenPanel need a live app and are covered by smoke. Path-escape guarding
/// itself lives in `PathPolicy` and is covered by `PathPolicyTests`.
struct ScopedFileToolsTests {

    // MARK: FolderGrants store

    @Test func grantsRoundTripThroughDefaults() {
        let suite = "ScopedFileToolsTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        #expect(FolderGrants.load(from: defaults).isEmpty)
        let grants = [
            FolderGrant(path: "/Users/me/Projects", bookmark: Data([1, 2])),
            FolderGrant(path: "/Users/me/Notes", bookmark: Data([3])),
        ]
        FolderGrants.save(grants, to: defaults)
        #expect(FolderGrants.load(from: defaults) == grants)
        #expect(FolderGrants.grantedPaths(from: defaults) == ["/Users/me/Notes", "/Users/me/Projects"])

        FolderGrants.removeGrant(path: "/Users/me/Notes", from: defaults)
        #expect(FolderGrants.grantedPaths(from: defaults) == ["/Users/me/Projects"])
    }

    // MARK: covering-grant matcher

    private func grant(_ path: String) -> FolderGrant { FolderGrant(path: path, bookmark: Data()) }

    @Test func coveringGrantMatchesInsideAndSelf() {
        let grants = [grant("/tmp/granted")]
        #expect(FolderGrants.grant(covering: "/tmp/granted/sub/file.txt", in: grants)?.path == "/tmp/granted")
        #expect(FolderGrants.grant(covering: "/tmp/granted", in: grants)?.path == "/tmp/granted")
    }

    @Test func coveringGrantRejectsOutsideAndNamePrefixSibling() {
        let grants = [grant("/tmp/granted")]
        #expect(FolderGrants.grant(covering: "/tmp/other/file.txt", in: grants) == nil)
        // "/tmp/granted-2" starts with the same characters but is a sibling, not inside.
        #expect(FolderGrants.grant(covering: "/tmp/granted-2/file.txt", in: grants) == nil)
    }

    @Test func coveringGrantPrefersTheLongestRoot() {
        let grants = [grant("/tmp/a"), grant("/tmp/a/deeper")]
        #expect(FolderGrants.grant(covering: "/tmp/a/deeper/file", in: grants)?.path == "/tmp/a/deeper")
        #expect(FolderGrants.grant(covering: "/tmp/a/other", in: grants)?.path == "/tmp/a")
    }

    // MARK: roots handed to PathPolicy

    @Test func rootURLsIncludeGrantsAndContainer() {
        let container = URL(fileURLWithPath: "/container", isDirectory: true)
        let roots = FolderGrants.rootURLs(grants: [grant("/tmp/granted")], container: container)
        #expect(roots.map(\.path) == ["/tmp/granted", "/container"])
    }

    // MARK: search walk (real temp directory)

    @Test func searchFindsMatchesRecursivelyCaseInsensitive() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("scoped-search-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("sub"), withIntermediateDirectories: true)
        try "x".write(to: root.appendingPathComponent("Notes.md"), atomically: true, encoding: .utf8)
        try "x".write(to: root.appendingPathComponent("sub/more-notes.txt"), atomically: true, encoding: .utf8)
        try "x".write(to: root.appendingPathComponent("sub/other.txt"), atomically: true, encoding: .utf8)

        let matches = ScopedSearchFilesTool.search(root: root, query: "notes")
        #expect(matches.sorted() == ["Notes.md", "sub/more-notes.txt"])
    }

    @Test func searchFormatsEmptyAndOverflow() {
        #expect(ScopedSearchFilesTool.format(matches: [], query: "q")
            == "No file names containing \"q\".")

        let many = (0...ScopedSearchFilesTool.maxResults).map { "file-\($0).txt" }  // cap + 1
        let out = ScopedSearchFilesTool.format(matches: many, query: "file")
        #expect(out.hasSuffix("… more matches not shown"))
    }

    // MARK: tool registration + conformance

    @Test func scopedSetShipsFiveToolsWithWriteAndEditGated() {
        let tools = ScopedFileTools.all()
        #expect(Set(tools.map(\.name)) == [
            "read_file", "write_file", "edit_file", "list_directory", "search_files",
        ])
        #expect(ScopedWriteFileTool().requiresApproval == true)
        #expect(ScopedEditFileTool().requiresApproval == true)
        #expect(ScopedReadFileTool().requiresApproval == false)
        #expect(ScopedListDirectoryTool().requiresApproval == false)
        #expect(ScopedSearchFilesTool().requiresApproval == false)
        // Both side-effecting tools are opt-in: unticked until the user turns them on.
        #expect(ScopedFileTools.disabledByDefault == ["write_file", "edit_file"])
    }

    @Test func scopedToolsAreSpecced() {
        let write = ScopedWriteFileTool().toolSpec["function"] as? [String: any Sendable]
        let writeParams = write?["parameters"] as? [String: any Sendable]
        #expect(Set(writeParams?["required"] as? [String] ?? []) == ["path", "contents"])

        let search = ScopedSearchFilesTool().toolSpec["function"] as? [String: any Sendable]
        let searchParams = search?["parameters"] as? [String: any Sendable]
        #expect(Set(searchParams?["required"] as? [String] ?? []) == ["path", "query"])

        // edit_file: three required strings plus the optional boolean (D1).
        let edit = ScopedEditFileTool().toolSpec["function"] as? [String: any Sendable]
        let editParams = edit?["parameters"] as? [String: any Sendable]
        #expect(Set(editParams?["required"] as? [String] ?? []) == ["path", "old_string", "new_string"])
        let editProperties = editParams?["properties"] as? [String: any Sendable]
        let replaceAll = editProperties?["replace_all"] as? [String: any Sendable]
        #expect(replaceAll?["type"] as? String == "boolean")
    }

    // MARK: rejection message guides the user to the grant UI

    @Test func rejectionMentionsGrantingWhenNoGrantsExist() {
        // No grant covers /nonexistent-root; the message should teach the model about the menu.
        let message = ScopedFileTools.rejection(for: "/nonexistent-root/file.txt")
        #expect(message.hasPrefix("Error:"))
        #expect(message.contains("wrench"))
    }

    // MARK: edit_file argument + path checks (EF-2)

    @Test func scopedEditFileRequiresBothStringsBeforeTouchingThePath() async throws {
        // A missing new_string is an error, not an empty string: deleting text means
        // sending new_string == "" explicitly (D1/D5).
        let tool = ScopedEditFileTool()
        let missingOld = try await tool.execute(arguments: [
            "path": .string("/nonexistent-root/file.txt"),
            "new_string": .string("x"),
        ])
        #expect(missingOld == "Error: 'old_string' is required.")

        let missingNew = try await tool.execute(arguments: [
            "path": .string("/nonexistent-root/file.txt"),
            "old_string": .string("x"),
        ])
        #expect(missingNew == "Error: 'new_string' is required.")
    }

    @Test func scopedEditFileRefusesAPathOutsideEveryGrantAndLeavesTheFileAlone() async throws {
        // A temp-directory path is outside every granted root and the app container, so the
        // rejection must come back before `TextEdit` ever sees the file.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("scoped-edit-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let sentinel = directory.appendingPathComponent("sentinel.txt")
        try "sentinel\n".write(to: sentinel, atomically: true, encoding: .utf8)
        let before = try Data(contentsOf: sentinel)

        let result = try await ScopedEditFileTool().execute(arguments: [
            "path": .string(sentinel.path),
            "old_string": .string("sentinel"),
            "new_string": .string("CHANGED"),
        ])
        let after = try Data(contentsOf: sentinel)

        #expect(result.hasPrefix("Error:"))
        #expect(result.contains("not permitted"))
        #expect(after == before)
    }
}
