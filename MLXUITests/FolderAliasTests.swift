import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// CW-FIX-2 — a relative path that starts with an attached folder's own name ("workfolder") means
/// that folder. Temp roots only; nothing here touches `~/Library`.
struct FolderAliasTests {

    private func makeTree(_ files: [String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("folder-alias-\(UUID().uuidString)", isDirectory: true)
        for file in files {
            let url = root.appendingPathComponent(file)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(file.utf8).write(to: url)
        }
        return root
    }

    private func list(_ path: String?, scope: FileScope, container: URL) async throws -> String {
        let tool = ScopedListDirectoryTool(
            injected: .init(container: container, roots: [container]))
        return try await FileScope.$current.withValue(scope) {
            try await tool.execute(arguments: path.map { ["path": .string($0)] } ?? [:])
        }
    }

    private func read(_ path: String, scope: FileScope, container: URL) async throws -> String {
        let tool = ScopedReadFileTool(injected: .init(container: container, roots: [container]))
        return try await FileScope.$current.withValue(scope) {
            try await tool.execute(arguments: ["path": .string(path)])
        }
    }

    @Test func folderNameListsTheFolderItself() async throws {
        let root = try makeTree(["workfolder/a.txt", "workfolder/sub/b.txt"])
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("workfolder").path
        let scope = FileScope(folders: [folder], workingFolder: folder)
        #expect(try await list("workfolder", scope: scope, container: root) == "a.txt\nsub/")
        #expect(try await list("workfolder/", scope: scope, container: root) == "a.txt\nsub/")
    }

    @Test func folderNamePrefixReadsInsideTheFolder() async throws {
        let root = try makeTree(["workfolder/sub/a.txt"])
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("workfolder").path
        let scope = FileScope(folders: [folder], workingFolder: folder)
        #expect(try await read("workfolder/sub/a.txt", scope: scope, container: root)
            == "workfolder/sub/a.txt")
    }

    @Test func aRealSubfolderOfTheSameNameWins() async throws {
        let root = try makeTree(["workfolder/workfolder/inner.txt", "workfolder/outer.txt"])
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("workfolder").path
        let scope = FileScope(folders: [folder], workingFolder: folder)
        #expect(try await list("workfolder", scope: scope, container: root) == "inner.txt")
    }

    @Test func twoFoldersWithOneNameAreNotGuessed() async throws {
        let root = try makeTree(["one/docs/a.txt", "two/docs/b.txt"])
        defer { try? FileManager.default.removeItem(at: root) }
        let one = root.appendingPathComponent("one/docs").path
        let two = root.appendingPathComponent("two/docs").path
        let scope = FileScope(folders: [one, two], workingFolder: one)
        #expect(PathPolicy.aliasedFolder(
            for: "docs", base: URL(fileURLWithPath: one), folders: scope.folders) == nil)
        let reply = try await list("docs", scope: scope, container: root)
        #expect(reply.hasPrefix("Error: there is no \"docs\" in the working folder docs (\(one))."))
        #expect(reply.contains("call list_directory with no path"))
        #expect(reply.contains("This chat can use: docs (\(one)), docs (\(two))."))
    }

    @Test func notFoundTextNamesTheWorkingFolderForRelativePathsOnly() async throws {
        let root = try makeTree(["workfolder/a.txt"])
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("workfolder").path
        let scope = FileScope(folders: [folder], workingFolder: folder)
        let relative = try await read("nope.txt", scope: scope, container: root)
        #expect(relative.hasPrefix("Error: there is no \"nope.txt\" in the working folder workfolder"))
        let absolute = try await read("\(folder)/nope.txt", scope: scope, container: root)
        #expect(!absolute.contains("in the working folder"))
        #expect(absolute.hasPrefix("Error:"))
    }

    @Test func theAliasNeverWidensAccess() throws {
        let root = try makeTree(["workfolder/a.txt", "secret/s.txt"])
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("workfolder")
        // A folder that is not attached is never an alias.
        #expect(PathPolicy.aliasedFolder(
            for: "secret", base: folder, folders: [folder.path]) == nil)
        // `..` after the alias still leaves the root and is refused.
        let escaped = PathPolicy.resolve(
            "workfolder/../secret/s.txt", roots: [folder], base: folder, folders: [folder.path])
        guard case .rejected = escaped else {
            Issue.record("expected the escape to be refused, got \(escaped)")
            return
        }
        // The alias itself is allowed only because the folder is a root.
        #expect(PathPolicy.resolve(
            "workfolder", roots: [folder], base: folder, folders: [folder.path])
            == .allowed(folder.standardizedFileURL))
    }
}

@MainActor
struct FolderAttachRoutesTests {
    @Test func bothAttachRoutesStoreTheSameFolderAndWorkingFolder() throws {
        let work = FileManager.default.temporaryDirectory
            .appendingPathComponent("attach-routes-\(UUID().uuidString)/workfolder", isDirectory: true)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        defer {
            FolderGrants.removeGrant(path: work.path)   // the URL route grants it in the App Store build
            try? FileManager.default.removeItem(at: work.deletingLastPathComponent())
        }

        func attached(_ attach: (ModelRunner) -> Void) -> ([String], String?) {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("attach-store-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let runner = ModelRunner()
            runner.prepare(for: makeEntry(id: "model-a", displayName: "Test Model"), in: directory)
            attach(runner)
            return (runner.activeAttachedFolders, runner.activeWorkingFolder)
        }

        let byPath = attached { $0.attachFolder(path: work.path) }
        let byURL = attached { $0.attachFolder(work) }
        #expect(byPath.0 == [work.path])
        #expect(byPath.1 == work.path)
        #expect(byURL.0 == byPath.0)
        #expect(byURL.1 == byPath.1)
    }
}
