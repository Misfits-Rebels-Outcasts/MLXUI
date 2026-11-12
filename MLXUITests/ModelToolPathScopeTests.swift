//
//  ModelToolPathScopeTests.swift
//  MT4-2 — owner ruling, Boo, 2026-11-09: "in both editions, a model tool's path input resolves only inside the chat's
//  attached folders or granted scope. Anything outside is refused, and the refusal names the attach action. Add a table
//  test for each edition."
//
//  One table, three runs:
//   - `appStoreRealTools` (`#if !DIRECT_BUILD`): the REAL `ScopedReadFileTool` against the REAL model-tool path resolution
//     (`ModelToolRunner.resolveInput`), same scope, same roots. A model tool must never reach what `read_file` cannot.
//   - `directPolicy` (always, so the Direct policy is exercised in the one test host that exists): `ModelToolPathScope.
//     directRoots` + `PathPolicy.resolve`, the exact calls the Direct branch of `resolvePath` makes.
//   - `directRealTools` (`#if DIRECT_BUILD`): the REAL Direct `ReadFileTool` against the REAL model-tool path. The Direct
//     scheme has no test action today, so this block compiles only where a Direct test host exists.
//

import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

@Suite("MT4-2 model-tool path scope")
struct ModelToolPathScopeTests {

    // MARK: fixture

    /// base/ctr/{work, Invoices} and base/elsewhere. `ctr` stands in for the App Store app container (always a root there);
    /// `work` is the working folder, `Invoices` a second attached folder, `elsewhere` is attached to nothing and outside `ctr`.
    struct Fixture {
        let base: URL
        let ctr: URL
        let work: URL
        let invoices: URL
        let elsewhere: URL
        let home: URL
        /// This chat's own folders inside `ctr`, another chat's attachments, and the registry file (the app container's contents).
        let chatID = UUID()
        let attachments: URL
        let assets: URL
        let otherAttachments: URL
        let installedJSON: URL
        var assetStore: ChatAssetStore { ChatAssetStore(conversationID: chatID, in: ctr.appendingPathComponent("conversations")) }

        init() throws {
            let fm = FileManager.default
            base = fm.temporaryDirectory.appendingPathComponent("mt42-\(UUID().uuidString)", isDirectory: true)
            ctr = base.appendingPathComponent("ctr", isDirectory: true)
            work = ctr.appendingPathComponent("work", isDirectory: true)
            invoices = ctr.appendingPathComponent("Invoices", isDirectory: true)
            elsewhere = base.appendingPathComponent("elsewhere", isDirectory: true)
            home = fm.homeDirectoryForCurrentUser
            let conversations = ctr.appendingPathComponent("conversations", isDirectory: true)
            attachments = ConversationStore.attachmentsDirectory(for: chatID, in: conversations)
            assets = ConversationStore.folderURL(for: chatID, in: conversations).appendingPathComponent("assets", isDirectory: true)
            otherAttachments = ConversationStore.attachmentsDirectory(for: UUID(), in: conversations)
            installedJSON = ctr.appendingPathComponent("installed.json")
            for dir in [work.appendingPathComponent("notes"), invoices, elsewhere, attachments, assets, otherAttachments] {
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            }
            for file in [work.appendingPathComponent("notes/a.png"), invoices.appendingPathComponent("inv.png"),
                         elsewhere.appendingPathComponent("b.png"), ctr.appendingPathComponent("scratch.png"), installedJSON,
                         attachments.appendingPathComponent("0-a.png"), assets.appendingPathComponent("a9.png"),
                         otherAttachments.appendingPathComponent("0-x.png")] {
                try Data("hello".utf8).write(to: file)
            }
            try fm.createSymbolicLink(at: work.appendingPathComponent("escape.png"),
                                      withDestinationURL: elsewhere.appendingPathComponent("b.png"))
        }

        func remove() { try? FileManager.default.removeItem(at: base) }
    }

    enum Scope { case working, noWorkingFolder, noFolders, home }

    struct Case {
        let name: String
        let scope: Scope
        let path: @Sendable (Fixture) -> String
        /// Whether a model tool may read it (Direct).
        let reaches: Bool
    }

    static let table: [Case] = [
        Case(name: "absolute inside the working folder", scope: .working, path: { $0.work.path + "/notes/a.png" }, reaches: true),
        Case(name: "absolute inside the second attached folder", scope: .working, path: { $0.invoices.path + "/inv.png" }, reaches: true),
        Case(name: "absolute outside every attached folder", scope: .working, path: { $0.elsewhere.path + "/b.png" }, reaches: false),
        Case(name: "relative, with a working folder", scope: .working, path: { _ in "notes/a.png" }, reaches: true),
        Case(name: "relative, folders attached but no working folder", scope: .noWorkingFolder, path: { _ in "notes/a.png" }, reaches: false),
        Case(name: "relative, no folder attached at all", scope: .noFolders, path: { _ in "notes/a.png" }, reaches: false),
        Case(name: "absolute, no folder attached at all", scope: .noFolders, path: { $0.elsewhere.path + "/b.png" }, reaches: false),
        Case(name: "dot-dot escape from the working folder", scope: .working, path: { _ in "../../elsewhere/b.png" }, reaches: false),
        Case(name: "dot-dot escape inside an absolute path", scope: .working,
             path: { $0.work.path + "/notes/../../../elsewhere/b.png" }, reaches: false),
        Case(name: "inside the app container but in no attached folder", scope: .working,
             path: { $0.ctr.path + "/scratch.png" }, reaches: false),
        Case(name: "installed.json in the app container", scope: .working, path: { $0.installedJSON.path }, reaches: false),
        Case(name: "another chat's attachments", scope: .working, path: { $0.otherAttachments.path + "/0-x.png" }, reaches: false),
        Case(name: "this chat's own attachments", scope: .working, path: { $0.attachments.path + "/0-a.png" }, reaches: true),
        Case(name: "this chat's own attachments, no folder attached", scope: .noFolders,
             path: { $0.attachments.path + "/0-a.png" }, reaches: true),
        Case(name: "this chat's own assets", scope: .working, path: { $0.assets.path + "/a9.png" }, reaches: true),
        Case(name: "denied subpath inside an attached home folder", scope: .home, path: { _ in "~/.ssh/config" }, reaches: false),
        Case(name: "attached-folder alias (the chip name)", scope: .working, path: { _ in "Invoices/inv.png" }, reaches: true),
        Case(name: "a symlink inside an attached folder that points outside", scope: .working,
             path: { $0.work.path + "/escape.png" }, reaches: false),
    ]

    private func fileScope(_ scope: Scope, _ f: Fixture) -> FileScope {
        switch scope {
        case .working: return FileScope(folders: [f.work.path, f.invoices.path], workingFolder: f.work.path, attachmentsFolder: f.attachments.path)
        case .noWorkingFolder: return FileScope(folders: [f.work.path, f.invoices.path], workingFolder: nil, attachmentsFolder: f.attachments.path)
        case .noFolders: return FileScope(folders: [], workingFolder: nil, attachmentsFolder: f.attachments.path)
        case .home: return FileScope(folders: [f.home.path], workingFolder: nil, attachmentsFolder: f.attachments.path)
        }
    }

    /// The roots the App Store resolution gets in a test (the live grant store is replaced): the container first, as
    /// `FolderGrants.rootURLs` makes it, then the attached folders.
    private func roots(_ scope: FileScope, _ f: Fixture) -> [URL] {
        [f.ctr] + scope.folders.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    private func modelToolRuntime(roots: [URL]?, container: URL?) -> ModelToolRuntime {
        ModelToolRuntime(catalog: [], installedModelIDs: [], claimableModelIDs: [], totalRAMGB: 16,
                         makeExecutor: { _, _ in MT13Executor.text("x") }, pathRoots: roots, pathContainer: container)
    }

    /// Whether the model tool's `path` resolution accepts `path` in `scope`, and the refusal when it does not.
    private func modelToolOutcome(path: String, scope: FileScope, roots: [URL]?, container: URL? = nil, scratch: URL, host: ModelToolHost?) async throws
        -> (reached: Bool, refusal: String?)
    {
        let spec = try #require(ModelToolSpec.spec(named: "ocr_image"))
        let runner = ModelToolRunner(spec: spec, runtime: modelToolRuntime(roots: roots, container: container))
        let result = await FileScope.$current.withValue(scope) {
            await runner.resolveInput(arguments: ["path": .string(path)], host: host, scratch: scratch)
        }
        switch result {
        case .success(let input):
            for file in input.temporaryFiles { try? FileManager.default.removeItem(at: file) }
            return (true, nil)
        case .failure(let failure): return (false, failure.message)
        }
    }

    private func namesTheAttachAction(_ refusal: String?) -> Bool {
        guard let refusal else { return false }
        return refusal.contains("+ button")
    }

    // MARK: Direct policy (pure; the exact calls the Direct branch makes)

    @Test func directPolicy() throws {
        let f = try Fixture(); defer { f.remove() }
        for c in Self.table {
            let scope = fileScope(c.scope, f)
            let roots = ModelToolPathScope.directRoots(scope: scope, assets: f.assets)
            let base = scope.workingFolder.map { URL(fileURLWithPath: $0, isDirectory: true) }
            let resolution = PathPolicy.resolve(c.path(f), roots: roots, base: base, folders: scope.folders)
            switch resolution {
            case .allowed: #expect(c.reaches, "Direct: \(c.name) was allowed, expected a refusal")
            case .rejected(let message):
                #expect(!c.reaches, "Direct: \(c.name) was refused, expected it to resolve")
                #expect(namesTheAttachAction(ModelToolPathScope.refusal(message, scope: scope)),
                        "Direct: the refusal for \(c.name) must name the attach action")
            }
        }
    }

    @Test func directRootsAreTheAttachedFoldersNotHome() throws {
        let f = try Fixture(); defer { f.remove() }
        let scope = FileScope(folders: [f.work.path], workingFolder: f.work.path, attachmentsFolder: f.base.appendingPathComponent("att").path)
        let roots = ModelToolPathScope.directRoots(scope: scope).map { $0.standardizedFileURL.path }
        #expect(roots == [f.work.standardizedFileURL.path, f.base.appendingPathComponent("att").standardizedFileURL.path])
        #expect(!roots.contains(f.home.standardizedFileURL.path))
        #expect(ModelToolPathScope.directRoots(scope: nil).isEmpty)
        // Home itself, outside every attached folder: refused (Direct's read_file may read it; a model tool may not).
        let inHome = f.home.appendingPathComponent("Documents/x.png").path
        guard case .rejected = PathPolicy.resolve(inHome, roots: ModelToolPathScope.directRoots(scope: scope), base: f.work, folders: scope.folders)
        else { Issue.record("a file in home outside the attached folders was allowed"); return }
        // This chat's own attachments copy is allowed.
        let attached = f.base.appendingPathComponent("att/1-x.png").path
        guard case .allowed = PathPolicy.resolve(attached, roots: ModelToolPathScope.directRoots(scope: scope), base: f.work, folders: scope.folders)
        else { Issue.record("the chat's attachments copy was refused"); return }
    }

    @Test func theRefusalNamesTheAttachActionWithAndWithoutFolders() {
        let none = ModelToolPathScope.refusal("Error: access to /x is not permitted.", scope: FileScope(folders: [], workingFolder: nil))
        #expect(none == FileScope.noFolderMessage)
        #expect(namesTheAttachAction(none))
        let some = ModelToolPathScope.refusal("Error: access to /x is not permitted.", scope: FileScope(folders: ["/a/work"], workingFolder: nil))
        #expect(some.hasPrefix("access to /x is not permitted."))
        #expect(some.contains("work (/a/work)"))
        #expect(some.hasSuffix(ModelToolPathScope.attachAction))
        #expect(namesTheAttachAction(ModelToolPathScope.refusal("Error: x", scope: nil)))
    }

    // MARK: App Store, real tools

    #if !DIRECT_BUILD
    @Test func appStoreRealTools() async throws {
        let f = try Fixture(); defer { f.remove() }
        for c in Self.table {
            let scope = fileScope(c.scope, f)
            let roots = roots(scope, f)
            let path = c.path(f)
            let expected = c.reaches
            // read_file, the real tool, under the same scope and roots.
            let readFile = ScopedReadFileTool(injected: .init(container: f.ctr, roots: roots))
            let read = try await FileScope.$current.withValue(scope) { try await readFile.execute(arguments: ["path": .string(path)]) }
            let readReaches = read == "hello"
            // The model tool's path resolution (the production code, with the grant store replaced by the same roots).
            let outcome = try await modelToolOutcome(path: path, scope: scope, roots: roots, container: f.ctr, scratch: f.base.appendingPathComponent("scratch"), host: ModelToolHost(assets: f.assetStore))
            #expect(outcome.reached == expected, "App Store: \(c.name): model tool reached=\(outcome.reached), expected \(expected)")
            // read_file is unchanged and reaches the whole container, so it only has to reach everything the model tool may.
            if c.reaches { #expect(readReaches, "App Store: \(c.name): read_file should reach it too") }
            // The stop condition: a model tool must never reach a file read_file cannot.
            #expect(!(outcome.reached && !readReaches), "App Store: \(c.name): the model tool reached a file read_file cannot")
            if !outcome.reached {
                #expect(namesTheAttachAction(outcome.refusal), "App Store: the refusal for \(c.name) must name the attach action, got: \(outcome.refusal ?? "nil")")
            }
        }
    }
    #endif

    // MARK: Direct, real tools (compiles only where a Direct test host exists)

    #if DIRECT_BUILD
    @Test func directRealTools() async throws {
        let f = try Fixture(); defer { f.remove() }
        for c in Self.table {
            let scope = fileScope(c.scope, f)
            let path = c.path(f)
            let read = try await FileScope.$current.withValue(scope) { try await ReadFileTool().execute(arguments: ["path": .string(path)]) }
            let readReaches = read == "hello"
            // No `pathRoots`: the production roots, i.e. `ModelToolPathScope.directRoots(scope:)`.
            let outcome = try await modelToolOutcome(path: path, scope: scope, roots: nil, scratch: f.base.appendingPathComponent("scratch"), host: ModelToolHost(assets: f.assetStore))
            #expect(outcome.reached == c.reaches, "Direct: \(c.name): model tool reached=\(outcome.reached), expected \(c.reaches)")
            #expect(!(outcome.reached && !readReaches), "Direct: \(c.name): the model tool reached a file read_file cannot")
            if !outcome.reached {
                #expect(namesTheAttachAction(outcome.refusal), "Direct: the refusal for \(c.name) must name the attach action, got: \(outcome.refusal ?? "nil")")
            }
        }
    }
    #endif
}
