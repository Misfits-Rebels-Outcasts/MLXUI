import Foundation
import Testing
@testable import MLXUI

/// CW-2 — attaching folders to a conversation: the runner's attach/detach/working-folder API, the
/// persistence trap (`saveActiveConversation` only writes when `items` changed), and the C5 text
/// naming only folders that are both attached and granted. The "+" menu, chips and drag-and-drop
/// are verified by build + smoke (CW-S3/S4).
@MainActor
struct ChatFolderAttachTests {

    private func makeRunner() -> (ModelRunner, URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-folders-\(UUID().uuidString)")
        let runner = ModelRunner()
        runner.prepare(for: makeEntry(id: "model-a", displayName: "Test Model"), in: directory)
        return (runner, directory)
    }

    // MARK: attach / detach / working folder

    @Test func theFirstAttachedFolderBecomesTheWorkingFolder() {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }

        runner.attachFolder(path: "/Users/me/a")
        runner.attachFolder(path: "/Users/me/b")
        #expect(runner.activeAttachedFolders == ["/Users/me/a", "/Users/me/b"])
        #expect(runner.activeWorkingFolder == "/Users/me/a")
    }

    @Test func attachingTheSameFolderTwiceKeepsOneCopy() {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        runner.attachFolder(path: "/Users/me/a")
        runner.attachFolder(path: "/Users/me/a")
        #expect(runner.activeAttachedFolders == ["/Users/me/a"])
    }

    /// C8: detaching the working folder promotes the next attached one.
    @Test func detachingTheWorkingFolderPromotesTheNext() {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        runner.attachFolder(path: "/Users/me/a")
        runner.attachFolder(path: "/Users/me/b")
        runner.detachFolder(path: "/Users/me/a")
        #expect(runner.activeAttachedFolders == ["/Users/me/b"])
        #expect(runner.activeWorkingFolder == "/Users/me/b")
    }

    @Test func detachingTheLastFolderLeavesNoWorkingFolder() {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        runner.attachFolder(path: "/Users/me/a")
        runner.detachFolder(path: "/Users/me/a")
        #expect(runner.activeAttachedFolders.isEmpty)
        #expect(runner.activeWorkingFolder == nil)
    }

    @Test func setWorkingFolderOnlyAcceptsAnAttachedFolder() {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        runner.attachFolder(path: "/Users/me/a")
        runner.attachFolder(path: "/Users/me/b")
        runner.setWorkingFolder(path: "/Users/me/b")
        #expect(runner.activeWorkingFolder == "/Users/me/b")
        runner.setWorkingFolder(path: "/Users/me/not-attached")
        #expect(runner.activeWorkingFolder == "/Users/me/b")
    }

    // MARK: the persistence trap

    @Test func newChatsStartWithNothingAttached() {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(runner.activeAttachedFolders.isEmpty)   // G2
    }

    /// The trap: `saveActiveConversation` returns early unless `items` changed, so an attach on a
    /// chat **with history** has to be persisted by the folder API itself.
    @Test func anAttachOnAConversationWithItemsIsPersisted() throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        runner.appendUserMessageForTesting("hello")
        // Persist the items the way the app does: `newConversation` saves the outgoing one.
        runner.newConversation()
        let firstID = runner.conversations.first { $0.items.isEmpty == false }?.id
        runner.selectConversation(try #require(firstID))

        runner.attachFolder(path: "/Users/me/a")
        let stored = ConversationStore.conversations(forModelID: "model-a", in: directory)
        #expect(stored.count == 1)
        #expect(stored.first?.attachedFolders == ["/Users/me/a"])
        #expect(stored.first?.workingFolder == "/Users/me/a")
    }

    /// A blank chat stays memory-only, exactly as before — the first send persists it along with
    /// what was attached.
    @Test func anAttachOnABlankConversationStaysInMemory() throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        runner.attachFolder(path: "/Users/me/a")
        #expect(runner.activeAttachedFolders == ["/Users/me/a"])
        #expect(ConversationStore.conversations(forModelID: "model-a", in: directory).isEmpty)
    }

    @Test func aDetachIsPersistedToo() throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        runner.appendUserMessageForTesting("hello")
        runner.newConversation()
        runner.selectConversation(try #require(runner.conversations.first { !$0.items.isEmpty }?.id))
        runner.attachFolder(path: "/Users/me/a")
        runner.attachFolder(path: "/Users/me/b")
        runner.detachFolder(path: "/Users/me/a")

        let stored = ConversationStore.conversations(forModelID: "model-a", in: directory)
        #expect(stored.first?.attachedFolders == ["/Users/me/b"])
        #expect(stored.first?.workingFolder == "/Users/me/b")
    }

    /// Folders survive a relaunch: the same directory is read back by a fresh runner.
    @Test func attachedFoldersSurviveAReload() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-folders-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }

        let first = ModelRunner()
        first.prepare(for: makeEntry(id: "model-a", displayName: "Test Model"), in: directory)
        first.appendUserMessageForTesting("hello")
        first.newConversation()
        first.selectConversation(try #require(first.conversations.first { !$0.items.isEmpty }?.id))
        first.attachFolder(path: "/Users/me/a")
        first.attachFolder(path: "/Users/me/b")
        first.setWorkingFolder(path: "/Users/me/b")

        let second = ModelRunner()
        second.prepare(for: makeEntry(id: "model-a", displayName: "Test Model"), in: directory)
        #expect(second.activeAttachedFolders == ["/Users/me/a", "/Users/me/b"])
        #expect(second.activeWorkingFolder == "/Users/me/b")

        // …and a brand-new chat in the same model still starts empty (G2).
        second.newConversation()
        #expect(second.activeAttachedFolders.isEmpty)
    }

    // MARK: C5 names only attached + granted folders

    /// The scope the tools use excludes an unattached grant, so a rejection cannot name it. The
    /// no-path wording is used here on purpose: it builds the list **without** echoing a requested
    /// path, so any appearance of the secret folder would be a genuine leak (the C5 text always
    /// quotes the path the model asked for, which is the model's own input).
    @Test func theRejectionTextCannotNameAnUnattachedGrant() {
        let grants = [
            FolderGrant(path: "/Users/me/attached", bookmark: Data()),
            FolderGrant(path: "/Users/me/secret", bookmark: Data()),
        ]
        let scope = FileScope.forConversation(
            attached: ["/Users/me/attached"], workingFolder: nil,
            grants: grants, edition: .appStore)
        let message = FileScope.$current.withValue(scope) {
            ScopedFileTools.rejection(for: nil)
        }
        #expect(message.contains("This chat can use:"))
        #expect(message.contains("/Users/me/attached"))
        #expect(!message.contains("/Users/me/secret"))
    }

    /// The environment block follows the same rule: one attached folder, so the unattached grant
    /// never appears in the prompt.
    @Test func theEnvironmentBlockCannotNameAnUnattachedGrant() {
        let grants = [
            FolderGrant(path: "/Users/me/attached", bookmark: Data()),
            FolderGrant(path: "/Users/me/secret", bookmark: Data()),
        ]
        let scope = FileScope.forConversation(
            attached: ["/Users/me/attached"], workingFolder: "/Users/me/attached",
            grants: grants, edition: .appStore)
        let block = ChatEnvironment.instructions(
            scope: scope, edition: .appStore,
            fileToolsEnabled: ["read_file"], writeEnabled: false)
        #expect(block?.contains("/Users/me/attached") == true)
        #expect(block?.contains("/Users/me/secret") == false)
    }

    /// And the roots the tools get are the attached ones only.
    @Test func theToolRootsComeFromTheAttachedSet() {
        let grants = [
            FolderGrant(path: "/Users/me/attached", bookmark: Data()),
            FolderGrant(path: "/Users/me/secret", bookmark: Data()),
        ]
        let scope = FileScope.forConversation(
            attached: ["/Users/me/attached"], workingFolder: "/Users/me/attached",
            grants: grants, edition: .appStore)
        let roots = FolderGrants.rootURLs(
            grants: grants, scope: scope,
            container: URL(fileURLWithPath: "/container", isDirectory: true))
        #expect(roots.map(\.path) == ["/Users/me/attached", "/container"])
    }
}
