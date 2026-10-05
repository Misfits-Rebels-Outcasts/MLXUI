import Foundation
import Testing
@testable import MLXUI

/// CW-2 — attaching folders to a conversation: the runner's attach/detach/working-folder API, the
/// persistence trap (`saveActiveConversation` only writes when `items` changed), and the C5 text
/// naming only folders that are both attached and granted. The "+" menu, chips and drag-and-drop
/// are verified by build + smoke (CW-S3/S4).
@MainActor
struct ChatFolderAttachTests {

    private let entry = makeEntry(id: "model-a", displayName: "Test Model")

    private func makeRunner() -> (ModelRunner, URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-folders-\(UUID().uuidString)")
        let runner = ModelRunner()
        runner.prepare(for: entry, in: directory)
        return (runner, directory)
    }

    /// A temp folder to hold the "attached" files themselves (not the conversation store).
    private func makeWork() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-work-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - B1: one turn at a time, across the attachment import

    /// A model directory that exists, so `send` gets past its "files not found" guard without
    /// weights. The turn is then ended by `afterImportForTesting`, so nothing ever loads.
    private func makeModelDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func imageFile(in directory: URL, named name: String) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: url)
        return url
    }

    /// A one-shot async gate: `wait` suspends until `raise` runs, and stays raised after that.
    private actor Signal {
        private var raised = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func raise() {
            raised = true
            let pending = waiters
            waiters = []
            for continuation in pending { continuation.resume() }
        }

        func wait() async {
            if raised { return }
            await withCheckedContinuation { continuation in
                waiters.append(continuation)
            }
        }
    }

    /// Every user message in every **conversation**. B1 is about what lands in a chat, and the
    /// live transcript is allowed to keep a turn the re-check abandoned (`saveActiveConversation`
    /// only writes from the transcript when the turn completes).
    private func storedUserTexts(_ runner: ModelRunner) -> [String] {
        runner.conversations.flatMap { conversation in
            conversation.items.compactMap { item in
                guard case .message(let message) = item, message.role == .user else { return nil }
                return message.text
            }
        }
    }

    /// Installs a hook that parks the turn until the test releases it, so the runner can be
    /// observed mid-send with no model and no timing luck. The turn then ends on the injected
    /// container (which throws), so the whole `send` body runs and finishes deterministically.
    private func parkAfterImport(
        _ runner: ModelRunner
    ) -> (reached: @Sendable () async -> Void, release: @Sendable () -> Void) {
        let reachedSignal = Signal()
        let releaseSignal = Signal()
        runner.containerForTesting = { throw AttachmentImporter.Failure.tooLarge(1) }
        runner.afterImportForTesting = {
            await reachedSignal.raise()
            await releaseSignal.wait()
        }
        return (
            reached: { await reachedSignal.wait() },
            release: { Task { await releaseSignal.raise() } }
        )
    }

    /// (a) A second `send` while the first is importing is a no-op, not a second turn.
    @Test func aSecondSendDuringTheImportIsANoOp() async throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let work = try makeWork()
        defer { try? FileManager.default.removeItem(at: work) }
        let modelDir = try makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }
        runner.modelDirectoryForTesting = { _ in modelDir }
        let image = try imageFile(in: work, named: "shot.png")

        let gate = parkAfterImport(runner)
        let first = Task { await runner.send("first", attachments: [image], for: self.entry) }
        await gate.reached()                       // the turn is fully set up
        #expect(runner.isRunning)

        await runner.send("second", for: entry)    // must return immediately
        gate.release()
        await first.value

        // The live transcript is where the one accepted turn lives; the second send added nothing.
        let live = runner.transcript.items.compactMap { item -> String? in
            guard case .message(let message) = item, message.role == .user else { return nil }
            return message.text
        }
        #expect(live == ["first"])
    }

    /// (b) Switching conversation during the import appends to **neither** chat.
    @Test func switchingChatsDuringTheImportAppendsToNeither() async throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let work = try makeWork()
        defer { try? FileManager.default.removeItem(at: work) }
        let modelDir = try makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }
        runner.modelDirectoryForTesting = { _ in modelDir }
        let image = try imageFile(in: work, named: "shot.png")

        // A first chat with history, then a second one, and go back to the first.
        runner.appendUserMessageForTesting("earlier")
        runner.newConversation()
        let secondID = try #require(runner.activeConversationID)
        let firstID = try #require(runner.conversations.first { !$0.items.isEmpty }?.id)
        runner.selectConversation(firstID)

        let gate = parkAfterImport(runner)
        let send = Task { await runner.send("mid-import", attachments: [image], for: self.entry) }
        await gate.reached()
        // Move away and **back** while the turn is parked. With the re-check missing, the append
        // then lands in the live transcript of the conversation the user returned to, and the
        // turn's own `saveActiveConversation` writes it there — the leak B1 describes.
        runner.selectConversation(secondID)
        runner.selectConversation(firstID)
        gate.release()
        await send.value

        #expect(!storedUserTexts(runner).contains("mid-import"))
        let first = try #require(runner.conversations.first { $0.id == firstID })
        #expect(first.items.count == 1)                 // only its own history, no mid-import turn
        let second = try #require(runner.conversations.first { $0.id == secondID })
        #expect(second.items.isEmpty)
    }

    /// (c) `stop()` during the import: nothing is appended and the runner is idle again.
    @Test func stopDuringTheImportAppendsNothing() async throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let work = try makeWork()
        defer { try? FileManager.default.removeItem(at: work) }
        let modelDir = try makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }
        runner.modelDirectoryForTesting = { _ in modelDir }
        let image = try imageFile(in: work, named: "shot.png")

        let gate = parkAfterImport(runner)
        let send = Task { await runner.send("stopped", attachments: [image], for: self.entry) }
        await gate.reached()
        runner.stop()
        gate.release()
        await send.value

        #expect(!runner.isRunning)
        #expect(!storedUserTexts(runner).contains("stopped"))
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

    /// Call 5's added case: A and B are both granted, only A is attached, and the model asks for a
    /// path in a **third** folder C. The C5 text echoes the path it asked for (its own input), and
    /// must still not name B — the unattached grant.
    @Test func theRejectionTextEchoesTheAskedPathButNeverAnUnattachedGrant() {
        let grants = [
            FolderGrant(path: "/Users/me/grant-a", bookmark: Data()),
            FolderGrant(path: "/Users/me/grant-b", bookmark: Data()),
        ]
        let scope = FileScope.forConversation(
            attached: ["/Users/me/grant-a"], workingFolder: "/Users/me/grant-a",
            grants: grants, edition: .appStore)
        let message = FileScope.$current.withValue(scope) {
            ScopedFileTools.rejection(
                for: "/Users/me/grant-c/file.txt",
                roots: [URL(fileURLWithPath: "/Users/me/grant-a", isDirectory: true)])
        }
        #expect(message.contains("/Users/me/grant-c/file.txt"))   // the echoed request
        #expect(message.contains("/Users/me/grant-a"))            // the attached folder
        #expect(!message.contains("/Users/me/grant-b"))           // never the unattached grant
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
