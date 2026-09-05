import Testing
import Foundation
@testable import MLXUI

/// MoC-5-2 (`RSI/DelegateMoCBacklog.md`) — uninstall learns to count. Two catalog cards
/// naming the same HF repo (`makeEntry`'s fixed `hfModelId`, "mlx-community/Test-4bit" —
/// exactly the future MoC-6 shape: distinct `id`s, one repo) must not let removing one break
/// the other, and removing the last one actually deletes the shared directory.
struct InstallManagerReferenceCountingTests {

    private func makeTempStore() -> (store: ModelStore, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-2-\(UUID().uuidString)", isDirectory: true)
        let store = ModelStore(baseDirectory: tempDir)
        return (store, { try? FileManager.default.removeItem(at: tempDir) })
    }

    /// Simulates a completed install by writing the directory + marker directly at the
    /// repo path — `downloadModel`'s actual network path isn't exercised here, only the
    /// storage layout it leaves behind.
    private func simulateInstalled(_ model: ModelEntry, store: ModelStore) throws {
        let dir = store.directory(forHFModelID: model.hfModelId)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forHFModelID: model.hfModelId).path, contents: nil)
    }

    @Test func removingTheFirstOfTwoCardsLeavesTheSecondsFilesIntact() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let cardA = makeEntry(id: "mlx-community--Test-Model-A")
        let cardB = makeEntry(id: "mlx-community--Test-Model-B")
        #expect(cardA.hfModelId == cardB.hfModelId, "the fixture assumes both cards share one repo")

        try simulateInstalled(cardA, store: store)
        let catalog = [cardA, cardB]
        let installedIDs: Set<String> = [cardA.id, cardB.id]

        manager.uninstall(cardA, catalog: catalog, installedModelIDs: installedIDs)

        // The shared repo directory must still exist — card B is still "installed".
        let sharedDir = store.directory(forHFModelID: cardA.hfModelId)
        #expect(FileManager.default.fileExists(atPath: sharedDir.path))
        #expect(FileManager.default.fileExists(atPath: store.installedMarker(forHFModelID: cardB.hfModelId).path))
        // Card A's own in-memory state clears even though the files stay.
        #expect(manager.modelStates[cardA.id] == .idle)
    }

    @Test func removingTheLastCardOnARepoDeletesTheDirectory() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let cardA = makeEntry(id: "mlx-community--Test-Model-A")
        let cardB = makeEntry(id: "mlx-community--Test-Model-B")
        try simulateInstalled(cardA, store: store)
        let catalog = [cardA, cardB]

        // Card B is NOT in installedModelIDs — cardA is the only installed card left.
        manager.uninstall(cardA, catalog: catalog, installedModelIDs: [cardA.id])

        let sharedDir = store.directory(forHFModelID: cardA.hfModelId)
        #expect(!FileManager.default.fileExists(atPath: sharedDir.path))
    }

    @Test func singleCardModelUninstallsExactlyAsBefore() throws {
        // The common case — no sibling card at all. Reference-counting must be a no-op here.
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let solo = makeEntry(id: "mlx-community--Solo-Model")
        try simulateInstalled(solo, store: store)

        manager.uninstall(solo, catalog: [solo], installedModelIDs: [solo.id])

        #expect(!FileManager.default.fileExists(atPath: store.directory(forHFModelID: solo.hfModelId).path))
    }

    /// The `.installed` marker semantics are unchanged: it's still a plain file-existence
    /// check, at whatever path the model's storage actually resolves to — `uninstall`'s
    /// reference-counting only decides whether to *delete* that path, never how "is this
    /// installed" is answered.
    @Test func installedMarkerStaysAPlainExistenceCheckAtTheRepoPath() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let cardA = makeEntry(id: "mlx-community--Test-Model-A")
        let cardB = makeEntry(id: "mlx-community--Test-Model-B")
        try simulateInstalled(cardA, store: store)

        manager.uninstall(cardA, catalog: [cardA, cardB], installedModelIDs: [cardA.id, cardB.id])

        // The marker at the shared repo path still exists — kept, not deleted, because
        // cardB is still considered installed. This is the file both cards' storage
        // resolves to (MoC-5-1: repo-keyed), independent of either card's own `id`.
        #expect(FileManager.default.fileExists(atPath: store.installedMarker(forHFModelID: cardA.hfModelId).path))
    }
}
