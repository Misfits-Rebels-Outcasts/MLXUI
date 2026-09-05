import Testing
import Foundation
@testable import MLXUI

/// MoC-5-FIX-2 (`RSI/DelegateMoCBacklog.md`) — `install()` refuses to re-download what is
/// already on disk. Independent of the read-path keying (`InstallManagerRepoReadPathTests`):
/// this is the guard that turns the worst symptom of a stale badge from a 6 GB transfer into
/// a no-op, and it holds whatever the marker resolution does.
struct InstallManagerAlreadyOnDiskTests {

    private func makeTempStore() -> (store: ModelStore, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-fix2-\(UUID().uuidString)", isDirectory: true)
        let store = ModelStore(baseDirectory: tempDir)
        return (store, { try? FileManager.default.removeItem(at: tempDir) })
    }

    private func simulateInstalled(_ hfModelId: String, in store: ModelStore) throws {
        let dir = store.directory(forHFModelID: hfModelId)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: store.installedMarker(forHFModelID: hfModelId).path, contents: nil)
    }

    /// A second `install` for a card whose repo is already installed on disk starts **no**
    /// download — the card goes straight to `.installed` (never `.resolving`) and the
    /// completion closure fires so `AppState` records it.
    @Test func installForAnAlreadyOnDiskRepoStartsNoDownload() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let card = makeEntry(id: "mlx-community--Already-Here-4bit",
                             hfModelId: "mlx-community/Already-Here-4bit")
        try simulateInstalled(card.hfModelId, in: store)

        var completedWith: [String] = []
        manager.install(card) { completedWith.append($0) }

        #expect(manager.modelStates[card.id] == .installed)
        #expect(manager.modelStates[card.id] != .resolving)
        #expect(completedWith == [card.id])
    }

    /// The shared-repo case the guard exists for: the `llm` card's download is on disk, and
    /// clicking Install on the `vision` sibling (id ≠ repo slug) does not start a 5.97 GB
    /// re-fetch.
    @Test func installForASiblingWhoseRepoIsAlreadyInstalledIsANoOp() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let llm = makeEntry(id: "mlx-community--Shared-Repo-4bit",
                            hfModelId: "mlx-community/Shared-Repo-4bit")
        let vision = makeEntry(id: "mlx-community--Shared-Repo-4bit-vision", modelType: .vision,
                               hfModelId: "mlx-community/Shared-Repo-4bit")
        try simulateInstalled(llm.hfModelId, in: store)

        var completedWith: [String] = []
        manager.install(vision) { completedWith.append($0) }

        #expect(manager.modelStates[vision.id] == .installed)
        #expect(completedWith == [vision.id])
    }

    /// A genuinely absent model is unaffected — `install` proceeds to `.resolving` and starts
    /// its download as before.
    @Test func aGenuinelyAbsentModelStillDownloads() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let card = makeEntry(id: "mlx-community--Not-Here-4bit", downloadSizeGB: 0.01,
                             hfModelId: "mlx-community/Not-Here-4bit")
        manager.install(card)
        #expect(manager.modelStates[card.id] == .resolving)
        manager.cancel(card)
    }
}
