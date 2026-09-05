import Testing
import Foundation
@testable import MLXUI

/// MoC-5-4 (`RSI/DelegateMoCBacklog.md`) — the progress UI fans out. One download now feeds
/// two cards: both must show its progress, and starting an install from either must not
/// start a second physical download.
///
/// `InstallManager` has no injectable network layer, so a genuine in-flight download can
/// only be observed by actually calling `install(_:)`, which fires a real (tiny, nonexistent
/// repo, fast-failing) HF API request in the background — the same tolerance this project
/// already extends to `scripts/validate_browser_json.py`. The assertions below only depend
/// on the **synchronous** state `install(_:)` sets before any `await` point, so they don't
/// wait on that network call at all; `cancel` stops it immediately afterward regardless.
struct InstallManagerProgressFanOutTests {

    private func makeTempStore() -> (store: ModelStore, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-4-\(UUID().uuidString)", isDirectory: true)
        let store = ModelStore(baseDirectory: tempDir)
        return (store, { try? FileManager.default.removeItem(at: tempDir) })
    }

    /// A second card naming the same repo joins the first's in-flight download — its
    /// `modelStates` entry mirrors the first's current state immediately, rather than
    /// starting its own independent download.
    @Test func secondCardOnTheSameRepoJoinsAndMirrorsCurrentState() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let cardA = makeEntry(id: "mlx-community--Test-Model-A", downloadSizeGB: 0.01)
        let cardB = makeEntry(id: "mlx-community--Test-Model-B", downloadSizeGB: 0.01)
        #expect(cardA.hfModelId == cardB.hfModelId, "the fixture assumes both cards share one repo")

        manager.install(cardA)
        #expect(manager.modelStates[cardA.id] == .resolving)

        manager.install(cardB)
        #expect(manager.modelStates[cardB.id] == .resolving,
                "cardB should mirror cardA's in-flight state, not sit at .idle waiting on its own download")

        manager.cancel(cardA)
        #expect(manager.modelStates[cardA.id] == .idle)
        #expect(manager.modelStates[cardB.id] == .idle, "cancelling the shared download clears both watchers")
    }

    /// The pre-existing guard (`InstallManager.swift`'s `install` entry check) still refuses
    /// a genuine double-start for the **same** card — unrelated to the new repo-joining path,
    /// confirmed unchanged.
    @Test func callingInstallTwiceForTheSameCardDoesNotRestart() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let card = makeEntry(id: "mlx-community--Test-Model-Solo", downloadSizeGB: 0.01)
        manager.install(card)
        #expect(manager.modelStates[card.id] == .resolving)

        manager.install(card)   // no-op: state isn't .idle/.error/.needsAuth
        #expect(manager.modelStates[card.id] == .resolving)

        manager.cancel(card)
    }

    /// A card with no sibling at all behaves exactly as before MoC-5-4 — install, cancel,
    /// no repo-sharing machinery visibly involved.
    @Test func soloCardInstallAndCancelUnaffectedByRepoTracking() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let solo = makeEntry(id: "mlx-community--Solo-Only", downloadSizeGB: 0.01)
        manager.install(solo)
        #expect(manager.modelStates[solo.id] == .resolving)
        manager.cancel(solo)
        #expect(manager.modelStates[solo.id] == .idle)
    }
}
