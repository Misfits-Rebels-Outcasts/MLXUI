import Testing
import Foundation
@testable import MLXUI

/// MoC-5-1 (`RSI/DelegateMoCBacklog.md`) — storage keyed by repo, not by card.
/// `ModelStore` gains `directory(forHFModelID:)`; every existing call site must resolve to
/// the same path it does today for single-card models.
struct ModelStoreRepoKeyingTests {

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    // MARK: - No path moved, for every shipping catalog entry

    @Test func noPathMovedForAnyShippingCatalogEntry() throws {
        let catalog = try loadCatalog()
        #expect(!catalog.isEmpty)
        let store = ModelStore(baseDirectory: FileManager.default.temporaryDirectory)
        for entry in catalog {
            let byCard = store.directory(forModelID: entry.id)
            let byRepo = store.directory(forHFModelID: entry.hfModelId)
            #expect(byCard == byRepo,
                    "\(entry.id): card path \(byCard.path) != repo path \(byRepo.path)")
        }
    }

    @Test func repoSlugReplacesSlashesWithDoubleDash() {
        #expect(ModelStore.repoSlug(for: "mlx-community/Qwen3-8B-4bit") == "mlx-community--Qwen3-8B-4bit")
    }

    @Test func downloadDirectoryAndInstalledMarkerAlsoUnaffected() throws {
        let catalog = try loadCatalog()
        let store = ModelStore(baseDirectory: FileManager.default.temporaryDirectory)
        for entry in catalog {
            let cardSlug = entry.id
            let repoSlug = ModelStore.repoSlug(for: entry.hfModelId)
            #expect(store.downloadDirectory(forModelID: cardSlug) == store.downloadDirectory(forModelID: repoSlug))
            #expect(store.installedMarker(forModelID: cardSlug) == store.installedMarker(forModelID: repoSlug))
        }
    }

    // MARK: - InstallManager routes through ModelStore, not a hand-rolled path

    /// Confirms the refactor: `InstallManager` no longer builds `models/{id}/...` paths
    /// itself — a marker written where `ModelStore` says the model lives is what
    /// `isInstalled`/`loadInstalled` actually check, using the injectable `store` (MoC-5-1),
    /// not the real `Application Support/AI Browser/`.
    @Test func installManagerFindsAMarkerWrittenAtTheModelStorePath() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let store = ModelStore(baseDirectory: tempDir)
        let manager = InstallManager(store: store)

        let cardID = "mlx-community--Test-Model-4bit"
        let modelDir = store.directory(forModelID: cardID)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forModelID: cardID).path, contents: nil)

        #expect(manager.isInstalled(cardID))
        let verified = manager.loadInstalled(modelIDs: [cardID])
        #expect(verified.contains(cardID))
    }

    @Test func installManagerReportsNotInstalledWhenMarkerMissing() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let store = ModelStore(baseDirectory: tempDir)
        let manager = InstallManager(store: store)

        #expect(!manager.isInstalled("mlx-community--Never-Installed-4bit"))
    }
}
