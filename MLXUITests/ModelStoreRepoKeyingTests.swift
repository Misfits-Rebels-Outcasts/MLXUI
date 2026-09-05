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
            // MoC-6 introduced the first intentional exception: `Qwen3.5-9B Vision`'s own id
            // diverges from its hfModelId's repo slug on purpose — it shares the `Qwen3.5-9B`
            // (`llm`) card's download (MoC-5's install-path sharing). Every entry whose id
            // still equals its own repo slug must keep resolving identically by card and by
            // repo; `sharedRepoEntriesResolveToOneDirectory` below covers the exception itself.
            guard entry.id == ModelStore.repoSlug(for: entry.hfModelId) else { continue }
            let byCard = store.directory(forModelID: entry.id)
            let byRepo = store.directory(forHFModelID: entry.hfModelId)
            #expect(byCard == byRepo,
                    "\(entry.id): card path \(byCard.path) != repo path \(byRepo.path)")
        }
    }

    /// The MoC-6 exception itself: the vision card's own id resolves to a *different*
    /// directory than its repo path, but the repo path is exactly the llm sibling's — i.e.
    /// they genuinely share one directory on disk (R6's reviewer check).
    @Test func sharedRepoEntriesResolveToOneDirectory() throws {
        let catalog = try loadCatalog()
        let vision = try #require(catalog.first { $0.id == "mlx-community--Qwen3.5-9B-MLX-4bit-vision" })
        let llm = try #require(catalog.first { $0.id == "mlx-community--Qwen3.5-9B-MLX-4bit" })
        #expect(vision.hfModelId == llm.hfModelId)
        let store = ModelStore(baseDirectory: FileManager.default.temporaryDirectory)
        #expect(store.directory(forHFModelID: vision.hfModelId) == store.directory(forHFModelID: llm.hfModelId))
        #expect(store.directory(forModelID: vision.id) != store.directory(forHFModelID: vision.hfModelId))
    }

    @Test func repoSlugReplacesSlashesWithDoubleDash() {
        #expect(ModelStore.repoSlug(for: "mlx-community/Qwen3-8B-4bit") == "mlx-community--Qwen3-8B-4bit")
    }

    @Test func downloadDirectoryAndInstalledMarkerAlsoUnaffected() throws {
        let catalog = try loadCatalog()
        let store = ModelStore(baseDirectory: FileManager.default.temporaryDirectory)
        for entry in catalog {
            // MoC-6's one intentional exception — see `noPathMovedForAnyShippingCatalogEntry`.
            guard entry.id == ModelStore.repoSlug(for: entry.hfModelId) else { continue }
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
    ///
    /// **MoC-5-FIX-1: this test cannot catch the card-keyed-read-path defect.** It uses a
    /// card whose `id` equals its own repo slug (`hfModelId` = `mlx-community/Test-Model-4bit`
    /// → slug `mlx-community--Test-Model-4bit` = `id`), so the card-keyed and repo-keyed
    /// marker paths coincide and it passes whichever way the read path resolves. The
    /// divergent case — two cards, one repo, one `id` ≠ its repo slug — is covered by
    /// `InstallManagerRepoReadPathTests`.
    @Test func installManagerFindsAMarkerWrittenAtTheModelStorePath() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let store = ModelStore(baseDirectory: tempDir)
        let manager = InstallManager(store: store)

        let entry = makeEntry(id: "mlx-community--Test-Model-4bit",
                              hfModelId: "mlx-community/Test-Model-4bit")
        let modelDir = store.directory(forHFModelID: entry.hfModelId)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: store.installedMarker(forHFModelID: entry.hfModelId).path, contents: nil)

        #expect(manager.isInstalled(entry))
        let verified = manager.loadInstalled(modelIDs: [entry.id], catalog: [entry])
        #expect(verified.contains(entry.id))
    }

    @Test func installManagerReportsNotInstalledWhenMarkerMissing() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let store = ModelStore(baseDirectory: tempDir)
        let manager = InstallManager(store: store)

        #expect(!manager.isInstalled(makeEntry(id: "mlx-community--Never-Installed-4bit",
                                               hfModelId: "mlx-community/Never-Installed-4bit")))
    }
}
