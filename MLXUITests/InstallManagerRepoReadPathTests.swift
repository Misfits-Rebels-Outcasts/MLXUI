import Testing
import Foundation
@testable import MLXUI

/// MoC-5-FIX-1 (`RSI/DelegateMoCBacklog.md`) — the installed-state **read** path resolves by
/// repo, matching where `downloadModel` **writes** the `.installed` marker.
///
/// The pair here is the **real shipping pair** from the bundled `browser.json`
/// (`Qwen3.5 9B` / `Qwen3.5 9B Vision`, one `hfModelId`), not a constructed fixture — a
/// fixture pair can be given any ids and would pass without proving the shipping case
/// (R5-FIX reviewer check). The one card whose `id` diverges from its repo slug is exactly
/// the entry the whole MoC-5/MoC-6 arc existed to make work.
struct InstallManagerRepoReadPathTests {

    private func loadBrowserData() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
    }

    private func makeTempStore() -> (store: ModelStore, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-fix-\(UUID().uuidString)", isDirectory: true)
        let store = ModelStore(baseDirectory: tempDir)
        return (store, { try? FileManager.default.removeItem(at: tempDir) })
    }

    /// Writes the on-disk result of a completed install for `hfModelId` — the repo directory
    /// plus the `.installed` marker, exactly where `downloadModel`'s atomic move puts them.
    private func simulateInstalled(_ hfModelId: String, in store: ModelStore) throws {
        let dir = store.directory(forHFModelID: hfModelId)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: store.installedMarker(forHFModelID: hfModelId).path, contents: nil)
    }

    private func sharedRepoPair(from data: BrowserData) throws -> (llm: ModelEntry, vision: ModelEntry) {
        let catalog = data.domains.flatMap { $0.allModels }
        let llm = try #require(catalog.first { $0.id == "mlx-community--Qwen3.5-9B-MLX-4bit" })
        let vision = try #require(catalog.first { $0.id == "mlx-community--Qwen3.5-9B-MLX-4bit-vision" })
        #expect(llm.hfModelId == vision.hfModelId)
        #expect(vision.id != ModelStore.repoSlug(for: vision.hfModelId))
        return (llm, vision)
    }

    // MARK: - isInstalled

    /// Installing the `llm` card leaves the `vision` card reading Installed too — they share
    /// one download, so one marker on disk answers for both. This is the symptom MoC-6-1's
    /// done-when named: "installing either shows the other as Installed".
    @Test func installingOneCardMakesBothReadInstalled() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let data = try loadBrowserData()
        let (llm, vision) = try sharedRepoPair(from: data)
        let manager = InstallManager(store: store)

        #expect(!manager.isInstalled(llm))
        #expect(!manager.isInstalled(vision))

        try simulateInstalled(llm.hfModelId, in: store)

        #expect(manager.isInstalled(llm))
        #expect(manager.isInstalled(vision), "the vision card shares the llm card's download — it must read Installed")
    }

    // MARK: - loadInstalled across a relaunch

    /// A fresh `InstallManager` (no in-memory `modelStates` — the relaunch case) reconciles
    /// the registry against disk and keeps **both** ids, because it resolves each to the
    /// shared repo via the catalog. On the card-keyed read path the vision id was logged as
    /// stale and dropped here.
    @Test func loadInstalledKeepsBothAcrossASimulatedRelaunch() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let data = try loadBrowserData()
        let (llm, vision) = try sharedRepoPair(from: data)
        try simulateInstalled(llm.hfModelId, in: store)

        // Fresh manager: nothing in modelStates, so this genuinely goes through the on-disk
        // marker check, not a cached "installed" flag.
        let relaunched = InstallManager(store: store)
        let verified = relaunched.loadInstalled(
            modelIDs: [llm.id, vision.id],
            catalog: data.domains.flatMap { $0.allModels })

        #expect(verified == [llm.id, vision.id])
    }

    /// The empty-catalog call (`AppState.init()` before `browser.json` is decoded) still
    /// resolves the single-card entry, and `AppState.loadBrowserData()`'s re-run with the
    /// catalog then picks up the shared-repo sibling.
    @Test func loadInstalledWithoutCatalogResolvesSingleCardThenTheReRunResolvesTheSibling() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let data = try loadBrowserData()
        let (llm, vision) = try sharedRepoPair(from: data)
        try simulateInstalled(llm.hfModelId, in: store)
        let manager = InstallManager(store: store)

        // Pass 1 — no catalog (init order). The llm card resolves card-keyed, which is also
        // its repo path; the vision card can't be mapped to the repo yet.
        let firstPass = manager.loadInstalled(modelIDs: [llm.id, vision.id], catalog: [])
        #expect(firstPass == [llm.id])

        // Pass 2 — catalog available (loadBrowserData re-run). Now the sibling resolves.
        let secondPass = manager.loadInstalled(
            modelIDs: [llm.id, vision.id],
            catalog: data.domains.flatMap { $0.allModels })
        #expect(secondPass == [llm.id, vision.id])
    }

    // MARK: - saveRegistry

    /// `saveRegistry` records **both** cards, each with the **repo** path — not `models/<own
    /// id>`, which for the vision card points at a directory nothing ever wrote.
    @Test func saveRegistryRecordsBothWithTheRepoPath() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let data = try loadBrowserData()
        let (llm, vision) = try sharedRepoPair(from: data)
        try simulateInstalled(llm.hfModelId, in: store)
        let manager = InstallManager(store: store)

        manager.saveRegistry(installedIDs: [llm.id, vision.id], browserData: data)

        let registry = try JSONDecoder().decode(
            InstalledModels.self, from: Data(contentsOf: store.installedRegistryURL))
        let repoSlug = ModelStore.repoSlug(for: llm.hfModelId)
        #expect(registry.models[llm.id]?.path == "models/\(repoSlug)")
        #expect(registry.models[vision.id]?.path == "models/\(repoSlug)",
                "the vision card's recorded path must be the shared repo directory, not models/\(vision.id)")
        #expect((registry.models[vision.id]?.sizeBytes ?? -1) >= 0)
    }

    // MARK: - single-card entries unaffected

    /// Every shipping entry whose `id` still equals its repo slug resolves through
    /// `isInstalled`/`loadInstalled` to the byte-identical marker path it did before — the
    /// regression that would silently orphan every existing user's downloads.
    @Test func singleCardEntriesResolveToTheSameMarkerAsBefore() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let data = try loadBrowserData()
        let catalog = data.domains.flatMap { $0.allModels }
        let manager = InstallManager(store: store)

        for entry in catalog where entry.id == ModelStore.repoSlug(for: entry.hfModelId) {
            try simulateInstalled(entry.hfModelId, in: store)
            #expect(manager.isInstalled(entry), "\(entry.id) should read Installed at its repo-keyed marker")
            let verified = InstallManager(store: store)
                .loadInstalled(modelIDs: [entry.id], catalog: catalog)
            #expect(verified == [entry.id], "\(entry.id) should survive reconciliation")
            try? FileManager.default.removeItem(at: store.directory(forHFModelID: entry.hfModelId))
        }
    }
}
