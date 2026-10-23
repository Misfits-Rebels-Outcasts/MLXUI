import Foundation
import Testing
@testable import MLXUI

/// WW3-1-FIX-1 — flows see a card as installed exactly when Browse does. Two catalog cards can share one download
/// (`Gemma 4 E4B` / `…-vision`, `Qwen3.5-9B-MLX-4bit` / `…-vision`, MoC-5) but `installed.json` registers only the card
/// that was installed, so `ModelSlot.readiness(installedModelIDs:)` called the sibling `.needsDownload` and the editor
/// said "Install Required Models" beside a Browse card reading "Installed" (smoke row 113 step 1, 2026-10-23).
/// `InstallManager.readyModelIDs` is the one rule; uninstall keeps counting the registry (MoC-5-2).
/// Written to extend, not edit, `InstallManagerReferenceCountingTests` and `Gemma4WorkflowWiringTests`.
struct FlowSharedDownloadReadinessTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
    }

    private func makeTempStore() -> (store: ModelStore, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ww3-1-fix-1-\(UUID().uuidString)", isDirectory: true)
        let store = ModelStore(baseDirectory: tempDir)
        return (store, { try? FileManager.default.removeItem(at: tempDir) })
    }

    /// The layout a finished install leaves: the repo-keyed directory and its marker.
    private func simulateInstalled(hfModelID: String, store: ModelStore) throws {
        try FileManager.default.createDirectory(at: store.directory(forHFModelID: hfModelID),
                                                withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forHFModelID: hfModelID).path, contents: nil)
    }

    private func describeImageFlow(_ display: String) -> FlowDocument {
        FlowDocument(version: "0.8", rows: [Row(task: "Describe Image", model: display, settings: nil)])
    }

    /// The preflight the editor runs, with `installed` as the set it is handed.
    private func preflight(_ display: String, catalog: [ModelEntry], installed: Set<String>) -> FlowPreflight.Result {
        FlowPreflight.run(describeImageFlow(display), catalog: catalog, installedModelIDs: installed,
                          totalRAMGB: 64, claimableModelIDs: Set(catalog.map(\.id)))
    }

    // MARK: - (a), (b) the two shared downloads

    @Test(arguments: [
        ("Gemma 4 E4B Vision", "mlx-community/gemma-4-e4b-it-4bit",
         "mlx-community--gemma-4-e4b-it-4bit", "mlx-community--gemma-4-e4b-it-4bit-vision"),
        ("Qwen3.5 9B Vision", "mlx-community/Qwen3.5-9B-MLX-4bit",
         "mlx-community--Qwen3.5-9B-MLX-4bit", "mlx-community--Qwen3.5-9B-MLX-4bit-vision"),
    ])
    func onlyTheChatCardRegisteredStillMakesTheVisionCardReady(display: String, hfID: String,
                                                               chatID: String, visionID: String) throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)
        let catalog = try loadCatalog()
        try simulateInstalled(hfModelID: hfID, store: store)
        let registered: Set<String> = [chatID]          // exactly what installed.json holds today

        // The bug, stated: the registry alone leaves the Vision card to download.
        let before = preflight(display, catalog: catalog, installed: registered)
        #expect(before.downloadSet.map(\.id) == [visionID])

        // The fix: the ready set carries the sibling, so the preflight installs nothing.
        let ready = manager.readyModelIDs(registered: registered, catalog: catalog)
        #expect(ready.contains(visionID))
        let after = preflight(display, catalog: catalog, installed: ready)
        #expect(after.downloadSet.isEmpty)
        #expect(after.installed.compactMap { $0.model?.id } == [visionID])
        // The slot readiness the picker and run sheet read agrees.
        let card = try #require(catalog.first { $0.id == visionID })
        #expect(ModelSlot.cataloged(card).readiness(installedModelIDs: ready) == .ready)
    }

    @Test func aCardWhoseDownloadIsNotOnDiskIsStillNotReady() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)
        let catalog = try loadCatalog()
        // Only the chat card is registered, and its marker is gone (a stale registry): nothing is ready by sharing.
        let ready = manager.readyModelIDs(registered: [], catalog: catalog)
        #expect(ready.isEmpty)
        let result = preflight("Gemma 4 E4B Vision", catalog: catalog, installed: ready)
        #expect(result.downloadSet.map(\.id) == ["mlx-community--gemma-4-e4b-it-4bit-vision"])
    }

    // MARK: - (c) uninstall still counts the registry, not the ready set

    /// The ready set is a read-model for flows. Uninstall is handed the registry, so the sibling the marker made
    /// "ready" must not keep a shared download alive on its own: removing the one *registered* card deletes the files.
    @Test func theReadySetDoesNotMakeASharedDownloadUndeletable() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)
        let catalog = try loadCatalog()
        let hfID = "mlx-community/gemma-4-e4b-it-4bit"
        try simulateInstalled(hfModelID: hfID, store: store)
        let chat = try #require(catalog.first { $0.id == "mlx-community--gemma-4-e4b-it-4bit" })
        let registry: Set<String> = [chat.id]

        let ready = manager.readyModelIDs(registered: registry, catalog: catalog)
        #expect(ready.contains("mlx-community--gemma-4-e4b-it-4bit-vision"))
        #expect(registry == [chat.id], "computing the ready set must not touch the registry")

        manager.uninstall(chat, catalog: catalog, installedModelIDs: registry)

        #expect(!FileManager.default.fileExists(atPath: store.directory(forHFModelID: hfID).path))
        // …and with the files gone, flows stop seeing the sibling as ready too.
        #expect(!manager.readyModelIDs(registered: [], catalog: catalog).contains("mlx-community--gemma-4-e4b-it-4bit-vision"))
    }

    /// Both cards registered (the MoC-5-4 fan-out case): removing one keeps the files, as before.
    @Test func uninstallStillKeepsFilesWhileAnotherRegisteredCardUsesThem() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)
        let catalog = try loadCatalog()
        let hfID = "mlx-community/Qwen3.5-9B-MLX-4bit"
        try simulateInstalled(hfModelID: hfID, store: store)
        let chat = try #require(catalog.first { $0.id == "mlx-community--Qwen3.5-9B-MLX-4bit" })
        let registry: Set<String> = [chat.id, "mlx-community--Qwen3.5-9B-MLX-4bit-vision"]

        manager.uninstall(chat, catalog: catalog, installedModelIDs: registry)

        #expect(FileManager.default.fileExists(atPath: store.installedMarker(forHFModelID: hfID).path))
    }

    // MARK: - (d) a single-card model is unchanged

    @Test func aSingleCardModelIsReadyExactlyWhenItWasBefore() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)
        let catalog = try loadCatalog()
        let solo = try #require(catalog.first { $0.id == "mlx-community--all-MiniLM-L6-v2-4bit" })
        #expect(catalog.filter { $0.hfModelId == solo.hfModelId }.count == 1, "the fixture assumes a single-card repo")

        // Registered and on disk: ready, as before.
        try simulateInstalled(hfModelID: solo.hfModelId, store: store)
        #expect(manager.readyModelIDs(registered: [solo.id], catalog: catalog).contains(solo.id))
        // Registered ids are never dropped by the helper.
        #expect(manager.readyModelIDs(registered: ["not-in-catalog"], catalog: catalog).contains("not-in-catalog"))
        // Nothing else is invented: a catalog of 47 with one download on disk yields that one card.
        #expect(manager.readyModelIDs(registered: [], catalog: catalog) == [solo.id])
    }
}
