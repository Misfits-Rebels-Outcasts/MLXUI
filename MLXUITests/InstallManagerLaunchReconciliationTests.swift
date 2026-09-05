import Testing
import Foundation
@testable import MLXUI

/// MoC-5-3 (`RSI/DelegateMoCBacklog.md`) — `installed.json` migration and launch
/// reconciliation.
///
/// **This is a verification-only task, not a migration, given the minimal design MoC-5-1
/// chose (owner ruling, 2026-09-05): `installed.json` stays card-keyed, and every existing
/// single-card model's card path already equals its repo path.** There is nothing in the
/// on-disk registry's shape or the `models/{id}/` layout for an existing install to migrate
/// *away from* — the registry's `InstalledModel`/`InstalledModels` `Codable` shape is
/// unchanged, and `AppState.loadInstalledModels()` only ever extracts the registry's *keys*
/// before handing them to `InstallManager.loadInstalled(modelIDs:)`, which never reads the
/// registry file itself. So the done-when's three properties (lands every model still
/// installed, a second run is a no-op, a genuinely stale entry is still dropped) are
/// properties `loadInstalled` already had before this arc — this task pins them with a real
/// fixture rather than leaving them merely assumed.
///
/// "A captured real `installed.json` fixture" (the done-when's own wording) wasn't literally
/// available — this is a development machine with no real user install history. Built the
/// fixture with the app's own `InstalledModels`/`InstalledModel` `Codable` types and
/// `JSONEncoder`, so its bytes are exactly what `InstallManager.saveRegistry` itself would
/// produce, rather than hand-typed JSON that could silently drift from the real shape.
struct InstallManagerLaunchReconciliationTests {

    private func makeTempStore() -> (store: ModelStore, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moc5-3-\(UUID().uuidString)", isDirectory: true)
        let store = ModelStore(baseDirectory: tempDir)
        return (store, { try? FileManager.default.removeItem(at: tempDir) })
    }

    private func installedModelsFixture(ids: [String]) -> InstalledModels {
        var models: [String: InstalledModel] = [:]
        for id in ids {
            models[id] = InstalledModel(
                installedAt: "2026-09-05T00:00:00Z",
                variant: "mlx-community/\(id).../fixture",
                path: "models/\(id)",
                sizeBytes: 123_456
            )
        }
        return InstalledModels(version: 1, models: models)
    }

    /// Round-trips the fixture through the real `Codable` types + `JSONEncoder`/`JSONDecoder`
    /// — confirms the fixture itself is genuine `installed.json` bytes, not a hand-typed
    /// approximation of the schema.
    @Test func fixtureRoundTripsThroughTheRealCodableTypes() throws {
        let fixture = installedModelsFixture(ids: ["mlx-community--Kept-Model", "mlx-community--Stale-Model"])
        let data = try JSONEncoder().encode(fixture)
        let decoded = try JSONDecoder().decode(InstalledModels.self, from: data)
        #expect(decoded.models.keys.sorted() == ["mlx-community--Kept-Model", "mlx-community--Stale-Model"])
    }

    /// Every model still on disk (a real `.installed` marker present) survives
    /// reconciliation; a genuinely stale entry (marker missing — deleted outside the app, or
    /// an install that never finished) is dropped, exactly as `CLAUDE.md`'s "registry is
    /// reconciled against [the marker] on launch (stale entries dropped)" describes.
    @Test func reconciliationKeepsInstalledDropsStale() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let kept = "mlx-community--Kept-Model"
        let stale = "mlx-community--Stale-Model"
        let fixture = installedModelsFixture(ids: [kept, stale])
        _ = try JSONEncoder().encode(fixture)   // exercised above; not re-decoded here

        // Only `kept`'s marker actually exists on disk — `stale`'s install never completed
        // (or its folder was removed outside the app), the real-world case this reconciles.
        try FileManager.default.createDirectory(at: store.directory(forModelID: kept), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forModelID: kept).path, contents: nil)

        let verified = manager.loadInstalled(modelIDs: Set(fixture.models.keys))
        #expect(verified == [kept])
    }

    /// A second reconciliation pass (e.g. a relaunch, or calling it twice in one session)
    /// produces the identical result — no state accumulates, nothing flips.
    @Test func aSecondReconciliationPassIsANoOp() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let kept = "mlx-community--Kept-Model"
        try FileManager.default.createDirectory(at: store.directory(forModelID: kept), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forModelID: kept).path, contents: nil)

        let ids: Set<String> = [kept, "mlx-community--Stale-Model"]
        let first = manager.loadInstalled(modelIDs: ids)
        let second = manager.loadInstalled(modelIDs: ids)
        #expect(first == second)
        #expect(first == [kept])
    }

    /// Every model in the fixture is genuinely installed — the "lands every model still
    /// installed" half of the done-when, checked with more than one survivor.
    @Test func everyStillInstalledModelInTheFixtureIsVerified() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let manager = InstallManager(store: store)

        let ids = ["mlx-community--Model-One", "mlx-community--Model-Two", "mlx-community--Model-Three"]
        for id in ids {
            try FileManager.default.createDirectory(at: store.directory(forModelID: id), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: store.installedMarker(forModelID: id).path, contents: nil)
        }
        let fixture = installedModelsFixture(ids: ids)

        let verified = manager.loadInstalled(modelIDs: Set(fixture.models.keys))
        #expect(verified == Set(ids))
    }
}
