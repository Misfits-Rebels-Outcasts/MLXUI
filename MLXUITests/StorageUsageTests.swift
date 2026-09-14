import Testing
import Foundation
@testable import MLXUI

/// SET-6's G2 seam (`RSI/DelegateSettingsBacklog.md`) — `ModelStore.directorySize(at:)`, the
/// pure, `nonisolated` function behind the Models pane's storage row. No main-actor work: the
/// pane calls this off the main actor (`Task.detached`) so a large `models/` tree can't stall
/// the window opening.
struct StorageUsageTests {

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test func emptyDirectoryIsZero() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelStore.directorySize(at: dir) == 0)
    }

    @Test func sumsFilesAcrossNestedSubdirectories() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let modelA = dir.appendingPathComponent("model-a", isDirectory: true)
        try FileManager.default.createDirectory(at: modelA, withIntermediateDirectories: true)

        try Data(repeating: 0, count: 100).write(to: dir.appendingPathComponent("top-level.bin"))
        try Data(repeating: 0, count: 250).write(to: modelA.appendingPathComponent("weights.bin"))

        #expect(ModelStore.directorySize(at: dir) == 350)
    }

    /// A subdirectory the enumerator can't descend into (permission denied) must not abort
    /// the whole count — a partial total beats a blank pane.
    @Test func skipsAnUnreadableChildRatherThanStoppingEnumeration() throws {
        let dir = tempDir()
        let locked = dir.appendingPathComponent("locked", isDirectory: true)
        try FileManager.default.createDirectory(at: locked, withIntermediateDirectories: true)
        try Data(repeating: 0, count: 999).write(to: locked.appendingPathComponent("hidden.bin"))
        try Data(repeating: 0, count: 100).write(to: dir.appendingPathComponent("readable.bin"))
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: locked.path)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path)
            try? FileManager.default.removeItem(at: dir)
        }

        // Never crashes, and still counts what it can read.
        #expect(ModelStore.directorySize(at: dir) >= 100)
    }

    @Test func missingDirectoryIsZero() {
        let dir = tempDir().appendingPathComponent("does-not-exist")
        #expect(ModelStore.directorySize(at: dir) == 0)
    }
}
