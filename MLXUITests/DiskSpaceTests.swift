import Testing
@testable import MLXUI

/// Covers the pure disk-space pre-flight logic gating model installs (backlog T1).
struct DiskSpaceTests {

    @Test func requiredSpaceDoublesDownloadAndAddsBuffer() {
        // 4 GB download → 4 × 2 + 2 = 10 GB required.
        #expect(DiskSpace.requiredGB(downloadSizeGB: 4) == 10)
    }

    @Test func passesWhenAmpleSpace() {
        #expect(DiskSpace.preflightError(downloadSizeGB: 4, availableDiskGB: 50) == nil)
    }

    @Test func failsWhenTooLittleSpace() {
        #expect(DiskSpace.preflightError(downloadSizeGB: 4, availableDiskGB: 5) != nil)
    }

    @Test func passesAtExactBoundary() {
        // Required for a 4 GB model is exactly 10 GB; 10 GB available must NOT be blocked.
        #expect(DiskSpace.preflightError(downloadSizeGB: 4, availableDiskGB: 10) == nil)
    }

    @Test func failsJustBelowBoundary() {
        #expect(DiskSpace.preflightError(downloadSizeGB: 4, availableDiskGB: 9.99) != nil)
    }
}
