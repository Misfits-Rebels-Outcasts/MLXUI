import Testing
@testable import MLXUI

/// Covers the pure download-progress clamp that feeds `ProgressView(value:)` (backlog B2).
struct DownloadProgressTests {

    @Test func normalRatioPassesThrough() {
        #expect(DownloadProgress.fraction(downloaded: 50, total: 100) == 0.5)
    }

    @Test func clampsAboveOneToOne() {
        // downloaded > total (estimated total too low) must not exceed 1.0.
        #expect(DownloadProgress.fraction(downloaded: 150, total: 100) == 1.0)
    }

    @Test func zeroTotalIsZero() {
        #expect(DownloadProgress.fraction(downloaded: 0, total: 0) == 0)
    }

    @Test func negativeTotalIsZero() {
        #expect(DownloadProgress.fraction(downloaded: 10, total: -5) == 0)
    }

    @Test func exactCompletionIsOne() {
        #expect(DownloadProgress.fraction(downloaded: 100, total: 100) == 1.0)
    }
}
