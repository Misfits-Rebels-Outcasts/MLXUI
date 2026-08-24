import Testing
import Foundation
@testable import MLXUI

/// CFM-R11-1 — the per-run memory recorder. The real measurement is a cold run (human smoke
/// row 30); these pin the recorder itself: the peak resets at run start, samples fold into
/// the run's peaks, and RSS is a real, positive process figure.
struct CatFlowRunMetricsTests {

    @Test func recordSamplesFoldIntoPeaks() {
        RunMetrics.resetPeak()
        var metrics = RunMetrics()
        for _ in 0..<3 {
            metrics.record(rowID: UUID())
        }
        // Sampling touches the real MLX snapshot and Mach RSS — both must be sane.
        #expect(metrics.peakGPU >= 0)
        #expect(metrics.rowSamples.count == 3)
        let peak = metrics.rowSamples.values.map(\.gpuPeak).max() ?? 0
        #expect(metrics.peakGPU == peak)
    }

    @Test func rssIsARealPositiveFigure() {
        let rss = currentRSSBytes()
        // The test process is alive; RSS must be well above zero.
        #expect(rss > 1_000_000)
    }

    @Test func summaryReadsAsBytes() {
        RunMetrics.resetPeak()
        var metrics = RunMetrics()
        metrics.record(rowID: UUID())
        let summary = metrics.summary()
        #expect(summary.contains("peak GPU"))
        #expect(summary.contains("peak RSS"))
        #expect(summary.contains("MB"))
    }
}
