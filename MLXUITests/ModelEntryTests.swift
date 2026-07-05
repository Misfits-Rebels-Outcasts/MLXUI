import Testing
@testable import MLXUI

/// Covers the pure formulas on `ModelEntry`: speed scaling, RAM fit, and quality score.
/// These are the highest-signal, UI-free pieces of logic in the app (see RSI/evals/eval-plan.md G2).
struct ModelEntryTests {

    // MARK: scaledSpeed(bandwidthGBps:)

    @Test func scaledSpeedAtBaselineBandwidthIsUnchanged() {
        // The stored speed is an M2 Max (400 GB/s) baseline; at 400 it should not scale.
        let entry = makeEntry(speedTokensPerSec: 100)
        #expect(entry.scaledSpeed(bandwidthGBps: 400) == 100)
    }

    @Test func scaledSpeedHalvesAtHalfBandwidth() {
        let entry = makeEntry(speedTokensPerSec: 100)
        #expect(entry.scaledSpeed(bandwidthGBps: 200) == 50)
    }

    @Test func scaledSpeedDoublesAtDoubleBandwidth() {
        let entry = makeEntry(speedTokensPerSec: 100)
        #expect(entry.scaledSpeed(bandwidthGBps: 800) == 200)
    }

    @Test func scaledSpeedIsNilWhenNoBaseline() {
        let entry = makeEntry(speedTokensPerSec: nil)
        #expect(entry.scaledSpeed(bandwidthGBps: 400) == nil)
    }

    // MARK: exceedsRAM(_:)

    @Test func exceedsRAMIsFalseWhenModelFits() {
        let entry = makeEntry(ramGB: 8)
        #expect(entry.exceedsRAM(16) == false)
    }

    @Test func exceedsRAMIsTrueWhenModelTooLarge() {
        let entry = makeEntry(ramGB: 20)
        #expect(entry.exceedsRAM(16) == true)
    }

    @Test func exceedsRAMIsFalseAtExactBoundary() {
        // ramGB > systemRAMGB — equal must NOT count as exceeding.
        let entry = makeEntry(ramGB: 16)
        #expect(entry.exceedsRAM(16) == false)
    }

    // MARK: qualityScore

    @Test func qualityScoreIsNilWithoutBenchmarks() {
        #expect(makeEntry(benchmarks: nil).qualityScore == nil)
    }

    @Test func qualityScoreAveragesAndClampsToFiveScale() {
        // mmlu 80 → 80/20 = 4.0 on the 1...5 scale.
        let b = ModelBenchmarks(mmlu: 80, humanEval: nil, gsm8k: nil,
                                hellaswag: nil, arc: nil, truthfulQA: nil)
        let score = makeEntry(benchmarks: b).qualityScore
        #expect(score == 4.0)
    }
}
