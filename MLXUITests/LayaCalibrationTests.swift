import Testing
@testable import MLXUI

struct LayaCalibrationTests {
    @Test func temperatureBucketsMatchPythonThresholds() {
        #expect(LayaCalibration.temperatureBucket(type: .choice, optionCount: 2) == "choice:2")
        #expect(LayaCalibration.temperatureBucket(type: .choice, optionCount: 5) == "choice:3-5")
        #expect(LayaCalibration.temperatureBucket(type: .choice, optionCount: 10) == "choice:6-10")
        #expect(LayaCalibration.temperatureBucket(type: .choice, optionCount: 20) == "choice:11+")
        #expect(LayaCalibration.temperatureBucket(type: .score, optionCount: 3) == "score:3-5")
        #expect(LayaCalibration.temperatureBucket(type: .noul, optionCount: 2) == "noul:2")
    }

    // The shipped checkpoint's `choice:11+` bucket is 0.1006 — clamps to `temperatureMin`
    // (§0 of `RSI/DelegateLayaBacklog.md`).
    @Test func clampTemperatureConfinesToRange() {
        #expect(LayaCalibration.clampTemperature(0.1006) == LayaCalibration.temperatureMin)
        #expect(LayaCalibration.clampTemperature(10) == LayaCalibration.temperatureMax)
        #expect(LayaCalibration.clampTemperature(1.25) == 1.25)
        #expect(LayaCalibration.clampTemperature(.nan) == 1.0)
    }

    @Test func confidenceFromProbsIsOneMinusNormalizedEntropy() {
        #expect(abs(LayaCalibration.confidence(fromProbabilities: [1, 0], optionCount: 2) - 1.0) < 1e-9)
        #expect(abs(LayaCalibration.confidence(fromProbabilities: [0.5, 0.5], optionCount: 2)) < 1e-9)
    }

    @Test func confidenceForSingleOptionIsAlwaysOne() {
        #expect(LayaCalibration.confidence(fromProbabilities: [1.0], optionCount: 1) == 1.0)
    }
}
