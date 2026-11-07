import Foundation
import CoreGraphics
import MLX
import Testing
@testable import MLXUI

/// MC3-11 — where does the non-MLX memory go? Footprint (`phys_footprint`) and MLX active/cache after each stage of
/// the Run path, one phase per process. Opt-in: `MLXUI_TEST_DEPTH=1`, `MLXUI_TEST_DEPTH_DIR`, `MLXUI_TEST_DEPTH_PROBE=1`.
@Suite(.serialized)
struct DepthAnythingStageProbeTests {
    static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return DepthAnythingRealLoad.canRun
            && (env["MLXUI_TEST_DEPTH_PROBE"] ?? env["TEST_RUNNER_MLXUI_TEST_DEPTH_PROBE"]) == "1"
    }

    @Test(.enabled(if: DepthAnythingStageProbeTests.enabled), arguments: DepthAnythingFootprintTests.phases)
    func stages(phase: String) throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let source = try DepthAnythingFootprintTests.source(phase: phase)
        Memory.clearCache()
        let tracker = FootprintTracker()
        let mib = 1_048_576.0
        var lines: [String] = []
        func mark(_ name: String) {
            let now = FootprintTracker.phys()
            tracker.note(now)
            let snap = Memory.snapshot()
            lines.append(String(format: "STAGEPROBE phase=%@ stage=%@ footMiB=%.0f runPeakMiB=%.0f mlxActiveMiB=%.0f mlxCacheMiB=%.0f mlxPeakMiB=%.0f",
                                phase, name, Double(now) / mib, Double(tracker.runPeak) / mib,
                                Double(snap.activeMemory) / mib, Double(snap.cacheMemory) / mib, Double(snap.peakMemory) / mib))
        }
        let sampler = Task.detached {
            while !Task.isCancelled { tracker.note(FootprintTracker.phys()); try? await Task.sleep(for: .milliseconds(10)) }
        }
        mark("start")
        let image = try #require(DepthAnythingImage.decoded(source)); mark("decoded")
        let prepared = try DepthAnythingInput.prepare(image: image); mark("prepared")
        let loaded = try DepthAnythingEngine.load(directory: directory); mark("loaded")
        let input = try DepthAnythingPreprocessor.preprocess(image: prepared.image); eval(input); mark("preprocessed")
        let prediction = loaded.model.predictedDepth(input); eval(prediction); mark("model")
        let gray = try DepthAnythingPostprocessor.grayscale(prediction: prediction, height: prepared.height, width: prepared.width); mark("grayscale")
        let grayImage = try DepthAnythingPostprocessor.image(gray: gray, width: prepared.width, height: prepared.height); mark("cgimage")
        let png = PNGEncoder.pngData(from: grayImage); mark("png")
        sampler.cancel()
        let out = lines.joined(separator: "\n") + "\n"
        print(out)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("DEPTHPROBE.txt")
        try? (out.data(using: .utf8) ?? Data()).write(to: url, options: .atomic)
        #expect(png != nil)
    }
}
