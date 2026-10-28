import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXUI

/// RIFE-3a: the IFNet port. The op tests need no weights; `RIFERealLoad` tests run when `MLXUI_TEST_RIFE_DIR=<folder with
/// config.json + model.safetensors>` (passed as `TEST_RUNNER_MLXUI_TEST_RIFE_DIR`) or the model is installed. A skipped run does not count.
enum RIFERealLoad {
    nonisolated static let directory: URL? = {
        let env = ProcessInfo.processInfo.environment
        let url = env["MLXUI_TEST_RIFE_DIR"].map { URL(fileURLWithPath: $0) }
            ?? ModelStore.shared.directory(forHFModelID: "mlx-community/RIFE-4.25")
        return FileManager.default.fileExists(atPath: url.appendingPathComponent("model.safetensors").path) ? url : nil
    }()
    nonisolated static let enabled = directory != nil
}

struct RIFENetOpsTests {

    private func values(_ a: MLXArray) -> [Float] { a.asType(.float32).asArray(Float.self) }

    @Test func parameterNamesAndCountMatchTheCheckpointLayout() {
        let net = RIFEIFNet()
        let keys = Set(net.parameters().flattened().map(\.0))
        #expect(keys.count == 158, "the published checkpoint has 158 tensors")
        #expect(keys.contains("block0.conv0.0.conv.weight"))
        #expect(keys.contains("block4.convblock.7.beta"))
        #expect(keys.contains("block2.lastconv.bias"))
        #expect(keys.contains("encode.cnn3.weight"))
        let count = net.parameters().flattened().reduce(0) { $0 + $1.1.size }
        #expect(count == 5_664_776, "parameter count from the safetensors header")
    }

    @Test func parameterShapesMatchTheCheckpointLayout() {
        let p = Dictionary(uniqueKeysWithValues: RIFEIFNet().parameters().flattened())
        #expect(p["block0.conv0.0.conv.weight"]?.shape == [96, 3, 3, 15])
        #expect(p["block1.conv0.0.conv.weight"]?.shape == [64, 3, 3, 28])
        #expect(p["block0.convblock.0.beta"]?.shape == [1, 1, 1, 192])
        #expect(p["block0.lastconv.weight"]?.shape == [52, 4, 4, 192])
        #expect(p["encode.cnn3.weight"]?.shape == [4, 4, 4, 16])
        #expect(p["encode.cnn0.weight"]?.shape == [16, 3, 3, 3])
    }

    /// (C, r, r) channel split: input channel c·r² + i·r + j lands at output pixel (i, j) of channel c.
    @Test func pixelShuffleMatchesTorchOrdering() {
        let x = MLXArray(0 ..< 8).asType(.float32).reshaped(1, 1, 1, 8)   // C = 2, r = 2
        let y = RIFEOps.pixelShuffle(x, 2)
        #expect(y.shape == [1, 2, 2, 2])
        #expect(values(y) == [0, 4, 1, 5, 2, 6, 3, 7])
    }

    @Test func resizeIsIdentityAtScaleOneAndAveragesWhenHalved() {
        let x = MLXArray(0 ..< 16).asType(.float32).reshaped(1, 4, 4, 1)
        #expect(values(RIFEOps.resizeBilinear(x, scaleFactor: 1)) == values(x))
        let half = RIFEOps.resizeBilinear(x, scaleFactor: 0.5)
        #expect(half.shape == [1, 2, 2, 1])
        // align_corners=False halving samples at the centre of each 2x2 block: the mean 2.5 + 4·row-ish. Pixel (0,0): mean of 0,1,4,5.
        #expect(abs(values(half)[0] - 2.5) < 1e-5)
        let up = RIFEOps.resizeBilinear(half, scaleFactor: 2)
        #expect(up.shape == [1, 4, 4, 1])
    }

    @Test func warpWithZeroFlowIsIdentityAndIntegerFlowShiftsWithBorderClamp() {
        let img = MLXArray(0 ..< 12).asType(.float32).reshaped(1, 3, 4, 1)   // rows of 0..3, 4..7, 8..11
        let zero = MLXArray.zeros([1, 3, 4, 2])
        // Not bit-exact: (g + 1)·0.5·(size-1) can land a ulp under an integer, as it does in torch; the weights absorb it.
        #expect(zip(values(RIFEOps.warp(img, flow: zero)), values(img)).allSatisfy { abs($0 - $1) < 1e-4 })
        // flow x = +1 samples the pixel one to the right; the last column clamps to the border.
        let shift = concatenated([MLXArray.ones([1, 3, 4, 1]), MLXArray.zeros([1, 3, 4, 1])], axis: -1)
        let out = values(RIFEOps.warp(img, flow: shift))
        let expected: [Float] = [1, 2, 3, 3, 5, 6, 7, 7, 9, 10, 11, 11]
        #expect(zip(out, expected).allSatisfy { abs($0 - $1) < 1e-4 })
    }

    @Test func gridSampleInterpolatesBilinearly() {
        let img = MLXArray([0, 10, 20, 30] as [Float]).reshaped(1, 2, 2, 1)
        // centre of the 2x2 image is the mean of the four corners.
        let grid = MLXArray([0, 0] as [Float]).reshaped(1, 1, 1, 2)
        #expect(abs(values(RIFEOps.gridSampleBorder(img, grid: grid))[0] - 15) < 1e-5)
    }

    @Test func configValidationRefusesAnotherNetwork() throws {
        let good = RIFEConfig(name: "4.25", scaleList: [16, 8, 4, 2, 1], blockChannels: [192, 128, 96, 64, 32],
                              blockInPlanes: [15, 28, 28, 28, 28], padTo: 64)
        try good.validate()
        let bad = RIFEConfig(name: "4.0", scaleList: [8, 4, 2, 1], blockChannels: [192, 128, 96, 64],
                             blockInPlanes: [15, 28, 28, 28], padTo: 32)
        #expect(throws: RIFEEngineError.self) { try bad.validate() }
    }

    @Test func missingFilesAreReportedByName() {
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("rife-empty-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(throws: RIFEEngineError.self) { _ = try RIFEEngine.load(directory: empty) }
    }
}

@Suite(.enabled(if: RIFERealLoad.enabled, "needs MLXUI_TEST_RIFE_DIR or an installed mlx-community/RIFE-4.25"))
struct RIFERealLoadTests {

    /// Strict load: every key present with the right shape and no extras (`verify: .all`), config accepted.
    @Test func checkpointLoadsStrictly() throws {
        let dir = try #require(RIFERealLoad.directory)
        let loaded = try RIFEEngine.load(directory: dir)
        #expect(loaded.config.name == "4.25")
        let count = loaded.net.parameters().flattened().reduce(0) { $0 + $1.1.size }
        #expect(count == 5_664_776)
    }

    /// Identical frames: the blend of two identical warps of the same image is that image, whatever the flow; and the output
    /// has the input's size for a size that needs padding (70 x 90 is not a multiple of 64).
    @Test func identicalFramesComeBackUnchangedAndPaddingIsCropped() throws {
        let dir = try #require(RIFERealLoad.directory)
        let loaded = try RIFEEngine.load(directory: dir)
        let frame = MLXRandom.uniform(low: Float(0), high: Float(1), [1, 70, 90, 3], key: MLXRandom.key(7))
        let out = try RIFEEngine.interpolate(loaded, frame, frame, timestep: 0.5, scale: 1)
        eval(out)
        #expect(out.shape == [1, 70, 90, 3])
        let drift = abs(out - frame).max().item(Float.self)
        #expect(drift < 0.35, "identical frames should stay near-identical, got max |Δ| \(drift)")
    }

    @Test func halfScaleRunsOnAPaddedSize() throws {
        let dir = try #require(RIFERealLoad.directory)
        let loaded = try RIFEEngine.load(directory: dir)
        let a = MLXRandom.uniform(low: Float(0), high: Float(1), [1, 100, 140, 3], key: MLXRandom.key(1))
        let b = MLXRandom.uniform(low: Float(0), high: Float(1), [1, 100, 140, 3], key: MLXRandom.key(2))
        let out = try RIFEEngine.interpolate(loaded, a, b, timestep: 0.5, scale: 0.5)
        eval(out)
        #expect(out.shape == [1, 100, 140, 3])
        #expect(out.min().item(Float.self) > -0.5 && out.max().item(Float.self) < 1.5)
    }
}
