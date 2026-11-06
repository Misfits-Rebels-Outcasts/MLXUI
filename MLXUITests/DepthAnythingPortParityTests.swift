import Foundation
import CoreGraphics
import MLX
import MLXNN
import Testing
@testable import MLXUI

/// MC3-5 — the DPT neck, the depth head, the pre/post-processing and full-model parity.
/// Reference: `transformers` 5.12.0 / `torch` 2.7.0 (fixture README). Real-weights suites are opt-in
/// (`MLXUI_TEST_DEPTH=1` + `MLXUI_TEST_DEPTH_DIR`).
@Suite(.enabled(if: DepthAnythingRealLoad.canRun, "needs MLXUI_TEST_DEPTH=1 and MLXUI_TEST_DEPTH_DIR"))
struct DepthAnythingPortParityTests {

    private func loaded() throws -> DepthAnythingModel.Loaded {
        try DepthAnythingModel.load(directory: try #require(DepthAnythingRealLoad.directory))
    }

    /// Pearson correlation, computed in Double.
    private func pearson(_ a: [Float], _ b: [Float]) -> Double {
        let n = Double(a.count)
        let meanA = a.reduce(0) { $0 + Double($1) } / n
        let meanB = b.reduce(0) { $0 + Double($1) } / n
        var covariance = 0.0, varianceA = 0.0, varianceB = 0.0
        for (x, y) in zip(a, b) {
            let dx = Double(x) - meanA, dy = Double(y) - meanB
            covariance += dx * dy
            varianceA += dx * dx
            varianceB += dy * dy
        }
        return covariance / (varianceA * varianceB).squareRoot()
    }

    /// Compare an array with a raw fixture, printing the max |Δ| RIFE-style.
    private func compare(_ array: MLXArray, file: String, label: String) throws -> Float {
        let difference = try DepthAnythingFixtures.maxDifference(array, file: file)
        let line = "DEPTH PARITY \(label) max |Δ| = \(difference)"
        print(line)
        DepthAnythingFixtures.record(line)
        return difference
    }

    /// The whole published checkpoint loads strictly: 287 tensors, none missing, unused or misshaped.
    @Test func checkpointLoadsAllPublishedTensorsStrictly() throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let weights = try DepthAnythingWeights.loadArrays(directory: directory)
        #expect(weights.count == DepthAnythingModel.checkpointTensorCount)
        let loaded = try loaded()
        let parameters = loaded.model.parameters().flattened()
        #expect(parameters.count == DepthAnythingModel.checkpointTensorCount)
        let total = parameters.reduce(0) { $0 + $1.1.size }
        #expect(total == 24_785_089, "the card's 24.8 M parameters")
    }

    /// Every stage on the non-square 112 x 154 probe: the four reassemble outputs, the final fusion map and
    /// the prediction. A shape or `align_corners` mistake shows up here, before end-to-end.
    @Test func neckHeadAndFusionMatchTheReferenceOnTheProbe() throws {
        let loaded = try loaded()
        let input = try DepthAnythingFixtures.preprocessed("probe")
        let patchHeight = 112 / 14, patchWidth = 154 / 14
        let features = loaded.model.backboneFeatures(input)

        let reassembled = loaded.model.neck.reassembleStage(features, patchHeight: patchHeight, patchWidth: patchWidth)
        var worst: Float = 0
        for (index, tensor) in reassembled.enumerated() {
            let shape = try DepthAnythingFixtures.shape("probe", ["reassemble", String(index)])
            #expect(tensor.shape == shape, "reassemble \(index): \(tensor.shape) vs \(shape)")
            let file = try DepthAnythingFixtures.file(for: "probe", ["reassemble", String(index)])
            worst = max(worst, try compare(tensor, file: file, label: "reassemble \(index)"))
        }

        let projected = zip(reassembled, loaded.model.neck.convs).map { feature, conv in conv(feature) }
        let fused = loaded.model.neck.fusionStage(projected)
        let fusionShape = try DepthAnythingFixtures.shape("probe", ["fusion"])
        #expect(fused.last?.shape == fusionShape)
        worst = max(worst, try compare(try #require(fused.last), file: try DepthAnythingFixtures.file(for: "probe", ["fusion"]),
                                       label: "fusion"))

        let predicted = loaded.model.head(fused, patchHeight: patchHeight, patchWidth: patchWidth)
        let predictionShape = try DepthAnythingFixtures.shape("probe", ["predicted_depth"])
        #expect(predicted.shape == [1] + predictionShape, "prediction \(predicted.shape) vs \(predictionShape)")
        worst = max(worst, try compare(predicted, file: try DepthAnythingFixtures.file(for: "probe", ["predicted_depth"]),
                                       label: "predicted_depth"))
        #expect(worst < 1e-2, "neck/head stages: max |Δ| \(worst)")
    }

    /// `predicted_depth` at the model's real preprocessed size, on the synthetic scene and on the repo's
    /// sample photo: Pearson r ≥ 0.999 and max |Δ| ≤ 1 % of the reference's range.
    @Test func predictedDepthMatchesTheReferenceAtFullSize() throws {
        let loaded = try loaded()
        for caseName in ["a", "b"] {
            let input = try DepthAnythingFixtures.preprocessed(caseName)
            let predicted = loaded.model.predictedDepth(input)
            eval(predicted)
            let shape = try DepthAnythingFixtures.shape(caseName, ["predicted_depth"])
            #expect(predicted.shape == [1] + shape, "\(caseName) prediction \(predicted.shape) vs \(shape)")
            let actual = try MLXErrorGuard.read(predicted.asType(.float32), as: Float.self)
            let reference = try DepthAnythingFixtures.floats(try DepthAnythingFixtures.file(for: caseName, ["predicted_depth"]))
            let r = pearson(actual, reference)
            let maxDelta = zip(actual, reference).map { abs($0 - $1) }.max() ?? .infinity
            let range = (reference.max() ?? 0) - (reference.min() ?? 0)
            let line = String(format: "DEPTH PARITY predicted_depth[%@] r = %.7f max |Δ| = %.6f (range %.4f, %.3f%%)",
                              caseName, r, maxDelta, range, range > 0 ? 100 * maxDelta / range : 0)
            print(line)
            DepthAnythingFixtures.record(line)
            #expect(r >= 0.999, "\(caseName): Pearson r \(r)")
            #expect(maxDelta <= 0.01 * range, "\(caseName): max |Δ| \(maxDelta) > 1% of range \(range)")
        }
    }

    /// The final PNG from the reference prediction (so only the post-processor is compared): min–max
    /// normalisation, bicubic upsample, 8-bit quantisation, within ±2 grey levels on 99.9 % of pixels.
    @Test func finalPNGMatchesTheReference() throws {
        for caseName in ["a", "b"] {
            let referenceDepth = try DepthAnythingFixtures.array(
                try DepthAnythingFixtures.file(for: caseName, ["predicted_depth"]),
                shape: try DepthAnythingFixtures.shape(caseName, ["predicted_depth"]))
            let png = try DepthAnythingFixtures.caseInfo(caseName)["final_png"] as? [String: Any]
            let height = try #require(png?["height"] as? Int)
            let width = try #require(png?["width"] as? Int)
            let actual = try DepthAnythingPostprocessor.grayscale(prediction: referenceDepth, height: height, width: width)
            let reference = try grayBytes(file: try #require(png?["file"] as? String), width: width, height: height)
            #expect(actual.count == reference.count)
            let deltas = zip(actual, reference).map { abs(Int($0) - Int($1)) }
            let within2 = deltas.filter { $0 <= 2 }.count
            let fraction = Double(within2) / Double(max(deltas.count, 1))
            let line = String(format: "DEPTH PARITY final PNG[%@] within ±2 = %.4f%% max |Δ| = %d",
                              caseName, 100 * fraction, deltas.max() ?? 0)
            print(line)
            DepthAnythingFixtures.record(line)
            #expect(fraction >= 0.999, "\(caseName): only \(fraction) within ±2 grey levels")
        }
    }

    /// Preprocessing has its own tolerance (Trap 2): Swift's Core Graphics `.high` resize is not PIL's
    /// bicubic, so the tensors differ at sharp edges; the model tests above feed the reference tensor, this
    /// one tests the resize. Measured on this fixture: median 0, p99 0.16, max 0.47 (normalised units).
    @Test func preprocessingMatchesTheReferenceWithinItsOwnTolerance() throws {
        let info = try DepthAnythingFixtures.caseInfo("a")
        let input = try #require(info["input"] as? [String: Any])
        let height = try #require(input["height"] as? Int)
        let width = try #require(input["width"] as? Int)
        let rgb = try [UInt8](Data(contentsOf: DepthAnythingFixtures.directory.appendingPathComponent(
            try #require(input["file"] as? String))))
        let image = try #require(rgbImage(rgb: rgb, height: height, width: width))
        let actual = try DepthAnythingPreprocessor.preprocess(image: image)
        let expectedShape = try DepthAnythingFixtures.shape("a", ["preprocessed"])
        #expect(actual.shape == expectedShape, "preprocessed \(actual.shape) vs \(expectedShape)")
        let file = try DepthAnythingFixtures.file(for: "a", ["preprocessed"])
        let stats = try DepthAnythingFixtures.differenceStats(actual, file: file)
        let line = String(format: "DEPTH PARITY preprocessed (a) median |Δ| = %.4f p99 = %.4f max |Δ| = %.4f",
                          stats.median, stats.p99, stats.max)
        print(line)
        DepthAnythingFixtures.record(line)
        #expect(stats.median <= 0.01, "preprocessing: median |Δ| \(stats.median)")
        #expect(stats.p99 <= 0.25, "preprocessing: p99 |Δ| \(stats.p99)")
        #expect(stats.max <= 0.6, "preprocessing: max |Δ| \(stats.max)")
    }

    // MARK: - Helpers

    private func rgbImage(rgb: [UInt8], height: Int, width: Int) -> CGImage? {
        var bytes = rgb
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        bytes = []
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                       bytesPerRow: width * 3, space: colorSpace,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// The gray bytes of a checked-in PNG.
    private func grayBytes(file: String, width: Int, height: Int) throws -> [UInt8] {
        let url = DepthAnythingFixtures.directory.appendingPathComponent(file)
        let image = try #require(DepthAnythingImage.decoded(try Data(contentsOf: url)))
        var bytes = [UInt8](repeating: 0, count: width * height)
        bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let context = CGContext(data: base, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return bytes
    }
}

/// Ungated: the preprocessing size table is pure arithmetic (Trap 2) and the fixture carries the reference's
/// own numbers for five inputs.
struct DepthAnythingPreprocessingTableTests {

    @Test func targetSizeMatchesTheReferenceTable() throws {
        let golden = try DepthAnythingFixtures.golden()
        let table = try #require(golden["size_table"] as? [[String: Any]])
        #expect(table.count == 5)
        for row in table {
            let inputHeight = try #require(row["input_height"] as? Int)
            let inputWidth = try #require(row["input_width"] as? Int)
            let outputHeight = try #require(row["output_height"] as? Int)
            let outputWidth = try #require(row["output_width"] as? Int)
            let target = DepthAnythingPreprocessor.targetSize(inputHeight: inputHeight, inputWidth: inputWidth)
            #expect(target.height == outputHeight && target.width == outputWidth,
                    "\(inputWidth)x\(inputHeight) -> \(target.width)x\(target.height), reference \(outputWidth)x\(outputHeight)")
        }
    }

    @Test func aConstantPredictionIsAllBlack() throws {
        let constant = MLXArray.full([1, 20, 30], values: MLXArray(Float(2.5)))
        let pixels = try DepthAnythingPostprocessor.grayscale(prediction: constant, height: 40, width: 60)
        #expect(pixels.count == 40 * 60)
        #expect(pixels.allSatisfy { $0 == 0 })
    }

    @Test func aRampPredictionIsNormalisedAndBrightWhereNear() throws {
        // A left-to-right ramp must become a 0..255 ramp: near (right, bright) above far (left, dark).
        var values: [Float] = []
        for _ in 0 ..< 8 { for column in 0 ..< 16 { values.append(Float(column)) } }
        let prediction = MLXArray(values).reshaped(1, 8, 16)
        let pixels = try DepthAnythingPostprocessor.grayscale(prediction: prediction, height: 8, width: 16)
        #expect(pixels.first == 0)
        #expect(pixels.last == 255)
        #expect(pixels[8] > 0 && pixels[8] < 255, "the middle of the ramp is mid-grey: \(pixels[8])")
    }
}
