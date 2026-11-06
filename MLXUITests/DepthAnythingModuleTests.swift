import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Testing
@testable import MLXUI

/// MC3-6 — the module wiring (claim, registry, stage), the input limits, the orientation case (Trap 6), the
/// main-actor probe (Trap 7), determinism and the PNG contract. The real-weights tests are opt-in
/// (`MLXUI_TEST_DEPTH=1` + `MLXUI_TEST_DEPTH_DIR`).
@MainActor
struct DepthAnythingModuleTests {

    private func catalog() throws -> BrowserData {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json")))
    }

    private func depthEntry() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.family == "depth-anything" })
    }

    @Test func sdkClaimsOnlyTheDepthAnythingEntry() throws {
        let entry = try depthEntry()
        #expect(DepthAnythingSDK().claim(entry) == .exact)
        #expect(DepthAnythingUI().claim(entry) == .exact)
        for other in try catalog().domains.flatMap({ $0.allModels }) where other.id != entry.id {
            #expect(DepthAnythingSDK().claim(other) == .no, "\(other.id) must not be claimed by Depth Anything")
        }
    }

    /// Trap 8: by family, never by architecture — a different family with the `depth` kind would not be claimed.
    @Test func sdkDoesNotClaimAnotherFamilyWithTheDepthKind() {
        var other = ModelEntry(
            id: "x--y", family: "some-other-depth", displayName: "Other", paramSize: "1B", paramCountB: 1,
            modelType: .depth, source: .mlx, format: "mlx-fp32", platforms: ["macOS 14+"], minMacOSVersion: "14.0",
            hfRepo: "x", hfModelId: "x/y", ramGB: 1, downloadSizeGB: 1, contextWindow: nil,
            license: "mit", licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil,
            architecture: "DepthAnythingForDepthEstimation", languages: nil, lastUpdated: nil,
            taskTags: nil, benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil, speedEstimated: nil,
            communityDownloads: nil, communityLikes: nil, variants: nil)
        #expect(DepthAnythingSDK().claim(other) == .no)
        other = ModelEntry(
            id: "depth-anything--x", family: "depth-anything", displayName: "X", paramSize: "1B", paramCountB: 1,
            modelType: .image, source: .mlx, format: "mlx-fp32", platforms: ["macOS 14+"], minMacOSVersion: "14.0",
            hfRepo: "depth-anything", hfModelId: "depth-anything/x", ramGB: 1, downloadSizeGB: 1, contextWindow: nil,
            license: "apache-2.0", licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil,
            architecture: nil, languages: nil, lastUpdated: nil, taskTags: nil, benchmarks: nil,
            speedTokensPerSec: nil, speedHardware: nil, speedEstimated: nil, communityDownloads: nil,
            communityLikes: nil, variants: nil)
        #expect(DepthAnythingSDK().claim(other) == .no)
    }

    @Test func registryResolvesTheEntryToTheDepthAnythingModule() throws {
        let entry = try depthEntry()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        #expect(registry.bestModule(for: entry)?.descriptor.id == "depth-anything")
        #expect(DepthAnythingModule.descriptor.packageLicense == "Apache-2.0")
        #expect(DepthAnythingModule.descriptor.modelTypes == [.depth])
        #expect(ModelSupport.unsupportedReason(for: entry) == nil)
    }

    @Test func makeStageIsAnImageToImageDepthStage() throws {
        let entry = try depthEntry()
        let stage = try DepthAnythingSDK().makeStage(for: entry, config: .default)
        #expect(stage.accepts == .image && stage.produces == .image)
        #expect(stage.id == "depth.\(entry.id)")
        #expect(stage.name == "Depth Anything V2 Small")
    }

    @Test func stageRefusesTheWrongInputKind() async throws {
        let stage = DepthAnythingStage(modelDirectory: FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-depth-\(UUID().uuidString)"))
        await #expect(throws: StageError.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: [0], sampleRate: 16_000)), progress: { _ in })
        }
    }

    /// Owner decision 5: the limits are refused with one plain sentence, before any work.
    @Test func inputLimitsRefuseLongSidesAndWidePanoramas() {
        #expect(DepthAnythingInput.refusal(width: 8192, height: 3000) == nil)
        #expect(DepthAnythingInput.refusal(width: 8193, height: 2000)?.contains("8,192") == true)
        #expect(DepthAnythingInput.refusal(width: 500, height: 9000)?.contains("8,192") == true)
        #expect(DepthAnythingInput.refusal(width: 600, height: 200) == nil)
        #expect(DepthAnythingInput.refusal(width: 601, height: 200)?.contains("3 : 1") == true)
        #expect(DepthAnythingInput.refusal(width: 200, height: 601)?.contains("3 : 1") == true)
    }

    /// An oversized image is refused by the limit sentence, proving the refusal comes before the model files are
    /// even looked for (the directory does not exist).
    @Test func theLimitRefusalHappensBeforeAnyModelWork() throws {
        let image = try solidImage(width: 9000, height: 100, level: 128)
        #expect(throws: DepthAnythingError.self) {
            _ = try DepthAnythingEngine.estimateDepth(image, modelDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("no-such-depth-\(UUID().uuidString)"))
        }
    }

    /// Trap 6: an EXIF-rotated photo is decoded with the orientation applied, so the map lines up with the
    /// picture the user sees. This needs no weights.
    @Test func exifOrientationIsAppliedWhenDecoding() throws {
        // A 40 x 20 image, black on the left half and white on the right half.
        let original = try splitImage(width: 40, height: 20)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("depth-exif-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            Issue.record("could not create the PNG destination"); return
        }
        CGImageDestinationAddImage(destination, original, [kCGImagePropertyOrientation: 6] as CFDictionary)   // rotate 90° CW
        #expect(CGImageDestinationFinalize(destination))

        let decoded = try #require(DepthAnythingImage.decoded(try Data(contentsOf: url)))
        #expect(decoded.width == 20 && decoded.height == 40, "orientation 6 swaps the sides: \(decoded.width)x\(decoded.height)")
        // Rotating 90° clockwise puts the original left (black) half on top.
        let pixels = try grayPixels(of: decoded)
        #expect(pixels[0] < 16, "top row after a 90° clockwise turn should be the black half: \(pixels[0])")
        #expect(pixels[(decoded.height - 1) * decoded.width] > 239,
                "bottom row should be the white half: \(pixels[(decoded.height - 1) * decoded.width])")
    }

    @Test func savedFileNameAndPreviewModes() {
        #expect(DepthRunModel.savedFileName(for: URL(fileURLWithPath: "/a/holiday.heic")) == "holiday-depth.png")
        #expect(DepthRunModel.savedFileName(for: nil) == "photo-depth.png")
        #expect(DepthPreviewMode.allCases == [.grayscale, .colour])
    }

    /// The colour map is display-only and near (bright) is lighter than far.
    @Test func colourMapIsBrightWhereNear() throws {
        let ramp = (0 ..< 256).map { UInt8($0) }
        let image = try DepthAnythingPostprocessor.colourImage(gray: ramp, width: 256, height: 1)
        let pixels = try rgbaPixels(of: image)
        let far = Int(pixels[0]) + Int(pixels[1]) + Int(pixels[2])
        let near = Int(pixels[255 * 4]) + Int(pixels[255 * 4 + 1]) + Int(pixels[255 * 4 + 2])
        #expect(near > far + 300, "near must be much brighter than far (\(near) vs \(far))")
    }

    // MARK: - Helpers

    private func solidImage(width: Int, height: Int, level: UInt8) throws -> CGImage {
        var bytes = [UInt8](repeating: level, count: width * height * 3)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw DepthAnythingError.badInput("could not build the test image")
        }
        bytes = []
        guard let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                                  bytesPerRow: width * 3, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw DepthAnythingError.badInput("could not build the test image")
        }
        return image
    }

    private func splitImage(width: Int, height: Int) throws -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 3)
        for row in 0 ..< height {
            for column in 0 ..< width where column < width / 2 {
                let base = (row * width + column) * 3
                bytes[base] = 0; bytes[base + 1] = 0; bytes[base + 2] = 0
            }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                                  bytesPerRow: width * 3, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw DepthAnythingError.badInput("could not build the test image")
        }
        return image
    }

    private func grayPixels(of image: CGImage) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: image.width * image.height)
        bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let context = CGContext(data: base, width: image.width, height: image.height, bitsPerComponent: 8,
                                          bytesPerRow: image.width, space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return bytes
    }

    private func rgbaPixels(of image: CGImage) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
        bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(data: base, width: image.width, height: image.height, bitsPerComponent: 8,
                                          bytesPerRow: image.width * 4, space: colorSpace,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return bytes
    }
}

@Suite(.enabled(if: DepthAnythingRealLoad.canRun, "needs MLXUI_TEST_DEPTH=1 and MLXUI_TEST_DEPTH_DIR"))
@MainActor
struct DepthAnythingEngineRealLoadTests {

    /// The whole Run path from the main actor (Trap 7): MLX work evaluates before any read, and no MLX error is
    /// recorded anywhere. The Run screen hops off the main actor, but the engine must also be safe if called here.
    @Test func engineRunsFromTheMainActorWithoutAnMLXError() throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let scene = try DepthAnythingFixtures.bytes("a_input.rgb")
        let height = 140, width = 196
        let image = try rgbImage(scene, width: width, height: height)
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        let result = try MLXErrorGuard.guardedRun {
            try DepthAnythingEngine.estimateDepth(image, modelDirectory: directory)
        }
        #expect(result.width == width && result.height == height)
        #expect(!result.pngData.isEmpty)
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }

    /// Owner decision 4/acceptance: two runs are byte-identical.
    @Test func twoRunsAreByteIdentical() throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let scene = try DepthAnythingFixtures.bytes("a_input.rgb")
        let image = try rgbImage(scene, width: 196, height: 140)
        let first = try DepthAnythingEngine.estimateDepth(image, modelDirectory: directory)
        let second = try DepthAnythingEngine.estimateDepth(image, modelDirectory: directory)
        #expect(first.pngData == second.pngData, "the output is not deterministic")
    }

    /// The 8-bit PNG is grayscale at the input's size and spans the full 0..255 range (near = bright is the
    /// min–max mapping, proven by the post-processing tests; a model's ordering on a synthetic scene is its own
    /// opinion, and the layer/end-to-end parity above is what proves the ordering is the reference's).
    @Test func pngIsGrayscaleAtTheInputSizeAndSpansTheRange() throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let scene = try DepthAnythingFixtures.bytes("a_input.rgb")
        let image = try rgbImage(scene, width: 196, height: 140)
        let result = try DepthAnythingEngine.estimateDepth(image, modelDirectory: directory)
        let decoded = try #require(DepthAnythingImage.decoded(result.pngData))
        #expect(decoded.width == 196 && decoded.height == 140)
        let gray = try grayPixels(of: decoded)
        #expect(gray.min() == 0 && gray.max() == 255, "min–max normalisation spans 0..255 (\(gray.min() ?? 0)..\(gray.max() ?? 0))")
        #expect(Set(gray).count > 32, "the depth map has contrast, not one flat value")
    }

    // MARK: - Helpers

    private func rgbImage(_ rgb: [UInt8], width: Int, height: Int) throws -> CGImage {
        guard let provider = CGDataProvider(data: Data(rgb) as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                                  bytesPerRow: width * 3, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw DepthAnythingError.badInput("could not build the test image")
        }
        return image
    }

    private func grayPixels(of image: CGImage) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: image.width * image.height)
        bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let context = CGContext(data: base, width: image.width, height: image.height, bitsPerComponent: 8,
                                          bytesPerRow: image.width, space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return bytes
    }
}
