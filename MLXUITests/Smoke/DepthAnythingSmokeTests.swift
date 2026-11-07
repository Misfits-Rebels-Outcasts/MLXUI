import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import Testing
@testable import MLXUI

/// AM-T — Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`), owner rulings R10 / R15 (MC3-12): the
/// depth map of `sample-image.png` and of a synthetic near-square scene (size = input, 8-bit grayscale, not constant,
/// near square brighter than its background, byte-identical twice), gallery flow 62's row through
/// `CachingExecutor(RealExecutor)` with the Depth Pro substitution shown, and memory through the **production path**
/// (`ServeMLXMemory.shared.scoped`, the 1 GiB cap the Run screen and flow rows both run under) at the sample image, a
/// 12 MP photo and the worst case the UI accepts, 8,192 x 2,731. Run with:
///
///     bash RSI/evals/model-smoke.sh run depth-anything/Depth-Anything-V2-Small-hf --suite DepthAnythingSmokeTests --weights <dir>
///
/// `MLXUI_SMOKE_DEPTH_PHASES` (comma list of `sample,12mp,worst`) limits the memory phases so each can run in a fresh process.
@Suite(.enabled(if: ModelSmoke.enabled("depth-anything/Depth-Anything-V2-Small-hf"), "run through RSI/evals/model-smoke.sh"), .serialized)
@MainActor
struct DepthAnythingSmokeTests {
    static let hfID = "depth-anything/Depth-Anything-V2-Small-hf"
    static let display = "Depth Anything V2 Small"
    static let auto = "MC3-12"
    static let review = "MC3-12r"

    // MARK: Images

    /// A far-to-near ground gradient with fine texture, and a textured square standing in the foreground.
    static let sceneWidth = 512, sceneHeight = 384
    static let square = CGRect(x: 192, y: 224, width: 128, height: 128)   // top-left origin, bottom-centre

    static func sceneRGB() -> [UInt8] {
        let w = sceneWidth, h = sceneHeight
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let inSquare = square.contains(CGPoint(x: x, y: y))
                var level: Double
                if inSquare {
                    let checker = ((x / 16) + (y / 16)) % 2 == 0
                    level = checker ? 235 : 150
                } else {
                    // top = far (dim, hazy), bottom = near (brighter, more contrast), plus deterministic texture
                    let t = Double(y) / Double(h - 1)
                    let texture = Double((x * 7 + y * 13) % 17) * (2 + 6 * t)
                    level = 90 + 70 * t + texture
                }
                let r = UInt8(max(0, min(255, level))), g = UInt8(max(0, min(255, level * 0.95))), b = UInt8(max(0, min(255, level * (inSquare ? 0.8 : 1.05))))
                let i = (y * w + x) * 3
                rgb[i] = r; rgb[i + 1] = g; rgb[i + 2] = b
            }
        }
        return rgb
    }

    static func cgImage(rgb: [UInt8], width: Int, height: Int) throws -> CGImage {
        let provider = try #require(CGDataProvider(data: Data(rgb) as CFData))
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        return try #require(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3, space: space,
                                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider, decode: nil,
                                    shouldInterpolate: false, intent: .defaultIntent))
    }

    static func sampleImage() throws -> CGImage {
        try #require(DepthAnythingImage.decoded(contentsOf: ModelSmoke.sample("sample-image.png")))
    }

    static func stats(_ gray: [UInt8]) -> (mean: Double, std: Double) {
        guard !gray.isEmpty else { return (0, 0) }
        let mean = gray.reduce(0.0) { $0 + Double($1) } / Double(gray.count)
        let variance = gray.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) } / Double(gray.count)
        return (mean, variance.squareRoot())
    }

    /// PNG properties: (width, height, bit depth, colour model).
    static func pngInfo(_ data: Data) -> (width: Int, height: Int, depth: Int, model: String)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let p = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else { return nil }
        return ((p[kCGImagePropertyPixelWidth] as? Int) ?? 0, (p[kCGImagePropertyPixelHeight] as? Int) ?? 0,
                (p[kCGImagePropertyDepth] as? Int) ?? 0, (p[kCGImagePropertyColorModel] as? String) ?? "?")
    }

    /// The production path: the same scoped 1 GiB cache cap the Run screen (`LeasingStage.perRun`) and flow rows run under.
    static func onPath(_ image: CGImage, _ dir: URL) async throws -> DepthAnythingEngine.Result {
        try await ServeMLXMemory.shared.scoped { try DepthAnythingEngine.estimateDepth(image, modelDirectory: dir) }
    }

    // MARK: Checks on a map

    private func judgeMap(_ name: String, result: DepthAnythingEngine.Result, input: CGImage) throws -> [UInt8] {
        let info = Self.pngInfo(result.pngData)
        ModelSmoke.check(Self.auto, "\(name).size-equals-input", result.width == input.width && result.height == input.height
                         && info?.width == input.width && info?.height == input.height,
                         "map \(result.width)x\(result.height), png \(info?.width ?? 0)x\(info?.height ?? 0), input \(input.width)x\(input.height)")
        ModelSmoke.check(Self.auto, "\(name).8-bit-grayscale", info?.depth == 8 && info?.model == "Gray", "depth \(info?.depth ?? 0), model \(info?.model ?? "?")")
        let gray = try #require(DepthAnythingPostprocessor.grayBytes(from: result.grayImage))
        let s = Self.stats(gray)
        ModelSmoke.check(Self.auto, "\(name).not-constant", s.std > 0, String(format: "mean %.1f std %.2f", s.mean, s.std))
        return gray
    }

    // MARK: Tests

    @Test func a_sampleImage() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let image = try Self.sampleImage()
        let r1 = try await Self.onPath(image, dir)
        _ = try judgeMap("sample", result: r1, input: image)
        let r2 = try await Self.onPath(image, dir)
        ModelSmoke.check(Self.auto, "sample.byte-identical-twice", r1.pngData == r2.pngData, "\(r1.pngData.count) vs \(r2.pngData.count) bytes")
        _ = try ModelSmoke.artifact(Self.review, "sample-input.png", copying: ModelSmoke.sample("sample-image.png"))
        _ = try ModelSmoke.artifact(Self.review, "sample-depth.png", data: r1.pngData,
                                    judge: "Is near bright and far dark, with the same layout as sample-input.png?")
    }

    @Test func b_syntheticNearSquare() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let w = Self.sceneWidth, h = Self.sceneHeight
        let image = try Self.cgImage(rgb: Self.sceneRGB(), width: w, height: h)
        let result = try await Self.onPath(image, dir)
        let gray = try judgeMap("scene", result: result, input: image)
        var inside: [UInt8] = [], outside: [UInt8] = []
        for y in 0 ..< h {
            for x in 0 ..< w {
                if Self.square.contains(CGPoint(x: x, y: y)) { inside.append(gray[y * w + x]) } else { outside.append(gray[y * w + x]) }
            }
        }
        let a = Self.stats(inside).mean, b = Self.stats(outside).mean
        ModelSmoke.check(Self.auto, "scene.near-square-brighter-than-background", a > b, String(format: "square mean %.1f, background mean %.1f", a, b))
        let twice = try await Self.onPath(image, dir)
        ModelSmoke.check(Self.auto, "scene.byte-identical-twice", result.pngData == twice.pngData)
        if let png = PNGEncoder.pngData(from: image) { _ = try ModelSmoke.artifact(Self.review, "scene-input.png", data: png) }
        _ = try ModelSmoke.artifact(Self.review, "scene-depth.png", data: result.pngData,
                                    judge: "Is the square in front of the background gradient (brighter)?")
    }

    private func catalog() throws -> [ModelEntry] {
        let data = try Data(contentsOf: ModelSmoke.repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    /// Gallery flow 62's shape (Read Image -> Estimate Depth `Depth Pro` -> Save Image) through `CachingExecutor(RealExecutor)`.
    @Test func c_flow62ThroughRealExecutor() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let catalog = try catalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        let sample = try Self.sampleImage()

        let galleryDoc = try GalleryLoader.loadDocument(flowID: try #require(GalleryLoader.loadMetadata().first { $0.number == 62 }).flowID)
        ModelSmoke.check(Self.auto, "flow62.gallery-row-names-depth-pro", galleryDoc.rows[1].model == "Depth Pro", galleryDoc.rows[1].model ?? "nil")
        guard case let .runnable(slot, equivalence, note) = CatalogBridge.resolve("Depth Pro", catalog: catalog) else {
            ModelSmoke.check(Self.auto, "flow62.depth-pro-resolves", false, "did not resolve"); return
        }
        ModelSmoke.check(Self.auto, "flow62.substitution-shown", slot.id == entry.id && equivalence == .substitute
                         && note == "running Depth Pro as \(entry.hfModelId)", "\(slot.id) \(equivalence) \(note ?? "nil")")

        let base = FileManager.default.temporaryDirectory.appendingPathComponent("mc3-12-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let flows = base.appendingPathComponent("flows")
        try FileManager.default.createDirectory(at: flows, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: flows)
        let flowDir = ws.directory(for: "62-SeeDepth")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: ModelSmoke.sample("sample-image.png"), to: flowDir.appendingPathComponent("photo.png"))
        let store = FlowCacheStore(root: base.appendingPathComponent("cache"))
        let make: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { _, _ in DepthAnythingStage(modelDirectory: dir) }
        let real = RealExecutor(workspace: ws, flowID: "62-SeeDepth", blobDirectory: base.appendingPathComponent("blobs"),
                                makeModelStage: make, installedModelIDs: [entry.id], catalog: catalog)
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real", catalog: catalog, runSeed: 0, workspace: ws, flowID: "62-SeeDepth")
        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Read Image       photo.png
        2. Estimate Depth   Depth Pro
        3. Save Image       photo-depth.png
        """)
        let events = try await ModelSmoke.measure("flow 62 row", detail: "sample-image.png") { try await FlowInterpreter.run(doc, executor: caching) }
        let failed = events.filter { $0.kind == .rowFailed }
        ModelSmoke.check(Self.auto, "flow62.no-row-failed", failed.isEmpty, failed.map { "\($0.path): \($0.error ?? $0.message ?? "")" }.joined(separator: "; "))
        let row2 = events.filter { $0.path == "2" }
        ModelSmoke.info(Self.auto, "flow62.row2-events", row2.map { "\($0.kind): \($0.message ?? "")" }.joined(separator: " | "))
        ModelSmoke.check(Self.auto, "flow62.row-completed", row2.contains { $0.kind == .rowCompleted })
        let written = flowDir.appendingPathComponent("photo-depth.png")
        let data = try? Data(contentsOf: written)
        let info = data.flatMap(Self.pngInfo)
        ModelSmoke.check(Self.auto, "flow62.saved-png-size-equals-input", info?.width == sample.width && info?.height == sample.height,
                         "\(info?.width ?? 0)x\(info?.height ?? 0) vs \(sample.width)x\(sample.height)")
        ModelSmoke.check(Self.auto, "flow62.saved-png-8-bit-gray", info?.depth == 8 && info?.model == "Gray", "\(info?.depth ?? 0) \(info?.model ?? "?")")
        if let data { _ = try ModelSmoke.artifact(Self.review, "flow62-photo-depth.png", data: data) }
    }

    /// R15: whole-process `phys_footprint` through the production path, three phases. The off-path figures are the earlier
    /// `DepthAnythingFootprintTests` runs of 2026-11-08 (default MLX cache limit), printed beside each result.
    @Test func d_memoryOnTheProductionPath() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let wanted = (ModelSmoke.value("MLXUI_SMOKE_DEPTH_PHASES") ?? "sample,12mp,worst").split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
        let offPath: [String: String] = ["sample": "2.751 GB (was 2.802 before MC3-11)", "12mp": "2.425 GB (was 2.901)", "worst": "3.778 GB (was 4.663)"]
        for phase in wanted {
            let source = try DepthAnythingFootprintTests.source(phase: phase)
            let image = try #require(DepthAnythingImage.decoded(source))
            let limit = MLX.Memory.cacheLimit
            let result = try await ModelSmoke.measure("depth \(phase)", detail: "\(image.width)x\(image.height), production path (ServeMLXMemory.scoped), off-path was \(offPath[phase] ?? "n/a")") {
                try await Self.onPath(image, dir)
            }
            ModelSmoke.check(Self.auto, "memory.\(phase).completes", result.width == image.width && result.height == image.height && !result.pngData.isEmpty,
                             "\(image.width)x\(image.height) -> \(result.width)x\(result.height), cacheLimit before \(limit / 1_048_576) MiB, after \(MLX.Memory.cacheLimit / 1_048_576) MiB")
        }
    }
}
