import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Testing
@testable import MLXUI

/// MC3-8 — the AI Workflows wiring: `Estimate Depth` is a served task, its pool is Depth Anything V2 Small,
/// gallery flow 62's "Depth Pro" row resolves to it with a shown substitution, settings are refused by name,
/// and a two-row flow writes an 8-bit grayscale PNG at the input's size. No weights are loaded.
@MainActor
struct DepthAnythingWorkflowWiringTests {

    private static let display = "Depth Anything V2 Small"

    private static func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func context() throws -> (catalog: [ModelEntry], claimable: Set<String>, entry: ModelEntry) {
        let data = try Data(contentsOf: Self.repoRoot().appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        let entry = try #require(catalog.first { $0.family == "depth-anything" })
        return (catalog, claimable, entry)
    }

    @Test func estimateDepthIsAvailableAndDefaultsToThisModel() throws {
        let (catalog, claimable, entry) = try context()
        let desc = try #require(TaskCatalog.get("Estimate Depth"))
        #expect(TaskAvailability.state(for: desc, catalog: catalog, claimableModelIDs: claimable) == .available)
        #expect(TaskModels.seededModels(forTask: "Estimate Depth") == [Self.display])
        #expect(TaskModels.defaultModel(forTask: "Estimate Depth", catalog: catalog,
                                        claimableModelIDs: claimable) == Self.display)
        #expect(TaskModels.isServedByExecutor("Estimate Depth"))
        let derived = TaskModels.derivedModels(for: "Estimate Depth", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.map(\.id) == [entry.id])
    }

    @Test func galleryS62ParsesAndRunsWithTheSubstitutionShown() throws {
        let (catalog, claimable, entry) = try context()
        let flow = try #require(GalleryLoader.loadMetadata().first { $0.number == 62 })
        let doc = try GalleryLoader.loadDocument(flowID: flow.flowID)
        #expect(doc.rows.map(\.task) == ["Read Image", "Estimate Depth", "Save Image"])
        #expect(doc.rows[1].model == "Depth Pro")

        guard case let .runnable(slot, equivalence, note) = CatalogBridge.resolve("Depth Pro", catalog: catalog) else {
            Issue.record("Depth Pro did not resolve"); return
        }
        #expect(slot.id == entry.id)
        #expect(equivalence == .substitute)
        #expect(note == "running Depth Pro as \(entry.hfModelId)")

        #expect(FlowRunnability.refusalReason(for: doc, catalog: catalog, installed: [],
                                              totalRAMGB: 32, claimableModelIDs: claimable) == nil,
                "gallery 62 must be runnable")
        #expect(!AppState.hiddenFlowNumbers.contains(62))
    }

    @Test func theRowWithItsOwnDisplayResolvesExactly() throws {
        let (catalog, _, entry) = try context()
        guard case let .runnable(slot, equivalence, note) = CatalogBridge.resolve(Self.display, catalog: catalog) else {
            Issue.record("the display did not resolve"); return
        }
        #expect(slot.id == entry.id)
        #expect(equivalence == .same)
        #expect(note == nil)
    }

    /// Owner decision 7: no settings. Each of the diffusion-family names is refused by name, not ignored.
    @Test func settingsOnTheRowAreRefusedByName() throws {
        let (catalog, _, entry) = try context()
        let executor = try makeExecutor(catalog: catalog, installed: [entry.id])
        let desc = try #require(TaskCatalog.get("Estimate Depth"))
        for setting in ["seed=1", "steps=4", "width=512", "height=512", "scale=2", "quantize=8"] {
            let row = Row(task: "Estimate Depth", model: Self.display, settings: setting)
            do {
                _ = try executor.stageConfig(for: desc, row: row, path: "2", model: entry)
                Issue.record("\(setting) was not refused")
            } catch let refusal as FlowSettingRefusal {
                let key = setting.split(separator: "=")[0]
                #expect(refusal.description.contains(String(key)), "the sentence must name \(key): \(refusal.description)")
                #expect(refusal.description.contains("Estimate Depth"))
            }
        }
        let clean = Row(task: "Estimate Depth", model: Self.display, settings: nil)
        #expect(try executor.stageConfig(for: desc, row: clean, path: "2", model: entry) == .default)
    }

    /// A flow Read Image → Estimate Depth → Save Image through a stub stage writes an 8-bit grayscale PNG at
    /// the input's size. (The real stage is the same `.image → .image` contract.)
    @Test func theFlowWritesAGrayscalePNGAtTheInputSize() async throws {
        let (catalog, _, entry) = try context()
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("depth-flow-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let inputURL = base.appendingPathComponent("photo.png")
        try inputPNG(width: 20, height: 12).write(to: inputURL)

        let executor = try makeExecutor(catalog: catalog, installed: [entry.id], blob: base.appendingPathComponent("blobs"))
        let row = Row(task: "Estimate Depth", model: Self.display, settings: nil)
        let input = Asset(items: [Item(kind: .image, value: nil, path: inputURL, sourceText: nil)])
        let output = try await executor.execute(path: "2", row: row, inputs: [input])
        let item = try #require(output.items.first)
        #expect(item.kind == .image)
        let url = try #require(item.path)
        let decoded = try #require(DepthAnythingImage.decoded(try Data(contentsOf: url)))
        #expect(decoded.width == 20 && decoded.height == 12)
        // Grayscale: every pixel's three channels are equal (the stub returns a grey ramp).
        let channels = try rgbPixels(of: decoded)
        let gray = stride(from: 0, to: channels.count, by: 3).allSatisfy {
            channels[$0] == channels[$0 + 1] && channels[$0 + 1] == channels[$0 + 2]
        }
        #expect(gray, "the saved PNG must be grayscale")
    }

    // MARK: - Helpers

    private func makeExecutor(catalog: [ModelEntry], installed: Set<String>, blob: URL? = nil) throws -> RealExecutor {
        let directory = blob ?? FileManager.default.temporaryDirectory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return RealExecutor(
            workspace: FlowWorkspace(root: directory),
            flowID: "62-SeeDepth",
            blobDirectory: directory,
            makeModelStage: { _, _ in StubGrayStage() },
            installedModelIDs: installed,
            catalog: catalog)
    }

    private func inputPNG(width: Int, height: Int) throws -> Data {
        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        for index in 0 ..< width * height {
            let level = UInt8((index * 255) / max(width * height - 1, 1))
            bytes[index * 3] = level; bytes[index * 3 + 1] = level; bytes[index * 3 + 2] = level
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                                  bytesPerRow: width * 3, space: space,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw DepthAnythingError.badInput("could not build the input image")
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw DepthAnythingError.badInput("could not encode the input image")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw DepthAnythingError.badInput("could not encode the input image")
        }
        return data as Data
    }

    private func rgbPixels(of image: CGImage) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 3)
        bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(data: base, width: image.width, height: image.height, bitsPerComponent: 8,
                                          bytesPerRow: image.width * 3, space: space,
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return bytes
    }
}

/// A stand-in for `DepthAnythingStage`: `.image → .image`, returning a grayscale ramp at the input's size.
private nonisolated struct StubGrayStage: PipelineStage {
    let id = "depth.stub"
    let name = "Depth Anything V2 Small"
    var accepts: MediaKind { .image }
    var produces: MediaKind { .image }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .image)
        guard case let .image(media) = input else { throw StageError.kindMismatch(expected: .image, got: input.kind) }
        let width = media.cgImage.width, height = media.cgImage.height
        var bytes = [UInt8](repeating: 0, count: width * height)
        for index in 0 ..< width * height { bytes[index] = UInt8((index * 255) / max(width * height - 1, 1)) }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw DepthAnythingError.badInput("could not build the stub depth map")
        }
        progress(1)
        return .image(ImageMedia(cgImage: image))
    }
}
