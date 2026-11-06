import Foundation
import SwiftUI
import AppKit
import Testing
@testable import MLXUI

/// MC3-6 — the PR's UI evidence: captures of the app's own views (`ImageRenderer`, the `SAMAudioScreenRenderTests`
/// method) at a fixed size in the light theme. Set `DP4D_SHOTS=<dir>` to write the PNGs the PR carries.
///
/// The **before** capture is Browse → Images with `main`'s bundled catalog (@ `6788e19`, supplied via
/// `DP4D_BASELINE_CATALOG`): `BrowseView`/`ModelCard`/`ModelDetailView` have no code changes between `main` and this
/// branch (`git diff origin/main -- MLXUI/Views` is empty), so the only difference from the after capture is the
/// catalog this ticket adds. The Run screen is new, so its "before" is that no such screen exists on `main`.
@Suite(.serialized)
@MainActor
struct DepthAnythingScreenRenderTests {

    nonisolated static var shotDirectory: String? {
        if let dir = ProcessInfo.processInfo.environment["DP4D_SHOTS"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_DP4D_SHOTS"] { return dir }
        return NSTemporaryDirectory() + "mlxui-depth-shots"
    }

    nonisolated static var baselineCatalog: String? {
        ProcessInfo.processInfo.environment["DP4D_BASELINE_CATALOG"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_DP4D_BASELINE_CATALOG"]
    }

    /// Render through AppKit's own drawing path (`NSHostingView.cacheDisplay`, the MC2-5 `TranslateGemmaScreenRenderTests`
    /// method): `ImageRenderer` misses lazy/AppKit-backed content (a `LazyVGrid`'s cards came out blank), so the PR's
    /// evidence would not be the real screen.
    private func render<V: View>(_ view: V, width: CGFloat, height: CGFloat, scheme: ColorScheme = .light) -> NSImage? {
        let host = NSHostingView(rootView:
            view
                .frame(width: width, height: height)
                .background(Color(nsColor: .windowBackgroundColor))
                .environment(\.colorScheme, scheme))
        host.frame = NSRect(x: 0, y: 0, width: width, height: height)
        host.layoutSubtreeIfNeeded()
        guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { return nil }
        host.cacheDisplay(in: host.bounds, to: rep)
        let image = NSImage(size: host.bounds.size)
        image.addRepresentation(rep)
        return image
    }

    @discardableResult
    private func save(_ image: NSImage?, _ name: String) -> Bool {
        guard let image,
              let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return false }
        // DP4D_SHOTS names the PR's folder; the sandboxed test host falls back to its own temporary directory
        // (the caller copies from there), so a plain test run still produces the evidence.
        for directory in [Self.shotDirectory, NSTemporaryDirectory() + "mlxui-depth-shots"] {
            guard let directory else { continue }
            try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
            if (try? png.write(to: URL(fileURLWithPath: directory).appendingPathComponent("\(name).png"))) != nil {
                return true
            }
        }
        return false
    }

    private func state() -> AppState {
        let state = AppState()
        state.loadBrowserData()
        state.selectedSection = .browse("images-videos")
        return state
    }

    private func depthEntry() throws -> ModelEntry {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        let all = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        return try #require(all.first { $0.family == "depth-anything" })
    }

    /// After: Browse → Images, with the Depth Estimation leaf and its card.
    @Test func browseImagesShowsDepthEstimation() throws {
        let state = state()
        #expect(state.browserData?.sidebarSections.first { $0.id == "images-videos" }?.domainIds.contains("depth") == true)
        save(render(BrowseView().environment(state), width: 1_100, height: 2_000), "after-browse-images")
    }

    /// Before: the same view with `main`'s catalog (no depth entry).
    @Test(.enabled(if: DepthAnythingScreenRenderTests.baselineCatalog != nil))
    func browseImagesOnTheBaselineCatalogHasNoDepthLeaf() throws {
        let path = try #require(Self.baselineCatalog)
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let baseline = try JSONDecoder().decode(BrowserData.self, from: data)
        let state = AppState()
        state.loadBrowserData()
        state.browserData = baseline
        state.selectedSection = .browse("images-videos")
        #expect(baseline.sidebarSections.first { $0.id == "images-videos" }?.domainIds.contains("depth") == false)
        save(render(BrowseView().environment(state), width: 1_100, height: 2_000), "before-browse-images")
    }

    /// The model page: title, facts, the Run action and the Apache-2.0 notice with its link.
    @Test func modelDetailShowsTheFactsAndLicence() throws {
        let state = state()
        let model = try depthEntry()
        state.installedModelIDs = [model.id]
        let image = render(ModelDetailView(model: model).environment(state), width: 1_000, height: 1_400)
        #expect(image != nil)
        save(image, "after-model-detail")
        #expect(DataNormalizer.normalizeLicense(model.license) == "Apache 2.0")
        #expect(model.licenseUrl != nil)
    }

    /// After (needs the real weights): the Run screen with the bundled sample photo and its real grayscale map,
    /// in both preview modes. Gated like the real-weights suites.
    @Test(.enabled(if: DepthAnythingRealLoad.canRun))
    func runScreenWithTheRealSamplePhotoOutput() throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let sampleURL = root.appendingPathComponent("MLXUI/Resources/Samples/sample-image.png")
        let data = try Data(contentsOf: sampleURL)
        let source = try #require(DepthAnythingImage.decoded(data))
        let result = try DepthAnythingEngine.estimateDepth(source, modelDirectory: directory)
        let bytes = try #require(DepthAnythingPostprocessor.grayBytes(from: result.grayImage))
        let colour = try DepthAnythingPostprocessor.colourImage(gray: bytes, width: result.width, height: result.height)

        let model = DepthRunModel()
        model.hostResult(source: source, gray: result.grayImage, colour: colour, png: result.pngData,
                         sourceLabel: "sample-image.png")
        let view = DepthAnythingRunView(modelDisplayName: "Depth Anything V2 Small", license: "apache-2.0",
                                       modelID: "depth-anything--Depth-Anything-V2-Small-hf",
                                       handedStage: depthProbe(), model: model)
        save(render(view, width: 760, height: 680), "after-run-grayscale")
        model.previewMode = .colour
        save(render(view, width: 760, height: 680), "after-run-colour")

        // The issue's "look at these" set: input + grayscale output for the sample and the two bundled sample photos.
        for name in ["sample-image", "sample-img-02", "sample-img-03"] {
            let url = name == "sample-image"
                ? sampleURL
                : root.appendingPathComponent("MLXUI/Resources/Samples/sample-images/\(name).png")
            guard let photo = DepthAnythingImage.decoded(try Data(contentsOf: url)) else { continue }
            let output = try DepthAnythingEngine.estimateDepth(photo, modelDirectory: directory)
            save(NSImage(cgImage: photo, size: NSSize(width: photo.width, height: photo.height)), "\(name)-input")
            save(NSImage(cgImage: output.grayImage, size: NSSize(width: output.width, height: output.height)),
                 "\(name)-depth")
        }
    }

    private func depthProbe() -> any PipelineStage {
        DepthAnythingStage(modelDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("depth-probe"))
    }
}
