import Testing
import Foundation
import SwiftUI
import AppKit
@testable import MLXUI

/// MC1-5 — the Run screen and the Browse surfaces render without crashing, and the PR's UI evidence is captured from
/// the real views: `ImageRenderer` on the app's own `BrowseView`, `ModelDetailView` and `SAMAudioRunView` (the
/// `ServerScreenRenderTests` pattern). Set `SAM4D_SHOTS` to a directory to write the PNGs the PR attaches.
///
/// No weights: the result state is produced by the Run model driving a stub stage, then rendered.
@Suite(.serialized)
@MainActor
struct SAMAudioScreenRenderTests {

    static var shotDirectory: String? { ProcessInfo.processInfo.environment["SAM4D_SHOTS"] ?? ProcessInfo.processInfo.environment["TEST_RUNNER_SAM4D_SHOTS"] }

    private func render<V: View>(_ view: V, width: CGFloat, height: CGFloat, scheme: ColorScheme = .light) -> NSImage? {
        let renderer = ImageRenderer(content:
            view
                .frame(width: width, height: height)
                .background(Color(nsColor: .windowBackgroundColor))
                .environment(\.colorScheme, scheme)
        )
        renderer.scale = 2
        return renderer.nsImage
    }

    private func save(_ image: NSImage?, _ name: String) {
        guard let dir = Self.shotDirectory, let image,
              let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return }
        try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
    }

    private func entry() throws -> ModelEntry { try SAMAudioCatalogTests.samAudio() }

    private func state(selecting section: String = "audio") -> AppState {
        let state = AppState()
        state.loadBrowserData()
        state.selectedSection = .browse(section)
        return state
    }

    /// Browse → Audio: the Sound Isolation leaf and its one card, in the section the issue names.
    @Test func browseAudioSectionShowsSoundIsolation() throws {
        let state = state()
        let image = render(BrowseView().environment(state), width: 1_100, height: 720)
        #expect(image != nil)
        save(image, "after-browse-audio")
        let audio = try #require(state.browserData?.sidebarSections.first { $0.id == "audio" })
        #expect(audio.domainIds.contains("separate"))
        #expect(state.visibleSections.contains { $0.id == "audio" })
    }

    /// The model page: title, facts, the Run action and the SAM License notice with its link.
    @Test func modelDetailShowsTheLicenseNotice() throws {
        let state = state()
        let model = try entry()
        state.installedModelIDs = [model.id]   // the Installed/Run state, so the page shows its action
        let image = render(ModelDetailView(model: model).environment(state), width: 1_000, height: 1_400)
        #expect(image != nil)
        save(image, "after-model-detail")
        #expect(DataNormalizer.normalizeLicense(model.license) == "SAM License")
        #expect(model.licenseUrl != nil)
    }

    /// The Run screen before a run: a chosen file, the "What to isolate" field with its suggestions, the length limit
    /// and the time estimate.
    @Test func runScreenBeforeARunShowsTheFieldLimitAndEstimate() async throws {
        let wav = try makeWAV(seconds: 12)
        defer { try? FileManager.default.removeItem(at: wav) }
        let view = SAMAudioRunView(modelDisplayName: "SAM-Audio Base", license: "sam-license",
                                   licenseURL: "https://huggingface.co/facebook/sam-audio-base/blob/main/LICENSE",
                                   modelID: "mlx-community--sam-audio-base",
                                   handedStage: stubProbe(), model: SAMAudioRunModel(),
                                   initialAudioURL: wav, initialDescription: "speech")
        let image = render(view, width: 600, height: 600)
        #expect(image != nil)
        save(image, "after-run-before")
        #expect(SAMAudioEngine.estimate(seconds: 12) > 12)
    }

    /// The Run screen after a successful run: both players with Save WAV and the result line.
    @Test func runScreenAfterARunShowsBothPlayers() async throws {
        let wav = try makeWAV(seconds: 3)
        defer { try? FileManager.default.removeItem(at: wav) }
        let box = SAMAudioTrackBox()
        let model = SAMAudioRunModel()
        model.start(audioURL: wav, stage: stubStage(box: box), output: box)
        let deadline = ContinuousClock.now + .seconds(10)
        while model.isRunning, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(20)) }
        #expect(model.hasResult, "the stub run must produce both tracks")
        let view = SAMAudioRunView(modelDisplayName: "SAM-Audio Base", license: "sam-license",
                                   licenseURL: "https://huggingface.co/facebook/sam-audio-base/blob/main/LICENSE",
                                   modelID: "mlx-community--sam-audio-base",
                                   handedStage: stubProbe(),
                                   model: model, initialAudioURL: wav, initialDescription: "speech")
        let image = render(view, width: 600, height: 600)
        #expect(image != nil)
        save(image, "after-run-with-results")
        #expect(model.other != nil)
    }

    // MARK: - Fixtures

    private func makeWAV(seconds: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sam-render-\(UUID().uuidString).wav")
        let samples = (0..<(48_000 * seconds)).map { 0.2 * Float(sin(2 * Double.pi * 330 * Double($0) / 48_000)) }
        try AudioWriter.writeWAV(AudioBuffer(samples: samples, sampleRate: 48_000), to: url)
        return url
    }

    /// The stage the sheet would hand; the view only unwraps it (`LeasingStage.perRun` turns it into the run's stage).
    private func stubProbe() -> any PipelineStage {
        SAMAudioStage(id: "probe", name: "SAM-Audio") { samples, _ in
            SAMAudioEngine.Result(isolated: samples, other: samples, sampleRate: 48_000)
        }
    }

    private func stubStage(box: SAMAudioTrackBox) -> SAMAudioStage {
        SAMAudioStage(id: "stub", name: "SAM-Audio", output: box) { samples, progress in
            progress(1)
            box.store(AudioBuffer(samples: samples.map { $0 * 0.5 }, sampleRate: 48_000))
            return SAMAudioEngine.Result(isolated: samples, other: samples.map { $0 * 0.5 }, sampleRate: 48_000)
        }
    }
}
