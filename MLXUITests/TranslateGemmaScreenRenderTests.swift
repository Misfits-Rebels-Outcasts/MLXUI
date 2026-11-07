import Testing
import Foundation
import SwiftUI
import AppKit
@testable import MLXUI

/// MC2-5 — the Translate screen and the model's Browse surfaces render without crashing, and the PR's visual evidence is
/// captured from the **real views** with `ImageRenderer` (the `SAMAudioScreenRenderTests` / `ServerScreenRenderTests`
/// pattern): the app's own `BrowseView`, `ModelDetailView` and `TranslateGemmaRunView`. Set `TGSHOTS` to a directory to
/// write the PNGs the PR attaches.
///
/// No weights: the filled and result states are set on the Run model directly, with a stub detector and counter.
@Suite(.serialized)
@MainActor
struct TranslateGemmaScreenRenderTests {

    /// `TGSHOTS` (or `TEST_RUNNER_TGSHOTS`) names the PR's screenshot folder; with neither set the captures go to a
    /// temp folder and the path is printed, so the evidence is still producible from a plain test run.
    static var shotDirectory: String {
        if let dir = ProcessInfo.processInfo.environment["TGSHOTS"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_TGSHOTS"] { return dir }
        return NSTemporaryDirectory() + "mlxui-translategemma-shots"
    }

    /// Render through AppKit's own drawing path (`NSHostingView.cacheDisplay`), the method `ServerScreenRenderTests`
    /// uses: `ImageRenderer` leaves AppKit-backed controls (`TextEditor`) as placeholder blocks and can miss a scrolled
    /// description, so the PR's evidence would not be the real screen.
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

    private func save(_ image: NSImage?, _ name: String) {
        let dir = Self.shotDirectory
        guard let image,
              let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return }
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
        print("TGSHOT \(dir)/\(name).png")
    }

    private func entry() throws -> ModelEntry { try TranslateGemmaCatalogTests.entry() }

    private func state(selecting section: String = "chat-text") -> AppState {
        let state = AppState()
        state.loadBrowserData()
        state.selectedSection = .browse(section)
        return state
    }

    private func runModel() -> TranslateGemmaRunModel {
        let model = TranslateGemmaRunModel(modelID: TranslateGemmaCatalogTests.cardID,
                                           displayName: "TranslateGemma 4B",
                                           modelDirectory: URL(fileURLWithPath: "/tmp/nonexistent-translategemma"),
                                           footprintBytes: 3_575_000_000)
        model.detect = { _ in .init(code: "de", name: "German", confidence: 0.93) }
        model.countTokens = { _ in 148 }
        return model
    }

    private func screen(_ model: TranslateGemmaRunModel) -> TranslateGemmaRunView {
        TranslateGemmaRunView(
            modelDisplayName: "TranslateGemma 4B",
            license: "gemma-terms-of-use",
            licenseURL: "https://ai.google.dev/gemma/terms",
            modelID: TranslateGemmaCatalogTests.cardID,
            handedStage: TranslateGemmaStage(id: "render", name: "TranslateGemma 4B",
                                             sourceCode: "en", targetCode: "ms") { _ in
                TranslateGemmaEngine.Outcome(text: "", promptTokens: 0, generatedTokens: 0, cancelled: true)
            },
            model: model)
    }

    // MARK: - Browse and detail

    @Test func browseChatAndTextShowsTheTranslator() throws {
        let state = state()
        let image = render(BrowseView().environment(state), width: 1_100, height: 720)
        #expect(image != nil)
        save(image, "after-browse-chat-text")
        let llm = try #require(state.browserData?.domains.first { $0.id == "llm" })
        #expect(llm.allModels.last?.hfModelId == TranslateGemmaCatalogTests.hfID)
    }

    @Test func modelDetailShowsTheFactsAndLicenseNotice() throws {
        let state = state()
        let model = try entry()
        state.installedModelIDs = [model.id]   // the Installed/Run state, so the page shows its action
        let image = render(ModelDetailView(model: model).environment(state), width: 1_000, height: 1_400)
        #expect(image != nil)
        save(image, "after-model-detail")
        #expect(DataNormalizer.normalizeLicense(model.license) == "Gemma Terms of Use")
    }

    // MARK: - The Translate screen

    @Test func translateScreenInitialState() throws {
        let model = runModel()
        model.sourceText = "Guten Morgen, wie geht es dir?"
        model.refreshDetection()
        model.promptTokenCount = 148
        let image = render(screen(model), width: 660, height: 640)
        #expect(image != nil)
        save(image, "after-translate-initial")
        #expect(model.detected?.name == "German")
        #expect(model.tokenCountLabel == "148 / 2048 tokens")
    }

    @Test func translateScreenFilledToChinese() throws {
        let model = runModel()
        model.targetSelection = "zh-Hans"
        model.sourceText = "Guten Morgen, wie geht es dir?"
        model.refreshDetection()
        model.promptTokenCount = 151
        let image = render(screen(model), width: 660, height: 640)
        #expect(image != nil)
        save(image, "after-translate-filled")
        #expect(model.targetLabel == "Chinese (Simplified)")
    }

    @Test func translateScreenShowsAResultWithCopy() throws {
        let model = runModel()
        model.sourceText = "Guten Morgen, wie geht es dir?"
        model.refreshDetection()
        model.promptTokenCount = 148
        model.resultText = "Good morning, how are you?"
        #expect(model.canCopy)
        let image = render(screen(model), width: 660, height: 640)
        #expect(image != nil)
        save(image, "after-translate-result")
    }

    @Test func translateScreenShowsTheOverLimitRefusal() throws {
        let model = runModel()
        model.sourceSelection = "en"
        model.targetSelection = "ms"
        model.sourceText = "A long passage."
        model.countTokens = { _ in 2049 }          // the view's `.task` re-counts on render
        model.promptTokenCount = 2049
        model.start(stage: TranslateGemmaStage(id: "render", name: "TranslateGemma 4B",
                                               sourceCode: "en", targetCode: "ms") { _ in
            TranslateGemmaEngine.Outcome(text: "never", promptTokens: 0, generatedTokens: 0, cancelled: false)
        })
        let image = render(screen(model), width: 660, height: 640)
        #expect(image != nil)
        save(image, "after-translate-over-limit")
        #expect(model.isOverLimit)
        #expect(model.errorText?.contains("2,048") == true)
    }
}
