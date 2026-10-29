import Testing
import AppKit
import SwiftUI
import AVFoundation
@testable import MLXUI

/// The finished-run screens embed a video preview. SwiftUI's `VideoPlayer` (`_AVKit_SwiftUI`) aborted the process on macOS 27 while
/// instantiating its generic metadata as the preview appeared (MLXUI Pro 0.26 (33), 2026-10-29, SIGABRT in `getSuperclassMetadata`).
/// Nothing rendered these views before, so these tests host the result state in a real window and lay it out.
@MainActor
struct VideoPreviewHostingTests {

    private func layOut<V: View>(_ view: V, size: NSSize) async {
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: view)
        window.contentView = host
        window.orderFront(nil)
        host.layoutSubtreeIfNeeded()
        for _ in 0 ..< 10 {
            try? await Task.sleep(for: .milliseconds(50))
            host.layoutSubtreeIfNeeded()
            host.displayIfNeeded()
        }
        #expect(host.fittingSize.width > 0)
        window.orderOut(nil)
    }

    @Test func theRIFEResultScreenRendersItsPreview() async throws {
        let url = try await RIFETestVideo.make(frames: 8, fps: 8)
        defer { try? FileManager.default.removeItem(at: url) }
        let model = RIFERunModel()
        model.showResult(at: url, summary: "Done: 15 frames at 16 fps")
        #expect(model.player != nil)
        let stage = RIFEStage(modelDirectory: FileManager.default.temporaryDirectory)
        await layOut(RIFERunView(modelDisplayName: "RIFE 4.25", license: "mit", modelID: "mlx-community--RIFE-4.25",
                                 handedStage: stage, model: model), size: NSSize(width: 640, height: 720))
    }
}
