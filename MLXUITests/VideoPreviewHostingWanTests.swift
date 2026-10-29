import Testing
import AppKit
import SwiftUI
@testable import MLXUI

/// See `VideoPreviewHostingTests`: the Wan result screen, with a finished video, in a real window.
@MainActor
struct VideoPreviewHostingWanTests {

    @Test func theWanResultScreenRendersItsPreview() async throws {
        let url = try await RIFETestVideo.make(frames: 8, fps: 8)
        defer { try? FileManager.default.removeItem(at: url) }
        let model = WanRunModel()
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 700, height: 800), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: WanVideoRunView(modelDisplayName: "Wan", modelID: "x", model: model))
        window.contentView = host
        window.orderFront(nil)
        host.layoutSubtreeIfNeeded()
        model.videoURL = url        // the view's onChange builds the player and shows the preview
        for _ in 0 ..< 10 {
            try? await Task.sleep(for: .milliseconds(50))
            host.layoutSubtreeIfNeeded()
            host.displayIfNeeded()
        }
        #expect(host.fittingSize.width > 0)
        window.orderOut(nil)
    }
}
