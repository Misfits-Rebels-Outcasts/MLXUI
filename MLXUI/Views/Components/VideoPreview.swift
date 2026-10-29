import SwiftUI
import AppKit
import AVKit

/// A video preview with the system playback controls: an AppKit `AVPlayerView` in an `NSViewRepresentable`. SwiftUI's `VideoPlayer`
/// (`_AVKit_SwiftUI`) aborted the process on macOS 27 as the preview appeared (SIGABRT in `getSuperclassMetadata`, MLXUI Pro 0.26 (33),
/// 2026-10-29), so no screen uses it; `VideoPreviewHostingTests` hosts the finished-run screens to keep it that way.
struct VideoPreview: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = false
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player { view.player = player }
    }
}
