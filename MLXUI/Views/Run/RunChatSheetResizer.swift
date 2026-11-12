import SwiftUI
import AppKit

/// CP-3-SPIKE: makes the Run sheet's window user-resizable.
///
/// **Relies on undocumented AppKit behaviour for SwiftUI sheets.** A SwiftUI `.sheet` is an AppKit sheet window that
/// is not resizable by default whatever its content's `maxWidth`/`maxHeight` say. Once this view is in a window we
/// insert `.resizable` into its `styleMask` and set a content minimum size. Nothing documents that a sheet honours
/// that; if a future macOS stops doing so the sheet just goes back to a fixed size (nothing else depends on it).
/// The last content size is kept under `RunChatSheetSize.storageKey` and re-applied (clamped to the screen) on open.
struct RunChatSheetResizer: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView { ResizerView() }
    func updateNSView(_ nsView: NSView, context: Context) {}

    final class ResizerView: NSView {
        private var endResize: NSObjectProtocol?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let window else { stopObserving(); return }
            // The sheet is still being assembled here; configure on the next turn.
            DispatchQueue.main.async { [weak self, weak window] in
                guard let self, let window else { return }
                self.configure(window)
            }
        }

        deinit { if let endResize { NotificationCenter.default.removeObserver(endResize) } }

        private func stopObserving() {
            if let endResize { NotificationCenter.default.removeObserver(endResize) }
            endResize = nil
        }

        private func configure(_ window: NSWindow) {
            window.styleMask.insert(.resizable)
            window.contentMinSize = RunChatSheetSize.minSize
            let screen = window.screen?.visibleFrame.size ?? NSScreen.main?.visibleFrame.size ?? .zero
            if let raw = UserDefaults.standard.string(forKey: RunChatSheetSize.storageKey),
               let saved = RunChatSheetSize.parse(raw) {
                window.setContentSize(RunChatSheetSize.clamp(saved, screen: screen))
            }
            stopObserving()
            endResize = NotificationCenter.default.addObserver(
                forName: NSWindow.didEndLiveResizeNotification, object: window, queue: .main
            ) { note in
                MainActor.assumeIsolated {
                    guard let window = note.object as? NSWindow else { return }
                    let size = window.contentRect(forFrameRect: window.frame).size
                    UserDefaults.standard.set(RunChatSheetSize.encode(size), forKey: RunChatSheetSize.storageKey)
                }
            }
        }
    }
}
