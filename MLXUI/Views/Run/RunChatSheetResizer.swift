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
        private var observers: [NSObjectProtocol] = []

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let window else { stopObserving(); return }
            // The sheet is still being assembled here; configure on the next turn.
            DispatchQueue.main.async { [weak self, weak window] in
                guard let self, let window else { return }
                self.configure(window)
            }
        }

        deinit { observers.forEach { NotificationCenter.default.removeObserver($0) } }

        private func stopObserving() {
            observers.forEach { NotificationCenter.default.removeObserver($0) }
            observers = []
        }

        /// SwiftUI re-lays out the sheet's window and can reset its limits, so both minimums are (re)applied here
        /// on every resize and window update, and a frame that is already below the floor is snapped back.
        private func enforceFloor(_ window: NSWindow) {
            let content = RunChatSheetSize.minSize
            let frameMin = window.frameRect(forContentRect: NSRect(origin: .zero, size: content)).size
            if window.contentMinSize != content { window.contentMinSize = content }
            if window.minSize != frameMin { window.minSize = frameMin }
            if !window.styleMask.contains(.resizable) { window.styleMask.insert(.resizable) }
            let current = window.contentRect(forFrameRect: window.frame).size
            if let target = RunChatSheetSize.snapTarget(for: current) {
                var frame = window.frame
                let newSize = window.frameRect(forContentRect: NSRect(origin: .zero, size: target)).size
                frame.origin.y = frame.maxY - newSize.height          // keep the top edge where it was
                frame.size = newSize
                window.setFrame(frame, display: true)
            }
        }

        private func configure(_ window: NSWindow) {
            enforceFloor(window)
            let screen = window.screen?.visibleFrame.size ?? NSScreen.main?.visibleFrame.size ?? .zero
            if let raw = UserDefaults.standard.string(forKey: RunChatSheetSize.storageKey),
               let saved = RunChatSheetSize.parse(raw) {
                window.setContentSize(RunChatSheetSize.clamp(saved, screen: screen))
            }
            stopObserving()
            let center = NotificationCenter.default
            for name in [NSWindow.didResizeNotification, NSWindow.didUpdateNotification] {
                observers.append(center.addObserver(forName: name, object: window, queue: .main) { note in
                    MainActor.assumeIsolated {
                        if let window = note.object as? NSWindow { self.enforceFloor(window) }
                    }
                })
            }
            observers.append(center.addObserver(
                forName: NSWindow.didEndLiveResizeNotification, object: window, queue: .main
            ) { note in
                MainActor.assumeIsolated {
                    guard let window = note.object as? NSWindow else { return }
                    let size = window.contentRect(forFrameRect: window.frame).size
                    // Never write a size below the floor (CP-3-SPIKE-FIX-1).
                    guard let value = RunChatSheetSize.storableValue(for: size) else { return }
                    UserDefaults.standard.set(value, forKey: RunChatSheetSize.storageKey)
                }
            })
        }
    }
}
