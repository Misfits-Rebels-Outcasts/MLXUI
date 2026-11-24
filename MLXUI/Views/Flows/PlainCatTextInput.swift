import SwiftUI
import AppKit

/// R3-FIX-1 (owner ruling, Boo, 2026-11-24) — a `.cat` file is text the runtime parses, so a text control that edits one must
/// store exactly what was typed. macOS's smart quotes, smart dashes and text replacement turn `only="person"` into
/// `only=“person”`, which the row refuses. `.plainCatTextInput()` switches those three (and autocorrect) off on the control's
/// underlying `NSTextView` / field editor.
///
/// SwiftUI's `TextEditor` / `TextField` expose no modifier for smart quotes, so the control's AppKit text view is found next to
/// an anchor view in the background: at once for a `TextEditor` (its `NSTextView` already exists) and again each time editing
/// begins (a `TextField` borrows the window's shared field editor only then).
extension View {
    func plainCatTextInput() -> some View {
        background(PlainCatTextAnchor())
    }
}

enum PlainCatText {
    /// Turn off every automatic substitution on one text view. Pure AppKit — the one place the list lives.
    static func configure(_ textView: NSTextView) {
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isAutomaticTextCompletionEnabled = false
        textView.smartInsertDeleteEnabled = false
    }

    /// Every `NSTextView` below `root` (a `TextEditor`'s lives inside an `NSScrollView`).
    static func textViews(in root: NSView) -> [NSTextView] {
        var found: [NSTextView] = []
        func walk(_ view: NSView) {
            if let textView = view as? NSTextView { found.append(textView) }
            view.subviews.forEach(walk)
        }
        walk(root)
        return found
    }
}

private struct PlainCatTextAnchor: NSViewRepresentable {
    func makeNSView(context: Context) -> AnchorView { AnchorView() }
    func updateNSView(_ view: AnchorView, context: Context) { view.applyNow() }

    final class AnchorView: NSView {
        private var observer: NSObjectProtocol?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            applyNow()
            guard observer == nil else { return }
            // A `TextField`'s editor appears when editing begins; configure it before the first character lands.
            observer = NotificationCenter.default.addObserver(
                forName: NSText.didBeginEditingNotification, object: nil, queue: .main
            ) { [weak self] note in
                guard let self, let textView = note.object as? NSTextView,
                      let scope = self.superview, textView.isDescendant(of: scope) else { return }
                MainActor.assumeIsolated { PlainCatText.configure(textView) }
            }
        }

        /// Configure whatever text views already sit beside this anchor (a `TextEditor`); deferred one turn so SwiftUI has
        /// finished attaching the control to the hierarchy.
        func applyNow() {
            DispatchQueue.main.async { [weak self] in
                guard let scope = self?.superview else { return }
                PlainCatText.textViews(in: scope).forEach(PlainCatText.configure)
            }
        }

        deinit { if let observer { NotificationCenter.default.removeObserver(observer) } }
    }
}
