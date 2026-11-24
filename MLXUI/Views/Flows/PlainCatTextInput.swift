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

    /// A `TextField`'s shared field editor is an `NSTextView` that sits inside the field; its frame in window coordinates is the field's.
    static func fieldEditorOwnerFrame(_ textView: NSTextView) -> NSRect? {
        guard let field = textView.superview?.superview as? NSTextField ?? textView.superview as? NSTextField else { return nil }
        return field.convert(field.bounds, to: nil)
    }

    /// The text views drawn where `anchor` is — the control the modifier was attached to. The anchor sits in the control's
    /// `.background`, so its own frame *is* the control's frame; its superview is only SwiftUI's wrapper and need not contain the
    /// text view at all (the first R7 run: the box showed `only=“person”` with the modifier applied). Matches by overlap in window
    /// coordinates, so it finds the `NSTextView` however SwiftUI nested it, and never a neighbouring control's.
    static func textViews(coveredBy anchor: NSView) -> [NSTextView] {
        guard let window = anchor.window, let content = window.contentView else { return [] }
        let area = anchor.convert(anchor.bounds, to: nil)
        guard area.width > 1, area.height > 1 else { return [] }
        return textViews(in: content).filter { textView in
            let frame = textView.convert(textView.bounds, to: nil)
            let overlap = frame.intersection(area)
            guard !overlap.isNull else { return false }
            return overlap.width * overlap.height >= 0.5 * min(frame.width * frame.height, area.width * area.height)
        }
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
                guard let self, let textView = note.object as? NSTextView else { return }
                MainActor.assumeIsolated {
                    guard PlainCatText.textViews(coveredBy: self).contains(where: { $0 === textView })
                            || (self.window != nil && textView.window === self.window
                                && PlainCatText.fieldEditorOwnerFrame(textView).map { $0.intersects(self.convert(self.bounds, to: nil)) } == true)
                    else { return }
                    EditorLog.step("didBeginEditing reached an anchored text view; quote substitution was", textView.isAutomaticQuoteSubstitutionEnabled)
                    PlainCatText.configure(textView)
                }
            }
        }

        /// Configure whatever text views already sit beside this anchor (a `TextEditor`); deferred one turn so SwiftUI has
        /// finished attaching the control to the hierarchy.
        func applyNow() {
            // A control inside a `DisclosureGroup` (Raw settings) has no frame the first turn after it appears, so try again once
            // layout has settled; `configure` is idempotent.
            for delay in [0.0, 0.15, 0.6] {
                DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                    guard let self else { return }
                    let found = PlainCatText.textViews(coveredBy: self)
                    EditorLog.step("anchor applyNow; text views found over the control", count: found.count)
                    found.forEach(PlainCatText.configure)
                }
            }
        }

        deinit { if let observer { NotificationCenter.default.removeObserver(observer) } }
    }
}
