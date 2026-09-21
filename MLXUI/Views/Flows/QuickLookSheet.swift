import SwiftUI
import QuickLookUI

/// OV-2: previews a saved file with Quick Look — the one viewer that needs no LaunchServices
/// hand-off, so it is the one path the sandbox can't defeat (`RSI/DelegateOutputViewerBacklog.md`
/// §7 trap 3). Presented as a `.sheet`, sized ~720×560 and resizable, with one Done button.
///
/// Wraps `QLPreviewView`, never `QLPreviewPanel` — the panel is driven off the responder chain
/// (`acceptsPreviewPanelControl`/`beginPreviewPanelControl`), which means fighting a SwiftUI
/// hierarchy for first responder. `QLPreviewView` in a sheet needs none of that (§2.4).
struct QuickLookSheet: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            QuickLookPreview(url: url)
                .frame(minWidth: 480, minHeight: 360)
            Divider()
            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(10)
        }
        .frame(minWidth: 480, idealWidth: 720, minHeight: 360, idealHeight: 560)
    }
}

/// `NSViewRepresentable` wrapper around `QLPreviewView` (`.compact` style) — the first
/// `NSViewRepresentable` in this target (`RSI/DelegateOutputViewerBacklog.md` fact 25).
///
/// `QLPreviewView`'s only initializer, `init?(frame:style:)`, is failable — represented as
/// plain `NSView` (rather than `QLPreviewView`) so the vanishingly rare nil case degrades to a
/// blank view instead of needing a force-unwrap (no force-unwraps, project convention).
private struct QuickLookPreview: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> NSView {
        guard let preview = QLPreviewView(frame: .zero, style: .compact) else { return NSView() }
        preview.autostarts = true
        preview.previewItem = url as QLPreviewItem
        return preview
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        guard let preview = nsView as? QLPreviewView else { return }
        if (preview.previewItem as? URL) != url {
            preview.previewItem = url as QLPreviewItem
        }
    }
}
