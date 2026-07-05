import SwiftUI
import AppKit

/// Error banner shared by the run views. The message is a **selectable** `Text` with a
/// one-click Copy button, so the full error (e.g. `unsupportedModelType("paddleocr_vl")`) can
/// be grabbed for a bug report instead of being locked inside a non-selectable `Label`.
struct RunErrorBar: View {
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.caption)
                .foregroundStyle(.red)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(message, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("Copy error")
        }
        .padding(8)
        .background(.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }
}
