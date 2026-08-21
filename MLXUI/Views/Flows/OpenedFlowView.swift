import SwiftUI

/// The detail view for a `.cat` file the user opened from disk (CFM-R5-5). Read-only:
/// shows the parsed document's rows, the validation result (or each issue with its
/// dotted row path and ⚠-free wording), and the raw `.cat` text. No Run — an opened
/// file is inspected, not executed.
struct OpenedFlowView: View {
    let opened: OpenedCatFlow

    @Environment(AppState.self) private var appState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                if opened.issues.isEmpty {
                    Label("This flow validates cleanly.", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                } else {
                    issuesList
                }
                rowsSummary
                rawText
            }
            .padding(20)
        }
        .navigationTitle(opened.displayName)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Label(opened.displayName, systemImage: "doc.plaintext")
                    .font(.title2.weight(.semibold))
                Text(opened.parsed.version.isEmpty ? "headerless" : "catflow \(opened.parsed.version)")
                    .font(.caption.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 4))
            }
            Text(opened.url.path)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var issuesList: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("\(opened.issues.count) issue\(opened.issues.count == 1 ? "" : "s")", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundStyle(.orange)
            ForEach(Array(opened.issues.enumerated()), id: \.offset) { _, issue in
                HStack(alignment: .top, spacing: 8) {
                    Text("row \(issue.row)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .frame(width: 70, alignment: .trailing)
                    Text(issue.message)
                        .textSelection(.enabled)
                }
                .font(.callout)
            }
        }
        .padding(12)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    private var rowsSummary: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Rows")
                .font(.headline)
            Text(opened.parsed.rows.enumerated()
                .map { "\($0.offset + 1). \($0.element.task ?? ($0.element.blockKind?.rawValue ?? "?"))" }
                .joined(separator: "\n"))
                .font(.callout.monospaced())
                .textSelection(.enabled)
        }
    }

    private var rawText: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Source")
                .font(.headline)
            Text(opened.rawText)
                .font(.callout.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
        }
    }
}
