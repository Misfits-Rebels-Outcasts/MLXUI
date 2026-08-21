import SwiftUI

/// One row of a flow list. Gray status dot (leftmost), the display number, the friendly
/// task name, and a subtitle — the model display name if the row has one, else a one-line
/// description. Block rows render their children indented beneath them.
///
/// Numbers are scoped to the enclosing block (matching the Python: rows number 1…n within
/// their own block). `referenceScope` is the sibling list a row's `(N)` references resolve
/// against — the document's top-level rows for a top-level row, a block's `children` for a
/// block child. Passed down rather than derived so block rows can hand their children their
/// own scope.
struct FlowRowView: View {
    let row: Row
    let number: Int
    let referenceScope: [Row]
    let status: FlowStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                FlowStatusDot(status: status)
                Text("\(number).")
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 24, alignment: .trailing)
                Text(FlowRowSummary.taskName(for: row))
                    .font(.callout)
                    .fontWeight(.medium)
                if let ref = FlowRowSummary.referenceLabel(for: row, in: referenceScope) {
                    Text(ref)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(FlowRowSummary.subtitle(for: row))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !row.children.isEmpty {
                    ForEach(Array(row.children.enumerated()), id: \.element.id) { index, child in
                        FlowRowView(row: child,
                                    number: index + 1,
                                    referenceScope: row.children,
                                    status: .notRun)
                            .padding(.leading, 28)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }
}
