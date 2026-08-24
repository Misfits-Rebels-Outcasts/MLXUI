import SwiftUI

/// One row of the flow list (CFM-R6-2 + R7-5): the status dot leftmost, then the row's
/// canonical serialized line(s) in a monospaced font, each in its own horizontal scroll
/// container (a long line scrolls sideways while the page never does). Selection uses a
/// `.simultaneousGesture` tap; a block row gets a disclosure chevron to expand/collapse its
/// (indented) children.
///
/// The row text is deliberately **not** `.textSelection(.enabled)` — that turns regions into
/// an I-beam insertion point that captures clicks (and an I-beam on a non-editable line is
/// wrong); the inspector pane shows the selected row's output as selectable text instead.
struct FlowSerializedRow: View {
    /// The row's serialized lines (from `CatSerializer.serializeLines`'s line ranges).
    let lines: [String]
    let status: FlowStatus
    /// Called when the row is clicked — selects it for the inspector.
    var onSelect: () -> Void
    /// R7-2: a `<parallel>` block runs its chains sequentially (semantic, not concurrent) —
    /// say so on hover rather than letting the word imply speed.
    var isSemanticParallel: Bool = false
    /// R7-5: a block row shows a disclosure chevron to expand/collapse its children.
    var isCollapsible: Bool = false
    var isCollapsed: Bool = false
    var onToggleCollapse: () -> Void = {}
    /// CFM-R11-2 visibility: this row's ✓ came from the flow's output cache this run, not a
    /// fresh computation — show it so a run's cache hits are obvious.
    var cached: Bool = false

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if isCollapsible {
                Button(action: onToggleCollapse) {
                    Image(systemName: isCollapsed ? "chevron.right" : "chevron.down")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 12)
                }
                .buttonStyle(.plain)
                .padding(.top, 5)
            }
            FlowStatusDot(status: status)
                .padding(.top, 4)
            ScrollView(.horizontal, showsIndicators: false) {
                Text(lines.joined(separator: "\n"))
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .simultaneousGesture(TapGesture().onEnded { onSelect() })
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if cached {
                Text("cached")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(.quaternary.opacity(0.5), in: Capsule())
                    .padding(.top, 3)
                    .help("This row's output was already computed — it was replayed from the flow cache, not recomputed.")
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .simultaneousGesture(TapGesture().onEnded { onSelect() })
        .help(isSemanticParallel ? "Runs its chains one at a time — semantic parallel, not concurrent." : "")
    }
}
