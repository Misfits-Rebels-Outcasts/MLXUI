import SwiftUI

/// The five-state status dot for a flow row: `○` gray (not run), `●` running, `✓` green
/// (succeeded), `✗` red (failed), `△` yellow (needs attention). R1 only ever shows gray;
/// R2 and R3 drive the rest. Five states defined once so the enum is stable (CFM-R1-5).
nonisolated enum FlowStatus: Sendable {
    case notRun, running, succeeded, failed, needsAttention

    /// The literal glyph the `.cat`-era spec draws.
    var glyph: String {
        switch self {
        case .notRun:         return "○"
        case .running:        return "●"
        case .succeeded:      return "✓"
        case .failed:         return "✗"
        case .needsAttention: return "△"
        }
    }

    var color: Color {
        switch self {
        case .notRun:         return .gray
        case .running:        return .blue
        case .succeeded:      return .green
        case .failed:         return .red
        case .needsAttention: return .yellow
        }
    }
}

/// The dot itself — a small circle that reads as a status glyph at list size.
struct FlowStatusDot: View {
    let status: FlowStatus

    var body: some View {
        Text(status.glyph)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(status.color)
            .frame(width: 16, height: 16)
            .accessibilityLabel(accessibilityLabel)
    }

    private var accessibilityLabel: String {
        switch status {
        case .notRun:         return "Not run"
        case .running:        return "Running"
        case .succeeded:      return "Succeeded"
        case .failed:         return "Failed"
        case .needsAttention: return "Needs attention"
        }
    }
}
