import Foundation

/// CP-4: the width of the conversations column. Pure, so the clamp rule is testable.
nonisolated enum ChatSidebarWidth {
    static let defaultWidth: Double = 210
    static let minWidth: Double = 160
    static let maxWidth: Double = 420

    /// NaN and infinities fall back to the default; everything else is clamped to `minWidth...maxWidth`.
    static func clamp(_ value: Double) -> Double {
        guard value.isFinite else { return defaultWidth }
        return min(max(value, minWidth), maxWidth)
    }
}
