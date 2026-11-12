import Foundation
import CoreGraphics

/// CP-3-SPIKE: the Run sheet's persisted size. Pure, so the minimum, the parse and the screen clamp are testable.
nonisolated enum RunChatSheetSize {
    /// The same floor `RunChatView` declares (`minWidth`/`minHeight`).
    static let minSize = CGSize(width: 840, height: 520)
    /// `@AppStorage` / `UserDefaults` key; the value is "w,h" in points (content size).
    static let storageKey = "runChatSheetSize"

    static func encode(_ size: CGSize) -> String {
        "\(Int(size.width.rounded())),\(Int(size.height.rounded()))"
    }

    /// `nil` for anything that is not two finite positive numbers (empty, garbage, NaN, negative).
    static func parse(_ raw: String) -> CGSize? {
        let parts = raw.split(separator: ",", omittingEmptySubsequences: false)
        guard parts.count == 2, let w = Double(parts[0]), let h = Double(parts[1]),
              w.isFinite, h.isFinite, w > 0, h > 0 else { return nil }
        return CGSize(width: w, height: h)
    }

    /// At least `minSize`, at most the screen's visible size (a screen smaller than the minimum never shrinks below it).
    static func clamp(_ size: CGSize, screen: CGSize) -> CGSize {
        let maxW = max(screen.width.isFinite ? screen.width : minSize.width, minSize.width)
        let maxH = max(screen.height.isFinite ? screen.height : minSize.height, minSize.height)
        let w = size.width.isFinite ? size.width : minSize.width
        let h = size.height.isFinite ? size.height : minSize.height
        return CGSize(width: min(max(w, minSize.width), maxW), height: min(max(h, minSize.height), maxH))
    }

    /// CP-3-SPIKE-FIX-1: whether `size` meets the floor in both dimensions.
    static func meetsMinimum(_ size: CGSize) -> Bool {
        size.width.isFinite && size.height.isFinite && size.width >= minSize.width && size.height >= minSize.height
    }

    /// CP-3-SPIKE-FIX-1: the string to write for a resize that just ended, or `nil` when it must not be written
    /// (below the floor, or not a real size). A size below the minimum is never stored.
    static func storableValue(for size: CGSize) -> String? {
        meetsMinimum(size) ? encode(size) : nil
    }

    /// CP-3-SPIKE-FIX-1: the content size to snap a window back to when a resize went below the floor, else `nil`.
    static func snapTarget(for contentSize: CGSize) -> CGSize? {
        meetsMinimum(contentSize) ? nil : clamp(contentSize, screen: CGSize(width: Double.greatestFiniteMagnitude, height: Double.greatestFiniteMagnitude))
    }
}
