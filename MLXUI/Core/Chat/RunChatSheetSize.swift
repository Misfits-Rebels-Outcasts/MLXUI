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
}
