import SwiftUI

/// CP-5: the four text-size steps for the chat, as explicit point sizes (not Dynamic Type).
/// `.standard` is "what the app drew before this existed": it overrides nothing, so every call
/// site keeps its original text style. The other steps shift each role's macOS point size.
nonisolated enum ChatTextSize: String, CaseIterable, Sendable, Equatable {
    case small, standard, large, extraLarge

    /// Where the size is read: the two `@AppStorage` keys.
    static let messageKey = "chatMessageTextSize"
    static let listKey = "chatListTextSize"

    /// What the text is. Base points are today's macOS sizes for the styles in use
    /// (`.body` 13, `.callout` 12, `.caption` 10, `.caption2` 10).
    enum Role: Sendable, Equatable {
        case message, toolResult, listTitle, listDetail

        var standardPoints: Double {
            switch self {
            case .message: return 13
            case .toolResult: return 10
            case .listTitle: return 12
            case .listDetail: return 10
            }
        }
    }

    /// Points added to every role's standard size.
    var delta: Double {
        switch self {
        case .small: return -2
        case .standard: return 0
        case .large: return 2
        case .extraLarge: return 4
        }
    }

    func points(for role: Role) -> Double { role.standardPoints + delta }

    /// `nil` for `.standard`: the caller's own font stays in force, so nothing shifts.
    func overridePoints(for role: Role) -> Double? {
        self == .standard ? nil : points(for: role)
    }

    /// The font for a role: the call site's own style at `.standard` (so nothing shifts), explicit points otherwise.
    /// Use this *instead of* `.font(x).chatTextSize(...)` — an inner `.font` beats an outer override (CP-5-FIX-1).
    func font(_ role: Role, standard: Font) -> Font {
        overridePoints(for: role).map { Font.system(size: $0) } ?? standard
    }

    var title: String {
        switch self {
        case .small: return "Small"
        case .standard: return "Standard"
        case .large: return "Large"
        case .extraLarge: return "Extra Large"
        }
    }

    /// A stored string that names no step (or nothing) reads as `.standard`.
    init(stored: String) { self = ChatTextSize(rawValue: stored) ?? .standard }

    /// One step bigger / smaller; clamps at the ends.
    var steppedUp: ChatTextSize {
        let all = Self.allCases
        let i = all.firstIndex(of: self) ?? 1
        return all[min(i + 1, all.count - 1)]
    }
    var steppedDown: ChatTextSize {
        let all = Self.allCases
        let i = all.firstIndex(of: self) ?? 1
        return all[max(i - 1, 0)]
    }
    static var reset: ChatTextSize { .standard }
}

extension View {
    /// Applies the chosen size to a role; at `.standard` it returns the view untouched.
    @ViewBuilder
    func chatTextSize(_ role: ChatTextSize.Role, _ size: ChatTextSize) -> some View {
        if let points = size.overridePoints(for: role) {
            self.font(.system(size: points))
        } else {
            self
        }
    }
}
