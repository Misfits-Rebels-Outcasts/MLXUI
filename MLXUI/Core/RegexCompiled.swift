import Foundation

extension NSRegularExpression {
    /// Compile a **static literal** regular expression. A literal pattern cannot fail at
    /// runtime, but `policies.md` bans bare `try!` flatly — this traps with the pattern in
    /// the message if one somehow does (CFM-FIX-6/L1). Replaces every `try!
    /// NSRegularExpression(pattern:)` in FlowKit/Modules.
    nonisolated static func compiled(
        _ pattern: String,
        options: NSRegularExpression.Options = []
    ) -> NSRegularExpression {
        do {
            return try NSRegularExpression(pattern: pattern, options: options)
        } catch {
            preconditionFailure("invalid regular expression literal '\(pattern)': \(error)")
        }
    }
}
