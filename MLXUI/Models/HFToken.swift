import Foundation

/// Pure validation/normalization for HuggingFace access tokens. Kept free of any I/O or
/// Keychain access so it can be unit-tested directly (see RSI/evals/eval-plan.md, gate G2).
enum HFTokenValidator {
    /// Trims surrounding whitespace/newlines (common when pasting a token).
    static func normalized(_ raw: String) -> String {
        raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A plausible HF token: non-empty, no embedded whitespace, and the `hf_` prefix HF uses.
    /// Used only to gate the Save button — it is a sanity check, not authentication.
    static func isPlausible(_ raw: String) -> Bool {
        let token = normalized(raw)
        return token.hasPrefix("hf_")
            && token.count > 3
            && !token.contains(where: \.isWhitespace)
    }
}
