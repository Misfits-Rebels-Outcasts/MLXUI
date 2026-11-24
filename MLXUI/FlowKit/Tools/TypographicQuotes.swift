import Foundation

/// R8 (owner ruling, Boo, 2026-11-24) — belt and braces for macOS smart quotes. A `.cat` row's settings are parsed text, so
/// `only=“person”` (curly) is a different, refused string from `only="person"`. Whatever slips through the text view's own
/// substitution switches is straightened here, **in the Raw settings box and the setting value fields only** — never in a prompt,
/// an instruction or Template text, where a typographic quote is legitimate content.
nonisolated enum TypographicQuotes {
    /// “ ” „ ‟ → `"`  ·  ‘ ’ → `'`
    static func straightened(_ text: String) -> String {
        guard text.contains(where: { double.contains($0) || single.contains($0) }) else { return text }
        return String(text.map { double.contains($0) ? "\"" : (single.contains($0) ? "'" : $0) })
    }
    private static let double: Set<Character> = ["\u{201C}", "\u{201D}", "\u{201E}", "\u{201F}"]
    private static let single: Set<Character> = ["\u{2018}", "\u{2019}"]
}
