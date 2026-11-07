import Foundation

/// MC2-3 — **the one language table** the Translate screen, the TranslateGemma engine and the `Translate` workflow row
/// all read (owner decision 6). It is generated from the pinned `chat_template.jinja`'s own `languages` dict — 581
/// codes for 161 display names — and checked against the pinned copy in `Fixtures/TranslateGemma/` by
/// `TranslateGemmaTemplateTests` (every code present, both directions, plus the three reference renders).
///
/// The table's two jobs:
///  * **the UI lists** show one entry per display name, with the name's base code (no region/script subtag where the
///    template has one; `Filipino` is the sole fallback, `fil-PH`). Chinese is split into Simplified (`zh-Hans`) and
///    Traditional (`zh-Hant`) because the output script differs — the template has one `Chinese` name for all 13 of
///    its Chinese codes.
///  * **`to=` resolution** accepts a display name (case-insensitive) or any code in the template (underscore written
///    for hyphen, any case) and answers the code the request should send.
///
/// An unknown token is **not** silently approximated: the caller refuses it, because the template's lookup renders a
/// broken prompt for an unknown code rather than raising (MC2 Trap 2).
nonisolated enum TranslateGemmaLanguages {

    /// One display-name entry. `code` is the code the lists show; `aliases` are every template code that names the
    /// same language, in template order (the Chinese pair carries only its own split code; `chineseCodes` has the rest).
    struct Language: Sendable, Hashable, Identifiable {
        let name: String
        let code: String
        let aliases: [String]

        var id: String { code }
    }

    /// The generated 162 entries: 161 template names with Chinese split in place, template order.
    static let all: [Language] = entries

    /// The short favourites block shown at the top of both lists (owner decision 6), in that order.
    static let favourites: [Language] = ["English", "Malay", "Chinese (Simplified)", "Chinese (Traditional)",
                                         "Indonesian", "Tamil", "Japanese", "Korean"]
        .compactMap { language(named: $0) }

    static let simplifiedChineseName = "Chinese (Simplified)"
    static let traditionalChineseName = "Chinese (Traditional)"
    static let simplifiedChineseCode = "zh-Hans"
    static let traditionalChineseCode = "zh-Hant"

    /// Owner decision 8: the card's stated input limit, in rendered prompt tokens.
    static let maxPromptTokens = 2048
    /// Owner decision 8: the output limit.
    static let maxOutputTokens = 2048

    // MARK: - Lookup

    /// The entry whose display name is `name` (`Chinese (Simplified)` / `Chinese (Traditional)` included), compared
    /// case-insensitively. `nil` for the template's bare `Chinese` name — the split is deliberate, so an ambiguous
    /// `Chinese` is refused and the person (or the row) must say which script.
    static func language(named name: String) -> Language? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return all.first { $0.name.caseInsensitiveCompare(trimmed) == .orderedSame }
    }

    /// The table's own spelling of a code written with `_` for `-`, or in another case (`en_gb` → `en-GB`,
    /// `ZH-HANS` → `zh-Hans`). `nil` when the code is not one the pinned template accepts.
    static func canonicalTemplateCode(_ code: String) -> String? {
        let normalized = code.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "_", with: "-")
        guard !normalized.isEmpty else { return nil }
        if let exact = templateCodes.first(where: { $0.code == normalized }) { return exact.code }
        return templateCodes.first(where: { $0.code.caseInsensitiveCompare(normalized) == .orderedSame })?.code
    }

    /// The UI list entry a `to=` token names: a display name, or a code in the table (underscore ≡ hyphen). A Chinese
    /// code maps to the entry for its script; the other Chinese codes are only ever reachable as codes, never as names.
    static func language(matching token: String) -> Language? {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if let named = language(named: trimmed) { return named }
        guard let code = canonicalTemplateCode(trimmed) else { return nil }
        if chineseCodes.contains(code) {
            return language(named: isTraditionalChinese(code) ? traditionalChineseName : simplifiedChineseName)
        }
        return all.first { $0.aliases.contains(code) }
    }

    /// The code a translation request should send for a `to=` token, or `nil` when the token is not in the table.
    /// A display name resolves to its base code; a code is kept as written (canonical table spelling), so `to=en-GB`
    /// asks for the regional English variant the template names — the same code `to=` accepts for the source.
    static func requestCode(for token: String) -> String? {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if let named = language(named: trimmed) { return named.code }
        return canonicalTemplateCode(trimmed)
    }

    /// The display name **the template itself** uses for a code (`zh-Hans` → "Chinese", not our split label). The
    /// request must say what the template's own table says; the UI labels are ours.
    static func templateName(for code: String) -> String? {
        guard let canonical = canonicalTemplateCode(code) else { return nil }
        return templateCodes.first { $0.code == canonical }?.name
    }

    /// The display name for a template code, for labels (`en` → "English", `zh-Hant` → "Chinese (Traditional)").
    static func name(forTemplateCode code: String) -> String? {
        guard let canonical = canonicalTemplateCode(code) else { return nil }
        if chineseCodes.contains(canonical) {
            return isTraditionalChinese(canonical) ? traditionalChineseName : simplifiedChineseName
        }
        return all.first { $0.aliases.contains(canonical) }?.name
    }

    // MARK: - Detection

    /// Maps a language code from `NLLanguageRecognizer` (BCP-47, e.g. `en`, `zh-Hans`, `fil`, `en-GB`, `de-AT`) onto a
    /// code the pinned template accepts, or `nil` when the template has no language for it.
    ///
    /// The resolver tries, in order: the code itself; a Chinese script the tag names; each progressively shorter
    /// subtag prefix (`zh-Hant-TW` → `zh-Hant`); then the base language among the template's aliases (`fil` → `fil-PH`,
    /// its only code). Chinese is answered with the **split** representative (`zh-Hans` / `zh-Hant`), never the bare
    /// `zh`, so the person is shown the script that will be produced. Implementer's call, pending owner confirmation:
    /// a bare `zh` (no script in the recogniser's answer) reads as Simplified, `zh-CH`/`*-TW` as Traditional.
    static func code(forDetectedLanguage raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "_", with: "-")
        guard !trimmed.isEmpty else { return nil }

        if let exact = canonicalTemplateCode(trimmed) {
            return chineseCodes.contains(exact) ? representativeChineseCode(for: exact) : exact
        }
        if trimmed.localizedCaseInsensitiveContains("Hant") { return traditionalChineseCode }
        if trimmed.localizedCaseInsensitiveContains("Hans") { return simplifiedChineseCode }

        var parts = trimmed.split(separator: "-").map(String.init)
        while parts.count > 1 {
            parts.removeLast()
            if let shorter = canonicalTemplateCode(parts.joined(separator: "-")) {
                return chineseCodes.contains(shorter) ? representativeChineseCode(for: shorter) : shorter
            }
        }

        guard let base = parts.first?.lowercased(), !base.isEmpty else { return nil }
        // The base language among the aliases: the template's bare code where it has one, else its only code.
        if let language = all.first(where: { $0.aliases.contains { alias in
            alias.split(separator: "-").first?.lowercased() == base
        } }) {
            return language.code
        }
        return nil
    }

    // MARK: - Chinese

    /// Whether a Chinese template code names the Traditional script. `zh-TW` and `zh-CH` are region spellings of the
    /// same two scripts; everything else without `Hant` (including the bare `zh`) reads as Simplified.
    static func isTraditionalChinese(_ code: String) -> Bool {
        code.localizedCaseInsensitiveContains("Hant")
            || code.caseInsensitiveCompare("zh-TW") == .orderedSame
    }

    static func representativeChineseCode(for code: String) -> String {
        isTraditionalChinese(code) ? traditionalChineseCode : simplifiedChineseCode
    }
}
