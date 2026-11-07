import Foundation

/// MC2-4 — the errors the translator raises, each a plain sentence a person can act on (the product voice: no raw
/// error dumps). The flow interpreter wraps a thrown row error as "Row N failed: …"; the Run screen shows the sentence
/// in its error bar.
nonisolated enum TranslateGemmaError: Error, CustomStringConvertible, LocalizedError, Equatable {
    /// Nothing to translate after trimming.
    case emptyText
    /// A language code the pinned template does not list. The template's own lookup is silent on a miss, so the engine
    /// refuses before rendering (Trap 2).
    case unknownLanguage(String)
    /// The rendered request is over the card's 2K-token input limit (owner decision 8).
    case promptTooLong(tokens: Int)
    /// The processor returned a prompt with no template in it — the silent `[BOS]`-only failure Trap 2 warns about.
    case unreadableTemplate
    /// The model produced no text at all.
    case noTranslation
    /// A request that is missing its languages — the Run sheet builds a probe stage this way, and the Translate screen
    /// always replaces it with a per-run stage that carries both. Never shown to a person in the app.
    case missingLanguages

    static var promptTokenLimit: Int { TranslateGemmaLanguages.maxPromptTokens }

    var description: String {
        switch self {
        case .emptyText:
            return "There's no text to translate."
        case .unknownLanguage(let code):
            return "\"\(code)\" isn't a language this model can translate — pick a language from its list."
        case .promptTooLong(let tokens):
            let limit = Self.promptTokenLimit.formatted(.number.grouping(.automatic))
            return "The text is too long: this request is \(tokens) tokens and the model accepts up to \(limit). Shorten it and try again."
        case .unreadableTemplate:
            return "The model's chat template couldn't be read — reinstall the model and try again."
        case .noTranslation:
            return "The model didn't return a translation. Try again."
        case .missingLanguages:
            return "Choose a source language and a target language before translating."
        }
    }

    var errorDescription: String? { description }
}
