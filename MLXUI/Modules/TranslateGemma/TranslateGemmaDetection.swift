import Foundation
import NaturalLanguage

/// MC2-4 / owner decision 5 — on-device source-language detection, in one place the Translate screen and the
/// `Translate` workflow row both call. The app already links `NaturalLanguage` (`TextChunker` uses it), so this adds no
/// dependency.
///
/// Detection is a **suggestion, never a guess**: a result below the confidence threshold, or a language the pinned
/// template does not list, returns `nil` and the caller asks the person to pick. Translating from a silently guessed
/// language is exactly what decision 5 forbids.
nonisolated enum TranslateGemmaLanguageDetector {

    struct Detection: Sendable, Equatable {
        /// A code from `TranslateGemmaLanguages`'s table (`en`, `ms`, `zh-Hans`, …).
        let code: String
        /// The display name for that code.
        let name: String
        /// The recogniser's own probability for the code, 0…1.
        let confidence: Double
    }

    /// Implementer's call, pending owner confirmation: a dominant hypothesis under 0.5 is treated as "not confident".
    /// `NLLanguageRecognizer`'s probabilities are cheap but noisy on very short strings (a two-word fragment can read
    /// 0.4-ish for its own language); asking the person to pick is the honest outcome there, per decision 5.
    static let confidenceThreshold: Double = 0.5

    /// The detected source language, or `nil` when the text is empty, the recogniser is not confident enough, or the
    /// language is outside the template's table (decision 5's "ask the user to pick" cases).
    static func detect(_ text: String, threshold: Double = confidenceThreshold) -> Detection? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(trimmed)
        let hypotheses = recognizer.languageHypotheses(withMaximum: 3)
        guard let dominant = hypotheses.max(by: { $0.value < $1.value }), dominant.value >= threshold else {
            return nil
        }
        guard let code = TranslateGemmaLanguages.code(forDetectedLanguage: dominant.key.rawValue) else {
            return nil
        }
        return Detection(code: code,
                         name: TranslateGemmaLanguages.name(forTemplateCode: code) ?? code,
                         confidence: dominant.value)
    }

    /// Convenience for the workflow row, which only needs the code.
    static func detectedCode(in text: String, threshold: Double = confidenceThreshold) -> String? {
        detect(text, threshold: threshold)?.code
    }
}
