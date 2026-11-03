import Foundation

/// MC2-6 — a `Translate` row naming TranslateGemma 4B. The row's only language setting is `to=` (the v0.9 language's
/// declaration); the source is detected from the row's input text. **One** function resolves both, so `RealExecutor`
/// (the run) and the Properties-tab preview cannot disagree about which request the row sends (FV-2).
///
/// There is no `from=` setting — the issue says not to invent one. A row that wants a fixed source language uses the
/// Translate screen instead.
nonisolated enum TranslateRowRequest {

    /// What a row sends: both codes from the pinned language table.
    struct Resolved: Sendable, Equatable {
        let sourceCode: String
        let targetCode: String
    }

    /// Refusals, each a plain sentence the flow interpreter wraps as "Row N failed: …".
    enum Failure: Error, CustomStringConvertible, Equatable, LocalizedError {
        case missingTarget
        case unknownTarget(String)
        case undetectableSource

        var description: String {
            switch self {
            case .missingTarget:
                return "This TranslateGemma row has no target language. Add to= with a language name or code, for example to=Malay."
            case .unknownTarget(let raw):
                return "\"\(raw)\" isn't a language TranslateGemma can translate into. Set to= to a language name or code from its list, for example to=Malay or to=ms."
            case .undetectableSource:
                return "This row can't tell which language its input is in. TranslateGemma detects the source itself, so use text in a clear language, or run it on the Translate screen and pick the source language."
            }
        }

        var errorDescription: String? { description }
    }

    /// The target code a row's `to=` names. Strict: an unknown name or code is a refusal, never a fallback.
    static func targetCode(settings: String?) throws -> String {
        let raw = (FlowSettings(settings).value(for: "to") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { throw Failure.missingTarget }
        guard let code = TranslateGemmaLanguages.requestCode(for: raw) else { throw Failure.unknownTarget(raw) }
        return code
    }

    /// The request a run sends. `detect` is injected (like `askLaya`) so the wiring tests are deterministic; production
    /// uses the on-device recogniser, never a guess — no detection is a refusal, not a fallback.
    static func resolve(settings: String?, inputText: String,
                        detect: (String) -> String? = { TranslateGemmaLanguageDetector.detectedCode(in: $0) })
    throws -> Resolved {
        let target = try targetCode(settings: settings)
        guard let detected = detect(inputText),
              let source = TranslateGemmaLanguages.canonicalTemplateCode(detected) else {
            throw Failure.undetectableSource
        }
        return Resolved(sourceCode: source, targetCode: target)
    }

    /// What the Properties tab shows for such a row.
    struct Preview: Sendable, Equatable {
        /// The rendered request: the same renderer a run's prompt goes through (the pure renderer is proven identical
        /// to the checkpoint's `chat_template.jinja`).
        let rendered: String
        let sourceCode: String
        let targetCode: String
        /// `false` when the edit-time stand-in text had no detectable language, so `sourceCode` is
        /// ``previewFallbackSourceCode`` rather than a real detection.
        let sourceWasDetected: Bool
    }

    /// The row's settings without the `to=` pair. `to=` is the row's target language, resolved here, not an engine
    /// setting; every other pair (and any bare instruction) is handed to the curated manifest's own resolution, so a
    /// row's `max_tokens=`/`temp=` still applies and an unknown key is still refused by the manifest.
    static func engineSettings(_ settings: String?) -> String? {
        let parsed = FlowSettings(settings)
        let kept = parsed.orderedPairs.filter { $0.key != "to" }
        guard !kept.isEmpty else { return nil }
        return kept.map { pair in
            let escaped = pair.value
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            return "\(pair.key)=\"\(escaped)\""
        }.joined(separator: " ")
    }

    /// The preview-only source when the Properties tab's stand-in text ("⟨the text from row 2⟩") has no detectable
    /// language. The request's *shape* is still this model's; a run replaces the source with the real detection.
    static let previewFallbackSourceCode = "en"

    static func preview(settings: String?, inputText: String,
                        detect: (String) -> String? = { TranslateGemmaLanguageDetector.detectedCode(in: $0) })
    throws -> Preview {
        let target = try targetCode(settings: settings)
        let detected = detect(inputText)
        let source = detected.flatMap { TranslateGemmaLanguages.canonicalTemplateCode($0) } ?? previewFallbackSourceCode
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let rendered = try TranslateGemmaPrompt.render(sourceCode: source, targetCode: target,
                                                       text: text.isEmpty ? "⟨the text to translate⟩" : text)
        return Preview(rendered: rendered, sourceCode: source, targetCode: target, sourceWasDetected: detected != nil)
    }
}
