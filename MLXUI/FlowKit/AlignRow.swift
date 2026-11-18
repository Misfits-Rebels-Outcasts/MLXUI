import Foundation

/// FA-6 (GitHub issue #13 decision 8, SPEC-Q252; owner ruling 2026-11-19): the `Align` row's settings.
///
/// One setting, `lang=` (the codes `Transcribe` already uses, mapped by `ForcedAlignerEngine` to the library's language
/// names). A missing `lang=` means English, the same as `Transcribe` today. **Any other key, a bare token, or an
/// unsupported language is refused by name, never ignored** — `FlowError.invalidSettings` renders as
/// "Row N's foo setting is malformed — …" through the interpreter.
nonisolated enum AlignRowRequest {
    /// The only setting the row accepts. The transcript is the row's second *input*, not a setting.
    static let acceptedSettings = ["lang"]

    /// The language code the row asks for: `lang=` when present, else `ForcedAlignerEngine.defaultLanguageCode` (English).
    /// `rowLabel` is the interpreter's row name ("Row 3") for the refusal sentence.
    static func languageCode(settings: String?, rowLabel: String = "This row") throws -> String {
        let flow = FlowSettings(settings)
        let codes = ForcedAlignerEngine.languageCodes.joined(separator: ", ")
        if let unknown = flow.orderedPairs.first(where: { !acceptedSettings.contains($0.key) }) {
            throw FlowError.invalidSettings(
                row: rowLabel, setting: unknown.key,
                detail: "the Align row takes only lang= (one of \(codes)); remove \(unknown.key)=")
        }
        if let bare = flow.firstBare() {
            throw FlowError.invalidSettings(
                row: rowLabel, setting: bare,
                detail: "the Align row takes only lang= (one of \(codes)); the transcript comes from the row's second input, not from a setting")
        }
        let code = (flow.value(for: "lang") ?? ForcedAlignerEngine.defaultLanguageCode)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard ForcedAlignerEngine.libraryLanguageName(for: code) != nil else {
            throw FlowError.invalidSettings(
                row: rowLabel, setting: "lang",
                detail: "'\(code)' isn't a language this model supports; use one of \(codes)")
        }
        return code
    }
}
