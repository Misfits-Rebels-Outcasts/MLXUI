import Foundation

/// MC2-4 — `text → text` stage backed by ``TranslateGemmaEngine``. The source and target language codes travel in the
/// stage's own values (not on the input text), so one flow row and one Run screen each build the stage they need.
///
/// The `translate` closure is the seam unit tests use: no weights, no folder. The two real inits bind the installed
/// model's directory (the process the app and the gated real-weights tests both run).
nonisolated struct TranslateGemmaStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    let sourceCode: String
    let targetCode: String
    /// One translation of a text; the argument is the source text.
    private let translate: @Sendable (String) async throws -> TranslateGemmaEngine.Outcome

    /// The seam tests use: an injectable translator, no weights, no folder.
    init(id: String, name: String, sourceCode: String, targetCode: String,
         translate: @escaping @Sendable (String) async throws -> TranslateGemmaEngine.Outcome) {
        self.id = id
        self.name = name
        self.sourceCode = sourceCode
        self.targetCode = targetCode
        self.translate = translate
    }

    /// The real engine on an explicit folder. The app's path (`init(modelID:…)`) and the owner-run tests (a `/private/tmp`
    /// copy of the installed folder) both go through here, so they call `TranslateGemmaEngine.translate` identically.
    init(modelDirectory: URL, id: String, name: String, sourceCode: String, targetCode: String,
         maxOutputTokens: Int = TranslateGemmaLanguages.maxOutputTokens, footprintBytes: Int64? = nil) {
        self.init(id: id, name: name, sourceCode: sourceCode, targetCode: targetCode) { text in
            try await TranslateGemmaEngine.translate(
                .init(text: text, sourceCode: sourceCode, targetCode: targetCode),
                modelDirectory: modelDirectory, maxOutputTokens: maxOutputTokens,
                footprintBytes: footprintBytes)
        }
    }

    /// The real engine on the installed folder of `modelID`.
    init(modelID: String, displayName: String, sourceCode: String, targetCode: String,
         maxOutputTokens: Int = TranslateGemmaLanguages.maxOutputTokens, footprintBytes: Int64? = nil) {
        self.init(modelDirectory: TranslateGemmaEngine.modelDirectory(for: modelID),
                  id: modelID, name: displayName, sourceCode: sourceCode, targetCode: targetCode,
                  maxOutputTokens: maxOutputTokens, footprintBytes: footprintBytes)
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        guard case .text(let text) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.05)
        let outcome = try await translate(text)
        if outcome.cancelled { throw CancellationError() }   // a Stop is not a translation
        progress(1)
        return .text(outcome.text)
    }
}
