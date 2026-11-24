import Foundation

/// PF-7 — `text → text` stage backed by ``PrivacyFilterEngine``: the row's text goes in, the redacted text comes out.
///
/// The `only=` subset travels in the stage's own value (not on the input text), so one flow row and one Run screen each
/// build the stage they need; the detected spans stay a Run-screen concern (the flow contract's output is the redacted
/// text only — decision 7).
///
/// The `redact` closure is the seam unit tests use: no weights, no folder. The two real inits bind the installed model's
/// directory (the process the app and the gated real-weights tests both run).
nonisolated struct PrivacyFilterStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    /// The categories to replace; `nil` = all eight.
    let only: Set<PrivacyFilterCategory>?

    private let redact: @Sendable (String) async throws -> PrivacyFilterEngine.Outcome

    /// The seam tests use: an injectable redactor, no weights, no folder.
    init(id: String, name: String, only: Set<PrivacyFilterCategory>? = nil,
         redact: @escaping @Sendable (String) async throws -> PrivacyFilterEngine.Outcome) {
        self.id = id
        self.name = name
        self.only = only
        self.redact = redact
    }

    /// The real engine on an explicit folder. The app's path (`init(modelID:…)`) and the owner-run real-weights tests
    /// (a `/private/tmp` copy of the installed folder) both go through here.
    init(modelDirectory: URL, id: String, name: String, only: Set<PrivacyFilterCategory>? = nil,
         footprintBytes: Int64? = nil, holder: MemoryBudget.HolderKind = .flow) {
        self.init(id: id, name: name, only: only) { text in
            try await PrivacyFilterEngine.redact(text, categories: only, modelDirectory: modelDirectory,
                                                 footprintBytes: footprintBytes, holder: holder)
        }
    }

    /// The real engine on the installed folder of `modelID`.
    init(modelID: String, displayName: String, only: Set<PrivacyFilterCategory>? = nil,
         footprintBytes: Int64? = nil, holder: MemoryBudget.HolderKind = .flow) {
        self.init(modelDirectory: PrivacyFilterEngine.modelDirectory(for: modelID),
                  id: modelID, name: displayName, only: only,
                  footprintBytes: footprintBytes, holder: holder)
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        guard case .text(let text) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.05)
        let outcome = try await redact(text)
        if outcome.cancelled { throw CancellationError() }   // a Stop is not a redacted text
        progress(1)
        return .text(outcome.redactedText)
    }
}
