import Foundation

/// Pure template substitution — how a Chat node gets both an instruction *and* the
/// upstream text. See open-pipeline-nodes §3a (text ops).
nonisolated enum TemplatePrompt {
    static let placeholder = "{input}"

    /// Substitute the upstream text into `template` at every `{input}`. If the template
    /// has no placeholder, append the input after a blank line, so a bare instruction
    /// (e.g. "Summarize the following:") still receives the text.
    static func apply(template: String, input: String) -> String {
        if template.contains(placeholder) {
            return template.replacingOccurrences(of: placeholder, with: input)
        }
        let trimmed = template.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? input : trimmed + "\n\n" + input
    }
}

/// A tool node (`text → text`, no model files, no RAM gate) that wraps the upstream
/// text in a fixed template — typically the summarize instruction feeding `LLMStage`.
nonisolated struct TemplatePromptStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    private let template: String

    init(template: String, id: String = "tool.template", name: String = "Template Prompt") {
        self.template = template
        self.id = id
        self.name = name
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        guard case let .text(text) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(1.0)
        return .text(TemplatePrompt.apply(template: template, input: text))
    }
}
