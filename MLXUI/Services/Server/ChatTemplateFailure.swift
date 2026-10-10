import Foundation
import MLXLMCommon

/// **A checkpoint's chat template refused the request** (S2-8): Llama 3.1's template raises when an
/// assistant turn carries more than one tool call (`ToolCallHistoryTable`), and other templates raise
/// on shapes they don't accept. That is the *checkpoint* declining, not this server failing, so it is
/// a `400` naming the template's own words — never S2's generic `500 generation_failed`, which told
/// the client to retry something that could never succeed (SV-6's Llama run retried it six times).
///
/// **How the exception is recognised.** The type is `swift-jinja`'s `TemplateException`, which this
/// target cannot `import` without adding a package dependency (a guardrail: no new dependencies
/// without approval). It is therefore matched **by type name** — the same name the server's error log
/// has shown for the live refusal — and its raise text is read from the value's stored `message` with
/// `Mirror`, because that property is internal upstream. A `Mirror` that finds nothing falls back to
/// a sentence rather than inventing a reason; both paths are covered by tests, the second against the
/// real template.
nonisolated enum ChatTemplateFailure {

    /// The checkpoint's own words, when this error is a template refusal; `nil` for anything else.
    static func raisedText(from error: Error) -> String? {
        guard isTemplateRefusal(error) else { return nil }
        let mirrored = Mirror(reflecting: error).children
            .first { $0.label == "message" }?.value as? String
        let trimmed = mirrored?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else { return nil }
        return trimmed
    }

    /// Whether this error came from the checkpoint's template.
    static func isTemplateRefusal(_ error: Error) -> Bool {
        String(describing: type(of: error)) == "TemplateException"
    }

    /// The wire error: a `400 invalid_request_error` with code `template_error` and the template's
    /// words (or, when it raised without one, a sentence that says exactly that).
    static func serveError(for error: Error) -> ServeError? {
        guard isTemplateRefusal(error) else { return nil }
        let raised = raisedText(from: error) ?? "the checkpoint's template raised an exception."
        return ServeError(status: 400,
                          message: "This model's chat template refused the request: \(raised)",
                          code: "template_error")
    }

    // MARK: - Render failures (S3-5b)

    /// What a thrown render error is. The first two existed before S3-5b; the third used to fall through
    /// to S2-6's fail-open (the count) or to a generic `500 generation_failed` (the generation).
    enum Classification {
        /// The checkpoint's template raised on purpose (`raise_exception`): S2-8's `400`.
        case refusal(ServeError)
        /// The checkpoint has no chat template at all: "can't count", not a failure of the request.
        case missingTemplate
        /// The template could not be *run* on this input (e.g. `swift-jinja` cannot convert a value in a
        /// tool schema): the request's content is the problem, and the client can fix it.
        case renderFailure
    }

    static func classify(_ error: Error) -> Classification {
        if let refusal = serveError(for: error) { return .refusal(refusal) }
        if let tokenizerError = error as? MLXLMCommon.TokenizerError, case .missingChatTemplate = tokenizerError {
            return .missingTemplate
        }
        return .renderFailure
    }

    /// The `400` for a render failure, naming the cause in one plain sentence: the tool whose schema
    /// could not be rendered when the probe found one (by name), else the conversation as a whole.
    static func renderFailure(toolNames: [String]) -> ServeError {
        let sentence: String
        switch toolNames.count {
        case 0:
            sentence = "This model's chat template couldn't render this request."
        case 1:
            sentence = "This model's chat template couldn't render the tool '\(toolNames[0])': its schema contains something the template can't serialise."
        default:
            let shown = toolNames.prefix(3).map { "'\($0)'" }.joined(separator: ", ")
            let more = toolNames.count > 3 ? " and \(toolNames.count - 3) more" : ""
            sentence = "This model's chat template couldn't render \(toolNames.count) of the tools (\(shown)\(more)): their schemas contain something the template can't serialise."
        }
        return ServeError(status: 400, message: sentence, code: "template_error")
    }

    /// Which tools fail to render **on their own**, by name, in the order given. The probe is injected so
    /// the search is testable without a tokenizer. Only called after a render has already failed.
    static func failingToolNames(in tools: [[String: any Sendable]],
                                 rendersAlone: ([String: any Sendable]) -> Bool) -> [String] {
        tools.compactMap { tool in
            rendersAlone(tool) ? nil : toolName(tool)
        }
    }

    static func toolName(_ tool: [String: any Sendable]) -> String? {
        ((tool["function"] as? [String: any Sendable])?["name"] as? String) ?? (tool["name"] as? String)
    }
}
