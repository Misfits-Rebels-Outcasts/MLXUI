import Foundation

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
}
