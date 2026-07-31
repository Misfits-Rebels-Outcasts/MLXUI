//
//  ChatTemplateSanitizer.swift
//  MLXUI
//
//  Works around a collision between `MLXLMCommon.ChatSession`'s tool loop and strict
//  Qwen3.5-style chat templates (e.g. prism-ml/Ternary-Bonsai-27B).
//
//  After dispatching a tool call, `ChatSession` restarts generation with ONLY the
//  tool-result message in the chat — the earlier conversation lives in the KV cache, not
//  the message list (see the `restart:` loop in ChatSession.swift). Qwen3.5's template
//  scans the messages for a real user query and calls
//  `raise_exception('No user query found in messages.')` when there isn't one, so on such
//  models every tool call died with `Jinja.TemplateException` right after the tool ran —
//  the model never saw the result (journal 2026-64).
//
//  The fix: neutralize exactly that guard before the template reaches the tokenizer.
//  Removing it is safe: the guard never fires on renders that contain a user message
//  (verified byte-identical output), and on the tool-only restart render the rest of the
//  template already emits the correct `<tool_response>…</tool_response>` continuation.
//  Model files on disk are not modified — `HFTokenizerBridge` passes the sanitized string
//  per call via `chatTemplate: .literal(…)`, which swift-transformers prefers over the
//  checkpoint's own template.
//

import Foundation

nonisolated enum ChatTemplateSanitizer {
    /// The strict no-user-query guard, whitespace-tolerant:
    ///
    ///     {%- if ns.multi_step_tool %}
    ///         {{- raise_exception('No user query found in messages.') }}
    ///     {%- endif %}
    ///
    /// Deliberately narrow — only this exact known-bad construct is touched, never
    /// templates in general.
    private static let guardPattern =
        #"\{%-?\s*if\s+ns\.multi_step_tool\s*-?%\}\s*"#
        + #"\{\{-?\s*raise_exception\(\s*'No user query found in messages\.'\s*\)\s*-?\}\}\s*"#
        + #"\{%-?\s*endif\s*-?%\}"#

    /// Returns the template with the guard removed, or nil when no change is needed.
    static func sanitized(_ template: String) -> String? {
        guard let regex = try? Regex(guardPattern),
              template.contains(regex)
        else { return nil }
        return template.replacing(regex, with: "")
    }

    /// The sanitized chat template for an installed model, or nil when the model has no
    /// `chat_template.jinja` or its template needs no fix (use the tokenizer's default).
    static func sanitizedTemplate(inModelDirectory directory: URL) -> String? {
        let url = directory.appendingPathComponent("chat_template.jinja")
        guard let template = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        return sanitized(template)
    }
}
