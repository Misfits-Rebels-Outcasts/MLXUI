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
//  Second workaround (MP1-3-FIX-1): swift-jinja ignores the `-` whitespace control on comment tags.
//  `{#- … -#}` is meant to trim the whitespace before and after the comment (the Jinja spec, and what
//  Python's jinja2 does); swift-jinja 2.3.6 keeps it, so a template that uses it — Gemma 4's has ten —
//  renders extra newlines the Python reference never produces (`<bos>\n<|turn>user…`). The fix applies
//  the `-` textually, for comment tags only, and drops the dashes. Every other tag type is left alone.
//  **Upstream bug:** `huggingface/swift-jinja`, `Sources/Jinja/Lexer.swift`, `extractCommentToken`
//  (it returns the comment body, dashes included, and never trims the neighbouring text). Remove
//  `applyingCommentWhitespaceControl` and its call once a pinned swift-jinja release honours it: on every
//  swift-jinja bump, run `CommentWhitespaceControlTests` with the call bypassed and see whether the
//  Python goldens still pass.
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

    /// Returns the template with the known-bad constructs fixed, or nil when no change is needed.
    static func sanitized(_ template: String) -> String? {
        var result = template
        var changed = false
        if let regex = try? Regex(guardPattern), result.contains(regex) {
            result = result.replacing(regex, with: "")
            changed = true
        }
        if let stripped = applyingCommentWhitespaceControl(result) {
            result = stripped
            changed = true
        }
        return changed ? result : nil
    }

    /// Applies Jinja's whitespace control to **comment tags only**: `{#-` removes the whitespace before the
    /// comment, `-#}` removes the whitespace after it, and the dashes are dropped from what is returned.
    /// Returns nil when the template has no dashed comment, so templates without one stay untouched.
    ///
    /// A small scanner rather than a regex, because a `{#` inside a string literal in `{{ … }}` / `{% … %}`
    /// is not a comment, and neither is anything inside `{% raw %}…{% endraw %}`. Whitespace is Unicode
    /// `White_Space`, the characters Python's `str.strip()` removes in practice.
    static func applyingCommentWhitespaceControl(_ template: String) -> String? {
        let s = Array(template.unicodeScalars)
        let n = s.count
        var out = String.UnicodeScalarView()
        var changed = false
        var i = 0

        func isWhitespace(_ c: Unicode.Scalar) -> Bool { c.properties.isWhitespace }

        while i < n {
            guard s[i] == "{", i + 1 < n else { out.append(s[i]); i += 1; continue }
            switch s[i + 1] {
            case "#":
                var end = i + 2
                while end + 1 < n, !(s[end] == "#" && s[end + 1] == "}") { end += 1 }
                guard end + 1 < n else {                      // unterminated: leave the rest verbatim
                    out.append(contentsOf: s[i...]); i = n; continue
                }
                let leftDash = i + 2 < end && s[i + 2] == "-"
                let bodyStart = i + 2 + (leftDash ? 1 : 0)
                let rightDash = end - 1 >= bodyStart && s[end - 1] == "-"
                if leftDash {
                    while let last = out.last, isWhitespace(last) { out.removeLast() }
                }
                out.append(contentsOf: "{#".unicodeScalars)
                out.append(contentsOf: s[bodyStart ..< (rightDash ? end - 1 : end)])
                out.append(contentsOf: "#}".unicodeScalars)
                i = end + 2
                if rightDash {
                    while i < n, isWhitespace(s[i]) { i += 1 }
                }
                if leftDash || rightDash { changed = true }
            case "{", "%":
                let isBlock = s[i + 1] == "%"
                let start = i
                i += 2
                var quote: Unicode.Scalar?
                var depth = 0                                  // `{…}` dict literals inside the expression
                tag: while i < n {
                    let c = s[i]
                    if let q = quote {
                        if c == "\\", i + 1 < n { i += 2; continue }
                        if c == q { quote = nil }
                    } else if c == "'" || c == "\"" {
                        quote = c
                    } else if c == "{" {
                        depth += 1
                    } else if c == "}" {
                        if depth > 0 { depth -= 1 }
                        else if !isBlock, i + 1 < n, s[i + 1] == "}" { i += 2; break tag }
                    } else if isBlock, c == "%", depth == 0, i + 1 < n, s[i + 1] == "}" {
                        i += 2; break tag
                    }
                    i += 1
                }
                i = min(i, n)
                out.append(contentsOf: s[start ..< i])
                if isBlock, isRawOpen(s[start ..< i]) {
                    // Copy through `{% endraw %}` verbatim: a `{#` in there is text.
                    let close = "endraw".unicodeScalars.map { $0 }
                    var j = i
                    while j < n {
                        if s[j] == "{", j + 1 < n, s[j + 1] == "%" {
                            var k = j + 2
                            if k < n, s[k] == "-" { k += 1 }
                            while k < n, isWhitespace(s[k]) { k += 1 }
                            if k + close.count <= n, Array(s[k ..< k + close.count]) == close { break }
                        }
                        j += 1
                    }
                    out.append(contentsOf: s[i ..< j])
                    i = j
                }
            default:
                out.append(s[i]); i += 1
            }
        }
        return changed ? String(out) : nil
    }

    /// True for `{% raw %}` / `{%- raw -%}`.
    private static func isRawOpen(_ tag: ArraySlice<Unicode.Scalar>) -> Bool {
        var inner = String(String.UnicodeScalarView(tag.dropFirst(2).dropLast(2)))
        if inner.hasPrefix("-") { inner.removeFirst() }
        if inner.hasSuffix("-") { inner.removeLast() }
        return inner.trimmingCharacters(in: .whitespacesAndNewlines) == "raw"
    }

    /// The sanitized chat template for an installed model, or nil when the model has no
    /// `chat_template.jinja` or its template needs no fix (use the tokenizer's default).
    static func sanitizedTemplate(inModelDirectory directory: URL) -> String? {
        let url = directory.appendingPathComponent("chat_template.jinja")
        guard let template = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        return sanitized(template)
    }
}
