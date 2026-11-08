//
//  FlowSettingsEscaper.swift
//  MLXUI — Model Tools, MT1-3.
//
//  The one place a value that came from outside a `.cat` file (a model's tool argument) is
//  turned into row-settings text. `FlowSettings` splits on whitespace and `;`, so an unescaped
//  `a "x"; seed=999` would become three tokens and smuggle in a setting. A quoted token is one
//  token whatever it holds, provided `\`, `"` and line breaks are escaped — which is all this
//  does. Values are always quoted, so nothing the model wrote can be read as a key or a `;`.
//
//  `TranslateRow.swift` has its own escaper (`\` and `"` only) and is deliberately untouched.
//

import Foundation

nonisolated enum FlowSettingsEscaper {
    /// `value` as a double-quoted settings token: `\` → `\\`, `"` → `\"`, a line break → `\n`,
    /// a tab → `\t`. A lone `\r` counts as a line break; `\r\n` is one.
    // SPEC-Q250: the read side, `FlowSettings.unquote`, replaces `\n` / `\t` BEFORE `\\`, so a quoted
    // value holding a backslash followed by n or t (`C:\new`) is altered on read (it comes back with
    // a newline). This escaper emits the right text; the asymmetry is in the shared reader. Filed as
    // Q250 in catflow-mlx/SPEC_QUESTIONS.md; `unquote` is deliberately unchanged.
    static func quoted(_ value: String) -> String {
        var out = "\""
        var previousWasCR = false
        for ch in value.unicodeScalars {
            switch ch {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": if !previousWasCR { out += "\\n" }
            case "\r": out += "\\n"
            case "\t": out += "\\t"
            default: out.unicodeScalars.append(ch)
            }
            previousWasCR = (ch == "\r")
        }
        return out + "\""
    }

    /// `key="value"`. `key` must already be a bare identifier (a spec row key, never user text).
    static func pair(_ key: String, _ value: String) -> String {
        "\(key)=\(quoted(value))"
    }
}
