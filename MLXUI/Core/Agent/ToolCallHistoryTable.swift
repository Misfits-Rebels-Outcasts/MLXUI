import Foundation

/// **How many tool calls a family's chat template accepts in one assistant turn** (S2-8), read from
/// the templates themselves — the 12 catalog chat models' `chat_template`, fetched 2026-10-07 and
/// kept in `RSI/notes/s2-1-templates/*.jinja`. Sits beside `ToolCallFormatTable` because the two
/// answer sibling questions: *does this family take tools* and *how may the history carry them*.
///
/// Two distinct limits came out of the grep, and they are different problems:
///
/// * **`oneCallPerTurn`** — the template **raises** on an assistant turn with more than one call. Only
///   `llama` does: `{%- if not message.tool_calls|length == 1 %}{{ raise_exception("This model only
///   supports single tool-calls at once!") }}`. A packed history is a `TemplateException`, i.e. a
///   failed request (S2-8 turned that into a `400`, see `ServeError.templateRefused`). *This is the
///   flag the assembly acts on.*
/// * **`rendersFirstCallOnly`** — the template silently draws `tool_calls[0]` and never looks at the
///   rest (`llama`: `{%- set tool_call = message.tool_calls[0].function %}`; `gpt_oss`:
///   `{%- set tool_call = message.tool_calls[0] %}`). `llama` is fixed by the same split (each turn
///   then holds one call); `gpt_oss` never reaches this code — its format has no parser, so the
///   server flattens its tool traffic (S2-1's `unsupportedFormat`).
///
/// The other nine families **loop** over every call (`{%- for tool_call in message.tool_calls %}` —
/// `qwen3_5` ×4, `qwen3`, `qwen2`, `mistral3` ×2) and are left alone: their history is already what
/// their template wants, and splitting it would change prompts that work.
///
/// Both limits are only about **history**. The wire response is untouched by any of this — a client
/// still receives every call the model produced.
nonisolated enum ToolCallHistoryTable {

    /// A family whose template refuses an assistant turn with more than one tool call.
    static func oneCallPerTurn(forModelType modelType: String) -> Bool {
        let t = modelType.lowercased()
        // `llama` (Llama 3.1/3.2's template). `llama4` is a different template and was not read, so
        // it keeps the conservative answer (no split) rather than inheriting this one — the same
        // rule S2-1 used for bare `gemma3`.
        return t == "llama"
    }

    /// A family whose template renders only the first call of a turn (recorded for the reader; the
    /// split above is what fixes `llama`, and `gpt_oss` never reaches the tool path).
    static func rendersFirstCallOnly(forModelType modelType: String) -> Bool {
        let t = modelType.lowercased()
        return t == "llama" || t.hasPrefix("gpt_oss")
    }

    /// The evidence behind `oneCallPerTurn`, for the caller that has to explain itself.
    static let llamaRaise = "This model only supports single tool-calls at once!"
}
