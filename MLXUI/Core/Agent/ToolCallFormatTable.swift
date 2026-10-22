//
//  ToolCallFormatTable.swift
//  MLXUI — S2-1: the per-family chat-template facts behind tool calling.
//
//  One small table answers the three questions a chat model family raises before tools are
//  offered or a `system` message is placed:
//
//    1. `resolved(forModelType:configData:)` — the `ToolCallFormat` the loaded container ends up
//       with. mlx-swift-lm 3.31.3 already infers this at load: `LLMModelFactory._load`
//       (`LLMModelFactory.swift:543`) calls `ToolCallFormat.infer(from:configData:)` whenever the
//       configuration carries no format, and `ModelContainerPool` hands the factory the
//       checkpoint's directory. So this table deliberately does **not** restate upstream's
//       mapping — `override(...)` is empty for every model in the catalog (see its comment) and
//       the resolution falls through to `ToolCallFormat.infer` by construction. "nil" here means
//       "upstream decides", not ".json": the container's generate path reads
//       `configuration.toolCallFormat ?? .json` (`Evaluate.swift:1445`).
//    2. `toolSupport(forModelType:)` — can a harness's `tools` array work for this family at all?
//       S2-2 keeps S1's "ignore tools" behaviour where the answer is not `.supported`, and the Connect
//       sheet's snippets (`tool_call`, limits) follow it (R51).
//    3. `systemPlacement(forModelType:)` — where the family's own chat template lets a `system`
//       message sit. `ServeMessageAssembly` uses it to hoist or fold system messages.
//
//  **Every answer below is "implementer's call, pending owner confirmation"** (backlog rule 12):
//  each is read off the checkpoint's own chat template, and the evidence — URL, fetch date and
//  the deciding template lines — is in `RSI/journal/2026-405-s2-1-tool-call-formats.md`. That
//  journal is the source of truth; the one-line citations here are pointers into it.
//
//  Keyed on the checkpoint's `model_type` from `config.json` (what upstream's `infer` uses), with
//  the catalog architecture string as the fallback. Templates were read for all 12 catalog chat
//  models on 2026-10-06.
//

import Foundation
import MLXLMCommon

nonisolated enum ToolCallFormatTable {

    // MARK: - Tool-call format

    /// The app's **overrides** on top of mlx-swift-lm's own `ToolCallFormat.infer`. `nil` means
    /// "upstream decides" and is the normal answer; `ModelContainerPool.loadFromDisk` installs a
    /// non-nil value on the loaded container, and it is the only place in the app that sets one.
    ///
    /// **Empty for all 12 catalog chat models (measured 2026-10-06).** Upstream's inference was
    /// checked against each family's own chat template and was right — or as right as the nine
    /// `ToolCallFormat` cases allow — everywhere:
    ///
    ///   - `qwen3_5*` → `.xmlFunction` (upstream): `<tool_call><function=…><parameter=k>v`.
    ///     https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit/raw/main/chat_template.jinja
    ///   - `mistral3` (Ministral 3, Devstral Small 2) → `.mistral` (upstream):
    ///     `[TOOL_CALLS]name[ARGS]{…}`.
    ///     https://huggingface.co/mlx-community/Ministral-3-3B-Instruct-2512-4bit/raw/main/chat_template.jinja
    ///   - `llama` → `.llama3` (upstream, from `vocab_size >= 128000` and
    ///     `rope_scaling.rope_type == "llama3"`): bare `{"name":…,"parameters":…}` JSON.
    ///   - `qwen3` / `qwen2` → `nil` (upstream) → container default `.json`:
    ///     `<tool_call>{"name":…,"arguments":…}</tool_call>`.
    ///     https://huggingface.co/mlx-community/Qwen3-8B-4bit/raw/main/tokenizer_config.json
    ///   - `gemma2` / `gemma3_text` → `nil` (upstream) → `.json`; neither template mentions
    ///     `tools` at all, so no format can fire. See `toolSupport`.
    ///   - `gpt_oss` → `nil` (upstream) → `.json`, which can never parse harmony output. No
    ///     `ToolCallFormat` case does, and **no harmony parser was written** for S2-1 (task
    ///     instruction 3). See `toolSupport`'s `.unsupportedFormat`.
    ///
    /// So the S1 workaround this replaces — `qwen3_5` → `.xmlFunction` applied here — was a
    /// duplicate of a row upstream already had (readme F3b).
    static func override(forModelType modelType: String, configData: Data?) -> ToolCallFormat? {
        _ = configData          // no catalog family needs a value the checkpoint doesn't supply
        return nil
    }

    /// The app's own override when there is one, else `nil` — the loaded container's own resolved
    /// format applies (mlx-swift-lm 3.32 resolves it at load from the model type and the checkpoint's
    /// chat template; that resolution is package-internal, so the app no longer repeats it).
    ///
    /// `configData` is the checkpoint's `config.json`, passed through to `override`.
    static func resolved(forModelType modelType: String, configData: Data?) -> ToolCallFormat? {
        override(forModelType: modelType, configData: configData)
    }

    // MARK: - Does a `tools` array do anything for this family?

    /// What a `tools` array in a request is worth for a family.
    enum ToolSupport: Equatable {
        /// The template renders tool schemas and a `ToolCallFormat` parser reads the family's
        /// output. `tools` can be honoured (S2-2) and the Connect snippets advertise `tool_call` (R51).
        case supported
        /// The family's chat template has no `tools` handling at all (`gemma2`, `gemma3_text`):
        /// it ignores the array and answers in prose. Serving `tools` would look like the model
        /// "chose not to call a tool" forever, so S2-2 keeps S1's behaviour for it.
        case templateIgnoresTools
        /// The template does render tool schemas, but no `ToolCallFormat` case parses the family's
        /// output — `gpt_oss` (harmony). Pinned by a test so S2-2/S2-4 can't mistake it for
        /// "supported" until a parser exists.
        case unsupportedFormat
        /// Not a family this table has evidence for. Conservative: not supported.
        case unknown

        /// The single boolean S2-2 and S2-4 should branch on.
        var acceptsTools: Bool { self == .supported }
    }

    /// The table. Keyed on `model_type`, matched only as broadly as the evidence allows: a family
    /// is listed when its own template was read (the 12 catalog chat models), or — for
    /// `qwen3_next` — when upstream's `infer` maps it and upstream documents the same
    /// `<tool_call><function=…>` markers the format's parser reads. Anything else is `.unknown`,
    /// which is not supported: the conservative answer for a badge.
    static func toolSupport(forModelType modelType: String) -> ToolSupport {
        let t = modelType.lowercased()
        // `qwen3_moe` (Qwen3-Coder and Qwen3-30B-A3B) is deliberately left unknown: upstream's
        // `infer` does not map it, and AG0 recorded that Coder emits the XML form, which the
        // resulting `.json` default silently drops. Not in the catalog; not guessed at here.
        if t == "qwen3_moe" { return .unknown }
        // Templates that render tool schemas and whose output a `ToolCallFormat` case parses.
        if t.hasPrefix("qwen3_5") || t.hasPrefix("qwen3.5") { return .supported }   // .xmlFunction
        if t.hasPrefix("qwen3_next") { return .supported }                          // .xmlFunction
        if t == "qwen3" { return .supported }                                       // .json
        if t.hasPrefix("qwen2") { return .supported }                               // .json
        // Exact: upstream's `.llama3` comes from `type == "llama"` plus a config signal, and
        // `llama4` (a different template and format) is not this family.
        if t == "llama" { return .supported }                                       // .llama3
        if t.hasPrefix("mistral3") || t.hasPrefix("ministral") { return .supported } // .mistral
        // OpenAI's harmony format. The gpt-oss template renders tools and emits harmony channels;
        // no `ToolCallFormat` case parses them (mlx-swift-lm 3.31.3 has none; the Swift sources
        // mention harmony only in comments about *downstream* parsers), so S2-1 records this as
        // "no supported tool format" rather than writing a parser (task instruction 3).
        if t.hasPrefix("gpt_oss") { return .unsupportedFormat }
        // Templates with no `tools` branch at all. Measured for gemma-2-9b-it (`gemma2`) and
        // gemma-3-1b-it-qat (`gemma3_text`); bare `gemma3` is Gemma 3's VLM spelling and was not
        // read, so it stays unknown rather than inheriting the 1B's answer.
        if t == "gemma2" || t.hasPrefix("gemma3_text") { return .templateIgnoresTools }
        // Gemma 4 (`gemma4`; MP1-S, `RSI/journal/2026-543-mp1-s-gemma4-serve.md`): deliberately `.unknown`. The
        // template renders tools (`<|tool>declaration:…<tool|>`, line 198) and writes calls as
        // `<|tool_call>call:name{key:<|"|>v<|"|>,…}<tool_call|>` (lines 244–246), and mlx-swift-lm 3.31.4's
        // `ToolCallFormat.infer` does map `gemma4*` → `.gemma4` with those exact tags. But that parser reads a
        // value up to the next comma, so a nested array or object — `tags:[<|"|>a<|"|>,<|"|>b<|"|>]`, which
        // coding-agent tool schemas use all the time — comes back as a garbled string with the rest dropped, and
        // even a bare number or boolean (`days:3`, `metric:true`) comes back as the *string* "3" / "true"
        // (`deserializeJSON` refuses a JSON fragment). Only string arguments round-trip. That is not an exact
        // match, and a wrong argument is worse than an ignored `tools` array. (The tool declaration the template
        // renders also differs from Python's by whitespace under swift-jinja: a second, smaller reason.)
        // `Gemma4ServeTests` (the round-trip tests) pins the evidence; re-read
        // it, and flip this line, when a parser that reads nested values exists (no parser is written in the Model
        // Pack, readme T1). https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit/raw/main/chat_template.jinja
        // (revision 475b9088d2…, fetched 2026-10-21).
        if t.hasPrefix("gemma4") { return .unknown }
        return .unknown
    }

    /// Convenience for the callers that only need the yes/no.
    static func acceptsTools(forModelType modelType: String) -> Bool {
        toolSupport(forModelType: modelType).acceptsTools
    }

    // MARK: - Where may a `system` message sit?

    /// What the family's chat template does with the `system` role. Drives `ServeMessageAssembly`.
    enum SystemPlacement: Equatable {
        /// The template reads or renders a `system` message wherever it sits (or ignores it
        /// harmlessly): leave the client's order alone.
        case anywhere
        /// Only `messages[0]` is meaningful; a later one either raises or is dropped. Hoist.
        case firstOnly
        /// The template has no `system` role at all and raises on one. Fold into the first user
        /// turn.
        case unsupported
    }

    /// The table, read off each family's template. Matched only as broadly as the evidence allows
    /// (see `toolSupport`).
    ///
    /// Families **not** listed default to `.anywhere` — the client's order is preserved, which is
    /// S1's behaviour and the conservative choice when a template is unknown.
    static func systemPlacement(forModelType modelType: String) -> SystemPlacement {
        let t = modelType.lowercased()
        // Qwen3.5/3.8: `{%- if message.role == "system" %}{%- if not loop.first %}
        // {{- raise_exception('System message must be at the beginning.') }}`.
        // https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit/raw/main/chat_template.jinja
        if t.hasPrefix("qwen3_5") || t.hasPrefix("qwen3.5") { return .firstOnly }
        // Ministral 3 / Devstral Small 2: `[SYSTEM_PROMPT]` only for `messages[0]`; a later one
        // hits `{{- raise_exception('Only user, assistant and tool roles are supported, got
        // system.') }}`.
        // https://huggingface.co/mlx-community/Ministral-3-3B-Instruct-2512-4bit/raw/main/chat_template.jinja
        if t.hasPrefix("mistral3") || t.hasPrefix("ministral") { return .firstOnly }
        // Gemma 2: the template's first statement is
        // `{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}`
        // — it refuses a system message even in the leading slot. Fold it into the first user
        // turn instead of failing the request.
        // https://huggingface.co/mlx-community/gemma-2-9b-it-4bit/raw/main/tokenizer_config.json
        if t == "gemma2" { return .unsupported }
        // Gemma 3 1B: folds a leading system message into the first user turn itself
        // (`{%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}`), but a later one
        // trips `Conversation roles must alternate user/assistant/...`, so it is first-only for
        // us. https://huggingface.co/mlx-community/gemma-3-1b-it-qat-4bit/raw/main/tokenizer_config.json
        if t.hasPrefix("gemma3_text") { return .firstOnly }
        // Gemma 4: the template reads `messages[0]` when its role is `system` or `developer` (line 179) and
        // renders any later `system` message as its own `<|turn>system` turn; it contains no
        // `raise_exception` at all, so nothing needs hoisting and the client's order is kept (`.anywhere`,
        // stated rather than left to the default).
        // https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit/raw/main/chat_template.jinja
        if t.hasPrefix("gemma4") { return .anywhere }
        // gpt-oss: `{%- if messages[0].role == "developer" or messages[0].role == "system" %}`
        // renders `<|start|>developer<|message|>…`; a system at index > 0 matches no branch and
        // is emitted nowhere — silent content loss, so hoist it.
        // https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8/raw/main/chat_template.jinja
        if t.hasPrefix("gpt_oss") { return .firstOnly }
        // `llama` is deliberately `.anywhere`: the Llama 3.1 template extracts a system message
        // from `messages[0]` and renders a later one as an ordinary inline `<|start_header_id|>
        // system` header — odd, but it does not raise, so R22's "templates that require it" does
        // not apply and S1's order is preserved.
        // https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit/raw/main/tokenizer_config.json
        return .anywhere
    }
}
