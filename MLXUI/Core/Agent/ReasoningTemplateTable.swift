//
//  ReasoningTemplateTable.swift
//  MLXUI — S2-3: which catalog families can think, and where their thinking comes from.
//
//  Read beside `ToolCallFormatTable`, and for the same reason: the model's own chat template decides
//  what a request-level control means, so the family fact is recorded once, cited, and asked by both
//  the route (for the log) and the backend (for the prompt and the split).
//
//  **Every answer here is "implementer's call, pending owner confirmation."** The evidence — template
//  URL, fetch date and the deciding lines — is in `RSI/journal/2026-408-s2-3-reasoning.md`.
//
//  The templates were read for all 12 catalog chat models on 2026-10-06 (S2-1's fetch, re-read for
//  this task). Five of the twelve read `enable_thinking`; the rest ignore it.
//

import Foundation

nonisolated enum ReasoningTemplateTable {

    /// Where the model's thinking comes from, once thinking is on. The server's
    /// `ServeReasoningSplitter` handles both without being told which is which — it decides by
    /// looking at the output — but the shapes are recorded here because they are the reason two
    /// tests exist and because S2-4's badge/Connect copy may want to say what a family does.
    enum ThinkBlockShape: Equatable {
        /// The **prompt** already opened the block: the generation prompt ends `<think>\n`, so the
        /// output starts inside it and only `</think>` has to be found.
        case promptOpened
        /// The **model** opens it: the prompt ends at the assistant header and the output starts
        /// `<think>\n…`.
        case modelEmitted
        /// The family has no thinking block at all.
        case none
    }

    /// The one place the thinking decision is made, so the route's log and the backend's prompt can
    /// never disagree (S2-3 decisions 1–2):
    ///
    /// - **off unless the request asked** — `nil` (nothing sent) and `false` both mean S1's
    ///   behaviour, `enable_thinking: false` plus a leading `<think>` strip;
    /// - **only for a family that reads `enable_thinking`** — for every other model the request is
    ///   **ignored**, never a `400`, and Apple Foundation Models is untouched.
    static func thinkingOn(requested: Bool?, forModelType modelType: String) -> Bool {
        (requested ?? false) && readsEnableThinking(forModelType: modelType)
    }

    /// Does the family's chat template read `enable_thinking`? Only these families get the control
    /// honoured; for every other model a request that asks for reasoning is **ignored**, never a
    /// `400` (S2-3 decision 2), and Apple Foundation Models is untouched.
    static func readsEnableThinking(forModelType modelType: String) -> Bool {
        thinkShape(forModelType: modelType) != .none
    }

    /// The table. Prefix-matched the way `ToolCallFormatTable` is.
    static func thinkShape(forModelType modelType: String) -> ThinkBlockShape {
        let t = modelType.lowercased()
        // Qwen3.5 / Qwen3.8 / Ternary-Bonsai: `{%- if enable_thinking is defined and
        // enable_thinking is false %}<think>\n\n</think>\n\n{%- else %}<think>\n{%- endif %}` in the
        // generation prompt — the prompt opens the block.
        // https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit/raw/main/chat_template.jinja
        // (lines 148–155; also Qwen3.8-27B-4bit lines 164–171 and Ternary-Bonsai lines 148–155)
        if t.hasPrefix("qwen3_5") || t.hasPrefix("qwen3.5") { return .promptOpened }
        // Qwen3 (the 8B in the catalog): the generation prompt ends `<|im_start|>assistant\n` and
        // only pre-closes an *empty* block when thinking is off, so the model emits `<think>` itself.
        // https://huggingface.co/mlx-community/Qwen3-8B-4bit/raw/main/tokenizer_config.json
        // (`chat_template`, lines 81–86)
        if t == "qwen3" { return .modelEmitted }
        // Everything else has no think block: Qwen2.5, Llama 3.1, Ministral 3 / Devstral, Gemma 2,
        // Gemma 3 1B, and gpt-oss. gpt-oss is the interesting one: its template reads
        // `reasoning_effort` four times, but that shapes *harmony channels*, not a `<think>` block,
        // and it never reads `enable_thinking` — S2-3 keeps S1 behaviour for it and does no harmony
        // work (decision 2). https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8/raw/main/chat_template.jinja
        return .none
    }
}
