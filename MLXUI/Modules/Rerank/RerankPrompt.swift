import Foundation

/// The causal yes/no-relevance prompt Qwen3-Reranker was packaged for — copied verbatim
/// from the model's own card (`mlx-community/Qwen3-Reranker-0.6B-4bit`'s README, which ships
/// a complete reference `rerank_score` implementation using exactly these three constants),
/// itself `catflow-mlx/src/catflow/engines/rerank.py`'s `_score_causal_yesno` recipe
/// (`_CAUSAL_YESNO_INSTRUCT`/`_CAUSAL_YESNO_PREFIX`/`_CAUSAL_YESNO_SUFFIX`). Pure string
/// assembly — no model, no tokenizer — so the exact byte shape is unit-testable without
/// weights (MoC-4-3, `RSI/DelegateMoCBacklog.md`: "a unit test pins the assembled prompt
/// string byte-for-byte against the Python constants — this is the test that catches a
/// paraphrase"). Do not reword any of the three strings below.
nonisolated enum RerankPrompt {
    /// `_CAUSAL_YESNO_INSTRUCT`, verbatim.
    static let instruct = "Given a web search query, retrieve relevant passages that answer the query"

    /// `_CAUSAL_YESNO_PREFIX`, verbatim.
    static let prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"

    /// `_CAUSAL_YESNO_SUFFIX`, verbatim.
    static let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    /// `<Instruct>: … \n<Query>: … \n<Document>: …` — the content assembled between
    /// `prefix` and `suffix` before tokenizing.
    static func content(query: String, document: String) -> String {
        "<Instruct>: \(instruct)\n<Query>: \(query)\n<Document>: \(document)"
    }
}
