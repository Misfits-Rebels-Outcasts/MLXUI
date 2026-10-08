import Foundation

// PC-2 — when the prompt cache may run at all. One place, one table (rule 10, ruling D1).

nonisolated enum PromptCache {
    /// Tests flip this to run the cache on or off without touching `AppState`.
    nonisolated(unsafe) static var overrideForTesting: Bool?

    /// The kill switch (rule 10) and the Local Server's own gate (Serve rule 14). Every
    /// prompt-cache path goes through here.
    static var isEnabled: Bool {
        overrideForTesting ?? (!AppState.disablePromptCache && LocalServerGate.isAvailable)
    }
}

/// The checkpoint `model_type`s whose cache-on output was **measured equal** to cache-off by a real
/// model test (ruling D1, PC-A0). Everything else bypasses the cache: no lookup, no insert.
///
/// A type joins this table only with the journal line that shows its real-model test passing. The
/// strings are the top-level `model_type` of the checkpoint's `config.json`
/// (`ModelContainerPool.modelTypeString`), never from memory. Pinned by a test.
nonisolated enum PromptCacheAllowlist {
    /// Tests substitute a table to prove the bypass.
    nonisolated(unsafe) static var overrideForTesting: Set<String>?

    /// Journal 2026-420 has each family's real-model line (B1: identical, or a logged near-tie with a
    /// top-two gap ≤ 1.0, on ≥ 256-token answers; two-turn, tool-call history, thinking-on where the
    /// family has them).
    /// - `qwen3` — Qwen3-8B. `qwen3_5` — Qwen3.5-9B; Qwen3.8-27B's configs say `qwen3_5` too
    ///   (`RSI/notes/qwen38/hf`), but Ternary-Bonsai-27B's config was **not** read, so its
    ///   membership is unverified. `gemma3_text` — Gemma 3 1B.
    /// - `llama` — Llama-3.1-8B. `gemma2` — gemma-2-9b.
    /// - **`mistral3` is deliberately absent.** Ministral-3-3B passes, but the same `model_type`
    ///   string would admit Devstral-Small-2-24B, which the PC-A0 ruling keeps off until someone
    ///   tests it (too big for the owner's Mac). Owner's call; see the journal.
    static let tested: Set<String> = ["qwen3", "qwen3_5", "gemma3_text", "llama", "gemma2"]

    static func allows(_ modelType: String) -> Bool { (overrideForTesting ?? tested).contains(modelType) }
}

/// Why a request did not use the cache. Metadata only (rule 9).
nonisolated enum PromptCacheBypass: String, Sendable, Equatable {
    case disabled                // the kill switch
    case modelNotTested          // not on the allowlist
    case seeded                  // `seed` is about reproducibility (implementer's call, pending owner confirmation)
    case customKVCache           // kvBits / maxKVSize change the cache classes
    case failed                  // the cache path threw; the request ran as if it were off
}

/// What a request did with the cache. Nothing in it is content: counts and token positions only.
/// PC-3 turns `reusedTokens` into `usage.cached_tokens`.
nonisolated struct PromptCacheOutcome: Sendable, Equatable {
    var bypass: PromptCacheBypass? = nil
    /// Prompt tokens served from a snapshot (0 on a miss).
    var reusedTokens = 0
    var promptTokens = 0
    /// The checkpoints this request prefilled and stored, by position.
    var storedAt: [Int] = []
    var hitStoredLength: Int? = nil
}

nonisolated enum PromptCachePolicy {
    /// `nil` = the cache may run.
    static func bypass(enabled: Bool, modelType: String, seed: Int?, kvBits: Int?, maxKVSize: Int?) -> PromptCacheBypass? {
        if !enabled { return .disabled }
        if !PromptCacheAllowlist.allows(modelType) { return .modelNotTested }
        if seed != nil { return .seeded }
        if kvBits != nil || maxKVSize != nil { return .customKVCache }
        return nil
    }
}
