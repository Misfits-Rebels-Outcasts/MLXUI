import Foundation

// PC-2 — when the prompt cache may run at all. One place, one table (rule 10, ruling D1).

nonisolated enum PromptCache {
    /// Tests flip this to run the cache on or off without touching `AppState`.
    nonisolated(unsafe) static var overrideForTesting: Bool?
    /// Tests make the cache path throw, to prove a failing cache falls back to the uncached path
    /// *and says so* (PC-3). Never set outside tests.
    nonisolated(unsafe) static var injectFailureForTesting = false

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

    /// **The allowlist is by checkpoint `model_type`, and a type that passed a real-model test admits
    /// every model of that type** (owner ruling 2026-10-08, amending D1: "Allow mistral3"). So
    /// Qwen3.8-27B and Ternary-Bonsai-27B ride on Qwen3.5-9B's pass, and Devstral-Small-2-24B rides
    /// on Ministral-3-3B's. Journal 2026-420 has each family's real-model line (B1: identical, or a
    /// logged near-tie with a top-two gap ≤ 1.0, on ≥ 256-token answers; two-turn, tool-call
    /// history, thinking-on where the family has them).
    ///
    /// - `qwen3` — Qwen3-8B. `qwen3_5` — Qwen3.5-9B; Qwen3.8-27B's configs say `qwen3_5`
    ///   (`RSI/notes/qwen38/hf`), and Ternary-Bonsai-27B is `qwen3_5` too (HF 2026-10-08:
    ///   `text_config` `qwen3_5_text`, `full_attention_interval` 4). `gemma3_text` — Gemma 3 1B.
    /// - `llama` — Llama-3.1-8B. `gemma2` — gemma-2-9b.
    /// - `mistral3` — Ministral-3-3B. Ministral and Devstral are both `mistral3` with
    ///   `text_config.model_type` `ministral3` and `sliding_window: null` (HF 2026-10-08), so a
    ///   `text_config` key could not separate them anyway.
    /// - Still off: `gpt_oss`, `qwen2` (and every other type) until a real-model test passes.
    static let tested: Set<String> = ["qwen3", "qwen3_5", "gemma3_text", "llama", "gemma2", "mistral3"]

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

/// Where a request's cache work went, in seconds (PC-3b). Durations only — metadata (rule 9).
nonisolated struct PromptCacheTimings: Sendable, Equatable {
    /// `pool.container` (a model load shows up here).
    var container = 0.0
    /// Rendering the real prompt through the chat template and tokenizer (`processor.prepare`).
    var render = 0.0
    /// The two probe renders (turn and system boundary) — each is a full render + tokenization.
    var probes = 0.0
    /// `store.lookup`.
    var lookup = 0.0
    /// Copying the hit, trimming it, and evaluating the copy.
    var restore = 0.0
    /// Manual prefill to the checkpoints (a miss's real work; ~0 on a hit that already covers T).
    var prefill = 0.0
    /// Copying + inserting snapshots after each prefilled segment.
    var snapshot = 0.0
    var total: Double { container + render + probes + lookup + restore + prefill + snapshot }
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
    var timings = PromptCacheTimings()
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

/// Process-wide counts for the pane and for tests (PC-3). Counts only — nothing about content.
nonisolated final class PromptCacheCounters: @unchecked Sendable {
    static let shared = PromptCacheCounters()

    struct Snapshot: Equatable, Sendable {
        var requests = 0          // requests that used the cache path (not bypassed)
        var hits = 0              // …of which reused a snapshot
        var fallbacks = 0         // …of which the cache path failed and the request ran uncached
        var reusedTokens = 0
    }

    private let lock = NSLock()
    private var value = Snapshot()

    var snapshot: Snapshot { lock.lock(); defer { lock.unlock() }; return value }

    func record(_ outcome: PromptCacheOutcome) {
        lock.lock(); defer { lock.unlock() }
        if outcome.bypass == .failed { value.requests += 1; value.fallbacks += 1; return }
        guard outcome.bypass == nil else { return }
        value.requests += 1
        if outcome.reusedTokens > 0 { value.hits += 1; value.reusedTokens += outcome.reusedTokens }
    }

    func reset() { lock.lock(); value = Snapshot(); lock.unlock() }
}

/// The pane's one line: "Prompt cache: 3 snapshots, 1.2 GB" (+ ", 2 fell back" when any did).
nonisolated enum PromptCacheSummary {
    static func line(snapshots: Int, bytes: Int64, fallbacks: Int) -> String {
        let size = String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
        var line = "Prompt cache: \(snapshots) \(snapshots == 1 ? "snapshot" : "snapshots"), \(size)"
        if fallbacks > 0 { line += ", \(fallbacks) fell back to no cache" }
        return line
    }
}

nonisolated extension ServeUsage {
    /// The usage a request reports, with what the prompt cache did folded in (PC-3).
    init(promptTokens: Int, completionTokens: Int, outcome: PromptCacheOutcome) {
        self.init(promptTokens: promptTokens, completionTokens: completionTokens,
                  cachedTokens: outcome.reusedTokens, cacheFellBack: outcome.bypass == .failed)
    }
}

nonisolated enum PromptCacheFault: Error { case injected }
