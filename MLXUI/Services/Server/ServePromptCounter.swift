import Foundation
import MLXLMCommon

/// R23 (S2-6): the pre-count's tokenizer cache — **tokenizers only, never weights**.
///
/// Counting a prompt needs the chat template and the tokenizer, and nothing else. Reading
/// `tokenizer.json` for a big vocabulary costs about a second (measured 1,043 ms for Qwen3-8B on
/// this Mac), which is far too much to put in front of every request's head; the template render
/// itself is milliseconds. A tokenizer is immutable and a few megabytes, so it is cached here by
/// directory, while the loaded **weights** stay `ModelContainerPool`'s business alone (S1-1's
/// one-owner rule is about weights, and this doesn't touch them).
///
/// It is an `actor` for two reasons: it serialises the tokenizer's use (swift-transformers'
/// tokenizers aren't documented as thread-safe, and two requests for one model would otherwise
/// render through the same object), and it keeps the dictionary itself safe. The tokenizer it hands
/// out is a **separate instance** from the one inside the pool's `ModelContainer`, so a generation
/// and a pre-count never share an object.
///
/// Known limits (implementer's call, pending owner confirmation): the cache holds four directories,
/// least-recently-used first (the same convention as `ResidentPool`), and it is keyed on the path
/// alone — re-downloading a model in place would need a relaunch (or four other models) before its
/// tokenizer is re-read. (S2-6 review: the first version popped `keys.first`, which in a Swift
/// dictionary is *arbitrary*, not oldest — now genuinely LRU.)
actor ServePromptCounter {
    nonisolated static let shared = ServePromptCounter()

    private var tokenizers: [String: any MLXLMCommon.Tokenizer] = [:]
    /// Most-recently-used first; eviction pops the end.
    private var recency: [String] = []
    private let capacity = 4

    /// Count the tokens the template renders for `messages`, with this request's tools and render
    /// context. `nil` when the checkpoint has no readable tokenizer or no chat template.
    /// - Throws: whatever the template raised (S2-8 turns a refusal into a `400`); a *missing*
    ///   tokenizer or chat template is `nil` — "can't count", not a refusal (S2-6's fail-open).
    func count(directory: URL,
               messages: [[String: any Sendable]],
               tools: [[String: any Sendable]]?,
               context: [String: any Sendable]) async throws -> Int? {
        guard let tokenizer = await tokenizer(for: directory) else { return nil }
        return try tokenizer.applyChatTemplate(messages: messages, tools: tools,
                                               additionalContext: context).count
    }

    /// S3-5b: after a render has failed, which tools fail **alone** (a one-message conversation), by name.
    /// Cheap enough for the failure path only: one render per tool.
    func failingToolNames(directory: URL, tools: [[String: any Sendable]],
                          context: [String: any Sendable]) async -> [String] {
        guard let tokenizer = await tokenizer(for: directory) else { return [] }
        let probe: [[String: any Sendable]] = [["role": "user", "content": "x"]]
        return ChatTemplateFailure.failingToolNames(in: tools) { tool in
            (try? tokenizer.applyChatTemplate(messages: probe, tools: [tool], additionalContext: context)) != nil
        }
    }

    /// For tests and for the journal's measurement: how many tokenizers are resident.
    var cachedCount: Int { tokenizers.count }

    private func tokenizer(for directory: URL) async -> (any MLXLMCommon.Tokenizer)? {
        let key = directory.standardizedFileURL.path
        if let cached = tokenizers[key] {
            touch(key)
            return cached
        }
        guard let loaded = try? await HFTokenizerLoader().load(from: directory) else { return nil }
        if tokenizers.count >= capacity, let leastRecent = recency.last {
            tokenizers[leastRecent] = nil
            recency.removeLast()
        }
        tokenizers[key] = loaded
        touch(key)
        return loaded
    }

    private func touch(_ key: String) {
        recency.removeAll { $0 == key }
        recency.insert(key, at: 0)
    }
}
