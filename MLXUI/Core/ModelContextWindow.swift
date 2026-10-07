import Foundation

/// A checkpoint's **declared context window**, from its `config.json` (S2-6, R23).
///
/// One parse, two readers: the Connect sheet's fitness line (S2-5) and the Local Server's
/// pre-count — which must agree, because a client that was told "Context length: 40,960 tokens"
/// should not then be refused at 32,000.
///
/// The key lives at the top level for most checkpoints and under `text_config` for the multimodal
/// ones (the catalog's Qwen3.5/3.8 rows), and the two spellings are checked in that order. Which
/// one each of the 12 catalog chat models uses is in
/// `RSI/journal/2026-410-s2-6-precount.md`.
nonisolated enum ModelContextWindow {

    /// The limit `config.json` declares, or `nil` when it declares none.
    static func limit(configData: Data?) -> Int? {
        limit(object: object(from: configData))
    }

    /// The same parse, from an already-decoded config object.
    static func limit(object: [String: Any]?) -> Int? {
        guard let object else { return nil }
        return positions(object) ?? (object["text_config"] as? [String: Any]).flatMap(positions)
    }

    private static func positions(_ dictionary: [String: Any]) -> Int? {
        (dictionary["max_position_embeddings"] as? NSNumber).flatMap { $0.intValue > 0 ? $0.intValue : nil }
    }

    private static func object(from data: Data?) -> [String: Any]? {
        guard let data else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
}
