import Foundation

/// Task-instruction prefixes for **asymmetric** retrieval embedding models. Some families
/// (nomic) are trained to prepend a task prefix to queries vs. documents; without it,
/// query↔document similarity is degraded. Symmetric models (all-MiniLM, bge-m3) need none
/// and return `nil`, so the run view hides the toggle for them.
enum EmbeddingPrefixes {
    /// The `(query, document)` prefixes for `family`, or `nil` if the model is symmetric.
    /// Matched case-insensitively on the family name.
    static func queryDocument(family: String) -> (query: String, document: String)? {
        let lower = family.lowercased()
        if lower.contains("nomic") {
            return (query: "search_query: ", document: "search_document: ")
        }
        return nil
    }
}
