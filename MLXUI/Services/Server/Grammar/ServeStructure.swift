import Foundation

/// What shape a reply must have — the neutral form of `response_format` (chat completions) and `text.format` (responses).
///
/// `{"type": "text"}` is **`.none`**: Codex may send it on every request with tools on, and it must behave exactly as if the
/// field were absent (owner ruling AO, 2026-10-24) — never constrained, never a 400.
nonisolated enum ServeStructure: Sendable, Equatable {
    case none
    /// `{"type": "json_object"}` — any JSON object.
    case jsonObject
    /// `{"type": "json_schema", …}`. `schema` is the schema **as the client sent it** (JSON text). `name` is the wire's label
    /// and `strict` the wire's flag; the server always enforces (a `strict: false` client gets the stricter behaviour), so
    /// neither changes what is compiled.
    case jsonSchema(name: String?, schema: String, strict: Bool)

    /// Whether the reply is constrained at all.
    var isConstrained: Bool {
        switch self {
        case .none: return false
        case .jsonObject, .jsonSchema: return true
        }
    }

    /// What `ServeGrammarCache` compiles; `nil` for `.none`.
    var grammarFormat: ServeGrammarFormat? {
        switch self {
        case .none: return nil
        case .jsonObject: return .jsonObject
        case .jsonSchema(_, let schema, _): return .jsonSchema(schema)
        }
    }
}
