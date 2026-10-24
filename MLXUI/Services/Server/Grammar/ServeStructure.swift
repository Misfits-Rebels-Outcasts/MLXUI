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

/// WW4-1-4 — how a dialect spells the field that carried the format, for the 400s that name it. `schemaParam` is the path
/// ServeSchemaCheck prefixes to every JSON pointer, so the client can find the construct in its own request.
nonisolated struct ServeStructureNames: Sendable, Equatable {
    /// The field itself: `response_format` / `text.format`.
    let field: String
    /// The field's schema: `response_format.json_schema.schema` / `text.format.schema`.
    let schemaParam: String

    static let chatCompletions = ServeStructureNames(field: "response_format", schemaParam: "response_format.json_schema.schema")
    static let responses = ServeStructureNames(field: "text.format", schemaParam: "text.format.schema")
}
