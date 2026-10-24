import Foundation

/// WW4-1-4 — `response_format` (Chat Completions) and `text.format` (Responses) → `ServeStructure`. Pure: it reads the parsed
/// request object and decides nothing about models, schemas or tools; `ServeStructureCheck` does that later, with the names
/// from `ServeStructureNames`.
///
/// - `{"type": "text"}`, an absent field and `null` are all `.none` — a no-op, never constrained, never a 400 (owner ruling AO:
///   Codex may send `text.format` text on every request, with tools on).
/// - `json_object` is any JSON object; `json_schema` carries the schema, which is handed on as JSON text.
/// - `name` is optional (accepted missing). **`strict: false` is enforced exactly like `true`**: this server never does
///   best-effort decoding, because a grammar either constrains the reply or it does not. The flag is kept only so the wire
///   value is not lost.
/// - `/v1/messages` is not read here: Anthropic's body has no such field and none is invented (later item, see the design note).
nonisolated enum ServeStructureDecoding {
    static let chatCode = "invalid_response_format"
    static let responsesCode = "invalid_text_format"

    // MARK: - Chat Completions

    /// `response_format`: `{"type":"json_schema","json_schema":{"name","schema","strict"}}` (nested) or `{"type":"json_object"}`.
    static func chatResponseFormat(_ value: Any?) throws -> ServeStructure {
        try decode(value, field: "response_format", code: chatCode, nested: true)
    }

    // MARK: - Responses

    /// The request's `text` object: `text.format` is `{"type":"json_schema","name","schema","strict"}` (flat, the Responses
    /// shape) or `{"type":"json_object"}`. `text` itself is otherwise ignored (`verbosity`, …), and a `text` that is not an
    /// object is ignored as before (R38: accepted, never a 400).
    static func responsesText(_ value: Any?) throws -> ServeStructure {
        guard let text = value as? [String: Any] else { return .none }
        return try decode(text["format"], field: "text.format", code: responsesCode, nested: false)
    }

    // MARK: -

    private static func decode(_ value: Any?, field: String, code: String, nested: Bool) throws -> ServeStructure {
        switch value {
        case nil, is NSNull: return .none
        default: break
        }
        guard let format = value as? [String: Any] else {
            throw invalid("'\(field)' must be an object such as {\"type\": \"json_schema\", …} or {\"type\": \"json_object\"}.", code: code)
        }
        guard let type = format["type"] as? String else {
            throw invalid("'\(field).type' is required and must be a string.", code: code)
        }
        switch type {
        case "text":
            return .none
        case "json_object":
            return .jsonObject
        case "json_schema":
            let body: [String: Any]
            let prefix: String
            if nested {
                guard let inner = format["json_schema"] as? [String: Any] else {
                    throw invalid("'\(field).json_schema' is required when the type is json_schema, and must be an object.", code: code)
                }
                body = inner
                prefix = "\(field).json_schema"
            } else {
                body = format
                prefix = field
            }
            var name: String?
            switch body["name"] {
            case nil, is NSNull: break
            case let text as String: name = text
            default: throw invalid("'\(prefix).name' must be a string.", code: code)
            }
            var strict = false
            switch body["strict"] {
            case nil, is NSNull: break
            // A real JSON boolean only: `as? Bool` would take the number 1 for `true`.
            case let flag as NSNumber where CFGetTypeID(flag) == CFBooleanGetTypeID(): strict = flag.boolValue
            default: throw invalid("'\(prefix).strict' must be true or false.", code: code)
            }
            guard let schema = body["schema"] as? [String: Any],
                  let data = try? JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys, .withoutEscapingSlashes]),
                  let text = String(data: data, encoding: .utf8) else {
                throw invalid("'\(prefix).schema' is required and must be a JSON Schema object.", code: code)
            }
            return .jsonSchema(name: name, schema: text, strict: strict)
        default:
            let shown = type.count > 40 ? String(type.prefix(40)) + "…" : type
            throw invalid("'\(field).type' must be 'text', 'json_object' or 'json_schema'; got '\(shown)'.", code: code)
        }
    }

    private static func invalid(_ message: String, code: String) -> ServeError {
        ServeError(status: 400, message: message, code: code)
    }
}
