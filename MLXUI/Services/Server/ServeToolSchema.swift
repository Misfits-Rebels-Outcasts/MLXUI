import Foundation

/// S3-5b part 2 — make a client's tool schemas **renderable** (journal `2026-441`).
///
/// **The cause** (verified in `swift-jinja`'s `Value.swift`, not from memory): the chat template is run by
/// `swift-jinja`, whose `Value(any:)` maps Swift `nil` to `null` but has no case for `NSNull` — what
/// `JSONSerialization` produces for a JSON `null`. So *any* JSON `null` anywhere in a tool schema
/// (`"default": null` is the common one) makes the render throw `JinjaError.runtime("Cannot convert value of
/// type Optional<Any> to Jinja Value")`. `$defs`, `$ref`, `allOf`, `oneOf`, `if/then`, `discriminator`,
/// `dependentRequired`, `unevaluatedProperties`, `examples`, type lists and the rest all render fine; the six
/// runpod MCP tools that failed carried a `null` (inside `$defs.ContainerConfig`). The minimal failing schema:
/// `{"type":"object","properties":{"a":{"type":"string","default":null}}}`.
///
/// **The fix, at our layer** (no package bump, T15): scrub the nulls from the object handed to the template and
/// to the tool-call parsers (`ServeToolSpec.object`); the client's `rawJSON` stays verbatim. What it changes,
/// all journaled as lossy:
///
/// * an object key whose value is `null` is **dropped** (`"default": null` carries no usable information for
///   a model; `"const": null` and `"example": null` likewise);
/// * a `null` **inside an array** (an `enum` or `examples` entry) becomes the string `"null"`, so a nullable
///   enum still reads as one;
/// * a `null` where a **schema** is expected, as a value in `properties`, `$defs`, `definitions`,
///   `patternProperties` or `dependentSchemas`, becomes `{}` (the empty schema), so a **property is never
///   dropped**. `required` is an array of names and is never touched.
nonisolated enum ServeToolSchema {

    /// Keys whose value is a map of *name → schema*: a `null` entry there keeps its name.
    private static let schemaMaps: Set<String> = ["properties", "$defs", "definitions", "patternProperties", "dependentSchemas"]

    /// `value` with every JSON `null` removed or replaced as described above. Everything else is returned as is.
    static func renderable(_ value: Any) -> Any {
        scrub(value, isSchemaMap: false)
    }

    private static func scrub(_ value: Any, isSchemaMap: Bool) -> Any {
        switch value {
        case let dictionary as [String: Any]:
            var out: [String: Any] = [:]
            for (key, child) in dictionary {
                if child is NSNull {
                    if isSchemaMap { out[key] = [String: Any]() }       // keep the property, with the empty schema
                    continue                                           // anywhere else: drop the key
                }
                // The entries of a schema map are schemas (their own keys are keywords again).
                out[key] = scrub(child, isSchemaMap: !isSchemaMap && schemaMaps.contains(key))
            }
            return out
        case let array as [Any]:
            return array.map { $0 is NSNull ? "null" : scrub($0, isSchemaMap: false) }
        default:
            return value
        }
    }
}
