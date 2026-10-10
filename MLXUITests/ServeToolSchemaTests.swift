import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

// S3-5b part 2 — tool schemas with JSON `null`s (the cause of the six unrenderable runpod tools).

/// Shapes copied from the six failing runpod schemas ($defs/$ref/allOf/oneOf+discriminator/if-then/
/// dependentRequired/unevaluatedProperties/uniqueItems/examples/type lists/enum) and the `null`s they carry.
/// No real content.
private let fixtureSchema = #"""
{"type":"object","required":["body"],
 "properties":{"body":{"$ref":"#/$defs/CreateRequest"}},
 "$defs":{
  "CreateRequest":{"type":"object","allOf":[{"$ref":"#/$defs/Base"}],"required":["name"],"unevaluatedProperties":false,
    "properties":{
      "name":{"type":"string","minLength":1,"default":null},
      "env":{"type":"object","additionalProperties":{"type":"string"},"examples":[{"JUPYTER_PASSWORD":"x"}]},
      "ports":{"type":"array","items":{"type":"string"},"uniqueItems":true},
      "category":{"enum":["NVIDIA","AMD",null]},
      "scaling":{"oneOf":[{"$ref":"#/$defs/A"},{"$ref":"#/$defs/B"}],"discriminator":{"propertyName":"t","mapping":{"a":"#/$defs/A"}}},
      "gone":null},
    "if":{"properties":{"k":{"const":1}}},"then":{"required":["x"]},"dependentRequired":{"a":["b"]}},
  "Base":{"type":"object","properties":{"disk":{"type":["integer","null"],"minimum":1,"default":null}}},
  "A":{"type":"object"},"B":{"type":"object"}}}
"""#

private func toolJSON(_ schema: String, name: String = "mcp__x__create-pod") -> String {
    #"{"type":"function","function":{"name":"\#(name)","description":"d","parameters":\#(schema)}}"#
}

private func containsNull(_ value: Any) -> Bool {
    switch value {
    case is NSNull: return true
    case let d as [String: Any]: return d.values.contains(where: containsNull)
    case let a as [Any]: return a.contains(where: containsNull)
    default: return false
    }
}

private func propertyPaths(_ value: Any, path: String = "") -> Set<String> {
    var out: Set<String> = []
    if let d = value as? [String: Any] {
        if let props = d["properties"] as? [String: Any] {
            for (name, child) in props { out.insert(path + "/" + name); out.formUnion(propertyPaths(child, path: path + "/" + name)) }
        }
        for (key, child) in d where key != "properties" { out.formUnion(propertyPaths(child, path: path + "/" + key)) }
    } else if let a = value as? [Any] {
        for (i, child) in a.enumerated() { out.formUnion(propertyPaths(child, path: path + "[\(i)]")) }
    }
    return out
}

/// A canonical (sorted-key) JSON text of any JSON value, so dictionaries compare by content, not iteration order.
private func canon(_ value: Any) -> String {
    let data = (try? JSONSerialization.data(withJSONObject: [value], options: [.sortedKeys])) ?? Data()
    return String(decoding: data, as: UTF8.self)
}

/// Every value stored under `key`, anywhere in the schema, canonicalised and sorted.
private func collect(_ key: String, in value: Any) -> [String] {
    var out: [String] = []
    func walk(_ v: Any) {
        if let d = v as? [String: Any] {
            if let found = d[key] { out.append(canon(found)) }
            d.values.forEach(walk)
        } else if let a = v as? [Any] { a.forEach(walk) }
    }
    walk(value)
    return out.sorted()
}

struct ServeToolSchemaTests {
    private func parameters(of spec: ServeToolSpec) -> [String: Any] {
        ((spec.object?["function"] as? [String: Any])?["parameters"] as? [String: Any]) ?? [:]
    }

    @Test func noNullSurvivesInTheObjectHandedToTheTemplate() throws {
        let spec = ServeToolSpec(name: "t", rawJSON: toolJSON(fixtureSchema))
        let object = try #require(spec.object)
        #expect(!containsNull(object))
        // …while the client's own JSON is kept verbatim
        #expect(spec.rawJSON.contains(#""default":null"#) && spec.rawJSON.contains("null]"))
    }

    /// The tool's meaning survives: every property (even the one that was `null`), every `required`, every type.
    @Test func propertiesRequiredAndTypesSurvive() throws {
        let original = try JSONSerialization.jsonObject(with: Data(fixtureSchema.utf8))
        let scrubbed = ServeToolSchema.renderable(original)
        #expect(propertyPaths(scrubbed).isSuperset(of: propertyPaths(original)))
        let props = try #require(((scrubbed as? [String: Any])?["$defs"] as? [String: Any])?["CreateRequest"] as? [String: Any])
        let names = Set((props["properties"] as? [String: Any] ?? [:]).keys)
        #expect(names == ["name", "env", "ports", "category", "scaling", "gone"])    // "gone" was null: kept, as {}
        #expect(((props["properties"] as? [String: Any])?["gone"] as? [String: Any])?.isEmpty == true)
        #expect(collect("required", in: scrubbed) == collect("required", in: original))
        #expect(collect("type", in: scrubbed) == collect("type", in: original))
        // the constructs that never needed touching are untouched
        for keyword in ["$ref", "allOf", "oneOf", "discriminator", "unevaluatedProperties", "dependentRequired", "uniqueItems",
                        "additionalProperties", "minimum", "minLength", "examples", "then", "if"] {
            #expect(collect(keyword, in: scrubbed) == collect(keyword, in: original), "\(keyword)")
        }
    }

    /// The lossy parts, pinned: a null-valued key is dropped, a null in an array is the string "null".
    @Test func theLossyRulesAreExactlyThese() throws {
        let scrubbed = try #require(ServeToolSchema.renderable(try JSONSerialization.jsonObject(with: Data(fixtureSchema.utf8))) as? [String: Any])
        let defs = try #require(scrubbed["$defs"] as? [String: Any])
        let name = try #require(((defs["CreateRequest"] as? [String: Any])?["properties"] as? [String: Any])?["name"] as? [String: Any])
        #expect(name["default"] == nil && name["minLength"] as? Int == 1 && name["type"] as? String == "string")
        let disk = try #require(((defs["Base"] as? [String: Any])?["properties"] as? [String: Any])?["disk"] as? [String: Any])
        #expect(disk["default"] == nil && disk["type"] as? [String] == ["integer", "null"])        // the string "null" in a type list was always fine
        let category = try #require(((defs["CreateRequest"] as? [String: Any])?["properties"] as? [String: Any])?["category"] as? [String: Any])
        #expect(category["enum"] as? [String] == ["NVIDIA", "AMD", "null"])
    }

    @Test func aSchemaWithoutNullsComesBackEqual() throws {
        let plain = try JSONSerialization.jsonObject(with: Data(#"{"type":"object","required":["a"],"properties":{"a":{"type":"string","enum":["x","y"]},"b":{"type":"array","items":{"type":"number"}}}}"#.utf8))
        #expect((ServeToolSchema.renderable(plain) as AnyObject).isEqual(plain))
    }

    /// Every dialect's tool lands in `ServeToolSpec`, so every dialect gets the scrub.
    @Test func eachDialectsToolIsScrubbed() throws {
        let schema = #"{"type":"object","properties":{"a":{"type":"string","default":null}}}"#
        let chat = try OpenAIChatDialect.decode(Data(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[\#(toolJSON(schema))]}"#.utf8))
        let anthropic = try AnthropicMessagesDialect.decode(Data(#"{"model":"m","max_tokens":9,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"t","input_schema":\#(schema)}]}"#.utf8))
        let responses = try OpenAIResponsesDialect.decode(Data(#"{"model":"m","input":"x","tools":[{"type":"function","name":"t","parameters":\#(schema)}]}"#.utf8)).request
        for request in [chat, anthropic, responses] {
            let tool = try #require(request.tools.first)
            #expect(tool.object.map { !containsNull($0) } == true)
        }
    }

    /// With the real Qwen3-8B tokenizer (run-if-installed): the scrubbed tool renders and counts; the raw one
    /// is exactly the failure the six runpod tools hit.
    @Test func theScrubbedToolRendersAndTheRawOneDoesNot() async throws {
        let dir = ModelStore.shared.directory(forHFModelID: "mlx-community/Qwen3-8B-4bit")
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) else { return }
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let messages: [[String: any Sendable]] = [["role": "user", "content": "hi"]]
        let spec = ServeToolSpec(name: "mcp__x__create-pod", rawJSON: toolJSON(fixtureSchema))
        let withTool = try tokenizer.applyChatTemplate(messages: messages, tools: [try #require(spec.object)], additionalContext: nil)
        let without = try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
        #expect(withTool.count > without.count + 100)                       // the tool's schema is in the prompt
        let raw = try #require(JSONSerialization.jsonObject(with: Data(spec.rawJSON.utf8)) as? [String: any Sendable])
        #expect(throws: (any Error).self) { try tokenizer.applyChatTemplate(messages: messages, tools: [raw], additionalContext: nil) }
    }
}
