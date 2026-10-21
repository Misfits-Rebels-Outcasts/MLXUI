import Foundation
import Jinja
import Testing
@testable import MLXUI

/// MP1-3-FIX-1 — `ChatTemplateSanitizer.applyingCommentWhitespaceControl` applies Jinja's `-` on comment tags
/// (`{#- … -#}`), which swift-jinja 2.3.6 ignores (upstream: `huggingface/swift-jinja`,
/// `Sources/Jinja/Lexer.swift`, `extractCommentToken`).
///
/// Three things are proven:
///  1. **Parity.** The pinned Gemma 4 template, sanitized and rendered the way the app renders it, is
///     byte-identical to Python's jinja2 for four conversations (`Fixtures/Gemma4/prompt_goldens.json`,
///     written by `Fixtures/Gemma4/make_prompt_goldens.py` with the provenance recorded inside it).
///  2. **The transform is exactly Jinja's rule.** Micro-templates whose expected output was produced by Python
///     jinja2 3.1.6 (`ImmutableSandboxedEnvironment(trim_blocks: True, lstrip_blocks: True)`, 2026-10-21).
///  3. **Nothing else moves.** A template with no dashed comment comes back nil (the caller then uses the
///     checkpoint's own template, as before) and every other pinned template renders identically.
struct CommentWhitespaceControlTests {

    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    /// The options swift-transformers compiles every chat template with (`Tokenizer.compiledTemplate`).
    private func render(_ template: String, _ context: [String: Value] = [:]) throws -> String {
        try Template(template, with: .init(lstripBlocks: true, trimBlocks: true)).render(context)
    }

    /// What the app does: render the sanitized template if there is one, else the original.
    private func renderAsTheAppDoes(_ template: String, _ context: [String: Value] = [:]) throws -> String {
        try render(ChatTemplateSanitizer.sanitized(template) ?? template, context)
    }

    // MARK: 1. Parity with Python on the real Gemma 4 template

    private struct GoldenFile: Decodable {
        struct Provenance: Decodable { let jinja2: String; let templateSHA256: String; let environment: String }
        struct Case: Decodable { let name: String; let context: [String: JSONValue]; let expected: String }
        let provenance: Provenance
        let cases: [Case]
    }

    /// Just enough JSON to turn a golden's context into jinja `Value`s.
    private enum JSONValue: Decodable {
        case string(String), bool(Bool), number(Double), array([JSONValue]), object([String: JSONValue]), null
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if c.decodeNil() { self = .null }
            else if let v = try? c.decode(Bool.self) { self = .bool(v) }
            else if let v = try? c.decode(Double.self) { self = .number(v) }
            else if let v = try? c.decode(String.self) { self = .string(v) }
            else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
            else { self = .object(try c.decode([String: JSONValue].self)) }
        }
        var any: Any? {
            switch self {
            case .string(let v): v
            case .bool(let v): v
            case .number(let v): v
            case .array(let v): v.map { $0.any as Any }
            case .object(let v): v.mapValues { $0.any as Any }
            case .null: nil
            }
        }
    }

    private func goldens() throws -> GoldenFile {
        try JSONDecoder().decode(GoldenFile.self,
            from: Data(contentsOf: root.appendingPathComponent("Fixtures/Gemma4/prompt_goldens.json")))
    }

    private func gemmaTemplate() throws -> String {
        try String(contentsOf: root.appendingPathComponent("Fixtures/Gemma4/chat_template.jinja"), encoding: .utf8)
    }

    @Test func goldensWerePinnedAgainstThisExactTemplate() throws {
        let g = try goldens()
        #expect(g.provenance.templateSHA256 == "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082")
        #expect(g.provenance.jinja2 == "3.1.6")
        #expect(g.cases.map(\.name) == ["a_one_user_turn", "b_system_and_user", "c_multi_turn", "d_system_image_text"])
    }

    @Test func gemma4TemplateRendersByteIdenticalToPythonJinja2() throws {
        let template = try gemmaTemplate()
        for c in try goldens().cases {
            var context: [String: Value] = [:]
            for (key, value) in c.context { context[key] = try Value(any: value.any) }
            let out = try renderAsTheAppDoes(template, context)
            #expect(out == c.expected, "\(c.name): got \(out.debugDescription), Python gave \(c.expected.debugDescription)")
        }
    }

    // MARK: 2. The transform is Jinja's rule (expected strings from Python jinja2 3.1.6)

    @Test(arguments: [
        ("both", "A\n\n{#- c -#}\nB", "AB"),
        ("left only", "A  \n  {#- c #}\nB", "AB"),
        ("right only", "A{# c -#}  \n\n  B", "AB"),
        ("no dashes: left alone", "A\n{# c #}\nB", "A\nB"),
        ("comment at both ends", "{#- c -#}\n x \n{#- d -#}", "x"),
        ("a `{#` inside a string literal is not a comment", "{{ '{#- not -#}' }}\n{#- real -#}\nZ", "{#- not -#}Z"),
        ("dict literal before a comment", "{{ {'a': 1}['a'] }}  {#- c -#}  {{ 2 }}", "12"),
        ("after a block tag", "{% if true %}\nX\n{% endif %}\n\n{#- c -#}\n  Y", "X\nY"),
        ("adjacent comments", "a {#- c -#}  {#- d -#} b", "ab"),
        ("`{#-#}` has a left dash only", "A  {#-#}  B", "A  B"),
    ] as [(String, String, String)])
    func commentWhitespaceControlMatchesPython(name: String, source: String, python: String) throws {
        #expect(try renderAsTheAppDoes(source) == python, "\(name)")
    }

    /// swift-jinja has no `{% raw %}` support at all (`Unknown statement: raw`), so this cannot be a render
    /// comparison; the transform alone must still leave a raw block's contents byte-for-byte alone. Python
    /// gives `{#- x -#}  Q` for this source.
    @Test func rawBlocksAreCopiedVerbatimAndTheCommentAfterThemIsStillApplied() throws {
        let source = "{% raw %}{#- x -#}  {% endraw %}\n{#- c -#}\nQ"
        let out = try #require(ChatTemplateSanitizer.applyingCommentWhitespaceControl(source))
        #expect(out == "{% raw %}{#- x -#}  {% endraw %}{# c #}Q")
    }

    @Test func anUnterminatedCommentIsLeftVerbatim() {
        let source = "A  {#- never closed"
        #expect(ChatTemplateSanitizer.applyingCommentWhitespaceControl(source) == nil)
    }

    // MARK: 3. Nothing else moves

    @Test func templatesWithoutADashedCommentAreReturnedAsNil() {
        for source in ["plain text", "{{ x }}", "{% if a %}b{% endif %}", "A\n{# plain comment #}\nB",
                       "{{ '{#- in a string -#}' }}", "{% raw %}{#- raw -#}{% endraw %}"] {
            #expect(ChatTemplateSanitizer.applyingCommentWhitespaceControl(source) == nil, "\(source)")
            #expect(ChatTemplateSanitizer.sanitized(source) == nil, "\(source)")
        }
    }

    @Test func theMiMoTemplateIsUntouchedAndItsRenderIsIdentical() throws {
        let template = try String(contentsOf: root.appendingPathComponent("Fixtures/MiMo/chat_template.jinja"),
                                  encoding: .utf8)
        #expect(!template.contains("{#"))
        #expect(ChatTemplateSanitizer.sanitized(template) == nil)
        let context: [String: Value] = [
            "messages": [["role": "system", "content": "Be brief."], ["role": "user", "content": "Hi"]],
            "add_generation_prompt": true, "enable_thinking": false,
        ]
        #expect(try renderAsTheAppDoes(template, context) == render(template, context))
    }

    @Test func theQwenGuardFixStillAppliesAndStaysTheOnlyChangeWhenThereAreNoComments() throws {
        let strict = ChatTemplateSanitizerTests.strictTemplate
        let sanitized = try #require(ChatTemplateSanitizer.sanitized(strict))
        #expect(!sanitized.contains("No user query found"))
        // The comment pass found nothing to do on it, so the guard removal is the whole difference.
        #expect(ChatTemplateSanitizer.applyingCommentWhitespaceControl(strict) == nil)
        #expect(ChatTemplateSanitizer.applyingCommentWhitespaceControl(sanitized) == nil)
    }

    @Test func aTemplateWithBothAGuardAndDashedCommentsGetsBothFixes() throws {
        let source = "{#- note -#}\n" + ChatTemplateSanitizerTests.strictTemplate
        let sanitized = try #require(ChatTemplateSanitizer.sanitized(source))
        #expect(!sanitized.contains("No user query found"))
        #expect(!sanitized.contains("{#-") && !sanitized.contains("-#}"))
    }

    @Test func sanitizingTwiceChangesNothingMore() throws {
        let once = try #require(ChatTemplateSanitizer.sanitized(try gemmaTemplate()))
        #expect(ChatTemplateSanitizer.sanitized(once) == nil)
    }
}
