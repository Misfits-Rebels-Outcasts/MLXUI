import Foundation
import Jinja
import Testing
@testable import MLXUI

/// The Bonsai tool-call fix (journal 2026-64). `ChatSession` restarts its tool loop with only
/// the tool-result message (the rest of the chat lives in the KV cache); strict Qwen3.5-style
/// templates `raise_exception('No user query found in messages.')` on that render, killing
/// every tool call after the tool ran. `ChatTemplateSanitizer` removes exactly that guard.
struct ChatTemplateSanitizerTests {

    /// A faithful miniature of the strict part of Bonsai's template: the reverse scan for a
    /// user query, the raising guard, and the tool-response branch.
    static let strictTemplate = """
        {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
        {%- for message in messages[::-1] %}
            {%- if ns.multi_step_tool and message.role == "user" %}
                {%- set ns.multi_step_tool = false %}
            {%- endif %}
        {%- endfor %}
        {%- if ns.multi_step_tool %}
            {{- raise_exception('No user query found in messages.') }}
        {%- endif %}
        {%- for message in messages %}
            {%- if message.role == "user" %}
                {{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
            {%- elif message.role == "tool" %}
                {{- '\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\\n' }}
        {%- endif %}
        """

    // MARK: - The string transform

    @Test func removesTheGuardAndNothingElse() throws {
        let sanitized = try #require(ChatTemplateSanitizer.sanitized(Self.strictTemplate))
        #expect(!sanitized.contains("raise_exception('No user query found in messages.')"))
        // Everything around the guard survives.
        #expect(sanitized.contains("namespace(multi_step_tool=true"))
        #expect(sanitized.contains("<tool_response>"))
        #expect(sanitized.contains("add_generation_prompt"))
    }

    @Test func matchesWhitespaceVariants() {
        let variant = "{% if ns.multi_step_tool %}{{ raise_exception('No user query found in messages.') }}{% endif %}"
        #expect(ChatTemplateSanitizer.sanitized(variant) != nil)
    }

    @Test func leavesTemplatesWithoutTheGuardAlone() {
        // No guard at all → nil (use the tokenizer's default template handling).
        #expect(ChatTemplateSanitizer.sanitized("{{ messages[0].content }}") == nil)
        // Uses the same namespace variable but never raises → untouched.
        let benign = "{%- if ns.multi_step_tool %}{{- '' }}{%- endif %}"
        #expect(ChatTemplateSanitizer.sanitized(benign) == nil)
        // A different raise_exception message → untouched (deliberately narrow).
        let other = "{%- if ns.multi_step_tool %}{{- raise_exception('Something else.') }}{%- endif %}"
        #expect(ChatTemplateSanitizer.sanitized(other) == nil)
    }

    @Test func sanitizationIsIdempotent() throws {
        let once = try #require(ChatTemplateSanitizer.sanitized(Self.strictTemplate))
        #expect(ChatTemplateSanitizer.sanitized(once) == nil)
    }

    // MARK: - End-to-end against the real Jinja engine

    @Test func strictTemplateRaisesOnChatSessionsToolOnlyRestartRender() throws {
        // Reproduces the bug: the exact message list ChatSession sends after a tool call.
        let template = try Template(Self.strictTemplate)
        #expect(throws: (any Error).self) {
            try template.render(Self.toolOnlyContext)
        }
    }

    @Test func sanitizedTemplateRendersTheToolOnlyRestart() throws {
        let sanitized = try #require(ChatTemplateSanitizer.sanitized(Self.strictTemplate))
        let output = try Template(sanitized).render(Self.toolOnlyContext)
        #expect(output.contains("<tool_response>\n17797\n</tool_response>"))
        #expect(output.contains("<|im_start|>assistant"))
    }

    @Test func sanitizedTemplateIsIdenticalForNormalChats() throws {
        // With a user message present the guard never fired, so removing it must not
        // change a single byte of normal renders.
        let context: [String: Value] = [
            "messages": [["role": "user", "content": "What is 37 x 481?"]],
            "add_generation_prompt": true,
        ]
        let original = try Template(Self.strictTemplate).render(context)
        let sanitized = try #require(ChatTemplateSanitizer.sanitized(Self.strictTemplate))
        #expect(try Template(sanitized).render(context) == original)
    }

    static let toolOnlyContext: [String: Value] = [
        "messages": [["role": "tool", "content": "17797"]],
        "add_generation_prompt": true,
    ]

    // MARK: - Model-directory loading

    @Test func sanitizedTemplateReadsChatTemplateJinja() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sanitizer-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // No template file → nil.
        #expect(ChatTemplateSanitizer.sanitizedTemplate(inModelDirectory: dir) == nil)

        // Strict template on disk → sanitized override.
        let url = dir.appendingPathComponent("chat_template.jinja")
        try Self.strictTemplate.write(to: url, atomically: true, encoding: .utf8)
        let override = try #require(ChatTemplateSanitizer.sanitizedTemplate(inModelDirectory: dir))
        #expect(!override.contains("No user query found"))

        // Benign template on disk → nil (no override, default path).
        try "{{ messages[0].content }}".write(to: url, atomically: true, encoding: .utf8)
        #expect(ChatTemplateSanitizer.sanitizedTemplate(inModelDirectory: dir) == nil)
    }
}
