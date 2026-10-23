import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

// WW2-3-3 — replay of the SV-8 failure on Qwen3.5-9B, with no model.
//
// SV-8 (Codex fixing `calc.py`, 2026-10-23, re-run through a recording proxy) ended with the model's final
// text being a literal `<tool_call><function=apply_patch>…` block. Codex had **not** offered `apply_patch`:
// the captured `/v1/responses` request carried `exec_command`, `write_stdin`, `request_user_input`,
// `view_image`, `get_goal`, `create_goal`, `update_goal` and the `multi_agent_v1` namespace (dropped by the
// dialect, R55). The same turns' `exec_command` calls were parsed fine under `.qwen35`.
//
// Cause (mlx-swift-lm 3.32 `Qwen35ToolCallParser.parse`): `.qwen35` returns nil for a call to a function that
// is not in the request's `tools` (`isDeclaredTool`); `.xmlFunction` (main's served format for `qwen3_5`) parses
// it, the client answers "unsupported call: apply_patch", and the model recovers with `exec_command` — which
// is exactly what SV-8 did on 2026-10-10 (journal 2026-448). These tests pin both halves: the replay under
// each format × each dialect's own `tools` list, and what the override gives up (Hermes-JSON payloads).

/// The raw text of SV-8's last turn (request 4 of the recorded run), verbatim.
private let sv8ApplyPatchText = """
The `mul` function is incorrectly using addition (`a + b`) instead of multiplication (`a * b`). Let me fix this:

<tool_call>
<function=apply_patch>
<parameter=command>
apply_patch
</parameter>
<parameter=arguments>
*** Begin Patch
*** Update File: calc.py
@@ def mul(a, b):
-    return a + b
+    return a * b
*** End Patch
</parameter>
</function>
</tool_call>
"""

/// A call the model makes to a declared tool, in the template's XML form. SV-7's turns come back from the server
/// already parsed, so the raw text is reconstructed from the parsed call (`Bash {command}`); SV-8's
/// `exec_command {cmd}` turns are the same shape.
private func xmlCall(_ name: String, _ key: String, _ value: String) -> String {
    "<tool_call>\n<function=\(name)>\n<parameter=\(key)>\n\(value)\n</parameter>\n</function>\n</tool_call>"
}

/// The Hermes-JSON dialect `.qwen35` additionally accepts.
private let hermesJSONCall = #"<tool_call>{"name": "exec_command", "arguments": {"cmd": "ls"}}</tool_call>"#

private func functionTool(_ name: String, param: String) -> [String: Any] {
    ["type": "function", "name": name, "description": name,
     "parameters": ["type": "object", "properties": [param: ["type": "string"]], "required": [param]]]
}

private enum Dialect: String, CaseIterable {
    case chatCompletions, messages, responses

    /// The `tools` each dialect hands the scanner for the clients of SV-7/8: Codex's recorded function set
    /// (and its `namespace`, which `/v1/responses` drops) on both OpenAI wires, Claude Code's recorded
    /// twelve on `/v1/messages`. Only names are from the capture; the schemas are minimal stand-ins.
    var tools: [[String: any Sendable]] {
        get throws {
            let codex = ["exec_command": "cmd", "write_stdin": "chars", "request_user_input": "questions",
                         "view_image": "path", "get_goal": "id", "create_goal": "objective", "update_goal": "status"]
            let claude = ["Agent", "Bash", "Edit", "ListAgents", "Read", "ReportFindings", "ScheduleWakeup", "Skill",
                          "ToolSearch", "Workflow", "DeferredToolPlaceholder", "Write"]
            let specs: [ServeToolSpec]
            switch self {
            case .responses:
                var list: [[String: Any]] = codex.sorted { $0.key < $1.key }.map { functionTool($0.key, param: $0.value) }
                list.append(["type": "namespace", "name": "multi_agent_v1", "tools": [Any]()])
                let body: [String: Any] = ["model": "m", "input": "x", "tools": list, "tool_choice": "auto"]
                specs = try OpenAIResponsesDialect.decode(JSONSerialization.data(withJSONObject: body)).request.tools
            case .chatCompletions:
                let list: [[String: Any]] = codex.sorted { $0.key < $1.key }.map {
                    ["type": "function", "function": ["name": $0.key, "description": $0.key,
                        "parameters": ["type": "object", "properties": [$0.value: ["type": "string"]]]]]
                }
                let body: [String: Any] = ["model": "m", "messages": [["role": "user", "content": "x"]], "tools": list]
                specs = try OpenAIChatDialect.decode(JSONSerialization.data(withJSONObject: body)).tools
            case .messages:
                let list: [[String: Any]] = claude.map {
                    ["name": $0, "description": $0,
                     "input_schema": ["type": "object", "properties": ["command": ["type": "string"]]]]
                }
                let body: [String: Any] = ["model": "m", "max_tokens": 64, "messages": [["role": "user", "content": "x"]],
                                           "tools": list]
                specs = try AnthropicMessagesDialect.decode(JSONSerialization.data(withJSONObject: body)).tools
            }
            return specs.compactMap(\.object)
        }
    }

    /// A declared tool of this dialect's list, with the parameter its schema names.
    var declared: (name: String, key: String) {
        self == .messages ? ("Bash", "command") : ("exec_command", "cmd")
    }
}

struct ServeQwen35ReplayTests {

    /// Everything the scanner lets through plus the calls it found, fed in small deltas the way a stream does.
    private func scan(_ format: ToolCallFormat, _ text: String, tools: [[String: any Sendable]])
        -> (text: String, calls: [ToolCall]) {
        var scanner = ServeToolCallScanner(format: format, tools: tools)
        var out = ""
        var index = text.startIndex
        while index < text.endIndex {
            let next = text.index(index, offsetBy: 7, limitedBy: text.endIndex) ?? text.endIndex
            out += scanner.push(String(text[index ..< next]))
            index = next
        }
        out += scanner.finish()
        return (out, scanner.parsedCalls)
    }

    /// The replay: SV-8's literal `apply_patch` text, each dialect's own tools list, both formats.
    @Test(arguments: Dialect.allCases)
    fileprivate func theApplyPatchTurnParsesUnderXmlFunctionAndNotUnderQwen35(dialect: Dialect) throws {
        let tools = try dialect.tools
        #expect(!tools.isEmpty)

        let strict = scan(.qwen35, sv8ApplyPatchText, tools: tools)
        #expect(strict.calls.isEmpty, "\(dialect)")
        #expect(strict.text == sv8ApplyPatchText, "undeclared call must come back verbatim, \(dialect)")

        let lenient = scan(.xmlFunction, sv8ApplyPatchText, tools: tools)
        #expect(lenient.calls.count == 1, "\(dialect)")
        #expect(lenient.calls.first?.function.name == "apply_patch")
        #expect(lenient.text.hasPrefix("The `mul` function"), "the sentence before the call still streams")
        #expect(!lenient.text.contains("<tool_call>"))
    }

    /// The fix itself: the table resolves `qwen3_5` to `.xmlFunction` and leaves the other families to upstream.
    @Test
    func theTableOverridesQwen35ToXmlFunctionOnly() {
        #expect(ToolCallFormatTable.override(forModelType: "qwen3_5", configData: nil) == .xmlFunction)
        #expect(ToolCallFormatTable.resolved(forModelType: "qwen3_5", configData: nil) == .xmlFunction)
        for other in ["qwen3", "qwen2", "llama", "mistral3", "gemma2", "gemma3_text", "gemma4", "gpt_oss"] {
            #expect(ToolCallFormatTable.override(forModelType: other, configData: nil) == nil, "\(other)")
        }
    }

    /// What the override must not lose: a call to a declared tool parses the same under both formats.
    @Test(arguments: Dialect.allCases)
    fileprivate func aDeclaredToolCallParsesUnderBothFormats(dialect: Dialect) throws {
        let tools = try dialect.tools
        let (name, key) = dialect.declared
        let text = "Running it.\n" + xmlCall(name, key, "pytest -q")
        for format in [ToolCallFormat.qwen35, .xmlFunction] {
            let r = scan(format, text, tools: tools)
            #expect(r.calls.count == 1, "\(format) \(dialect)")
            #expect(r.calls.first?.function.name == name)
            #expect(r.calls.first?.function.arguments[key] == .string("pytest -q"))
            #expect(r.text == "Running it.\n")
        }
    }

    /// What the override gives up, stated so the trade is on the record: `.xmlFunction` does not read the
    /// Hermes-JSON dialect that `.qwen35` accepts — it is returned as text, never dropped or misparsed.
    @Test
    fileprivate func xmlFunctionDoesNotReadTheHermesJSONDialect() throws {
        let tools = try Dialect.responses.tools
        #expect(scan(.qwen35, hermesJSONCall, tools: tools).calls.count == 1)
        let lenient = scan(.xmlFunction, hermesJSONCall, tools: tools)
        #expect(lenient.calls.isEmpty)
        #expect(lenient.text == hermesJSONCall)
    }
}
