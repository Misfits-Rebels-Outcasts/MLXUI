import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG0 — the agent core: `AgentTool` schema building and `AgentSession.dispatch`
/// (approval / budget / unknown-tool / error capture). Model-free: `dispatch` is exercised
/// directly with synthesized `ToolCall`s, no weights required.
struct AgentToolTests {

    // MARK: fixtures

    /// Echoes its `text` argument.
    nonisolated struct EchoTool: AgentTool {
        let name = "echo"
        let toolDescription = "Echo the text back."
        let parameters: [ToolParameter] = [
            .required("text", type: .string, description: "text to echo")
        ]
        func execute(arguments: [String: JSONValue]) async throws -> String {
            "echo: \(arguments.string("text") ?? "")"
        }
    }

    /// Requires approval; returns a sentinel if it ever runs.
    nonisolated struct GatedTool: AgentTool {
        let name = "gated"
        let toolDescription = "A side-effecting tool."
        let parameters: [ToolParameter] = []
        var requiresApproval: Bool { true }
        func execute(arguments: [String: JSONValue]) async throws -> String { "ran" }
    }

    struct StubError: Error {}
    /// Always throws.
    nonisolated struct FailingTool: AgentTool {
        let name = "boom"
        let toolDescription = "Always fails."
        let parameters: [ToolParameter] = []
        func execute(arguments: [String: JSONValue]) async throws -> String { throw StubError() }
    }

    private func call(_ name: String, _ args: [String: JSONValue] = [:]) -> ToolCall {
        ToolCall(function: .init(name: name, arguments: args))
    }

    // MARK: schema

    @Test func toolSpecHasFunctionSchema() {
        let spec = EchoTool().toolSpec
        #expect(spec["type"] as? String == "function")
        let fn = spec["function"] as? [String: any Sendable]
        #expect(fn?["name"] as? String == "echo")
        let params = fn?["parameters"] as? [String: any Sendable]
        #expect(params?["required"] as? [String] == ["text"])
        let properties = params?["properties"] as? [String: any Sendable]
        #expect(properties?["text"] != nil)
    }

    // MARK: dispatch

    @Test func dispatchRunsTool() async {
        let result = await AgentSession.dispatch(
            call("echo", ["text": .string("hi")]),
            registry: ToolRegistry([EchoTool()]), budget: ToolBudget(limit: 4),
            approve: { _, _ in true }, onEvent: { _ in })
        #expect(result == .ok("echo: hi"))
    }

    @Test func dispatchUnknownTool() async {
        let result = await AgentSession.dispatch(
            call("nope"),
            registry: ToolRegistry([EchoTool()]), budget: ToolBudget(limit: 4),
            approve: { _, _ in true }, onEvent: { _ in })
        #expect(result == .unknownTool("nope"))
    }

    @Test func dispatchRespectsBudget() async {
        let registry = ToolRegistry([EchoTool()])
        let budget = ToolBudget(limit: 1)
        let first = await AgentSession.dispatch(
            call("echo", ["text": .string("a")]), registry: registry, budget: budget,
            approve: { _, _ in true }, onEvent: { _ in })
        let second = await AgentSession.dispatch(
            call("echo", ["text": .string("b")]), registry: registry, budget: budget,
            approve: { _, _ in true }, onEvent: { _ in })
        #expect(first == .ok("echo: a"))
        #expect(second == .budgetExceeded)
    }

    @Test func dispatchDeniedWhenApprovalRefused() async {
        let result = await AgentSession.dispatch(
            call("gated"),
            registry: ToolRegistry([GatedTool()]), budget: ToolBudget(limit: 4),
            approve: { _, _ in false }, onEvent: { _ in })
        #expect(result == .denied)   // execute never ran (would have returned .ok("ran"))
    }

    @Test func dispatchCapturesToolError() async {
        let result = await AgentSession.dispatch(
            call("boom"),
            registry: ToolRegistry([FailingTool()]), budget: ToolBudget(limit: 4),
            approve: { _, _ in true }, onEvent: { _ in })
        if case .failed = result { } else { Issue.record("expected .failed, got \(result)") }
    }

    // MARK: tool-call format mapping

    @Test func qwenUsesXMLFunctionFormat() {
        #expect(AgentSession.toolCallFormat(forModelType: "qwen3_5") == .xmlFunction)
        #expect(AgentSession.toolCallFormat(forModelType: "qwen2") == .xmlFunction)
        #expect(AgentSession.toolCallFormat(forModelType: "llama") == nil)
    }
}
