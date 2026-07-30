//
//  AgentTool.swift
//  MLXUI — agentic chat tools, slice AG0.
//
//  A tool the chat agent can call during a `ChatSession` tool loop. Implementations are
//  stateless + `Sendable`; side-effecting tools set `requiresApproval` so the UI can gate
//  them before `execute` runs (AG1). The `toolSpec` is the OpenAI-style function schema
//  handed to the model — it mirrors the shape `MLXLMCommon.Tool` builds.
//
//  No tools ship in AG0 — only the protocol, schema builder, and argument helpers. Real
//  tools (web / files / MLX / compute) arrive in AG2–AG5.
//

import Foundation
import MLXLMCommon

/// A callable tool exposed to the chat model. Conformers are `nonisolated` (see `PipelineStage`).
protocol AgentTool: Sendable {
    /// Function name the model calls (must be unique in a `ToolRegistry`).
    var name: String { get }
    /// Human/model-facing description of what the tool does.
    var toolDescription: String { get }
    /// Typed parameters, used to build the JSON schema.
    var parameters: [ToolParameter] { get }
    /// When true, `AgentSession` awaits UI approval before executing (side-effecting tools).
    /// `nonisolated` so the nonisolated dispatcher can read it synchronously.
    nonisolated var requiresApproval: Bool { get }
    /// Max wall-clock (seconds) for one `execute` before the agent loop abandons the call and
    /// reports a timeout — the AG6 safety rail. `nil` means no cap (the default): most tools are
    /// bounded by their own I/O timeouts or by model generation, so only a tool that can spin
    /// unbounded (e.g. `run_javascript`, which has no in-tool JSC interrupt) opts into a cap.
    /// `nonisolated` so the nonisolated dispatcher can read it synchronously.
    nonisolated var executionTimeout: TimeInterval? { get }
    /// Run the tool. Return a string result to feed back to the model. Throwing is caught by
    /// the dispatcher and surfaced to the model as an error string (never crashes the run).
    func execute(arguments: [String: JSONValue]) async throws -> String
}

extension AgentTool {
    nonisolated var requiresApproval: Bool { false }
    nonisolated var executionTimeout: TimeInterval? { nil }

    /// OpenAI-style function schema (`{type:function, function:{name, description, parameters}}`)
    /// handed to the model as a `ToolSpec`.
    var toolSpec: ToolSpec {
        var properties: [String: any Sendable] = [:]
        var required: [String] = []
        for p in parameters {
            properties[p.name] = p.schema
            if p.isRequired { required.append(p.name) }
        }
        let parametersSchema: [String: any Sendable] = [
            "type": "object",
            "properties": properties,
            "required": required,
        ]
        let function: [String: any Sendable] = [
            "name": name,
            "description": toolDescription,
            "parameters": parametersSchema,
        ]
        return ["type": "function", "function": function]
    }
}

/// Convenience typed accessors for reading `ToolCall` arguments inside `execute`.
extension Dictionary where Key == String, Value == JSONValue {
    func string(_ key: String) -> String? {
        if case .string(let s) = self[key] { return s }
        return nil
    }
    func int(_ key: String) -> Int? {
        switch self[key] {
        case .int(let i): return i
        case .double(let d): return Int(d)
        default: return nil
        }
    }
    func double(_ key: String) -> Double? {
        switch self[key] {
        case .double(let d): return d
        case .int(let i): return Double(i)
        default: return nil
        }
    }
    func bool(_ key: String) -> Bool? {
        if case .bool(let b) = self[key] { return b }
        return nil
    }
}
