//
//  ToolRegistry.swift
//  MLXUI — agentic chat tools, slice AG0.
//
//  An immutable, `Sendable` collection of `AgentTool`s plus the dispatch outcome type and a
//  per-turn call budget. The registry produces `[ToolSpec]` for the model and resolves a
//  `ToolCall` back to its tool; the actual dispatch logic (approval, budget, error capture)
//  lives in `AgentSession.dispatch` so it is unit-testable without a model.
//

import Foundation
import MLXLMCommon

/// Immutable set of tools available to an agent turn.
nonisolated struct ToolRegistry: Sendable {
    private let tools: [String: any AgentTool]

    init(_ tools: [any AgentTool] = []) {
        var map: [String: any AgentTool] = [:]
        for tool in tools { map[tool.name] = tool }
        self.tools = map
    }

    var isEmpty: Bool { tools.isEmpty }

    /// Function schemas handed to the model (nil-able caller: pass `nil` when empty).
    var toolSpecs: [ToolSpec] { Array(tools.values.map(\.toolSpec)) }

    func tool(named name: String) -> (any AgentTool)? { tools[name] }
}

/// Outcome of dispatching a single `ToolCall`. `modelResult` is the string fed back to the
/// model; the case also drives the UI/audit log.
nonisolated enum ToolDispatchResult: Sendable, Equatable {
    case ok(String)
    case denied
    case unknownTool(String)
    case budgetExceeded
    case failed(String)

    var modelResult: String {
        switch self {
        case .ok(let s): return s
        case .denied: return "The user declined to run this tool."
        case .unknownTool(let name): return "Error: no tool named \"\(name)\"."
        case .budgetExceeded: return "Error: tool-call budget exceeded for this turn."
        case .failed(let message): return "Error: \(message)"
        }
    }
}

/// Observable tool-call lifecycle events (for the chat UI + audit log in AG1).
nonisolated enum AgentToolEvent: Sendable {
    case started(name: String, arguments: [String: JSONValue])
    case finished(name: String, result: String)
    case failed(name: String, message: String)
    case denied(name: String)
    case unknownTool(name: String)
    case budgetExceeded(name: String)
}

/// Per-turn tool-call budget: caps how many tools a single response may invoke so an agent
/// loop can't run away.
actor ToolBudget {
    private var remaining: Int
    init(limit: Int) { self.remaining = max(0, limit) }
    func tryConsume() -> Bool {
        guard remaining > 0 else { return false }
        remaining -= 1
        return true
    }
}
