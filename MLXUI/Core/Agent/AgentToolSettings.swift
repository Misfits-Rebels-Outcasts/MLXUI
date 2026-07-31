//
//  AgentToolSettings.swift
//  MLXUI — agentic chat tools, slice AG6b (safety rails, part 1).
//
//  Persisted per-tool approval policy. A tool that `requiresApproval` normally prompts the user
//  each time (AG1's approval sheet); this lets the user set a standing policy per tool:
//    • .ask    — prompt every time (default)
//    • .always — auto-approve without prompting
//    • .never  — auto-deny (the model is told it may not run the tool)
//
//  The gate decision is a pure function; persistence is a thin UserDefaults wrapper. Both are
//  `nonisolated` so they're testable synchronously and readable off the MainActor.
//

import Foundation

/// Standing user policy for a tool that requires approval.
nonisolated enum ToolApprovalPolicy: String, Codable, Sendable, CaseIterable {
    case ask, always, never
}

/// What the approval gate should do for a given tool call.
nonisolated enum ApprovalDecision: Equatable, Sendable {
    case prompt      // show the approval sheet
    case autoApprove // policy == .always
    case autoDeny    // policy == .never
}

/// User-configured agent-tool policy, persisted across launches.
nonisolated struct AgentToolSettings: Codable, Sendable, Equatable {
    /// Per-tool approval overrides; a tool absent here uses the default (`.ask`).
    private(set) var policies: [String: ToolApprovalPolicy]

    init(policies: [String: ToolApprovalPolicy] = [:]) {
        self.policies = policies
    }

    /// The standing policy for a tool (default `.ask`).
    func policy(for toolName: String) -> ToolApprovalPolicy {
        policies[toolName] ?? .ask
    }

    /// The gate decision for a tool that requires approval.
    func decision(for toolName: String) -> ApprovalDecision {
        switch policy(for: toolName) {
        case .ask:    return .prompt
        case .always: return .autoApprove
        case .never:  return .autoDeny
        }
    }

    /// Returns a copy with `toolName`'s policy set. `.ask` (the default) is stored as an absence
    /// so the persisted dictionary stays minimal.
    func setting(_ policy: ToolApprovalPolicy, for toolName: String) -> AgentToolSettings {
        var next = policies
        if policy == .ask { next.removeValue(forKey: toolName) }
        else { next[toolName] = policy }
        return AgentToolSettings(policies: next)
    }

    // MARK: Persistence

    private static let defaultsKey = "agentToolSettings"

    static func load(from defaults: UserDefaults = .standard) -> AgentToolSettings {
        guard let data = defaults.data(forKey: defaultsKey),
              let decoded = try? JSONDecoder().decode(AgentToolSettings.self, from: data)
        else { return AgentToolSettings() }
        return decoded
    }

    func save(to defaults: UserDefaults = .standard) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        defaults.set(data, forKey: Self.defaultsKey)
    }
}
