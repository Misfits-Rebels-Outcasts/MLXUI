//
//  AgentToolSettings.swift
//  MLXUI — agentic chat tools, slice AG6b (safety rails, parts 1 + 2).
//
//  Persisted per-tool approval policy. A tool that `requiresApproval` normally prompts the user
//  each time (AG1's approval sheet); this lets the user set a standing policy per tool:
//    • .ask    — prompt every time (default)
//    • .always — auto-approve without prompting
//    • .never  — auto-deny (the model is told it may not run the tool)
//
//  AG6b-2 adds two more persisted knobs:
//    • enabledOverrides — which tools the user has toggled on/off, stored as *deviations* from
//      the build default so a new build's defaults (and its new tools) keep applying.
//    • toolCallLimit    — the per-reply `ToolBudget` cap handed to `AgentSession`.
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
    /// Per-tool enabled overrides; a tool absent here uses its build default (everything on
    /// except the edition's `disabledByDefault` set).
    private(set) var enabledOverrides: [String: Bool]
    /// Per-reply tool-call budget (the `ToolBudget` limit), clamped to `toolCallLimitRange`.
    private(set) var toolCallLimit: Int

    static let defaultToolCallLimit = 8
    static let toolCallLimitRange = 1...32

    init(
        policies: [String: ToolApprovalPolicy] = [:],
        enabledOverrides: [String: Bool] = [:],
        toolCallLimit: Int = AgentToolSettings.defaultToolCallLimit
    ) {
        self.policies = policies
        self.enabledOverrides = enabledOverrides
        self.toolCallLimit = Self.clampedLimit(toolCallLimit)
    }

    /// Back-compat decoding: blobs persisted by AG6b-1 carry only `policies`.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            policies: try container.decodeIfPresent(
                [String: ToolApprovalPolicy].self, forKey: .policies) ?? [:],
            enabledOverrides: try container.decodeIfPresent(
                [String: Bool].self, forKey: .enabledOverrides) ?? [:],
            toolCallLimit: try container.decodeIfPresent(
                Int.self, forKey: .toolCallLimit) ?? Self.defaultToolCallLimit)
    }

    private enum CodingKeys: String, CodingKey {
        case policies, enabledOverrides, toolCallLimit
    }

    static func clampedLimit(_ limit: Int) -> Int {
        min(max(limit, toolCallLimitRange.lowerBound), toolCallLimitRange.upperBound)
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
        return AgentToolSettings(
            policies: next, enabledOverrides: enabledOverrides, toolCallLimit: toolCallLimit)
    }

    // MARK: Enabled tools (AG6b-2)

    /// The enabled set for this build: `defaults` with the user's persisted toggles applied.
    func enabledToolNames(defaults: Set<String>) -> Set<String> {
        var names = defaults
        for (name, on) in enabledOverrides {
            if on { names.insert(name) } else { names.remove(name) }
        }
        return names
    }

    /// Returns a copy with `toolName` toggled. A value matching the build default is stored as
    /// an absence, so future builds' defaults (and brand-new tools) keep applying.
    func settingEnabled(
        _ enabled: Bool, for toolName: String, defaultEnabled: Bool
    ) -> AgentToolSettings {
        var next = enabledOverrides
        if enabled == defaultEnabled { next.removeValue(forKey: toolName) }
        else { next[toolName] = enabled }
        return AgentToolSettings(
            policies: policies, enabledOverrides: next, toolCallLimit: toolCallLimit)
    }

    // MARK: Tool-call limit (AG6b-2)

    /// Returns a copy with the per-reply tool-call budget set (clamped by the initializer).
    func settingToolCallLimit(_ limit: Int) -> AgentToolSettings {
        AgentToolSettings(
            policies: policies, enabledOverrides: enabledOverrides, toolCallLimit: limit)
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
