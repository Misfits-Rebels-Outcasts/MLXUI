//
//  ModelToolHost.swift
//  MLXUI — Model Tools, MT1-3.
//
//  Per-conversation context for model tools, reaching them by `@TaskLocal` exactly as `FileScope`
//  does (`AgentTool` keeps its shape). `ModelToolHost.current == nil` — flows, the Local Server,
//  tests — means the tools behave as they did before Model Tools: plain-text results, no assets.
//

import Foundation

/// What a download would need, handed to `ModelToolHost.installRequest`.
nonisolated struct ModelInstallNeed: Sendable, Equatable {
    let toolName: String
    let modelDisplayName: String
    let modelID: String
    let downloadGB: Double
    /// MT4-3 ("Choose another..."): the other fitting, not-installed candidates for the same tool, in preference order.
    var alternatives: [ModelInstallOption] = []
}

/// One model the install card can offer (the need's own, or an alternative).
nonisolated struct ModelInstallOption: Sendable, Equatable, Identifiable {
    /// The catalog id (`ModelEntry.id`).
    let id: String
    let displayName: String
    let downloadGB: Double
}

/// MT4-3: what one turn remembers about the installs it was refused (plan §3.8: "the policy then drops that tool for the
/// rest of the turn, so the router can't loop on it"). A host is built per turn, so this lives exactly one turn. The tool
/// set a session holds is fixed when the turn starts, so "dropped" is enforced where the call lands: a later call to a
/// declined tool answers with the same sentence and asks nothing.
nonisolated final class ModelToolTurnState: @unchecked Sendable {
    private let lock = NSLock()
    private var declined: [String: String] = [:]

    init() {}

    func markDeclined(tool: String, sentence: String) {
        lock.lock(); defer { lock.unlock() }
        declined[tool] = sentence
    }

    func declinedSentence(for tool: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        return declined[tool]
    }
}

/// A progress line for the chip (`Transcribing…`). The runner reports the phases it owns; from
/// MT1-3b the executor can add a `detail` line and a `fraction` through `RowProgress.report`
/// (nothing emits yet).
nonisolated struct ModelToolProgress: Sendable, Equatable {
    enum Phase: String, Sendable { case preparing, running, saving }
    let toolName: String
    let phase: Phase
    /// A line reported from inside the row (`RowProgress.message`); nil for a runner phase.
    var detail: String? = nil
    /// 0…1 when the row knows it.
    var fraction: Double? = nil
}

nonisolated struct ModelToolHost: Sendable {
    let assets: ChatAssetStore
    /// Asked when the chosen model must be downloaded first. `true` = installed, go on. MT1 binds
    /// `declineAllInstalls`; MT4-3 binds the install card (`ModelInstallCoordinator`).
    let installRequest: @Sendable (ModelInstallNeed) async -> Bool
    let progress: @Sendable (ModelToolProgress) -> Void
    /// MT4-3: the runtime as it is now, asked once after an install so the same call re-resolves (and re-runs preflight) against
    /// the new state. `nil` = the runtime the call started with is used (tests, MT1 behaviour).
    let refreshRuntime: (@Sendable () async -> ModelToolRuntime?)?
    /// MT4-3: this turn's declined installs.
    let turn: ModelToolTurnState
    /// The inline budget for a text result, in estimated tokens (plan §3.7 step 6). A parameter,
    /// not a constant: Apple Intelligence 1,000, MLX the CW D6 budget.
    let inlineBudgetTokens: Int

    static let declineAllInstalls: @Sendable (ModelInstallNeed) async -> Bool = { _ in false }
    static let ignoreProgress: @Sendable (ModelToolProgress) -> Void = { _ in }

    init(assets: ChatAssetStore,
         installRequest: @escaping @Sendable (ModelInstallNeed) async -> Bool = ModelToolHost.declineAllInstalls,
         progress: @escaping @Sendable (ModelToolProgress) -> Void = ModelToolHost.ignoreProgress,
         refreshRuntime: (@Sendable () async -> ModelToolRuntime?)? = nil,
         turn: ModelToolTurnState = ModelToolTurnState(),
         inlineBudgetTokens: Int = 1000) {
        self.assets = assets
        self.installRequest = installRequest
        self.progress = progress
        self.refreshRuntime = refreshRuntime
        self.turn = turn
        self.inlineBudgetTokens = inlineBudgetTokens
    }

    @TaskLocal static var current: ModelToolHost?

    /// The sentence a declined download returns to the router (plan §3.8).
    static func declineSentence(for model: String) -> String {
        "The user chose not to download \(model). Answer without it, or say what it would need."
    }
}
