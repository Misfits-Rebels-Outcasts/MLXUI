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
    /// `declineAllInstalls`; MT4 binds the card.
    let installRequest: @Sendable (ModelInstallNeed) async -> Bool
    let progress: @Sendable (ModelToolProgress) -> Void
    /// The inline budget for a text result, in estimated tokens (plan §3.7 step 6). A parameter,
    /// not a constant: Apple Intelligence 1,000, MLX the CW D6 budget.
    let inlineBudgetTokens: Int

    static let declineAllInstalls: @Sendable (ModelInstallNeed) async -> Bool = { _ in false }
    static let ignoreProgress: @Sendable (ModelToolProgress) -> Void = { _ in }

    init(assets: ChatAssetStore,
         installRequest: @escaping @Sendable (ModelInstallNeed) async -> Bool = ModelToolHost.declineAllInstalls,
         progress: @escaping @Sendable (ModelToolProgress) -> Void = ModelToolHost.ignoreProgress,
         inlineBudgetTokens: Int = 1000) {
        self.assets = assets
        self.installRequest = installRequest
        self.progress = progress
        self.inlineBudgetTokens = inlineBudgetTokens
    }

    @TaskLocal static var current: ModelToolHost?

    /// The sentence a declined download returns to the router (plan §3.8).
    static func declineSentence(for model: String) -> String {
        "The user chose not to download \(model). Answer without it, or say what it would need."
    }
}
