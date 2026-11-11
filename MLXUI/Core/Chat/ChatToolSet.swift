//
//  ChatToolSet.swift
//  MLXUI — Model Tools, MT4-2: the tools one Chat turn hands its router.
//
//  `ToolExposurePolicy` is pure and works on names; this is the thin layer that feeds it the facts of a turn
//  (what Model Tools can do on this Mac, which assets the chat holds, what the user typed) and turns the chosen
//  names back into `AgentTool`s. Ruling D: Model Tools are on in the Chat page and off in the bench sheet; off
//  means this returns today's set, with only the tool-picking policy applied.
//

import Foundation

nonisolated enum ChatToolSet {

    struct Request: Sendable {
        var router: ToolExposurePolicy.Router
        /// Ruling D: on in the Chat page, off in the bench sheet.
        var modelToolsEnabled: Bool
        /// The tool-picking policy: on for Apple Intelligence, off for MLX (the MT4-6 eval decides MLX).
        var gateAmbiguousTools: Bool
        /// What the user typed this turn.
        var prompt: String
        /// What Model Tools can do on this Mac (statuses). `nil` = nothing known: every model tool reads as unavailable.
        var runtime: ModelToolRuntime?
        /// The conversation's assets (this turn's attachments are already registered in it).
        var assets: ChatAssetStore?
        /// This turn's attachments.
        var turnAttachments: [ChatAttachment] = []
        /// The inline budget the router reads a text result in (Apple Intelligence 1,000 tokens).
        var inlineBudgetTokens: Int
        /// The chat has folders attached: the file tools lead the user's own tools, so a 10-tool MLX budget never cuts them.
        var foldersAttached = false
    }

    struct Selection {
        let tools: [any AgentTool]
        let output: ToolExposurePolicy.Output
    }

    /// The tools for one turn. `enabled` is the user's enabled tools in registry order.
    static func select(enabled: [any AgentTool], _ request: Request) -> Selection {
        var statuses: [String: ModelToolStatus] = [:]
        var turnKinds = Set<Kind>()
        var conversationKinds = Set<Kind>()
        var oversized = false

        if request.modelToolsEnabled {
            if let runtime = request.runtime {
                for spec in ModelToolSpec.modelTools {
                    statuses[spec.name] = CapabilityResolver.status(
                        for: spec, catalog: runtime.catalog, installedModelIDs: runtime.installedModelIDs,
                        claimableModelIDs: runtime.claimableModelIDs, totalRAMGB: runtime.totalRAMGB)
                }
            }
            for attachment in request.turnAttachments {
                if let raw = attachment.assetKind, let kind = Kind(rawValue: raw) { turnKinds.insert(kind) }
            }
            for record in request.assets?.all ?? [] {
                guard let kind = Kind(rawValue: record.kind) else { continue }
                conversationKinds.insert(kind)
                if kind == .text, ChatAttachment.estimatedTokens(forUTF8Count: record.bytes) > request.inlineBudgetTokens {
                    oversized = true
                }
            }
        }

        var names = enabled.map(\.name)
        if request.modelToolsEnabled, request.foldersAttached {
            let files = ChatEnvironment.fileToolNames.union(ChatEnvironment.writeToolNames).union([ChatEnvironment.shellToolName])
            names = names.filter { files.contains($0) } + names.filter { !files.contains($0) }
        }
        let output = ToolExposurePolicy.select(ToolExposurePolicy.Input(
            router: request.router, modelToolsEnabled: request.modelToolsEnabled,
            conversationAssetKinds: conversationKinds, turnAssetKinds: turnKinds, hasOversizedTextAsset: oversized,
            prompt: request.prompt, statuses: statuses, enabledTools: names,
            gateAmbiguousTools: request.gateAmbiguousTools))

        let byName = Dictionary(enabled.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        let tools: [any AgentTool] = output.tools.compactMap { name in
            // A model tool is built for a host-bound chat (it advertises the `asset` argument); everything else is the user's tool.
            if request.modelToolsEnabled, let spec = ModelToolSpec.spec(named: name) {
                return ModelTool(spec: spec, advertisesAssets: true)
            }
            return byName[name]
        }
        return Selection(tools: tools, output: output)
    }
}
