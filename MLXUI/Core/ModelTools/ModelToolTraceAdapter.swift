//
//  ModelToolTraceAdapter.swift
//  MLXUI
//
//  MT5-2: the live chat → a `ModelToolTrace`. The transcript gives the order, the router's model and the arguments a tool
//  was called with; the chat's `ChatAssetStore` gives each asset's provenance (row text, model), its name and file, and
//  whether the file is still there. Joined here, once, so `ModelToolFlowBuilder` stays a pure function of plain values.
//  `@MainActor` only because `ChatMessage` / `ToolActivity` / `TranscriptItem` are.
//

import Foundation
import MLXLMCommon

@MainActor
enum ModelToolTraceAdapter {

    static func trace(items: [TranscriptItem], store: ChatAssetStore) -> ModelToolTrace {
        var events: [ModelToolTrace.Event] = []
        for item in items {
            switch item {
            case .message(let message):
                switch message.role {
                case .user:
                    let handles = message.attachments.compactMap { $0.assetHandle }
                    events.append(ModelToolTrace.Event(kind: .user, text: message.text, attachments: handles.isEmpty ? nil : handles))
                case .assistant:
                    events.append(ModelToolTrace.Event(kind: .assistant, text: message.text, modelID: message.modelID))
                }
            case .tool(let activity):
                events.append(toolEvent(activity, store: store))
            }
        }
        let assets = store.allRecords.map { record in
            ModelToolTrace.Asset(
                handle: record.handle, kind: record.kind, name: record.name, fileName: record.fileName,
                text: record.kind == "text" ? store.text(for: record.handle) : nil,
                rowText: record.provenance.rowText, model: record.provenance.model,
                missing: store.isMissing(record) ? true : nil)
        }
        return ModelToolTrace(events: events, assets: assets)
    }

    private static func toolEvent(_ activity: ToolActivity, store: ChatAssetStore) -> ModelToolTrace.Event {
        let finished = activity.status == .finished
        let result = activity.result ?? ""
        let refused = result.hasPrefix("Error:") || result.hasPrefix("The tool did not run")
        let output = finished && !refused ? producedHandle(in: result) : nil
        var assetArgument = activity.arguments.string("asset")
        if let given = assetArgument, let record = store.resolve(given) { assetArgument = record.handle }
        let isModelTool = ModelToolSpec.spec(named: activity.name)?.task != nil
        return ModelToolTrace.Event(
            kind: .tool, text: activity.arguments.string("text"), tool: activity.name, assetArgument: assetArgument,
            output: output, ok: finished && !refused && (!isModelTool || output != nil))
    }

    /// The handle in a model tool's result JSON (`{"asset":"a2",…}`).
    static func producedHandle(in result: String) -> String? {
        guard let data = result.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return object["asset"] as? String
    }

    // MARK: Catalog

    /// What the builder needs from the live catalog: the chat models' row displays, and the HF id behind every display a
    /// `models:` pin may need (the rule `FlowEditorModel.reconcileModelsBlock` applies).
    static func catalog(models: [ModelEntry]) -> ModelToolFlowCatalog {
        var routers: [String: String] = [:]
        var hf: [String: String] = [:]
        for entry in models {
            let display = TaskModels.displayName(for: entry)
            if entry.runnerKind == .llm { routers[entry.id] = display }
            if hf[display] == nil { hf[display] = entry.hfModelId }
            if hf[entry.displayName] == nil { hf[entry.displayName] = entry.hfModelId }
        }
        return ModelToolFlowCatalog(routerDisplays: routers, hfModelIDs: hf)
    }

    /// Build the flow for the chat on screen.
    static func flow(items: [TranscriptItem], store: ChatAssetStore, models: [ModelEntry]) -> ModelToolFlow {
        ModelToolFlowBuilder.build(trace(items: items, store: store), catalog: catalog(models: models))
    }
}
