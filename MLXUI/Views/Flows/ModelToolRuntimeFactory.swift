import Foundation

/// MT1-3: the live `ModelToolRuntime` — the catalog and install state as they are *now* (rebuilt each turn, so a
/// finished download is seen), and a factory that builds a fresh flow executor through the one place the view layer
/// wires the registry into FlowKit (`AppFlowExecutorFactory`). The executor has no served-load gate (nothing in a chat tool
/// call can park) and writes under the conversation's assets folder.
extension ModelToolRuntime {
    static func live(appState: AppState) -> ModelToolRuntime {
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        return ModelToolRuntime(
            catalog: catalog,
            installedModelIDs: appState.flowReadyModelIDs,
            claimableModelIDs: appState.claimableModelIDs,
            totalRAMGB: appState.systemInfo.totalRAMGB,
            makeExecutor: { blobDirectory, conversationID in
                await MainActor.run {
                    let scope: FlowScope
                    if let conversationID {
                        scope = FlowScope(identity: "chat:\(conversationID.uuidString)",
                                          workspace: FlowWorkspace(root: ConversationStore.defaultDirectory()),
                                          locationID: conversationID.uuidString)
                    } else {
                        scope = FlowScope(identity: "chat:model-tools",
                                          workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
                                          locationID: "model-tools")
                    }
                    return AppFlowExecutorFactory.make(scope: scope, appState: appState,
                                                       blobDirectory: blobDirectory, servedLoadGate: false)
                }
            })
    }
}
