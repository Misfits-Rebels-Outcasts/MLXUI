import Foundation

    /// Builds a `RealExecutor` for a flow from the app's `AppState` — the one place the view
    /// layer wires the `ModelRegistry` into FlowKit (FlowKit itself never imports a model
    /// module). CFM-R11-2: the `makeModelStage` closure routes through the shared
    /// `EngineCache`, so a model's engine stays warm across rows (and runs) within the RAM
    /// budget instead of reloading per row.
    nonisolated enum AppFlowExecutorFactory {

    /// Build the real executor for `flowID`, resolving model stages through `appState`.
    static func make(
        flowID: String,
        appState: AppState,
        transforms: [String: TransformDef] = [:]
    ) -> RealExecutor {
        let workspace = FlowWorkspace(root: ModelStore.shared.flowsDirectory)
        let flowDir = workspace.directory(for: flowID)
        let blobDir = flowDir.appendingPathComponent(".blobs", isDirectory: true)
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        let installed = appState.installedModelIDs

        let cache = EngineCache.shared
        let makeStage: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            try await cache.stage(for: model, config: config) { model, config in
                try await MainActor.run {
                    guard let resolved = appState.registry.bestModule(for: model) else {
                        throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind)
                    }
                    return try resolved.sdk.makeStage(for: model, config: config)
                }
            }
        }

        var executor = RealExecutor(workspace: workspace, flowID: flowID, blobDirectory: blobDir,
                                    makeModelStage: makeStage, installedModelIDs: installed,
                                    catalog: catalog)
        executor.transforms = transforms
        return executor
    }

    /// The app-wide engine cache lives at `EngineCache.shared` (R11-2): warm engines across
    /// rows and flows, bounded by `max(4 GB, total RAM × 0.6)`; the registry builder is
    /// captured per-executor in `make`.

    /// The `RunContext` for a run, using the real executor.
    static func context(flowID: String, appState: AppState) -> FlowRunner.RunContext {
        let workspace = FlowWorkspace(root: ModelStore.shared.flowsDirectory)
        let flowDir = workspace.directory(for: flowID)
        let blobDir = flowDir.appendingPathComponent(".blobs", isDirectory: true)
        return FlowRunner.RunContext(flowID: flowID, workspace: workspace,
                                     blobDirectory: blobDir,
                                     executor: make(flowID: flowID, appState: appState))
    }

    /// The `RunContext` with the real executor wrapped in the content-addressed cache
    /// (CFM-R3-2/4): a deterministic run seed is derived from the flow's raw text, so cached
    /// keys are stable across runs and `_with_seed` derives the same concrete seeds.
    static func cachingContext(flowID: String, appState: AppState,
                               transforms: [String: TransformDef] = [:]) -> FlowRunner.RunContext {
        let workspace = FlowWorkspace(root: ModelStore.shared.flowsDirectory)
        let flowDir = workspace.directory(for: flowID)
        let blobDir = flowDir.appendingPathComponent(".blobs", isDirectory: true)
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        let runSeed = (try? GalleryLoader.rawCatText(flowID: flowID)).map { FlowSeed.runSeed(for: $0) } ?? 0
        let inner = make(flowID: flowID, appState: appState, transforms: transforms)
        let caching = CachingExecutor(inner: inner, store: FlowCacheStore.shared,
                                      cacheTier: "real", catalog: catalog, runSeed: runSeed,
                                      workspace: workspace, flowID: flowID)
        return FlowRunner.RunContext(flowID: flowID, workspace: workspace,
                                     blobDirectory: blobDir, executor: caching)
    }
}
