import SwiftUI

/// Bundles the LLM SDK + UI and registers them. Backed by `ml-explore/mlx-swift-lm`
/// (MIT). Runs the **installed** LLM safetensors (the files `InstallManager` downloads)
/// via `LLMStage`/`LLMEngine` — the same engine the chat sheet uses. Before this module
/// existed, LLM models ran through `AppState.runningModel` → the `RunChatView` sheet and
/// had no registry entry; this makes `.llm` models answer to `ModelRegistry.bestModule`.
enum ChatModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "chat",
        displayName: "Chat (LLM)",
        modalities: ["text"],
        modelTypes: [.llm],
        backingPackage: "ml-explore/mlx-swift-lm",
        packageLicense: "MIT",
        maintainers: ["Apple MLX"],
        notes: """
        Text → text via mlx-swift-lm's LLMModelFactory, loading the installed \
        mlx-community LLM safetensors. Claims every .llm entry (only LLM SDK, \
        no tie to lose). The Run UI is the existing chat sheet (RunChatView).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: ChatSDK(), ui: ChatUI(), descriptor: descriptor)
    }
}

/// `ModelUI` for LLM: presents the existing `RunChatView` sheet. `claim` mirrors
/// `ChatSDK` so UI resolution matches SDK resolution.
struct ChatUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .llm else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        // The chat sheet drives `ModelRunner` itself (streaming transcript, tool calls),
        // so the pre-built `stage` is intentionally unused — same shape as `KokoroUI`.
        AnyView(RunChatView(model: model))
    }
}
