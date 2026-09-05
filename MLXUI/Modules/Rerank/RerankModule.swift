import Foundation

/// Bundles the Qwen3-Reranker SDK + UI and registers them. Backed by `ml-explore/mlx-swift-lm`'s
/// `MLXLLM`/`MLXLMCommon` (the same `LLMModelFactory`/`qwen3` registration the Chat module
/// uses) — the reranker is a plain `Qwen3ForCausalLM` checkpoint, scored by reading yes/no
/// logits rather than sampling text. Feeds the list-shaped `RerankStage` (`AssetStage`,
/// MoC-3-3) through `RealExecutor.runModel`'s `engines.rerank.` branch, never through the
/// generic `SingleMediaStage` adapter.
enum RerankModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "rerank",
        displayName: "Rerank (MLX)",
        modalities: ["text"],
        modelTypes: [.rerank],
        backingPackage: "ml-explore/mlx-swift-lm",
        packageLicense: "MIT",
        maintainers: ["Apple MLX"],
        notes: """
        Causal-LM yes/no relevance scoring via mlx-swift-lm's LLMModelFactory (qwen3), loading \
        the installed mlx-community safetensors. One forward pass per candidate, never \
        batched. Headless — Rerank is a flow-only task with no chat-style Run surface \
        (RerankUI explains this rather than presenting one). Claims source==.mlx rerank \
        entries (Qwen3-Reranker).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: RerankSDK(), ui: RerankUI(), descriptor: descriptor)
    }
}
