import Foundation

/// Bundles the MLX embedding SDK + UI and registers them. Backed by `ml-explore/mlx-swift-lm`'s
/// `MLXEmbedders` (MIT). Runs the **installed** mlx-community embedding safetensors (the files
/// `InstallManager` downloads) — text → L2-normalized vector. A model whose architecture
/// `MLXEmbedders` doesn't support (e.g. some ModernBERT variants) surfaces a graceful engine
/// error, not a crash.
enum EmbeddingModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "embedding",
        displayName: "Embeddings (MLX)",
        modalities: ["text", "embedding"],
        modelTypes: [.embedding],
        backingPackage: "ml-explore/mlx-swift-lm",
        packageLicense: "MIT",
        maintainers: ["Apple MLX"],
        notes: """
        Text → embedding via mlx-swift-lm's MLXEmbedders (BERT/RoBERTa, NomicBERT, Gemma3, \
        Qwen3 families), loading the installed mlx-community safetensors. Vectors are \
        L2-normalized, so cosine similarity is a dot product. Claims source==.mlx embedding \
        entries (all-MiniLM, embeddinggemma, nomic, bge-m3).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: EmbeddingSDK(), ui: EmbeddingUI(), descriptor: descriptor)
    }
}
