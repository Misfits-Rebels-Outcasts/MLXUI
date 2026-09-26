import Foundation

/// Bundles the Laya SDK + UI and registers them. Backed by `ml-explore/mlx-swift` directly
/// (no `MLXLLM`/`MLXEmbedders` — Laya is a non-generative ModernBERT encoder + decision head,
/// ported from the Python `laya-mlx` 0.2.0 package, Apache-2.0, LY-4). Runs the **installed**
/// `aac6fef/laya-mlx` FP16 checkpoint — text + a typed question → a label, a probability per
/// option, and a confidence.
enum LayaModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "laya",
        displayName: "Laya (Decision)",
        modalities: ["text"],
        modelTypes: [.decision],
        backingPackage: "ml-explore/mlx-swift",
        packageLicense: "MIT",
        maintainers: ["Apple MLX"],
        notes: """
        A ModernBERT-large encoder + a small decision head (LY-4), answering one typed \
        question (choice/score/yes-no) with a label, a probability per option, and a \
        confidence — never generative text. Ported from the Python `laya-mlx` 0.2.0 package \
        (Apache-2.0). Claims source==.mlx decision entries (Laya 0.4B).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: LayaSDK(), ui: LayaUI(), descriptor: descriptor)
    }
}
