import Foundation

/// Bundles the MLX VLM SDK + UI and registers them. Backed by `ml-explore/mlx-swift-lm`'s
/// `MLXVLM` (MIT). Runs the **installed** mlx-community VLM safetensors (the files
/// `InstallManager` downloads) — image + prompt → text, with no Apple Vision dependency.
enum VLMModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "vlm",
        displayName: "Vision-Language (VLM, MLX)",
        modalities: ["image", "text"],
        modelTypes: [.vision],
        backingPackage: "ml-explore/mlx-swift-lm",
        packageLicense: "MIT",
        maintainers: ["Apple MLX"],
        notes: """
        Image + prompt → text via mlx-swift-lm's MLXVLM (VLMModelFactory), loading the \
        installed mlx-community VLM safetensors. The CGImage is fed directly as \
        UserInput.Image.ciImage — no Apple Vision, no temp file. Claims source==.mlx \
        vision entries (LFM2-VL, gemma-3-VL, Qwen3-VL).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: VLMSDK(), ui: VLMUI(), descriptor: descriptor)
    }
}
