import Foundation

/// Bundles the FLUX.1 text → image SDK + UI and registers them. A from-reference Swift/MLX
/// port of `ml-explore/mlx-examples/flux` (MMDiT + T5/CLIP encoders + flow-matching sampler +
/// VAE), loading the installed mlx-community 4-bit safetensors (the files `InstallManager`
/// downloads). First diffusion (text → image) module in the app. See `RSI/plan-flux.md` (AM4).
enum FluxModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "flux",
        displayName: "FLUX (Image Generation, MLX)",
        modalities: ["text", "image"],
        modelTypes: [.image],
        backingPackage: "ml-explore/mlx-examples/flux (Swift port)",
        packageLicense: "FLUX.1-dev Non-Commercial",
        maintainers: ["Apple MLX", "Freepik"],
        notes: """
        Text → image via a from-reference Swift/MLX port of Apple's mlx-examples/flux: T5 + CLIP \
        text encoders, rectified-flow MMDiT transformer, flow-matching sampler (dev-style: 50 \
        steps, guidance 4), and VAE decode. Runs the installed mlx-community/Flux-1.lite-8B-MLX-Q4 \
        safetensors. Non-commercial license (inherited from FLUX.1-dev).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: FluxSDK(), ui: FluxUI(), descriptor: descriptor)
    }
}
