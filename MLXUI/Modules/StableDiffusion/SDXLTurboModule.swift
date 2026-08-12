import Foundation

/// Bundles the SDXL-Turbo text → image SDK + UI and registers them. A from-reference Swift/MLX
/// port of `ml-explore/mlx-examples/stable_diffusion` (SDXL-Turbo: dual CLIP encoders + SDXL UNet
/// + DDIM + AutoencoderKL), loading the installed fp16 weights from `stabilityai/sdxl-turbo` (the
/// files `InstallManager` downloads). Second diffusion module in the app (after Flux). SD-MOD1.
enum SDXLTurboModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "sdxl-turbo",
        displayName: "SDXL-Turbo (Image Generation, MLX)",
        modalities: ["text", "image"],
        modelTypes: [.image],
        backingPackage: "ml-explore/mlx-examples/stable_diffusion (Swift port)",
        packageLicense: "Stability AI Community License (RAIL++-M)",
        maintainers: ["Stability AI", "Apple MLX"],
        notes: """
        Text → image via a from-reference Swift/MLX port of Apple's mlx-examples/stable_diffusion, \
        adapted for SDXL-Turbo: CLIP-L + OpenCLIP ViT-G/14 text encoders, the SDXL \
        UNet2DConditionModel, a 4-step DDIM sampler (no CFG), and the AutoencoderKL VAE decoder. \
        Runs the installed stabilityai/sdxl-turbo fp16 safetensors. Adversarially distilled for \
        1–4 denoising steps. Non-commercial-friendly Stability AI Community License (RAIL++-M).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: SDXLTurboSDK(), ui: SDXLTurboUI(), descriptor: descriptor)
    }
}
