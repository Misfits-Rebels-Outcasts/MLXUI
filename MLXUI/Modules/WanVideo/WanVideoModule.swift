import Foundation

/// Bundles the WAN 2.1 T2V-1.3B text → video SDK + UI and registers them.
/// Swift/MLX port of the WAN-AI/Wan2.1-T2V-1.3B-Diffusers pipeline:
/// UMT5-XXL text encoder · 30-layer DiT backbone with 3-D RoPE ·
/// 3-D causal VAE decoder · Euler ODE sampler (30 steps). WAN-AM4.
enum WanVideoModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "wanvideo",
        displayName: "WAN 2.1 (Video Generation, MLX)",
        modalities: ["text", "video"],
        modelTypes: [.video],
        backingPackage: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers (Swift port)",
        packageLicense: "Apache 2.0",
        maintainers: ["Wan-AI"],
        notes: """
        Text → video via a from-reference Swift/MLX port of WAN 2.1 T2V-1.3B: \
        UMT5-XXL text encoder, WanTransformer3DModel DiT (30 L / 12 H / 1536 d) \
        with 3-D RoPE and adaLN-zero, 3-D causal VAE decoder, and a 30-step Euler ODE sampler. \
        AM4 delivers first-frame preview; full video + MP4 export in AM5.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: WanVideoSDK(), ui: WanVideoUI(), descriptor: descriptor)
    }
}
