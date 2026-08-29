import Foundation

/// Bundles the SeedVR2 3B super-resolution SDK + UI and registers them.
/// Swift/MLX port of ByteDance SeedVR2 (ICLR 2026): single-step Euler diffusion SR —
/// causal 3D VAE encoder/decoder · 32-layer multi-modal DiT (windowed attn, int8) ·
/// fixed pos_emb conditioning. SV-AM4.
enum SeedVR2Module: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "seedvr2",
        displayName: "SeedVR2 3B (Image SR, MLX int8)",
        modalities: ["image"],
        modelTypes: [.upscale],
        backingPackage: "mlx-community/SeedVR2-3B-mlx-int8 (Swift port)",
        packageLicense: "Apache 2.0",
        maintainers: ["ByteDance Research"],
        notes: """
        Image super-resolution via a Swift/MLX port of ByteDance SeedVR2 3B: \
        causal 3D VAE encoder/decoder (ch=128, 8× spatial), \
        32-layer multi-modal Diffusion Transformer (20 H / 128 d, int8 quantized), \
        and a single-step Euler scheduler. \
        Accepts an image and returns a 2× upscaled result. \
        SV-AM5 adds the full Run UI with image picker, scale selector, and before/after toggle.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: SeedVR2SDK(), ui: SeedVR2UI(), descriptor: descriptor)
    }
}
