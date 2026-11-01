import Foundation

/// Bundles the FLUX.2 Klein 4B text -> image SDK + UI and registers them (MP7-4). A hand-port (MP7-3a...3h) of VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT) onto
/// mlx-swift: Qwen3-4B conditioner, 5 + 20 block MMDiT, FLUX.2 VAE, run one component at a time. Loads the installed mlx-community/FLUX.2-klein-4B-bf16 files and never
/// downloads. Not served by the Local Server.
enum Flux2KleinModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "flux2-klein",
        displayName: "FLUX.2 Klein 4B (Image Generation, MLX)",
        modalities: ["text", "image"],
        modelTypes: [.image],
        backingPackage: "VincentGourbin/flux-2-swift-mlx (Swift port)",
        packageLicense: "Apache 2.0 (model); MIT (reference port)",
        maintainers: ["Black Forest Labs", "Vincent Gourbin"],
        notes: """
        Text to image with FLUX.2 Klein 4B, distilled to 4 steps. The Qwen3-4B text encoder (8-bit), the image model (quantized to 8 or 4 bits as it loads) and the VAE \
        (bf16) are loaded one at a time, so a 1024 x 1024 picture needs about 6 GB. Sizes are multiples of 16 up to what this Mac's memory allows. Claims only the Klein 4B \
        entry; the FLUX.1 module never sees it. Not served by the Local Server.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: Flux2KleinSDK(), ui: Flux2KleinUI(), descriptor: descriptor)
    }
}
