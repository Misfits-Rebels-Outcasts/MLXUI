import Foundation

/// Bundles the RIFE SDK and UI and registers them. A Swift/MLX port of Practical-RIFE 4.25 (IFNet, MIT, © hzwer) from the `rife-mlx`
/// reference (MIT). Runs the **installed** mlx-community/RIFE-4.25 files (`config.json` + `model.safetensors`, fp32). Claims only
/// RIFE-family `.interpolate` models; nothing else claims that kind.
enum RIFEModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "rife",
        displayName: "RIFE 4.25 (frame interpolation, MLX)",
        modalities: ["video"],
        modelTypes: [.interpolate],
        backingPackage: "hzwer/Practical-RIFE 4.25 (Swift port, via xocialize/rife-mlx)",
        packageLicense: "MIT",
        maintainers: ["hzwer"],
        notes: """
        Video frame interpolation: IFNet with five coarse-to-fine blocks, flow warp and a mask blend, run in fp32 (the flow is \
        fp16-sensitive). A clip becomes 2x or 4x its frame rate, the original audio is copied unchanged, and a Scale of 0.5 \
        trades flow detail for memory at 4K. Loads the files InstallManager downloaded; never downloads weights itself. Claims only \
        RIFE-family .interpolate models. No AI Workflows task names the kind (RIFE-W skipped).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: RIFESDK(), ui: RIFEUI(), descriptor: descriptor)
    }
}
