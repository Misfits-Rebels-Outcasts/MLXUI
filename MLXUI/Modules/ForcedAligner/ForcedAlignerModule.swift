import Foundation

/// Bundles the ForcedAligner SDK and the Run screen and registers them. Backed by `Blaizzy/mlx-audio-swift` (MIT) — runs
/// the **installed** mlx-community Qwen3-ForcedAligner safetensors via `Qwen3ForcedAlignerModel.generate`. Claims only the
/// `Qwen3-ForcedAligner` family at kind `.align`; Qwen3-ASR and every other module decline it (and it declines theirs).
enum ForcedAlignerModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "forced-aligner",
        displayName: "Qwen3-ForcedAligner (forced alignment, MLX)",
        modalities: ["audio", "text"],
        modelTypes: [.align],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Forced alignment via mlx-audio-swift's Qwen3ForcedAlignerModel.fromModelDirectory + generate(audio:text:language:) \
        (16 kHz mono; audio plus the known transcript in, word/character timestamps out; up to about five minutes per \
        call). Runs the installed mlx-community/Qwen3-ForcedAligner-0.6B-8bit safetensors -- the files InstallManager \
        downloads. Claims only the Qwen3-ForcedAligner family at .align; the Qwen3-ASR SDK's `qwen3-asr` claim cannot \
        match that family, and this SDK claims nothing else.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: ForcedAlignerSDK(), ui: ForcedAlignerUI(), descriptor: descriptor)
    }
}
