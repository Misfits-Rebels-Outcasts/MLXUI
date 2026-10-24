import Foundation

/// Bundles the Parakeet ASR SDK and registers it. Reuses `WhisperUI` (the run surface is the
/// same `ASRRunView`), exactly as `VoxtralModule` does. Backed by `Blaizzy/mlx-audio-swift`
/// (MIT) — runs the **installed** mlx-community Parakeet safetensors via `ParakeetModel.generate`.
/// Claims only Parakeet-family ASR; everything else falls through.
enum ParakeetModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "parakeet",
        displayName: "Parakeet (ASR, MLX)",
        modalities: ["audio", "text"],
        modelTypes: [.asr],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Speech-to-text via mlx-audio-swift's ParakeetModel.fromDirectory + generate(audio:) \
        (16 kHz). Runs the installed mlx-community/parakeet-* safetensors — the files \
        InstallManager downloads. Reuses ASRStage/ASRRunView (WhisperUI). Claims only \
        Parakeet-family ASR; Whisper and Voxtral decline that family, so no conflict.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: ParakeetSDK(), ui: WhisperUI(), descriptor: descriptor)
    }
}
