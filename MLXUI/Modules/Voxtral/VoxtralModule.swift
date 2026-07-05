import Foundation

/// Bundles the Voxtral ASR SDK and registers it. Reuses `WhisperUI` (the run surface is the
/// same `ASRRunView`), exactly as `MLXWhisperModule` does. Backed by `Blaizzy/mlx-audio-swift`
/// (MIT) — runs the **installed** mlx-community Voxtral safetensors via the model's offline
/// `generate(audio:)`. Claims only Voxtral-family ASR; everything else falls through.
enum VoxtralModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "voxtral",
        displayName: "Voxtral (ASR, MLX)",
        modalities: ["audio", "text"],
        modelTypes: [.asr],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Speech-to-text via mlx-audio-swift's VoxtralRealtimeModel.fromDirectory + offline \
        generate(audio:) (16 kHz). Runs the installed mlx-community/Voxtral-* safetensors — \
        the files InstallManager downloads. Reuses ASRStage/ASRRunView (WhisperUI). Claims \
        only Voxtral-family ASR; MLX-Whisper declines non-whisper families, so no conflict.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: VoxtralSDK(), ui: WhisperUI(), descriptor: descriptor)
    }
}
