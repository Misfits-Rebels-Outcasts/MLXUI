import Foundation

/// Bundles the Qwen3-ASR SDK and registers it. Reuses `WhisperUI` (the run surface is the same `ASRRunView`), exactly
/// as `VoxtralModule` and `ParakeetModule` do. Backed by `Blaizzy/mlx-audio-swift` (MIT) -- runs the **installed**
/// mlx-community Qwen3-ASR safetensors via `Qwen3ASRModel.generate`. Claims only Qwen3-ASR-family ASR; everything else
/// falls through. `taskFamilies` stays empty (readme T3).
enum Qwen3ASRModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "qwen3-asr",
        displayName: "Qwen3-ASR (ASR, MLX)",
        modalities: ["audio", "text"],
        modelTypes: [.asr],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Speech-to-text via mlx-audio-swift's Qwen3ASRModel.fromModelDirectory + generate(audio:generationParameters:) \
        (16 kHz, language left to the model). Runs the installed mlx-community/Qwen3-ASR-* safetensors -- the files \
        InstallManager downloads. Reuses ASRStage/ASRRunView (WhisperUI). Claims only Qwen3-ASR-family ASR; Whisper, \
        Voxtral and Parakeet decline that family, so no conflict.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: Qwen3ASRSDK(), ui: WhisperUI(), descriptor: descriptor)
    }
}
