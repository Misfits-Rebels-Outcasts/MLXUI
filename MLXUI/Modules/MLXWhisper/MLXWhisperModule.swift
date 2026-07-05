import Foundation

/// Bundles the MLX-native Whisper SDK and registers it. Reuses `WhisperUI` (the run view is
/// the same `ASRRunView`). Backed by `Blaizzy/mlx-audio-swift` (MIT). Unlike the WhisperKit
/// module, this runs the **installed MLX safetensors** directly — no separate CoreML download.
enum MLXWhisperModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "mlx-whisper",
        displayName: "Whisper (ASR, MLX)",
        modalities: ["audio", "text"],
        modelTypes: [.asr],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Runs the installed mlx-community/whisper-*-asr-fp16 safetensors via \
        mlx-audio-swift's WhisperModel.fromDirectory — the files InstallManager already \
        downloads. No separate CoreML model (unlike the WhisperKit module). Registered \
        before WhisperModule so it wins the .exact tie for source==.mlx whisper entries.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: MLXWhisperSDK(), ui: WhisperUI(), descriptor: descriptor)
    }
}
