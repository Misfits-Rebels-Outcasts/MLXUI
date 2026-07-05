import Foundation

/// Bundles the generic MLX TTS SDK + UI and registers them. Backed by
/// `Blaizzy/mlx-audio-swift` (MIT) via its `TTS.loadModel` factory. Runs the **installed**
/// non-Kokoro TTS safetensors (Qwen3-TTS, chatterbox, orpheus) — text → audio. Kokoro keeps
/// its dedicated module (claimed `.exact`); this claims everything else TTS at `.generic`.
enum MLXAudioTTSModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "mlx-audio-tts",
        displayName: "TTS (MLX, mlx-audio-swift)",
        modalities: ["text", "audio"],
        modelTypes: [.tts],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Text-to-speech for non-Kokoro MLX families via mlx-audio-swift's TTS.loadModel, \
        which auto-detects the architecture from the installed config.json (Qwen3-TTS, \
        chatterbox, orpheus/Llama, …). Voice defaults to the model's own speaker; per-family \
        voice pickers arrive in AT2. A family the package can't load surfaces a graceful \
        engine error, not a crash.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: MLXAudioTTSSDK(), ui: MLXAudioTTSUI(), descriptor: descriptor)
    }
}
