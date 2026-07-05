import Foundation

/// Bundles the Kokoro TTS SDK + UI and registers them. Backed by `Blaizzy/mlx-audio-swift`
/// (MIT). Runs the **installed** Kokoro safetensors + voices (`InstallManager` fetches the
/// full repo, incl. `voices/`) — no separate download at Run.
enum KokoroModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "kokoro",
        displayName: "Kokoro (TTS)",
        modalities: ["text", "audio"],
        modelTypes: [.tts],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Text-to-speech via mlx-audio-swift's KokoroModel, loading the installed
        mlx-community/Kokoro-82M-bf16 files (config + kokoro-v1_0.safetensors + voices/).
        Claims only Kokoro-family TTS entries; other TTS families fall through to unsupported.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: KokoroSDK(), ui: KokoroUI(), descriptor: descriptor)
    }
}
