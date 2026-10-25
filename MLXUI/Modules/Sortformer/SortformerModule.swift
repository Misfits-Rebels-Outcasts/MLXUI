import Foundation

/// Bundles the Sortformer SDK and UI and registers them. Backed by `Blaizzy/mlx-audio-swift` (MIT) -- runs the **installed**
/// mlx-community Sortformer safetensors via `SortformerModel.generate`. Claims only Sortformer-family diarization models.
enum SortformerModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "sortformer",
        displayName: "Sortformer (speaker diarization, MLX)",
        modalities: ["audio", "text"],
        modelTypes: [.diarization],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Speaker diarization via mlx-audio-swift's SortformerModel.fromModelDirectory + generate(audio:threshold:) \
        (16 kHz mono, up to 4 speakers). Runs the installed mlx-community/diar_streaming_sortformer_* safetensors -- the \
        files InstallManager downloads. Offline single pass; the streaming API is not used. Claims only \
        Sortformer-family diarization; no other module claims that kind.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: SortformerSDK(), ui: SortformerUI(), descriptor: descriptor)
    }
}
