import Foundation

/// Bundles the MusicGen text → music SDK + UI and registers them. A from-reference Swift/MLX
/// port of `ml-explore/mlx-examples/musicgen` (T5-base encoder + causal decoder + EnCodec),
/// loading the installed `jasonvassallo/mlx-musicgen-small` weights (the files `InstallManager`
/// downloads). First text-to-music module in the app. MG-MOD1.
enum MusicGenModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "musicgen",
        displayName: "MusicGen (Music Generation, MLX)",
        modalities: ["text", "audio"],
        modelTypes: [.music],
        backingPackage: "ml-explore/mlx-examples/musicgen (Swift port)",
        packageLicense: "CC-BY-NC-4.0",
        maintainers: ["Meta", "Apple MLX", "jasonvassallo"],
        notes: """
        Text → music via a from-reference Swift/MLX port of Apple's mlx-examples/musicgen: \
        T5-base text encoder, 24-layer causal decoder (4 interleaved codebooks, CFG, delay \
        pattern), and EnCodec 32 kHz decode. Runs the installed \
        jasonvassallo/mlx-musicgen-small float32 safetensors (the 32 kHz EnCodec codec is \
        bundled into the install). Non-commercial CC-BY-NC-4.0 license (T5 component Apache-2.0).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: MusicGenSDK(), ui: MusicGenUI(), descriptor: descriptor)
    }
}
