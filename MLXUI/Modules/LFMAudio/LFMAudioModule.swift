import Foundation

/// Bundles the LFM2.5-Audio SDK and UI and registers them. Backed by `Blaizzy/mlx-audio-swift` (MIT), pinned at 5c4c12b.
enum LFMAudioModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "lfmaudio",
        displayName: "LFM2.5-Audio (voice, MLX)",
        modalities: ["audio", "text"],
        modelTypes: [.voice],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Voice conversation, speech-to-text and text-to-speech via mlx-audio-swift's LFM2AudioModel.fromModelDirectory + \
        generateInterleaved / generateSequential and LFM2AudioDetokenizer (16 kHz in, 24 kHz out). Runs the installed \
        mlx-community/LFM2.5-Audio-1.5B-6bit files -- the files InstallManager downloads, without the root \
        tokenizer-*.safetensors. Claims only LFM2.5-Audio-family .voice models; no other module claims that kind. The model \
        itself is under the LFM Open License v1.0, separate from the package's MIT.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: LFMAudioSDK(), ui: LFMAudioUI(), descriptor: descriptor)
    }
}
