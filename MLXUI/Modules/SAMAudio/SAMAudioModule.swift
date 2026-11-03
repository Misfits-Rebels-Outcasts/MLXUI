import Foundation

/// Bundles the SAM-Audio SDK and UI and registers them. Backed by `Blaizzy/mlx-audio-swift` (MIT) — runs the
/// **installed** `mlx-community/sam-audio-base` safetensors through `SAMAudio(config:)` + `loadConvertedWeights`, with
/// the T5-base text encoder the installer puts in `t5/` (never a download at run time). Claims only SAM-Audio-family
/// `.separate` models.
enum SAMAudioModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "samaudio",
        displayName: "SAM-Audio (sound isolation, MLX)",
        modalities: ["audio"],
        modelTypes: [.separate],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Sound isolation via mlx-audio-swift's SAMAudio: a recording plus a short description of a sound goes in, the \
        isolated sound and everything else come out (48 kHz mono). Runs the installed mlx-community/sam-audio-base \
        files — the ones InstallManager downloads, with the google-t5/t5-base encoder in t5/ — loading them through \
        SAMAudio(config:) + loadConvertedWeights so no path can download weights. Clips up to 30 s go through one \
        separate() pass; longer clips are windowed 30 s / 3 s with the library's cosine crossfade so Stop works \
        between windows. Claims only SAM-Audio-family .separate models; no other module claims that kind.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: SAMAudioSDK(), ui: SAMAudioUI(), descriptor: descriptor)
    }
}
