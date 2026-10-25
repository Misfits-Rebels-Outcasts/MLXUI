import Foundation

/// Bundles the DeepFilterNet SDK and UI and registers them. Backed by `Blaizzy/mlx-audio-swift` (MIT) -- runs the **installed**
/// mlx-community DeepFilterNet v3 safetensors via `DeepFilterNetModel.fromLocal` + `enhance`. Claims only DeepFilterNet-family
/// `.enhance` models.
enum DeepFilterNetModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "deepfilternet",
        displayName: "DeepFilterNet (noise removal, MLX)",
        modalities: ["audio"],
        modelTypes: [.enhance],
        backingPackage: "Blaizzy/mlx-audio-swift",
        packageLicense: "MIT",
        maintainers: ["Prince Canuma"],
        notes: """
        Speech enhancement via mlx-audio-swift's DeepFilterNetModel.fromLocal(dir, subfolder: "v3") + enhance (48 kHz mono in \
        and out; the input is resampled to 48 kHz first). Runs the installed mlx-community/DeepFilterNet-mlx files -- the \
        files InstallManager downloads (v1, v2 and v3; v3 is used). Offline pass, optionally in crossfaded windows. Claims \
        only DeepFilterNet-family .enhance models; no other module claims that kind.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: DeepFilterNetSDK(), ui: DeepFilterNetUI(), descriptor: descriptor)
    }
}
