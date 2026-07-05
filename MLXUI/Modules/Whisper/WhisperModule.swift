import Foundation

/// Bundles the Whisper SDK + UI and registers them into the `ModelRegistry`. Backed by
/// `argmaxinc/WhisperKit` (MIT, on-device).
enum WhisperModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "whisper",
        displayName: "Whisper (ASR)",
        modalities: ["audio", "text"],
        modelTypes: [.asr],
        backingPackage: "argmaxinc/WhisperKit",
        packageLicense: "MIT",
        maintainers: ["Argmax"],
        notes: """
        WhisperKit ignores the catalog's MLX fp16 whisper files and downloads its own \
        CoreML model (argmaxinc/whisperkit-coreml), keyed by size name. The install gate \
        is therefore cosmetic for Whisper, and the first Run incurs a separate network \
        download. The catalog entry only selects the size (see WhisperKitSize). Voxtral \
        is ASR but not Whisper, so it does not claim here.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: WhisperSDK(), ui: WhisperUI(), descriptor: descriptor)
    }
}
