import Foundation

/// Bundles the ACE-Step 1.5 text -> music SDK + UI and registers them (MC5-3i). A Swift/MLX port of the turbo no-LM path of `ace-step/ACE-Step-1.5`:
/// Qwen3-Embedding-0.6B conditioner, condition encoder, 24-layer DiT (8 steps, DCW), Oobleck VAE. Loads the installed `ACE-Step/Ace-Step1.5` files one
/// component at a time and never downloads.
enum AceStepModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "acestep",
        displayName: "ACE-Step 1.5 (Music Generation, MLX)",
        modalities: ["text", "audio"],
        modelTypes: [.music],
        backingPackage: "ace-step/ACE-Step-1.5 (Swift port)",
        packageLicense: "MIT (model); Apache-2.0 (Qwen3-Embedding-0.6B text encoder)",
        maintainers: ["ACE Studio", "StepFun"],
        notes: """
        Text to stereo 48 kHz music with ACE-Step 1.5 turbo (8 denoising steps) on the no-language-model path: Qwen3-Embedding text encoder, condition encoder, \
        diffusion transformer with wavelet correction, and the Oobleck audio VAE. The components load one at a time. Songs of 10 to 240 seconds. Claims only \
        entries of the ace-step family; the MusicGen module never sees them.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: AceStepSDK(), ui: AceStepUI(), descriptor: descriptor)
    }
}
