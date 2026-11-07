import Foundation

/// Bundles the TranslateGemma SDK and UI and registers them. Backed by `ml-explore/mlx-swift-lm` (MIT) — Google's
/// TranslateGemma 4B, an MLX 4-bit Gemma 3 fine-tune converted by mlx-community, run through the pinned package's
/// `gemma3` text path on the **installed** files (`InstallManager` downloads them; nothing is downloaded at run time).
/// Claims only the `translategemma` family's `.translate` entries.
enum TranslateGemmaModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "translategemma",
        displayName: "TranslateGemma (translation, MLX)",
        modalities: ["text"],
        modelTypes: [.llm],
        backingPackage: "ml-explore/mlx-swift-lm",
        packageLicense: "MIT",
        maintainers: ["Google", "Apple MLX"],
        notes: """
        Text → text translation via mlx-swift-lm's gemma3 runner: a source language, a target language and some text \
        go in, the translation comes out. Builds the strict structured request TranslateGemma's chat template accepts \
        (one-item content list with source/target codes), counts the rendered prompt with the model's own tokenizer \
        and refuses over 2,048 tokens before generating, stops at <end_of_turn>, and decodes greedily. Runs only the \
        installed mlx-community/translategemma-4b-it-4bit files — no vision tower in this conversion, so text only. \
        Claims only `.translate` translategemma-family entries; every `.llm`-only path (chat, the Local Server, other \
        workflow tasks) leaves it out by kind.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: TranslateGemmaSDK(), ui: TranslateGemmaUI(), descriptor: descriptor)
    }
}
