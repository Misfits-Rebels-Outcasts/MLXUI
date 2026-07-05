import Foundation

/// Bundles the DeepSeek-OCR-2 SDK + UI and registers them. DeepSeek-OCR-2 uses a custom
/// `deepseekocr_2` architecture (SAM ViT → Qwen2-0.5B decoder-as-encoder → DeepSeek-V2 MoE LM) that
/// MLXVLM's stock type registry can't load, so it runs on a from-reference Swift/MLX port that
/// registers a custom `deepseekocr_2` model + processor into MLXVLM and reuses the shared VLM generate
/// path (`RSI/plan-deepseek-ocr.md`, SUP-2). The run surface is the same reusable `OCRUI`/`OCRRunView`
/// every OCR model uses (image → extracted text).
enum DeepSeekOCRModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "deepseek-ocr",
        displayName: "DeepSeek-OCR",
        modalities: ["image", "text"],
        modelTypes: [.ocr],
        backingPackage: "ml-explore/mlx-swift-lm (custom deepseekocr_2 model type)",
        packageLicense: "MIT",
        maintainers: ["Apple MLX", "DeepSeek"],
        notes: """
        Image → text for DeepSeek-OCR-2, whose `deepseekocr_2` architecture (SAM ViT → Qwen2-0.5B \
        encoder → DeepSeek-V2 MoE LM) isn't in MLXVLM's stock type registry. Runs on a from-reference \
        Swift/MLX port that registers a custom `deepseekocr_2` model + processor into MLXVLM. \
        Registered before OCRModule so it wins the `.exact` tie for DeepSeek-OCR checkpoints; all other \
        OCR repos still route to OCRModule.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: DeepSeekOCRSDK(), ui: OCRUI(), descriptor: descriptor)
    }
}
