import Foundation

/// Bundles the dots.ocr SDK + UI and registers them. dots.ocr / dots.mocr use a custom `dots_ocr`
/// architecture (a `dots_vit` NaViT vision tower + a Qwen2 LM) that MLXVLM's stock type registry
/// can't load, so it runs on a from-reference Swift/MLX port that registers a custom `dots_ocr`
/// model + processor into MLXVLM and reuses the shared VLM generate path (`RSI/plan-dots-ocr.md`,
/// SUP-3). The run surface is the same reusable `OCRUI`/`OCRRunView` every OCR model uses
/// (image → extracted text).
enum DotsOCRModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "dots-ocr",
        displayName: "dots.ocr",
        modalities: ["image", "text"],
        modelTypes: [.ocr],
        backingPackage: "ml-explore/mlx-swift-lm (custom dots_ocr model type)",
        packageLicense: "MIT",
        maintainers: ["Apple MLX", "rednote-hilab"],
        notes: """
        Image → text for dots.ocr / dots.mocr, whose `dots_ocr` architecture isn't in MLXVLM's \
        stock type registry. Runs on a from-reference Swift/MLX port that registers a custom \
        `dots_ocr` model + processor into MLXVLM and reuses the shared VLM generate path. \
        Registered before OCRModule so it wins the `.exact` tie for dots checkpoints; all other \
        OCR repos still route to OCRModule.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: DotsOCRSDK(), ui: OCRUI(), descriptor: descriptor)
    }
}
