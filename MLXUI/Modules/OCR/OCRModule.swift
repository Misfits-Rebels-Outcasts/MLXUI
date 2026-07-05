import Foundation

/// Bundles the OCR SDK + UI and registers them. OCR runs on `mlx-swift-lm`'s `MLXVLM`
/// (the catalog OCR repos are VLMs), reusing `VLMStage`/`VLMEngine` with a fixed
/// transcription prompt — image → extracted text, no Apple Vision.
enum OCRModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "ocr",
        displayName: "OCR (MLX)",
        modalities: ["image", "text"],
        modelTypes: [.ocr],
        backingPackage: "ml-explore/mlx-swift-lm",
        packageLicense: "MIT",
        maintainers: ["Apple MLX"],
        notes: """
        Image → text. The catalog OCR repos (PaddleOCR-VL, DeepSeek-OCR, dots.ocr, olmOCR) \
        are MLX VLMs, so OCR runs on the same MLXVLM engine (VLMStage) with a fixed \
        transcription prompt — no Apple Vision. A model whose architecture MLXVLM doesn't \
        yet support surfaces a graceful engine error, not a crash.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: OCRSDK(), ui: OCRUI(), descriptor: descriptor)
    }
}
