import Foundation

/// Bundles the PaddleOCR-VL SDK + UI and registers them. PaddleOCR-VL uses a custom
/// architecture `MLXVLM` can't load, so it runs on the vendored `PaddleOCRVL` package
/// (`Vendor/PaddleOCRVL`, MIT © lulzx) rather than the shared `VLMStage`. The run surface is
/// the same reusable `OCRUI`/`OCRRunView` every OCR model uses (image → extracted text).
/// See backlog SUP-1 / journal 2026-42.
enum PaddleOCRModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "paddleocr",
        displayName: "PaddleOCR-VL",
        modalities: ["image", "text"],
        modelTypes: [.ocr],
        backingPackage: "mlx-community/paddleocr-vl.swift (vendored → Vendor/PaddleOCRVL)",
        packageLicense: "MIT",
        maintainers: ["lulzx"],
        notes: """
        Image → text for PaddleOCR-VL, the 0.9B document VLM whose `paddleocr_vl` architecture \
        MLXVLM doesn't support. Runs on a from-scratch Swift/MLX port (vendored, pins bumped to \
        this app's mlx-swift 0.31 / swift-transformers 1.x). Registered before OCRModule so it \
        wins the `.exact` tie for PaddleOCR-VL; all other OCR repos still route to OCRModule.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: PaddleOCRSDK(), ui: OCRUI(), descriptor: descriptor)
    }
}
