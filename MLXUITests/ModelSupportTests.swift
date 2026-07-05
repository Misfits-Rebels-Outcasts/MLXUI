import Testing
@testable import MLXUI

/// Covers `ModelSupport.unsupportedReason` — the flag for architectures with no MLX runner yet
/// (backlog § "Model support gaps").
struct ModelSupportTests {
    @Test func allCatalogArchitecturesNowHaveRunners() {
        // Every OCR / embedder architecture is now runnable — the gap list is empty:
        // olmOCR (Qwen2-VL), gemma VLM, MiniLM, ModernBERT (SUP-4), PaddleOCR-VL (SUP-1),
        // dots.ocr / dots.mocr (SUP-3), and DeepSeek-OCR-2 (SUP-2) all resolve to a runner.
        for id in [
            "mlx-community--olmOCR-2-7B-1025-mlx-4bit",
            "mlx-community--gemma-3-4b-it-4bit",
            "mlx-community--all-MiniLM-L6-v2-4bit",
            "mlx-community--nomicai-modernbert-embed-base-bf16",
            "mlx-community--PaddleOCR-VL-0.9B-4bit",
            "mlx-community--dots.mocr-4bit",
            "mlx-community--DeepSeek-OCR-2-bf16",
        ] {
            #expect(ModelSupport.unsupportedReason(for: makeEntry(id: id)) == nil)
        }
    }
}
