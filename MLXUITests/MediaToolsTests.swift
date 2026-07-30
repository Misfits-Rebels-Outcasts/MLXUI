import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG4c — media tools (`transcribe_audio`, `ocr_image`). The network transfer + audio/image
/// decode + inference aren't unit-testable (offline CI, no weights); the pure OCR engine-routing
/// helper + protocol conformance are tested synchronously.
struct MediaToolsTests {

    // MARK: OCR engine routing

    @Test func ocrRoutesPaddleByModelID() {
        #expect(OCRImageTool.backend(forModelID: "mlx-community--PaddleOCR-VL-0.9B-4bit") == .paddle)
    }

    @Test func ocrRoutesDeepSeekByModelID() {
        #expect(OCRImageTool.backend(forModelID: "mlx-community--DeepSeek-OCR-2-bf16") == .deepseek)
    }

    @Test func ocrRoutesDotsByModelID() {
        #expect(OCRImageTool.backend(forModelID: "mlx-community--dots.mocr-4bit") == .dots)
    }

    @Test func ocrRoutesOtherToSharedVLMEngine() {
        // olmOCR (Qwen2-VL arch) and any generic vision model run on the shared MLXVLM engine.
        #expect(OCRImageTool.backend(forModelID: "mlx-community--olmOCR-2-7B-1025-mlx-4bit") == .vlm)
        #expect(OCRImageTool.backend(forModelID: "mlx-community--Qwen3-VL-4B-Instruct-4bit") == .vlm)
    }

    @Test func ocrRoutingIsCaseInsensitive() {
        #expect(OCRImageTool.backend(forModelID: "SOME-PADDLEOCR-MODEL") == .paddle)
    }

    // MARK: URL validation is shared with fetch_url (rejects non-http schemes)

    @Test func mediaFetchRejectsNonHTTPURL() {
        // MediaFetch validates via FetchURLTool.validate; a file:// URL must be rejected.
        if case .ok = FetchURLTool.validate("file:///etc/passwd") {
            Issue.record("file:// URL should not validate for a media fetch")
        }
        if case .invalid = FetchURLTool.validate("https://example.com/a.mp3") {
            Issue.record("a well-formed https URL should validate")
        }
    }

    // MARK: protocol conformance

    @Test func mediaToolsAreReadOnly() {
        #expect(TranscribeAudioTool().requiresApproval == false)
        #expect(OCRImageTool().requiresApproval == false)
    }

    @Test func mediaToolsAreSpecced() {
        for tool: any AgentTool in [TranscribeAudioTool(), OCRImageTool()] {
            let fn = tool.toolSpec["function"] as? [String: any Sendable]
            let params = fn?["parameters"] as? [String: any Sendable]
            #expect(params?["required"] as? [String] == ["url"])
        }
    }
}
