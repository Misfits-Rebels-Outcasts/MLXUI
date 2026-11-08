import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG4c — media tools (`transcribe_audio`, `ocr_image`). The network transfer + audio/image
/// decode + inference aren't unit-testable (offline CI, no weights); protocol conformance is tested
/// synchronously (engine choice moved to the resolver in MT1-3).
struct MediaToolsTests {

    // MT1-3: the id-substring OCR routing (`OCRImageTool.backend(forModelID:)`) and its five tests are retired. The
    // engine is picked by the registry's claim through the one-row flow; see ModelToolRunnerTests
    // (`ocrPicksTheInstalledModelByClaim`).

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
            // MT1-3: the input is `url` OR `asset` OR `path`, so none of them is individually required.
            #expect(params?["required"] as? [String] == [])
            let properties = params?["properties"] as? [String: any Sendable]
            #expect(properties?["url"] != nil && properties?["asset"] != nil && properties?["path"] != nil)
        }
    }
}
