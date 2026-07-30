import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG2 — `fetch_url` tool. Network I/O isn't exercised here (offline CI + the AG0-b async-suite
/// anomaly); instead the pure, synchronous helpers that hold the risk — URL validation, size
/// truncation, and HTML→text extraction — are tested directly.
struct WebToolsTests {

    // MARK: schema / metadata

    @Test func fetchUrlIsReadOnlyAndSpecced() {
        let tool = FetchURLTool()
        #expect(tool.name == "fetch_url")
        #expect(tool.requiresApproval == false)
        let fn = tool.toolSpec["function"] as? [String: any Sendable]
        let params = fn?["parameters"] as? [String: any Sendable]
        #expect(params?["required"] as? [String] == ["url"])
    }

    // MARK: URL validation

    @Test func acceptsAbsoluteHttpUrls() {
        #expect(isSuccess(FetchURLTool.validate("https://example.com/page")))
        #expect(isSuccess(FetchURLTool.validate("http://example.com")))
        // Surrounding whitespace is tolerated.
        #expect(isSuccess(FetchURLTool.validate("  https://example.com  ")))
    }

    @Test func rejectsNonHttpSchemes() {
        #expect(!isSuccess(FetchURLTool.validate("file:///etc/passwd")))
        #expect(!isSuccess(FetchURLTool.validate("ftp://example.com")))
        #expect(!isSuccess(FetchURLTool.validate("javascript:alert(1)")))
    }

    @Test func rejectsRelativeOrHostlessUrls() {
        #expect(!isSuccess(FetchURLTool.validate("/just/a/path")))
        #expect(!isSuccess(FetchURLTool.validate("example.com")))   // no scheme
        #expect(!isSuccess(FetchURLTool.validate("https://")))       // no host
    }

    // MARK: size truncation

    @Test func truncatesOversizedBodies() {
        let big = Data(repeating: 0x41, count: FetchURLTool.maxBytes + 10)
        let (out, truncated) = FetchURLTool.truncate(big)
        #expect(truncated)
        #expect(out.count == FetchURLTool.maxBytes)
    }

    @Test func keepsSmallBodiesIntact() {
        let small = Data(repeating: 0x41, count: 128)
        let (out, truncated) = FetchURLTool.truncate(small)
        #expect(!truncated)
        #expect(out.count == 128)
    }

    // MARK: HTML → text

    @Test func stripsTagsScriptsAndStyles() {
        let html = """
            <!doctype html><html><head><style>.x{color:red}</style>
            <script>var a = 1 < 2;</script></head>
            <body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>
            """
        let text = FetchURLTool.htmlToText(html)
        #expect(text.contains("Title"))
        #expect(text.contains("Hello world."))
        #expect(!text.contains("color:red"))
        #expect(!text.lowercased().contains("var a"))
        #expect(!text.contains("<"))
    }

    @Test func decodesCommonEntities() {
        let text = FetchURLTool.htmlToText("<p>Fish &amp; Chips &mdash; &quot;tasty&quot;</p>")
        #expect(text.contains("Fish & Chips"))
        #expect(text.contains("—"))
        #expect(text.contains("\"tasty\""))
    }

    @Test func breakTagsBecomeNewlines() {
        let text = FetchURLTool.htmlToText("<p>line one<br>line two</p>")
        #expect(text.contains("line one"))
        #expect(text.contains("line two"))
        #expect(text.contains("\n"))
    }

    @Test func plainTextPassesThroughWhenNotHtml() {
        let data = Data("just some plain text".utf8)
        let out = FetchURLTool.extractText(from: data, contentType: "text/plain")
        #expect(out == "just some plain text")
    }

    @Test func sniffsHtmlWhenContentTypeMissing() {
        let data = Data("<html><body><p>Sniffed</p></body></html>".utf8)
        let out = FetchURLTool.extractText(from: data, contentType: nil)
        #expect(out.contains("Sniffed"))
        #expect(!out.contains("<"))
    }

    // MARK: helpers

    private func isSuccess(_ result: FetchURLTool.Validation) -> Bool {
        if case .ok = result { return true }
        return false
    }
}
