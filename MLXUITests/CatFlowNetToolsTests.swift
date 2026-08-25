import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-9 (approved scope) — the networked tools. The pure logic (HTML→text, feed
/// parsing, headers, availability flips) is pinned here; the URLSession GET itself is thin
/// platform code with a redirect cap + timeout + size cap.
struct CatFlowNetToolsTests {

    @Test func htmlToTextExtractsPlainAndMarkdown() {
        let html = """
        <html><head><title>x</title></head><body>
        <h1>Hello</h1>
        <p>Some <b>text</b> and a <a href="https://example.com">link</a>.</p>
        <ul><li>one</li><li>two</li></ul>
        <script>var x = 1;</script>
        </body></html>
        """
        let plain = HTMLToText.extract(html, markdown: false)
        #expect(plain.contains("Hello"))
        #expect(plain.contains("Some text and a link."))
        #expect(plain.contains("one"))
        #expect(!plain.contains("var x"))          // script stripped
        #expect(!plain.contains("<b>"))
        let markdown = HTMLToText.extract(html, markdown: true)
        #expect(markdown.contains("# Hello"))
        #expect(markdown.contains("[link](https://example.com)"))
        #expect(markdown.contains("- one"))
    }

    @Test func feedParserReadsRSSAndAtom() throws {
        let rss = """
        <rss version="2.0"><channel><title>C</title>
          <item><title>First</title><link>http://a/1</link><description>D1</description></item>
          <item><title>Second</title><link>http://a/2</link><description>D2</description></item>
        </channel></rss>
        """
        let rssEntries = try FeedParser.parse(rss)
        #expect(rssEntries.count == 2)
        #expect(rssEntries[0] == ("First", "http://a/1", "D1"))

        let atom = """
        <feed xmlns="http://www.w3.org/2005/Atom"><title>T</title>
          <entry><title>E1</title><link href="http://a/e1"/><summary>S1</summary></entry>
        </feed>
        """
        let atomEntries = try FeedParser.parse(atom)
        #expect(atomEntries.count == 1)
        #expect(atomEntries[0] == ("E1", "http://a/e1", "S1"))
    }

    @Test func headersParseTheSemicolonGrammar() {
        let h = NetTools.parseHeaders("Accept:application/json;X-Api-Key:k")
        #expect(h == ["Accept": "application/json", "X-Api-Key": "k"])
        #expect(NetTools.parseHeaders(nil).isEmpty)
    }

    @Test func availabilityFlipsForTheApprovedNetTools() {
        #expect(TaskAvailability.isAvailable("Web Fetch"))
        #expect(TaskAvailability.isAvailable("HTTP Get"))
        #expect(TaskAvailability.isAvailable("Fetch Feed"))
        #expect(TaskAvailability.isAvailable("Download File"))
        // Web Search stays honestly refused — no provider.
        #expect(!TaskAvailability.isAvailable("Web Search"))
        if case .refusedByChannel = TaskAvailability.state(for: TaskCatalog.get("Web Search")!) {} else {
            Issue.record("Web Search should be channel-refused")
        }
    }

    @Test func feedWatchAndPageDiffAreRunnableNow() throws {
        // 32-FeedWatch needed Fetch Feed; 33-PageDiff needed Web Fetch.
        let runnable = ["32-FeedWatch", "33-PageDiff"]
        for fid in runnable {
            let doc = try GalleryLoader.loadDocument(flowID: fid)
            #expect(FlowRunner.canRun(doc) == .runnable, "\(fid) should be runnable after R12-9")
        }
        // A flow that needs Web Search stays refused, naming the task.
        let doc = try GalleryLoader.loadDocument(flowID: "31-ResearchBrief")
        guard case .notRunnable(let reason) = FlowRunner.canRun(doc) else {
            Issue.record("31 should stay refused (Web Search unported)")
            return
        }
        #expect(reason.contains("Web Search"))
    }

    @Test func noNetworkURLIsRefusedByTheSchemeCheck() async throws {
        await #expect(throws: FlowError.self) {
            _ = try await NetTools.httpGet("file:///etc/passwd")
        }
    }
}
