import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-9 (approved scope) — the networked tools. The pure logic (HTML→text, feed
/// parsing, headers, availability flips) is pinned here; the URLSession GET itself is thin
/// platform code with a redirect cap + timeout + size cap.
struct CatFlowNetToolsTests {

    /// CFM-R12-FIX-9: pinned against Python-generated goldens (`html_golden.json`), not a
    /// hand-written `contains()`.
    @Test func htmlToTextMatchesThePythonGoldens() throws {
        let filePath = #filePath
        let url = URL(fileURLWithPath: filePath)
        var dir = url.deletingLastPathComponent()
        while dir.lastPathComponent != "MLXUITests" { dir = dir.deletingLastPathComponent() }
        let goldenURL = dir.deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow/tools/html_golden.json")
        let goldens = try JSONDecoder().decode([String: HTMLGolden].self, from: Data(contentsOf: goldenURL))
        var mismatches: [String] = []
        for (name, golden) in goldens {
            for (format, expected) in [(true, golden.markdown), (false, golden.plain)] {
                let got = HTMLToText.extract(html(name), markdown: format)
                if got != expected {
                    mismatches.append("\(name)/\(format ? "md" : "plain"): got \(String(reflecting: got)) expected \(String(reflecting: expected))")
                }
            }
        }
        #expect(mismatches.isEmpty, Comment(rawValue: mismatches.joined(separator: "; ")))
    }

    private struct HTMLGolden: Decodable {
        let markdown: String
        let plain: String
    }

    private func html(_ name: String) -> String {
        switch name {
        case "alpha_beta": return "<p>Alpha</p><p>Beta</p>"
        case "entities": return "<p>Ben &amp; Jerry&#39;s &lt;tag&gt; &nbsp;end</p>"
        case "br": return "<p>a<br/>b</p>"
        case "links": return "<h1>Hi</h1><p>See <a href=\"https://x.com\">here</a>.</p><ul><li>one</li><li>two</li></ul>"
        case "script": return "<p>Keep</p><script>var x=1;</script><p>This</p>"
        case "wrapped": return "<h2>Title</h2><p>A longer paragraph that keeps going and going and wraps.</p>"
        case "entities2": return "<p>a &mdash; b &rsquo;s &copy; &hellip;</p>"
        case "crlf": return "<p>line one</p>\r\n<p>line two</p>"
        case "trailing_nbsp": return "<p>tail&nbsp;</p><p>next</p>"
        case "nbsp_lead": return "<p>&nbsp;leading&nbsp;space</p>"
        case "tabs": return "<p>a\tb\tc</p>"
        case "deep": return "<ul><li>alpha</li><li><ul><li>nested</li></ul></li></ul>"
        default: return ""
        }
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
