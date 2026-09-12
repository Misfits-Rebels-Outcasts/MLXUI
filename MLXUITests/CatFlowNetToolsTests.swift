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
        // Net verdicts are catalog-free, so explicit empties are correct (CFM-R14-FIX-3).
        let emptyCatalog: [ModelEntry] = []
        let emptyClaim: Set<String> = []
        #expect(TaskAvailability.isAvailable("Web Fetch", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
        #expect(TaskAvailability.isAvailable("HTTP Get", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
        #expect(TaskAvailability.isAvailable("Fetch Feed", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
        #expect(TaskAvailability.isAvailable("Download File", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
    }

    /// Phase WS: `Web Search` is ported (in `RealExecutor`/`supportedNetTools`), but with no
    /// Tavily/Brave key it's `.needsSetup`, not `.available` and not `.refusedByChannel` (a
    /// dead end this app could never clear) — distinct states this test pins directly rather
    /// than folding into the blanket "approved net tools" check above, since Web Search is
    /// the one net tool whose verdict depends on Keychain state, not just being ported.
    @Test func webSearchIsNeedsSetupWithNoKeyAndAvailableWithOne() {
        let account = KeychainHelper.providerAccount("tavily")
        let braveAccount = KeychainHelper.providerAccount("brave")
        let originalTavily = KeychainHelper.get(account: account)
        let originalBrave = KeychainHelper.get(account: braveAccount)
        KeychainHelper.delete(account: account)
        KeychainHelper.delete(account: braveAccount)
        defer {
            if let originalTavily { KeychainHelper.save(originalTavily, account: account) }
            if let originalBrave { KeychainHelper.save(originalBrave, account: braveAccount) }
        }

        let emptyCatalog: [ModelEntry] = []
        let emptyClaim: Set<String> = []
        let desc = TaskCatalog.get("Web Search")!
        #expect(!TaskAvailability.isAvailable("Web Search", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
        guard case .needsSetup(_, let action) = TaskAvailability.state(
            for: desc, catalog: emptyCatalog, claimableModelIDs: emptyClaim) else {
            Issue.record("Web Search with no key should be .needsSetup")
            return
        }
        #expect(action == .openSettings(.providers))

        KeychainHelper.save("test-key-\(UUID().uuidString)", account: account)
        #expect(TaskAvailability.isAvailable("Web Search", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
    }

    @Test func feedWatchAndPageDiffAreRunnableNow() throws {
        // 32-FeedWatch needed Fetch Feed; 33-PageDiff needed Web Fetch; 31-ResearchBrief
        // needed Web Search — all three are `canRun`-runnable now (Phase WS ports the
        // task; a missing key is `.needsSetup`, which `canRun` no longer treats as a
        // dead end — see `webSearchFlowsAreRunnableNowNeedingOnlyAKey` in
        // `CatFlowNotRunnableTests` for the `FlowPreflight`-inclusive version of this
        // same check).
        let runnable = ["32-FeedWatch", "33-PageDiff", "31-ResearchBrief"]
        for fid in runnable {
            let doc = try GalleryLoader.loadDocument(flowID: fid)
            #expect(FlowRunner.canRun(doc) == .runnable, "\(fid) should be runnable")
        }
    }

    @Test func noNetworkURLIsRefusedByTheSchemeCheck() async throws {
        await #expect(throws: FlowError.self) {
            _ = try await NetTools.httpGet("file:///etc/passwd")
        }
    }

    // MARK: - Download File overwrite (R13-8)

    /// R13-8: `DownloadFileTool`'s overwrite now ends in `FlowParity.replace(dest:with:)` —
    /// the same §14.1 seam the Save-* tools' overwrites use, so the old file is **trashed,
    /// never deleted**. A full-fetch version of this test can't run in the sandboxed test
    /// scheme (it may not exec a `python3` fixture server or accept loopback), so the seam
    /// itself is pinned here, exactly as `saveImageTrashesTheOverwrite` pins `moveToTrash`.
    @Test func downloadOverwriteTrashesThePreviousFile() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-download-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        let dest = base.appendingPathComponent("out.bin")
        try Data("payload v1".utf8).write(to: dest)
        let tmp = base.appendingPathComponent(".out.bin.tmp")
        try Data("payload v2".utf8).write(to: tmp)

        try FlowParity.replace(dest: dest, with: tmp)

        #expect(try String(contentsOf: dest, encoding: .utf8) == "payload v2")
        let trash = base.appendingPathComponent(".trash", isDirectory: true)
        let trashed = (try FileManager.default.contentsOfDirectory(atPath: trash.path))
            .filter { $0.hasPrefix("out.bin.") }
        #expect(trashed.count == 1, "the overwritten file must be trashed, not deleted")
        #expect(!FileManager.default.fileExists(atPath: tmp.path), "the temp file is consumed by the move")
    }
}
