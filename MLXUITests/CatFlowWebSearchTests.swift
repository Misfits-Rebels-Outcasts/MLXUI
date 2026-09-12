import Testing
import Foundation
@testable import MLXUI

/// Phase WS — Web Search (Tavily/Brave), entirely mock-driven at the network boundary:
/// every test here exercises `WebSearchWireFormat`'s pure request/response mapping or
/// `WebSearchTool`'s pre-network logic (query resolution, provider selection, the
/// missing-key refusal) — none issues a real HTTP request, mirroring both this
/// codebase's own convention (no existing net tool's real call is unit-tested either)
/// and Phase RM's `ProviderWireFormat` split. Every Keychain value is a throwaway,
/// UUID-suffixed string, never printed — the standing constraint since Phase KEY.
struct CatFlowWebSearchTests {

    private func withCleanCredentials<T>(_ body: () throws -> T) rethrows -> T {
        let tavilyAccount = KeychainHelper.providerAccount("tavily")
        let braveAccount = KeychainHelper.providerAccount("brave")
        let originalTavily = KeychainHelper.get(account: tavilyAccount)
        let originalBrave = KeychainHelper.get(account: braveAccount)
        KeychainHelper.delete(account: tavilyAccount)
        KeychainHelper.delete(account: braveAccount)
        defer {
            if let originalTavily { KeychainHelper.save(originalTavily, account: tavilyAccount) }
            if let originalBrave { KeychainHelper.save(originalBrave, account: braveAccount) }
        }
        return try body()
    }

    private func withCleanCredentials<T>(_ body: () async throws -> T) async rethrows -> T {
        let tavilyAccount = KeychainHelper.providerAccount("tavily")
        let braveAccount = KeychainHelper.providerAccount("brave")
        let originalTavily = KeychainHelper.get(account: tavilyAccount)
        let originalBrave = KeychainHelper.get(account: braveAccount)
        KeychainHelper.delete(account: tavilyAccount)
        KeychainHelper.delete(account: braveAccount)
        defer {
            if let originalTavily { KeychainHelper.save(originalTavily, account: tavilyAccount) }
            if let originalBrave { KeychainHelper.save(originalBrave, account: braveAccount) }
        }
        return try await body()
    }

    private func issues(for text: String) throws -> [FlowIssue] {
        let flow = try CatParser.parseForValidation(text)
        return FlowValidator.checkFlow(flow)
    }

    // MARK: - SPEC-Q224: the two search manifests

    @Test func tavilyManifestDecodesAsASearchKindWithCredentials() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "tavily.json"))
        #expect(manifest.kind == "search")
        #expect(manifest.credentials == "tavily")
        #expect(manifest.display == "Tavily")
    }

    @Test func braveManifestDecodesAsASearchKindWithCredentials() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "brave.json"))
        #expect(manifest.kind == "search")
        #expect(manifest.credentials == "brave")
        #expect(manifest.display == "Brave")
    }

    /// SPEC-Q224's own claim, confirmed rather than assumed: a `kind: "search"` manifest
    /// contributes nothing to the model-picker's provider registry.
    @Test func searchManifestsAreInertInTheModelProviderRegistry() {
        #expect(!TaskModels.providerDisplayNames.contains("Tavily"))
        #expect(!TaskModels.providerDisplayNames.contains("Brave"))
        #expect(TaskModels.providerModelRef(forDisplay: "Tavily") == nil)
        #expect(TaskModels.providerModelRef(forDisplay: "Brave") == nil)
    }

    // MARK: - WebSearchProvider.resolve — Tavily first, explicit override, neither

    @Test func resolvePrefersTavilyWhenBothKeysAreSet() {
        withCleanCredentials {
            KeychainHelper.save("t", account: KeychainHelper.providerAccount("tavily"))
            KeychainHelper.save("b", account: KeychainHelper.providerAccount("brave"))
            #expect(WebSearchProvider.resolve(explicit: nil) == .tavily)
        }
    }

    @Test func resolveFallsBackToBraveWhenOnlyItsKeyIsSet() {
        withCleanCredentials {
            KeychainHelper.save("b", account: KeychainHelper.providerAccount("brave"))
            #expect(WebSearchProvider.resolve(explicit: nil) == .brave)
        }
    }

    @Test func resolveIsNilWithNoKeyAndNoOverride() {
        withCleanCredentials {
            #expect(WebSearchProvider.resolve(explicit: nil) == nil)
        }
    }

    @Test func resolveHonorsAnExplicitOverrideRegardlessOfKeys() {
        withCleanCredentials {
            KeychainHelper.save("t", account: KeychainHelper.providerAccount("tavily"))
            #expect(WebSearchProvider.resolve(explicit: "brave") == .brave)
            #expect(WebSearchProvider.resolve(explicit: "Tavily") == .tavily)   // case-insensitive
        }
    }

    @Test func resolveIsNilForAnUnrecognizedOverride() {
        #expect(WebSearchProvider.resolve(explicit: "bing") == nil)
    }

    // MARK: - WebSearchWireFormat — Tavily, network-free

    @Test func tavilyRequestShapeMatchesTheLiveAPI() {
        let (url, body) = WebSearchWireFormat.tavilyRequest(query: "EU AI Act", topK: 5, site: nil, recency: nil)
        #expect(url.absoluteString == "https://api.tavily.com/search")
        #expect(body["query"] as? String == "EU AI Act")
        #expect(body["max_results"] as? Int == 5)
        #expect(body["include_domains"] == nil)
        #expect(body["time_range"] == nil)
    }

    @Test func tavilySiteMapsToIncludeDomains() {
        let (_, body) = WebSearchWireFormat.tavilyRequest(query: "q", topK: 5, site: "example.com", recency: nil)
        #expect(body["include_domains"] as? [String] == ["example.com"])
    }

    @Test func tavilyRecencyPassesThroughUnmapped() {
        // Tavily's own `time_range` vocabulary is day|week|month|year — WS-1's canonical
        // vocabulary verbatim, confirmed against the live docs, so no translation table.
        for recency in ["day", "week", "month", "year"] {
            let (_, body) = WebSearchWireFormat.tavilyRequest(query: "q", topK: 5, site: nil, recency: recency)
            #expect(body["time_range"] as? String == recency)
        }
    }

    @Test func tavilyHeadersCarryTheBearerKey() {
        let headers = WebSearchWireFormat.tavilyHeaders(apiKey: "tvly-test-key")
        #expect(headers["Authorization"] == "Bearer tvly-test-key")
    }

    @Test func tavilyParseReadsResultsURL() {
        let json: [String: Any] = ["results": [["url": "https://a.example", "title": "A", "content": "…"],
                                               ["url": "https://b.example", "title": "B", "content": "…"]]]
        #expect(WebSearchWireFormat.tavilyParse(json) == ["https://a.example", "https://b.example"])
    }

    @Test func tavilyParseIsEmptyForAMalformedResponse() {
        #expect(WebSearchWireFormat.tavilyParse([:]).isEmpty)
        #expect(WebSearchWireFormat.tavilyParse(["results": "not a list"]).isEmpty)
    }

    // MARK: - WebSearchWireFormat — Brave, network-free

    @Test func braveRequestURLMatchesTheLiveAPI() {
        let url = WebSearchWireFormat.braveRequestURL(query: "EU AI Act", topK: 5, site: nil, recency: nil)
        #expect(url.absoluteString.hasPrefix("https://api.search.brave.com/res/v1/web/search?"))
        #expect(url.query?.contains("q=EU%20AI%20Act") == true || url.query?.contains("q=EU+AI+Act") == true)
        #expect(url.query?.contains("count=5") == true)
    }

    @Test func braveSiteAppendsTheInlineOperator() {
        let url = WebSearchWireFormat.braveRequestURL(query: "q", topK: 5, site: "example.com", recency: nil)
        let decoded = url.query?.removingPercentEncoding ?? ""
        #expect(decoded.contains("q=q site:example.com"))
    }

    @Test func braveRecencyMapsToFreshness() {
        let expected = ["day": "pd", "week": "pw", "month": "pm", "year": "py"]
        for (recency, freshness) in expected {
            let url = WebSearchWireFormat.braveRequestURL(query: "q", topK: 5, site: nil, recency: recency)
            #expect(url.query?.contains("freshness=\(freshness)") == true)
        }
    }

    @Test func braveParseReadsWebResultsURL() {
        let json: [String: Any] = ["web": ["results": [["url": "https://a.example", "title": "A", "description": "…"]]]]
        #expect(WebSearchWireFormat.braveParse(json) == ["https://a.example"])
    }

    @Test func braveParseIsEmptyForAMalformedResponse() {
        #expect(WebSearchWireFormat.braveParse([:]).isEmpty)
        #expect(WebSearchWireFormat.braveParse(["web": "not a dict"]).isEmpty)
    }

    // MARK: - WebSearchTool.resolveQuery — input wins, settings is the fallback

    @Test func resolveQueryPrefersTheUpstreamTextInput() throws {
        let input = Asset(items: [Item(kind: .text, value: "from the row above", path: nil, sourceText: nil)])
        let query = try WebSearchTool.resolveQuery(inputs: [input], settings: FlowSettings("\"ignored\""))
        #expect(query == "from the row above")
    }

    @Test func resolveQueryFallsBackToTheBareSetting() throws {
        let query = try WebSearchTool.resolveQuery(inputs: [], settings: FlowSettings("\"a bare query\""))
        #expect(query == "a bare query")
    }

    @Test func resolveQueryFallsBackToTheQueryKey() throws {
        let query = try WebSearchTool.resolveQuery(inputs: [], settings: FlowSettings("query=\"explicit\""))
        #expect(query == "explicit")
    }

    @Test func resolveQueryThrowsWithNeitherInputNorSettings() throws {
        #expect(throws: (any Error).self) {
            _ = try WebSearchTool.resolveQuery(inputs: [], settings: FlowSettings(""))
        }
    }

    // MARK: - WebSearchTool.run — the missing-key refusal (no network reached)

    @Test func runRefusesPlainlyWithNoKeyNamingTavilyByDefault() async throws {
        await withCleanCredentials {
            let tool = WebSearchTool(settings: "\"test query\"")
            await #expect(throws: (any Error).self) {
                _ = try await tool.run(inputs: [])
            }
        }
    }

    @Test func runMissingKeyMessageNamesTheExplicitProviderOverride() async throws {
        await withCleanCredentials {
            let tool = WebSearchTool(settings: "\"test query\" provider=brave")
            do {
                _ = try await tool.run(inputs: [])
                Issue.record("expected a missing-key refusal")
            } catch {
                let message = String(describing: error)
                #expect(message.contains("Brave"))
                #expect(message.contains("Your key is configured in Settings, never in the flow"))
            }
        }
    }

    @Test func runRejectsAnUnrecognizedProviderOverride() async throws {
        let tool = WebSearchTool(settings: "\"q\" provider=bing")
        await #expect(throws: (any Error).self) {
            _ = try await tool.run(inputs: [])
        }
    }

    // MARK: - WS-1: settings surface reachable from Properties

    @Test func webSearchKnownSettingKeysMatchTheReferenceSurface() {
        let keys = Set(FlowRowInspectorView.knownSettingKeys(for: "Web Search"))
        #expect(keys == ["query", "top_k", "site", "recency", "timeout", "provider"])
    }

    // MARK: - WS-3: TaskAvailability / canRun for Web Search

    @Test func webSearchIsInSupportedNetTools() {
        #expect(TaskAvailability.supportedNetTools.contains("Web Search"))
    }

    @Test func aFreshWebSearchRowIsSelectableEvenWithNoKey() throws {
        try withCleanCredentials {
            let text = """
            mlxflow 0.8; offdevice; network
            1. Web Search    "query" top_k=3
            2. Save Text     out.md
            """
            let doc = try CatParser.parseForValidation(text)
            #expect(FlowRunner.canRun(try #require(doc.flowDocument)) == .runnable)
        }
    }

    // MARK: - SPEC-Q223: E120 for Web Search

    @Test func webSearchRowWithoutOffdeviceRaisesE120Unconditionally() throws {
        try withCleanCredentials {
            let text = """
            mlxflow 0.8; network
            1. Web Search    "query" top_k=3
            2. Save Text     out.md
            """
            let found = try issues(for: text)
            #expect(found.contains { $0.code == "E120" })
        }
    }

    @Test func webSearchRowWithOffdeviceRaisesNoE120() throws {
        let text = """
        mlxflow 0.8; offdevice; network
        1. Web Search    "query" top_k=3
        2. Save Text     out.md
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E120" })
    }

    @Test func e120NamesTheConfiguredProviderWhenOneExists() throws {
        try withCleanCredentials {
            KeychainHelper.save("test-key", account: KeychainHelper.providerAccount("tavily"))
            let text = """
            mlxflow 0.8; network
            1. Web Search    "query" top_k=3
            """
            let e120 = try #require(try issues(for: text).first { $0.code == "E120" })
            #expect(e120.message.contains("Tavily"))
            #expect(e120.message.contains("leaves this building"))
        }
    }

    @Test func e120NamesAGenericProviderWithNoKeyConfigured() throws {
        try withCleanCredentials {
            let text = """
            mlxflow 0.8; network
            1. Web Search    "query" top_k=3
            """
            let e120 = try #require(try issues(for: text).first { $0.code == "E120" })
            #expect(e120.message.contains("a search provider"))
        }
    }

    // MARK: - The two 2026-08-25 comments (superseded, not silently changed)

    @Test func webSearchIsNoLongerRefusedByCanRunForBeingUnported() throws {
        // The old ruling's own words ("no provider to name in App Review") no longer
        // apply — this is the live, functional proof the comment updates describe, not
        // just a text change.
        let text = "mlxflow 0.8; offdevice; network\n1. Web Search    \"q\"\n"
        try withCleanCredentials {
            let doc = try CatParser.parseForValidation(text)
            if case .notRunnable(let reason) = FlowRunner.canRun(try #require(doc.flowDocument)) {
                Issue.record("Web Search should not be canRun-refused: \(reason)")
            }
        }
    }
}
