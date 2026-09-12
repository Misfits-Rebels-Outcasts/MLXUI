import Foundation

/// WS-2 — which search provider serves a `Web Search` row. An explicit `provider=`
/// setting on the row wins; otherwise Tavily when its key is set, else Brave — "Tavily
/// first when both are" (the backlog's own wording; Tavily is also the free-tier
/// default the UI suggests, per §0 ruling 2's own note that Brave has had no free tier
/// since February 2026).
nonisolated enum WebSearchProvider: String, Sendable, CaseIterable {
    case tavily, brave

    var displayName: String { self == .tavily ? "Tavily" : "Brave" }
    /// The `credentials` name / Keychain account suffix — `KeychainHelper
    /// .providerAccount(credentialName)`. SPEC-Q224: the same name a `kind: "search"`
    /// manifest's own `credentials:` field carries (`tavily.json`/`brave.json`).
    var credentialName: String { rawValue }

    /// `explicit` is the row's own `provider=` setting, if any. Returns `nil` only when
    /// `explicit` names neither known provider — the caller reports that as an invalid
    /// setting, distinct from "no key configured yet" (a `WebSearchProvider` value with
    /// no key is still a resolvable, nameable provider — see `WebSearchTool`).
    static func resolve(explicit: String?) -> WebSearchProvider? {
        if let explicit, !explicit.isEmpty {
            return WebSearchProvider(rawValue: explicit.lowercased())
        }
        if KeychainHelper.get(account: KeychainHelper.providerAccount(tavily.credentialName)) != nil { return .tavily }
        if KeychainHelper.get(account: KeychainHelper.providerAccount(brave.credentialName)) != nil { return .brave }
        return nil
    }
}

/// WS-2 — the pure, network-free half of the port: build each provider's request, parse
/// each provider's response. Directly unit-tested, mirroring RM's own `ProviderWireFormat`
/// split (isolate the real network call, test the orchestration around it). Verified
/// against both vendors' live API docs before writing (standing rule 9), not guessed —
/// see the journal for the exact pages read.
nonisolated enum WebSearchWireFormat {
    /// Tavily's `/search`. `POST`, JSON body, `Authorization: Bearer`. `site` has no
    /// dedicated Tavily param — `include_domains` (an array) is the documented
    /// equivalent. `recency`'s canonical vocabulary (`day|week|month|year`, WS-1) is
    /// Tavily's own `time_range` vocabulary verbatim — no translation needed.
    static func tavilyRequest(query: String, topK: Int, site: String?, recency: String?)
        -> (url: URL, body: [String: Any]) {
        var body: [String: Any] = ["query": query, "max_results": topK]
        if let site { body["include_domains"] = [site] }
        if let recency { body["time_range"] = recency }
        return (URL(string: "https://api.tavily.com/search")!, body)
    }

    static func tavilyHeaders(apiKey: String) -> [String: String] {
        ["Authorization": "Bearer \(apiKey)", "content-type": "application/json"]
    }

    /// `results[].url`, in the order Tavily returned them.
    static func tavilyParse(_ json: [String: Any]) -> [String] {
        let results = json["results"] as? [[String: Any]] ?? []
        return results.compactMap { $0["url"] as? String }
    }

    /// `recency`'s canonical vocabulary → Brave's `freshness` vocabulary (WS-1's own
    /// mapping table).
    static let braveFreshness: [String: String] = ["day": "pd", "week": "pw", "month": "pm", "year": "py"]

    /// Brave's `/res/v1/web/search`. `GET`, query-string params, `X-Subscription-Token`.
    /// `site` has no dedicated Brave param either — `site:` is a documented inline query
    /// operator, the same idiom the (unported) DDG reference used.
    static func braveRequestURL(query: String, topK: Int, site: String?, recency: String?) -> URL {
        var q = query
        if let site { q += " site:\(site)" }
        var comps = URLComponents(string: "https://api.search.brave.com/res/v1/web/search")!
        var items = [URLQueryItem(name: "q", value: q), URLQueryItem(name: "count", value: String(topK))]
        if let recency, let freshness = braveFreshness[recency] {
            items.append(URLQueryItem(name: "freshness", value: freshness))
        }
        comps.queryItems = items
        return comps.url!
    }

    /// `web.results[].url`, in the order Brave returned them.
    static func braveParse(_ json: [String: Any]) -> [String] {
        let web = json["web"] as? [String: Any]
        let results = web?["results"] as? [[String: Any]] ?? []
        return results.compactMap { $0["url"] as? String }
    }
}

/// `Web Search` (text → `[text]`, SPEC-Q222: URLs only, not the content both APIs
/// already return — see the entry for the reasoning). WS-1's settings surface, ported
/// exactly: `query · top_k (default 5) · site · recency · timeout`, plus `provider=` to
/// override WS-2's Tavily-first selection.
///
/// **Never logs, echoes, or persists a key or a request body** — the same standing
/// constraint as Phase RM. The key is read from the Keychain fresh inside `run`, never
/// stored; the missing-key message names the provider, never a key.
nonisolated struct WebSearchTool {
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let s = FlowSettings(settings)
        let query = try Self.resolveQuery(inputs: inputs, settings: s)
        let topK = Int(s.value(for: "top_k") ?? "5") ?? 5
        let site = s.value(for: "site")
        let recency = s.value(for: "recency")
        let timeout = NetTools.parseTimeout(s.value(for: "timeout"))

        let providerSetting = s.value(for: "provider")
        if let providerSetting, !providerSetting.isEmpty, WebSearchProvider(rawValue: providerSetting.lowercased()) == nil {
            throw FlowError.invalidSettings(row: "Web Search", setting: "provider",
                                            detail: "must be \"tavily\" or \"brave\", not \"\(providerSetting)\"")
        }
        // Tavily is the suggested default when nothing is configured at all (§0 ruling
        // 2's own free-tier note) — naming it, not a generic "a search provider", is
        // more actionable in the missing-key refusal below.
        let provider = WebSearchProvider.resolve(explicit: providerSetting) ?? .tavily
        let account = KeychainHelper.providerAccount(provider.credentialName)
        guard let apiKey = KeychainHelper.get(account: account) else {
            throw FlowError.stageFailure(row: "Web Search",
                message: "Web Search needs a \(provider.displayName) key, and none is set. Add it in Settings, then run this row again. Your key is configured in Settings, never in the flow.")
        }

        let urls: [String]
        switch provider {
        case .tavily:
            let (url, body) = WebSearchWireFormat.tavilyRequest(query: query, topK: topK, site: site, recency: recency)
            let bodyData = try JSONSerialization.data(withJSONObject: body)
            let (data, _) = try await NetTools.request(url.absoluteString, method: "POST",
                                                        headers: WebSearchWireFormat.tavilyHeaders(apiKey: apiKey),
                                                        body: bodyData, timeout: timeout)
            urls = WebSearchWireFormat.tavilyParse(Self.decodeJSON(data))
        case .brave:
            let url = WebSearchWireFormat.braveRequestURL(query: query, topK: topK, site: site, recency: recency)
            let (data, _) = try await NetTools.httpGet(url.absoluteString,
                                                       headers: ["X-Subscription-Token": apiKey], timeout: timeout)
            urls = WebSearchWireFormat.braveParse(Self.decodeJSON(data))
        }
        guard !urls.isEmpty else {
            throw FlowError.stageFailure(row: "Web Search", message: "Web Search for \"\(query)\" returned no results")
        }
        // SPEC-Q222: URLs only, not the page text both APIs already return alongside
        // them — the conservative reading, matching `TaskCatalog`'s `[text]`-of-URLs
        // typing on both runtimes.
        return Asset(items: urls.prefix(max(topK, 0)).map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    private static func decodeJSON(_ data: Data) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    /// `net.py::_resolve_text`'s "upstream item, else `query=`/bare settings" resolution
    /// — the same shape `NetTools.resolveURL` already uses for a URL instead of a query.
    static func resolveQuery(inputs: [Asset], settings: FlowSettings) throws -> String {
        if let first = inputs.first?.items.first, first.kind == .text, let value = first.value, !value.isEmpty {
            return value
        }
        guard let raw = settings.value(for: "query") ?? settings.firstBare(), !raw.isEmpty else {
            throw FlowError.missingInlineValue(row: "Web Search", kind: .text)
        }
        return raw
    }
}
