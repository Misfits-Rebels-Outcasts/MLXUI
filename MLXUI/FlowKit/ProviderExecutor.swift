import Foundation

/// RM-2 — the "one executor arm" for `ModelSlot.provider`: a `URLSession`-only HTTP
/// client speaking each provider manifest's `engine`. `Foundation` + `URLSession`, no new
/// package (backlog/readme standing rule 9). Ported from
/// `catflow-mlx/src/catflow/engines/provider.py`'s `generate()`/`decide()` — the A1
/// (text-in, text-out) adapters only (`anthropic-api`, `deepseek-api`, `openai-api`,
/// `openai-compatible`); the reference's audio/vision/embed/rerank adapters (RA-12/13/
/// 14/15/17) serve tasks no ported manifest names yet and are out of this cycle's scope.
///
/// **Never logs, echoes, or persists a key or a request body.** Not in a trace, not in an
/// error, not in a journal — R904's own wording already sets the standard: "Your key is
/// configured in Settings, never in the flow." Concretely: the API key is read from the
/// Keychain fresh inside `generate()` and never stored as a stored property (so no
/// `Equatable`/`CustomStringConvertible` synthesis, no debugger `po`, ever holds it
/// incidentally); request headers/bodies are built, sent, and discarded, never written to
/// a file or a `print`/`os_log` call; `ProviderRequestError.status` (which becomes R904's
/// `{status}` fill value) carries only the HTTP status code and its standard reason
/// phrase — **deliberately not** the response body text the Python reference includes
/// (`provider.py::_post`'s `detail`). That is a flagged divergence, not an oversight: a
/// provider error body could, rarely but plausibly, echo a fragment of the request back
/// (some gateways include the offending payload in a 400 body), and this cycle's own
/// safety directive outweighs byte-parity here.
nonisolated protocol ProviderExecuting: Sendable {
    func generate(instructions: String, prompt: String, maxTokens: Int, temperature: Double) async throws -> String
}

/// Surfaced as R904 (`{provider} answered: {status}. …`) — never the response body.
nonisolated struct ProviderRequestError: Error, Sendable {
    let status: String   // e.g. "401 Unauthorized" — never response-body text
}

/// RM-2 — the pure, network-free half of the port: build each engine's request, parse
/// each engine's response. Unit-tested directly (mirrors the Python's own test strategy,
/// which "stubs `urllib.request.urlopen`", i.e. isolates the real network call and tests
/// the orchestration around it — `URLSessionProviderExecutor.generate` is that call, this
/// type is everything around it).
nonisolated enum ProviderWireFormat {
    /// `manifest.id` is `org/model` (Registry §7's own worked example,
    /// `"anthropic/claude-sonnet-4"`) — the part after the slash is the literal wire model
    /// name each provider's own API expects. Verbatim from `provider.py::_api_model_name`.
    static func apiModelName(_ manifestID: String) -> String {
        guard let slash = manifestID.firstIndex(of: "/") else { return manifestID }
        return String(manifestID[manifestID.index(after: slash)...])
    }

    /// Anthropic's Messages API. Verbatim from `provider.py::_anthropic_request`.
    static func anthropicRequest(prompt: String, apiModelName: String, maxTokens: Int,
                                 temperature: Double, apiKey: String) -> (url: URL, headers: [String: String], body: [String: Any]) {
        let url = URL(string: "https://api.anthropic.com/v1/messages")!
        let headers = ["x-api-key": apiKey, "anthropic-version": "2023-06-01", "content-type": "application/json"]
        let body: [String: Any] = [
            "model": apiModelName, "max_tokens": maxTokens, "temperature": temperature,
            "messages": [["role": "user", "content": prompt]],
        ]
        return (url, headers, body)
    }

    /// Verbatim from `provider.py::_anthropic_parse`.
    static func anthropicParse(_ json: [String: Any]) -> String {
        let blocks = json["content"] as? [[String: Any]] ?? []
        return blocks.filter { ($0["type"] as? String) == "text" }
            .map { $0["text"] as? String ?? "" }.joined()
    }

    /// DeepSeek's Chat Completions API (OpenAI-compatible wire shape). Verbatim from
    /// `provider.py::_deepseek_request` — note the base path has no `/v1` segment,
    /// confirmed against DeepSeek's own docs (their comment, carried forward here).
    static func deepseekRequest(prompt: String, apiModelName: String, maxTokens: Int,
                                temperature: Double, apiKey: String) -> (url: URL, headers: [String: String], body: [String: Any]) {
        let url = URL(string: "https://api.deepseek.com/chat/completions")!
        let headers = ["Authorization": "Bearer \(apiKey)", "content-type": "application/json"]
        let body: [String: Any] = [
            "model": apiModelName, "max_tokens": maxTokens, "temperature": temperature,
            "messages": [["role": "user", "content": prompt]],
        ]
        return (url, headers, body)
    }

    /// OpenAI's Chat Completions API. Verbatim from `provider.py::_openai_request`,
    /// including its two documented, verified (not guessed) GPT-5.6-family quirks: no
    /// `temperature` key at all (any value but the default 400s), and
    /// `max_completion_tokens` rather than `max_tokens` as the output-length key — the
    /// manifest's own `maps_to` stays `"max_tokens"` (the Registry's abstract-setting
    /// name); the wire-key rename is this adapter's own business, per the Python's
    /// comment.
    static func openAIRequest(prompt: String, apiModelName: String, maxTokens: Int,
                              apiKey: String) -> (url: URL, headers: [String: String], body: [String: Any]) {
        let url = URL(string: "https://api.openai.com/v1/chat/completions")!
        let headers = ["Authorization": "Bearer \(apiKey)", "content-type": "application/json"]
        let body: [String: Any] = [
            "model": apiModelName, "max_completion_tokens": maxTokens,
            "messages": [["role": "user", "content": prompt]],
        ]
        return (url, headers, body)
    }

    /// The generic OpenAI-shaped Chat Completions adapter — Groq, Together, OpenRouter,
    /// and every self-hosted server (a LAN box), one implementation. `base_url` from the
    /// manifest; a LAN endpoint (`egress: "lan"`) may declare no `credentials` at all, so
    /// `apiKey` is optional and the request simply omits `Authorization` when absent.
    /// Verbatim from `provider.py::_openai_compatible_request`.
    static func openAICompatibleRequest(prompt: String, apiModelName: String, maxTokens: Int,
                                        temperature: Double, baseURL: String?, apiKey: String?)
        -> (url: URL, headers: [String: String], body: [String: Any]) {
        let base = (baseURL ?? "https://api.openai.com/v1")
        let trimmed = base.hasSuffix("/") ? String(base.dropLast()) : base
        let url = URL(string: "\(trimmed)/chat/completions")!
        var headers: [String: String] = [:]
        if let apiKey { headers["Authorization"] = "Bearer \(apiKey)" }
        let body: [String: Any] = [
            "model": apiModelName, "max_tokens": maxTokens, "temperature": temperature,
            "messages": [["role": "user", "content": prompt]],
        ]
        return (url, headers, body)
    }

    /// Shared by `deepseek-api`, `openai-api`, and `openai-compatible` — all three are
    /// genuinely the same wire shape. Verbatim from `provider.py::_openai_compatible_parse`.
    static func openAICompatibleParse(_ json: [String: Any]) -> String {
        let choices = json["choices"] as? [[String: Any]] ?? []
        guard let first = choices.first, let message = first["message"] as? [String: Any] else { return "" }
        return message["content"] as? String ?? ""
    }
}

/// RM-2 — the real HTTP boundary, one per manifest. Built fresh per row (no caching, no
/// engine reuse across rows — matching `RealExecutor`'s own "sequential load/release"
/// discipline for the MLX path).
nonisolated struct URLSessionProviderExecutor: ProviderExecuting {
    let manifest: CuratedManifest

    func generate(instructions: String, prompt: String, maxTokens: Int, temperature: Double) async throws -> String {
        let apiModelName = ProviderWireFormat.apiModelName(manifest.id)
        let apiKey = manifest.credentials.flatMap { KeychainHelper.get(account: KeychainHelper.providerAccount($0)) }
        let (url, headers, body): (URL, [String: String], [String: Any])
        switch manifest.engine {
        case "anthropic-api":
            guard let apiKey else { throw ProviderRequestError(status: "no key configured") }
            (url, headers, body) = ProviderWireFormat.anthropicRequest(
                prompt: prompt, apiModelName: apiModelName, maxTokens: maxTokens,
                temperature: temperature, apiKey: apiKey)
        case "deepseek-api":
            guard let apiKey else { throw ProviderRequestError(status: "no key configured") }
            (url, headers, body) = ProviderWireFormat.deepseekRequest(
                prompt: prompt, apiModelName: apiModelName, maxTokens: maxTokens,
                temperature: temperature, apiKey: apiKey)
        case "openai-api":
            guard let apiKey else { throw ProviderRequestError(status: "no key configured") }
            (url, headers, body) = ProviderWireFormat.openAIRequest(
                prompt: prompt, apiModelName: apiModelName, maxTokens: maxTokens, apiKey: apiKey)
        case "openai-compatible":
            (url, headers, body) = ProviderWireFormat.openAICompatibleRequest(
                prompt: prompt, apiModelName: apiModelName, maxTokens: maxTokens,
                temperature: temperature, baseURL: manifest.baseURL, apiKey: apiKey)
        default:
            throw ProviderRequestError(status: "no adapter for engine \"\(manifest.engine ?? "?")\"")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        for (field, value) in headers { request.setValue(value, forHTTPHeaderField: field) }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ProviderRequestError(status: "no response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw ProviderRequestError(status: "\(http.statusCode) \(HTTPURLResponse.localizedString(forStatusCode: http.statusCode))")
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return ""
        }
        switch manifest.engine {
        case "anthropic-api":
            return ProviderWireFormat.anthropicParse(json)
        default:
            return ProviderWireFormat.openAICompatibleParse(json)
        }
    }
}

/// RM-2 — the AFM-1 pattern applied to remote providers: a process-global override seam
/// (`executorOverride`) so tests never touch the network, mirroring
/// `AppleFoundationAvailability.executorOverride` exactly. Production never sets it.
nonisolated enum ProviderAvailability {
    static var executorOverride: (any ProviderExecuting)?

    static func makeExecutor(for manifest: CuratedManifest) -> any ProviderExecuting {
        executorOverride ?? URLSessionProviderExecutor(manifest: manifest)
    }
}

/// RM-2 — wraps a `ProviderExecuting` as a `PipelineStage` so `RealExecutor.fireTag`
/// (F004's strict-parse-plus-one-retry) runs against a remote provider exactly as it
/// already does against a local MLX stage — no second tag-parsing implementation. Every
/// provider manifest declares `capabilities.constrained_decoding: false` (never guided
/// generation, unlike AFM's on-device path), so this is genuinely the same F004 path a
/// local decider takes, never suppressed.
nonisolated struct ProviderStage: PipelineStage {
    let id: String
    let name: String
    let executor: any ProviderExecuting
    let maxTokens: Int
    let temperature: Double
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        guard case .text(let prompt) = input else {
            throw StageError.unsupportedModel(id: id, kind: .llm)
        }
        let text = try await executor.generate(instructions: "", prompt: prompt,
                                                maxTokens: maxTokens, temperature: temperature)
        return .text(text)
    }
}
