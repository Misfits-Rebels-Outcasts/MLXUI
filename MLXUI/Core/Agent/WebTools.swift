//
//  WebTools.swift
//  MLXUI — agentic chat tools, slice AG2.
//
//  The first *real* agent tool: `fetch_url`. Performs a read-only HTTP GET over the app's
//  `network.client` entitlement (no sandbox change), caps response size + time, and reduces
//  HTML to readable text so the model gets usable content to summarize. Read-only ⇒ no
//  approval gate (`requiresApproval == false`).
//
//  Network I/O is not unit-testable in CI (offline, and the AG0-b async-suite anomaly), so the
//  risky logic — URL validation, size truncation, and HTML→text extraction — lives in pure
//  `static` helpers exercised directly by synchronous tests. `execute` is a thin wrapper that
//  does the actual transfer and calls the helpers.
//
//  `web_search` (a provider-backed search tool) is intentionally deferred: it needs a product
//  decision on a search provider/endpoint (see plan-agentic-tools.md, AG2 notes).
//

import Foundation
import MLXLMCommon

/// Fetches a web page or HTTP API over HTTPS/HTTP and returns its body as text.
nonisolated struct FetchURLTool: AgentTool {
    let name = "fetch_url"
    let toolDescription = """
        Fetch the contents of a web page or HTTP API using a GET request over http(s). \
        Returns the response body as text; HTML pages are reduced to readable plain text. \
        Read-only — it cannot submit forms, log in, or send data.
        """
    let parameters: [ToolParameter] = [
        .required("url", type: .string, description: "The absolute http(s) URL to fetch.")
    ]
    // Read-only fetch — safe to run without user approval.
    var requiresApproval: Bool { false }

    /// Largest response body kept (bytes). Larger bodies are truncated; the time cap is the
    /// real runaway guard.
    static let maxBytes = 512 * 1024
    /// Per-request + whole-resource timeout (seconds).
    static let timeout: TimeInterval = 20

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let raw = arguments.string("url"), !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "Error: missing required 'url' argument."
        }
        let url: URL
        switch Self.validate(raw) {
        case .ok(let u): url = u
        case .invalid(let message): return "Error: \(message)"
        }

        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = Self.timeout
        config.timeoutIntervalForResource = Self.timeout
        config.httpCookieStorage = nil
        config.urlCache = nil
        let session = URLSession(configuration: config)
        defer { session.finishTasksAndInvalidate() }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Mozilla/5.0 (compatible; AI-Browser-Agent/1.0)", forHTTPHeaderField: "User-Agent")

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return "Error: non-HTTP response from \(url.absoluteString)."
            }
            guard (200..<300).contains(http.statusCode) else {
                return "Error: HTTP \(http.statusCode) fetching \(url.absoluteString)."
            }
            let contentType = http.value(forHTTPHeaderField: "Content-Type")
            let (body, truncated) = Self.truncate(data)
            let text = Self.extractText(from: body, contentType: contentType)
            let header = "Fetched \(url.absoluteString) (HTTP \(http.statusCode))"
                + (truncated ? " — truncated to \(Self.maxBytes / 1024) KB" : "")
            return "\(header)\n\n\(text)"
        } catch let error as URLError where error.code == .timedOut {
            return "Error: request to \(url.absoluteString) timed out after \(Int(Self.timeout))s."
        } catch {
            return "Error: could not fetch \(url.absoluteString): \(error.localizedDescription)"
        }
    }

    // MARK: - Pure helpers (unit-tested)

    /// Outcome of `validate` — a URL or a human-readable reason it was rejected.
    enum Validation: Equatable {
        case ok(URL)
        case invalid(String)
    }

    /// Accepts only absolute http(s) URLs; returns a descriptive reason otherwise.
    static func validate(_ urlString: String) -> Validation {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), let scheme = url.scheme?.lowercased() else {
            return .invalid("'\(trimmed)' is not a valid URL.")
        }
        guard scheme == "http" || scheme == "https" else {
            return .invalid("only http and https URLs are allowed (got '\(scheme)').")
        }
        guard url.host?.isEmpty == false else {
            return .invalid("URL is missing a host: '\(trimmed)'.")
        }
        return .ok(url)
    }

    /// Caps `data` at `maxBytes`, flagging whether truncation occurred.
    static func truncate(_ data: Data) -> (data: Data, truncated: Bool) {
        guard data.count > maxBytes else { return (data, false) }
        return (data.prefix(maxBytes), true)
    }

    /// Decodes the body to text; if it looks like HTML, reduces it to readable plain text.
    static func extractText(from data: Data, contentType: String?) -> String {
        let text = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .isoLatin1)
            ?? ""
        let isHTML = (contentType?.lowercased().contains("html") ?? false)
            || looksLikeHTML(text)
        return isHTML ? htmlToText(text) : text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Cheap sniff for markup when the Content-Type is missing or generic.
    static func looksLikeHTML(_ s: String) -> Bool {
        let head = s.prefix(1024).lowercased()
        return head.contains("<!doctype html") || head.contains("<html") || head.contains("<body")
    }

    /// Strips `<script>`/`<style>` blocks and tags, decodes common entities, and collapses
    /// whitespace — enough for a model to read/summarize a page.
    static func htmlToText(_ html: String) -> String {
        var s = html
        for tag in ["script", "style", "head", "noscript", "svg"] {
            s = removeBlocks(named: tag, in: s)
        }
        // Turn block-level break tags into newlines before stripping the rest.
        for br in ["<br>", "<br/>", "<br />", "</p>", "</div>", "</li>", "</tr>", "</h1>", "</h2>", "</h3>"] {
            s = s.replacingOccurrences(of: br, with: "\n", options: .caseInsensitive)
        }
        s = s.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
        s = decodeEntities(s)
        // Collapse runs of spaces/tabs, drop spaces a stripped tag left before punctuation,
        // then trim excess blank lines.
        s = s.replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)
        s = s.replacingOccurrences(of: " +([.,;:!?])", with: "$1", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\n[ \\t]*\\n[ \\t]*(\\n[ \\t]*)+", with: "\n\n", options: .regularExpression)
        let lines = s.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Removes `<tag …>…</tag>` blocks (case-insensitive, across newlines).
    private static func removeBlocks(named tag: String, in s: String) -> String {
        s.replacingOccurrences(
            of: "<\(tag)[^>]*>[\\s\\S]*?</\(tag)>",
            with: " ",
            options: [.regularExpression, .caseInsensitive]
        )
    }

    /// Decodes the handful of HTML entities common in body text.
    private static func decodeEntities(_ s: String) -> String {
        var out = s
        let map = [
            "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": "\"",
            "&#39;": "'", "&apos;": "'", "&nbsp;": " ", "&mdash;": "—",
            "&ndash;": "–", "&hellip;": "…",
        ]
        for (entity, replacement) in map {
            out = out.replacingOccurrences(of: entity, with: replacement, options: .caseInsensitive)
        }
        return out
    }
}
