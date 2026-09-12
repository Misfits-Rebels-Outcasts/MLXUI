import Foundation

// CFM-R12-9 — the networked tools, **approved scope** (owner 2026-08-25): Web Fetch, HTTP
// Get, Fetch Feed, Download File — GET-only, `URLSession`, no new entitlement (the app
// already carries `network.client`), a visible timeout and an explicit **redirect cap**.
// `Web Search` stayed unported per that ruling ("no provider to name in App Review") —
// **superseded by `RSI/DelegateOffMachineBacklog.md` §0 ruling 2, 2026-09-11** ("Tavily"
// and "Brave", both BYO-key), which names exactly the providers the 2026-08-25 ruling
// said didn't exist. Phase WS (journal `2026-09-12`) ports it; see `WebSearchTool` below.

nonisolated enum NetTools {
    static let defaultTimeout: TimeInterval = 30
    static let maxRedirects = 5
    /// Download File's size cap (no doc names one — the reviewer's "needs a size cap").
    static let defaultMaxBytes: Int64 = 512 * 1024 * 1024

    /// GET a URL → `(body, contentType)`. A plain-sentence error on failure, a redirect cap
    /// (no unbounded following), a streaming size cap, and **2xx only** (CFM-R12-FIX-10 — a
    /// 3xx returned by the capped delegate is an error, not content). A thin wrapper over
    /// `request(_:method:…)` — WS-2 reuses that shared function for Tavily's `POST`, rather
    /// than a second HTTP path, per the backlog's own instruction.
    static func httpGet(_ urlString: String, headers: [String: String] = [:],
                        timeout: TimeInterval = defaultTimeout,
                        maxBytes: Int64 = defaultMaxBytes,
                        maxRedirects: Int = maxRedirects) async throws -> (Data, String) {
        try await request(urlString, method: "GET", headers: headers, body: nil,
                          timeout: timeout, maxBytes: maxBytes, maxRedirects: maxRedirects)
    }

    /// WS-2: the same redirect-cap / timeout / streaming-size-cap / 2xx-only machinery as
    /// `httpGet`, generalized to any method + an optional body — Tavily's search endpoint
    /// is `POST` with a JSON body; everything else about the safety envelope is identical.
    static func request(_ urlString: String, method: String, headers: [String: String] = [:],
                        body: Data? = nil, timeout: TimeInterval = defaultTimeout,
                        maxBytes: Int64 = defaultMaxBytes,
                        maxRedirects: Int = maxRedirects) async throws -> (Data, String) {
        guard let url = URL(string: urlString) else {
            throw FlowError.stageFailure(row: "network", message: "'\(urlString)' isn't a valid URL")
        }
        guard url.scheme == "http" || url.scheme == "https" else {
            throw FlowError.stageFailure(row: "network",
                                         message: "'\(urlString)' isn't http(s) — only those are fetched")
        }
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.httpMethod = method
        request.httpBody = body
        var merged = ["User-Agent": "AI Browser/1.0"]
        for (k, v) in headers { merged[k] = v }
        request.allHTTPHeaderFields = merged

        let delegate = RedirectCappingDelegate(limit: maxRedirects)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let (bytes, response): (URLSession.AsyncBytes, URLResponse)
        do {
            (bytes, response) = try await session.bytes(for: request)
        } catch {
            throw FlowError.stageFailure(row: "network",
                                         message: "couldn't fetch \(urlString): \(error.localizedDescription)")
        }
        guard let http = response as? HTTPURLResponse else {
            throw FlowError.stageFailure(row: "network", message: "non-HTTP response for \(urlString)")
        }
        guard (200..<300).contains(http.statusCode) else {
            if (300..<400).contains(http.statusCode), delegate.capped {
                throw FlowError.stageFailure(row: "network",
                                             message: "\(urlString) followed more than \(maxRedirects) redirects — the fetch stopped")
            }
            throw FlowError.stageFailure(row: "network",
                                         message: "\(method) \(urlString) returned HTTP \(http.statusCode)")
        }
        // Stream, stopping at the cap — a chunked response with no Content-Length must not
        // be buffered to exhaustion (FIX-10).
        var bytesOut: [UInt8] = []
        var total: Int64 = 0
        if http.expectedContentLength > 0 {
            bytesOut.reserveCapacity(Int(min(http.expectedContentLength, maxBytes)))
        }
        for try await byte in bytes {
            total += 1
            if total > maxBytes {
                throw FlowError.stageFailure(row: "network",
                                             message: "\(urlString) exceeded the \(maxBytes / 1_048_576) MB cap")
            }
            bytesOut.append(byte)
        }
        let data = Data(bytesOut)
        let contentType = http.allHeaderFields["Content-Type"] as? String ?? ""
        return (data, contentType.components(separatedBy: ";").first ?? contentType)
    }

    /// Cap redirects at `limit` (an unbounded redirect chain is how a fetch becomes a loop).
    private final class RedirectCappingDelegate: NSObject, URLSessionTaskDelegate {
        let limit: Int
        private(set) var capped = false
        private var count = 0
        init(limit: Int) { self.limit = limit }

        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest,
                        completionHandler: @escaping (URLRequest?) -> Void) {
            count += 1
            if count > limit { capped = true }
            completionHandler(count <= limit ? request : nil)
        }
    }
}

/// `Web Fetch` (text → text): GET a URL; `as=markdown|plain` (default markdown); HTML bodies
/// are converted, everything else returned verbatim.
nonisolated struct WebFetchTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let url = try NetTools.resolveURL(inputs: inputs, settings: settings)
        let s = FlowSettings(settings)
        let asFormat = s.value(for: "as") ?? "markdown"
        guard asFormat == "markdown" || asFormat == "plain" else {
            throw FlowError.invalidSettings(row: "Web Fetch", setting: "as",
                                            detail: "must be \"markdown\" or \"plain\"")
        }
        let timeout = NetTools.parseTimeout(s.value(for: "timeout"))
        let (body, contentType) = try await NetTools.httpGet(url, timeout: timeout)
        var text = String(data: body, encoding: .utf8) ?? String(decoding: body, as: UTF8.self)
        if contentType == "text/html" {
            text = HTMLToText.extract(text, markdown: asFormat == "markdown")
        }
        return Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }
}

/// `HTTP Get` (text → text): GET a URL and return the raw body, `headers="K:V;K:V"`.
nonisolated struct HTTPGetTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let url = try NetTools.resolveURL(inputs: inputs, settings: settings)
        let s = FlowSettings(settings)
        let headers = NetTools.parseHeaders(s.value(for: "headers"))
        let timeout = NetTools.parseTimeout(s.value(for: "timeout"))
        let (body, _) = try await NetTools.httpGet(url, headers: headers, timeout: timeout)
        let text = String(data: body, encoding: .utf8) ?? String(decoding: body, as: UTF8.self)
        return Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }
}

/// `Download File` (text → file): GET a URL and write the body to `path=`/bare (default the
/// URL's last path segment), inside the flow folder, with a size cap + status check.
nonisolated struct DownloadFileTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let url = try NetTools.resolveURL(inputs: inputs, settings: settings)
        let s = FlowSettings(settings)
        let rawPath = s.value(for: "path")
            ?? URL(string: url)?.lastPathComponent ?? "download"
        guard !rawPath.isEmpty else {
            throw FlowError.missingInlineValue(row: "Download File", kind: .file)
        }
        let dest = try workspace.resolve(rawPath, flowID: flowID)
        let timeout = NetTools.parseTimeout(s.value(for: "timeout"))
        let (body, contentType) = try await NetTools.httpGet(url, timeout: timeout)
        // CFM-R12-FIX-10: the content-type check the approval named — a raw download that
        // returns HTML is usually a captive/login page, not the file.
        if contentType == "text/html" {
            throw FlowError.stageFailure(row: "Download File",
                                         message: "\(url) returned an HTML page, not a file — check the URL")
        }
        // Copy-to-temp-then-replace (journal 2026-121's Save-* discipline): a cancelled
        // download must never leave a truncated file that looks complete.
        let fm = FileManager.default
        try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = dest.deletingLastPathComponent()
            .appendingPathComponent(".\(dest.lastPathComponent).tmp\(UUID().uuidString)")
        try body.write(to: tmp)
        // R13-8: the overwrite follows the same §14.1 rule every other tool uses — the old
        // file is trashed, never deleted.
        try FlowParity.replace(dest: dest, with: tmp)
        return Asset(items: [Item(kind: .file, value: nil, path: dest, sourceText: nil)])
    }
}

/// `Fetch Feed` (text → text list): parse an RSS/Atom feed, `max_items=` (20).
nonisolated struct FetchFeedTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let url = try NetTools.resolveURL(inputs: inputs, settings: settings)
        let s = FlowSettings(settings)
        let maxItems = max(0, Int(s.value(for: "max_items") ?? "20") ?? 20)
        let timeout = NetTools.parseTimeout(s.value(for: "timeout"))
        let (body, _) = try await NetTools.httpGet(url, timeout: timeout)
        let text = String(data: body, encoding: .utf8) ?? String(decoding: body, as: UTF8.self)
        let entries = try FeedParser.parse(text)
        guard !entries.isEmpty else {
            throw FlowError.stageFailure(row: "Fetch Feed", message: "\(url) has no RSS/Atom items")
        }
        let items = entries.prefix(maxItems).map { title, link, summary in
            Item(kind: .text, value: "\(title)\n\(link)\n\n\(summary)".trimmingCharacters(in: .whitespacesAndNewlines),
                 path: nil, sourceText: nil)
        }
        return Asset(items: items)
    }
}

// MARK: - Shared helpers

extension NetTools {
    static func resolveURL(inputs: [Asset], settings: String) throws -> String {
        if let first = inputs.first?.items.first, first.kind == .text, let value = first.value, !value.isEmpty {
            return value
        }
        let s = FlowSettings(settings)
        guard let raw = s.value(for: "url") ?? s.firstBare(), !raw.isEmpty else {
            throw FlowError.missingInlineValue(row: "network", kind: .text)
        }
        return raw
    }

    /// SPEC-Q115's `headers=` grammar: `Key:Value` pairs separated by `;`.
    static func parseHeaders(_ raw: String?) -> [String: String] {
        guard let raw else { return [:] }
        var out: [String: String] = [:]
        for pair in raw.split(separator: ";") {
            let parts = pair.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                out[String(parts[0]).trimmingCharacters(in: .whitespaces)] =
                    String(parts[1]).trimmingCharacters(in: .whitespaces)
            }
        }
        return out
    }

    static func parseTimeout(_ raw: String?) -> TimeInterval {
        guard let raw, let t = Double(raw), t > 0 else { return NetTools.defaultTimeout }
        return t
    }
}

/// A compact HTML → text extractor mirroring `net.py::_HTMLToText`'s rules (CFM-R12-FIX-9):
/// skip script/style/head/noscript, block tags break lines, `markdown` adds heading `#`s and
/// `[text](href)` links, character references are decoded in text runs, blank lines are kept
/// (Python's `splitlines()` keeps them), and a self-closing tag (`<br/>`) fires **both** the
/// start and end handlers (Python's `handle_startendtag`). Pinned against Python goldens.
nonisolated enum HTMLToText {
    static let skipTags: Set<String> = ["script", "style", "head", "noscript"]
    static let blockTags: Set<String> = ["p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"]

    static func extract(_ html: String, markdown: Bool) -> String {
        Parser(markdown: markdown).run(html)
    }

    private final class Parser {
        let markdown: Bool
        var parts: [String] = []
        var skipDepth = 0
        var linkHref: String?

        init(markdown: Bool) { self.markdown = markdown }

        func run(_ html: String) -> String {
            var i = html.startIndex
            while i < html.endIndex {
                if html[i] == "<" {
                    guard let close = html[i...].firstIndex(of: ">") else {
                        parts.append(Self.unescape(String(html[i...])))
                        break
                    }
                    let tagText = String(html[html.index(after: i)..<close]).lowercased()
                    let isClosing = tagText.hasPrefix("/")
                    let isSelfClosing = !isClosing && tagText.hasSuffix("/")   // FIX-9: `<br/>`
                    var name = tagText.trimmingCharacters(in: .whitespaces)
                    name = name.replacingOccurrences(of: "^/", with: "", options: .regularExpression)
                    name = name.replacingOccurrences(of: "/$", with: "", options: .regularExpression)
                    name = name.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
                    if isClosing {
                        endTag(name)
                    } else {
                        startTag(name, tagText: tagText)
                        // Python's HTMLParser fires handle_startendtag → starttag **and**
                        // endtag for `<br/>`, which is why a self-closing block adds two breaks.
                        if isSelfClosing { endTag(name) }
                    }
                    i = html.index(after: close)
                } else {
                    if skipDepth == 0 {
                        // FIX-9: decode character references in text runs (Python convert_charrefs).
                        let next = html[i...].firstIndex(of: "<") ?? html.endIndex
                        parts.append(Self.unescape(String(html[i..<next])))
                        i = next
                    } else {
                        i = html.index(after: i)
                    }
                }
            }
            let raw = parts.joined()
            // FIX-9: keep empty lines (Python splitlines), then collapse blank runs to one.
            var lines: [String] = []
            var blankRun = 0
            // QR12R2-3: Python's splitlines() also splits on \r and \r\n (CRLF is what
            // most servers send); and Python's str.strip() treats U+00A0 as whitespace.
            let splitLines = raw.split(maxSplits: .max, omittingEmptySubsequences: false,
                                       whereSeparator: { $0 == "\n" || $0 == "\r" })
            for rawLine in splitLines {
                // Python's per-line `re.sub(r"[ \t]+", " ", ln).strip()`: interior
                // \xa0 survives (it's not a `[ \t]` char), edges strip it (Python's
                // `'\xa0'.isspace()` is True).
                let collapsed = rawLine.split(maxSplits: .max, omittingEmptySubsequences: true,
                                              whereSeparator: { $0 == " " || $0 == "\t" }).joined(separator: " ")
                let line = collapsed.trimmingCharacters(in: Parser.PythonWhitespace)
                if line.isEmpty {
                    blankRun += 1
                    if blankRun > 1 { continue }
                } else {
                    blankRun = 0
                }
                lines.append(line)
            }
            return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        }

        private func startTag(_ name: String, tagText: String) {
            if HTMLToText.skipTags.contains(name) { skipDepth += 1 }
            else if skipDepth == 0 {
                if name == "a", markdown {
                    linkHref = Self.href(of: tagText)
                    parts.append("[")
                } else if name == "li" {
                    parts.append(markdown ? "\n- " : "\n")
                } else if let h = Self.heading(name) {
                    parts.append("\n" + (markdown ? String(repeating: "#", count: h) + " " : ""))
                } else if HTMLToText.blockTags.contains(name) {
                    parts.append("\n")
                }
            }
        }

        private func endTag(_ name: String) {
            if HTMLToText.skipTags.contains(name) { skipDepth = max(0, skipDepth - 1) }
            else if skipDepth == 0 {
                if name == "a", markdown, let href = linkHref {
                    parts.append("](\(href))"); linkHref = nil
                } else if HTMLToText.blockTags.contains(name) {
                    parts.append("\n")
                }
            }
        }

        /// FIX-9: decode HTML character references (`&amp;`, `&#39;`, `&lt;`, `&nbsp;`, …)
        /// the way the Python's `convert_charrefs=True` does. `CFXML…UnescapingEntities`
        /// covers XML entities + numeric refs; `&nbsp;` (HTML-only) is mapped by hand.
        static func unescape(_ text: String) -> String {
            guard text.contains("&") else { return text }
            // QR12R2-3: CFXML decodes only the five XML entities + numeric refs; Python's
            // convert_charrefs decodes all ~2,000 HTML5 named references. The ones real pages
            // actually reach are a small table.
            var out = text
            for (name, scalar) in namedEntities {
                out = out.replacingOccurrences(of: "&\(name);", with: String(scalar))
            }
            out = out.replacingOccurrences(of: "&#160;", with: "\u{00A0}")
            let cf = CFXMLCreateStringByUnescapingEntities(kCFAllocatorDefault, out as CFString, nil)
            return (cf as String?) ?? out
        }

        /// Python's `str.strip()` whitespace — includes U+00A0, which Foundation's
        /// `whitespacesAndNewlines` does not.
        static let PythonWhitespace: CharacterSet = {
            var cs = CharacterSet.whitespacesAndNewlines
            cs.insert(Unicode.Scalar(0x00A0)!)
            return cs
        }()

        /// The HTML5 named references Python decodes that CFXML does not (prose punctuation).
        static let namedEntities: [(String, Unicode.Scalar)] = [
            ("mdash", "\u{2014}"), ("ndash", "\u{2013}"), ("rsquo", "\u{2019}"),
            ("lsquo", "\u{2018}"), ("ldquo", "\u{201C}"), ("rdquo", "\u{201D}"),
            ("hellip", "\u{2026}"), ("copy", "\u{00A9}"), ("reg", "\u{00AE}"),
            ("trade", "\u{2122}"), ("bull", "\u{2022}"), ("middot", "\u{00B7}"),
            ("nbsp", "\u{00A0}"), ("laquo", "\u{00AB}"), ("raquo", "\u{00BB}"),
            ("deg", "\u{00B0}"), ("plusmn", "\u{00B1}"), ("times", "\u{00D7}"),
            ("divide", "\u{00F7}"), ("eacute", "\u{00E9}"), ("egrave", "\u{00E8}"),
            ("agrave", "\u{00E0}"), ("ccedil", "\u{00E7}"), ("uuml", "\u{00FC}"),
            ("ouml", "\u{00F6}"), ("auml", "\u{00E4}"),
        ].compactMap { ($0.0, Unicode.Scalar($0.1)) }

        static func heading(_ name: String) -> Int? {
            guard name.count == 2, name.hasPrefix("h"), let n = Int(String(name.suffix(1))), (1...6).contains(n) else { return nil }
            return n
        }

        static func href(of tag: String) -> String? {
            guard let r = tag.range(of: "href=") else { return nil }
            let after = tag[tag.index(r.upperBound, offsetBy: 0)...]
            let trimmed = after.drop(while: { $0 == " " || $0 == "\t" })
            guard let first = trimmed.first else { return nil }
            let value: String
            if first == "\"" {
                value = String(trimmed.dropFirst().prefix(while: { $0 != "\"" }))
            } else if first == "'" {
                value = String(trimmed.dropFirst().prefix(while: { $0 != "'" }))
            } else {
                value = String(trimmed.prefix(while: { $0 != " " && $0 != "\t" && $0 != ">" }))
            }
            return value.isEmpty ? nil : value
        }
    }
}

/// An RSS/Atom feed parser (the `net.py::_parse_feed` shape): `(title, link, summary)` per
/// entry. Uses `XMLDocument` (Foundation), namespace-insensitive.
nonisolated enum FeedParser {
    static func parse(_ xml: String) throws -> [(String, String, String)] {
        guard let data = xml.data(using: .utf8),
              let doc = try? XMLDocument(data: data, options: []) else {
            throw FlowError.stageFailure(row: "Fetch Feed", message: "not a valid RSS/Atom feed")
        }
        let root = doc.rootElement()?.name ?? ""
        var entries: [(String, String, String)] = []
        func elements(_ node: XMLNode) -> [XMLElement] {
            node.children?.compactMap { $0 as? XMLElement } ?? []
        }
        if root == "rss" {
            let channel = elements(doc.rootElement()!).first { stripNS($0.name ?? "") == "channel" }
            for child in elements(channel ?? XMLNode()) where stripNS(child.name ?? "") == "item" {
                entries.append((childText(child, "title"), childText(child, "link"),
                                childText(child, "description")))
            }
        } else if root == "feed" {
            for entry in elements(doc.rootElement()!) where stripNS(entry.name ?? "") == "entry" {
                let title = childText(entry, "title")
                var link = ""
                if let linkEl = elements(entry).first(where: { stripNS($0.name ?? "") == "link" }) {
                    link = linkEl.attribute(forName: "href")?.stringValue ?? ""
                }
                let summary = childText(entry, "summary")
                entries.append((title, link, summary))
            }
        } else {
            throw FlowError.stageFailure(row: "Fetch Feed",
                                         message: "not a recognized RSS/Atom feed (root <\(root)>)")
        }
        return entries
    }

    private static func stripNS(_ name: String) -> String {
        name.split(separator: "}").last.map(String.init) ?? name
    }

    private static func childText(_ el: XMLElement, _ name: String) -> String {
        let child = el.children?.compactMap { $0 as? XMLElement }.first { stripNS($0.name ?? "") == name }
        return child?.stringValue ?? ""
    }
}
