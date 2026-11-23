import Foundation
import CryptoKit
import ImageIO

/// ER-11 — `Stage Print` (SPEC-Q258, owner rulings R8 and R10, 2026-11-22): a staged effect beside `Stage Send` / `Stage Post`.
///
/// **Staging never prints.** It validates the one PDF or image it was handed, keeps the outbox entry's *own stored copy* under
/// `outbox/files/`, and stops. **Approval is a person at the standard macOS print panel** (`PrintPresenting`): the entry is marked
/// committed only when the person completes the print. This runtime never runs `lp` or any process, and never prints silently.
nonisolated struct StagePrintSettingsError: Error, Equatable {
    /// Completes "Row N's `Stage Print` ..." (E722).
    var detail: String
}

nonisolated enum StagePrintTool {
    static let destination = "default printer"

    /// Python's `parse_stage_print`: no settings in v1 (no printer name, no options).
    static func parseSettings(_ raw: String?) throws {
        if let given = AudioRowSettings.firstGiven(raw) {
            throw StagePrintSettingsError(detail:
                "doesn't take settings -- the file goes to the system's default printer, and v1 has no printer or option settings; remove \(given)")
        }
    }

    /// What staging decided about one file, before anything is written.
    struct Prepared: Equatable {
        let data: Data
        let name: String
        let mime: String
        let ext: String
        let digest: String
        /// `sha256("Stage Print|" + sha256(normalised bytes))[:10]` (Q259): the content alone decides it, so the same bytes staged twice are one pending entry.
        let id: String
        var summary: String { "to \(StagePrintTool.destination): \(name)" }
        var status: String { "staged for print (\(id))" }
    }

    private static let printableImages: [String: (mime: String, ext: String)] = [
        "public.png": ("image/png", ".png"), "public.jpeg": ("image/jpeg", ".jpg"), "com.compuserve.gif": ("image/gif", ".gif"),
        "public.tiff": ("image/tiff", ".tiff"), "com.microsoft.bmp": ("image/bmp", ".bmp"),
    ]

    // SPEC-Q259 (owner ruling, Boo, 2026-11-22): the id of a staged PDF ignores /CreationDate, /ModDate and the trailer /ID, so a regenerated
    // identical document is the same entry. The byte rule is the Python's (`tools/outbox.py::pdf_digest_bytes`), pinned by the synced golden
    // `goldens/check/q259_pdf_digest.json`: over the bytes read as ISO-8859-1, `/CreationDate|/ModDate` + PDF whitespace + a literal `(...)` (backslash
    // escapes honoured) or hex `<...>` string becomes `/<key>()`, and `/ID` + whitespace + `[...]` becomes `/ID[]`. Only the digest is normalised.
    private static let pdfWhitespace = "[\\x00\\t\\n\\f\\r ]*"
    private static let pdfDateExpression = try? NSRegularExpression(
        pattern: "/(CreationDate|ModDate)" + pdfWhitespace + "(?:\\((?:\\\\(?s:.)|[^\\\\)])*\\)|<[0-9A-Fa-f\\x00\\t\\n\\f\\r ]*>)")
    private static let pdfIDExpression = try? NSRegularExpression(pattern: "/ID" + pdfWhitespace + "\\[[^\\]]*\\]")

    /// The bytes a PDF is hashed over for its Stage Print id. Anything that does not start with `%PDF-` is returned unchanged.
    static func pdfDigestBytes(_ data: Data) -> Data {
        guard data.starts(with: Data("%PDF-".utf8)),
              let text = String(data: data, encoding: .isoLatin1),
              let dates = pdfDateExpression, let ids = pdfIDExpression else { return data }
        var result = text
        result = dates.stringByReplacingMatches(in: result, range: NSRange(result.startIndex..., in: result), withTemplate: "/$1()")
        result = ids.stringByReplacingMatches(in: result, range: NSRange(result.startIndex..., in: result), withTemplate: "/ID[]")
        return result.data(using: .isoLatin1) ?? data
    }

    /// Python's `_minimal_pdf`: what a file the *mock* fabricated stands in as.
    static func minimalPDF(seed: Data) -> Data {
        let digest = SHA256.hash(data: seed).map { String(format: "%02x", $0) }.joined()
        return Data("%PDF-1.4\n% catflow mock print \(digest)\n%%EOF\n".utf8)
    }

    /// Validate and decide. Every refusal names `Stage Print` and starts the sentence, as `Make PDF`'s do. `mockBlobDirectory` is passed
    /// only by `MockExecutor`: a hex-digest `.bin` file it fabricated under that directory is replaced by a minimal PDF; a real file
    /// is never altered or excused.
    static func prepare(settings: String?, inputs: [Asset], path: String, mockBlobDirectory: URL? = nil) throws -> Prepared {
        func refuse(_ message: String) -> FlowError { FlowError.stageFailure(row: path, message: message) }
        do { try parseSettings(settings) } catch let e as StagePrintSettingsError {
            throw refuse("Stage Print \(e.detail)")
        }
        let items = inputs.flatMap(\.items)
        guard !items.isEmpty else { throw refuse("Stage Print got nothing to print -- feed it a PDF file or an image") }
        guard items.count == 1 else {
            throw refuse("Stage Print got \(items.count) items -- it prints one file at a time; wrap it in `<each>`")
        }
        let item = items[0]
        guard item.kind == .file || item.kind == .image, let url = item.path else {
            throw refuse("Stage Print takes a PDF file or an image, not a \(item.kind.rawValue)")
        }
        let name = url.lastPathComponent
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue,
              var data = try? Data(contentsOf: url) else {
            throw refuse("Stage Print can't read \(name) -- check the file exists")
        }
        if let mock = mockBlobDirectory, url.deletingLastPathComponent().standardizedFileURL == mock.standardizedFileURL,
           url.pathExtension == "bin", item.kind == .file {
            data = minimalPDF(seed: data)
        }
        guard !data.isEmpty else { throw refuse("Stage Print can't print \(name) -- the file is empty") }
        let mime: String, ext: String
        if data.starts(with: Data("%PDF-".utf8)) {
            (mime, ext) = ("application/pdf", ".pdf")  // by the bytes, never the name
        } else {
            guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) > 0,
                  let type = CGImageSourceGetType(source) as String? else {
                throw refuse("Stage Print can't print \(name) -- it takes a PDF or an image (PNG, JPEG, GIF, TIFF, BMP); this file is neither")
            }
            guard let known = printableImages[type] else {
                let label = type.split(separator: ".").last.map { $0.uppercased() } ?? "unknown-format"
                throw refuse("Stage Print can't print a \(label) image (\(name)) -- the printing system takes PNG, JPEG, GIF, TIFF and BMP; convert it first (`Convert` format=png)")
            }
            (mime, ext) = known
        }
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        // Q259: the id's digest is over the date/ID-normalised bytes; `digest` (the entry's `sha256`) stays the stored bytes' own hash.
        let idDigest = SHA256.hash(data: pdfDigestBytes(data)).map { String(format: "%02x", $0) }.joined()
        let id = String(SHA256.hash(data: Data("Stage Print|\(idDigest)".utf8)).map { String(format: "%02x", $0) }.joined().prefix(10))
        // ER-FIX-7: the label is the file's real name, never a cache blob's; `id` above never reads it.
        var label = item.originalName ?? name
        if label.hasSuffix(".bin") { label = "\(digest.prefix(12))\(ext)" }
        return Prepared(data: data, name: label, mime: mime, ext: ext, digest: digest, id: id)
    }

    /// Queue one `print` entry (nothing prints). The same bytes staged again, while still pending, change nothing.
    static func stage(settings: String?, inputs: [Asset], path: String, workspace: FlowWorkspace, flowID: String,
                      now: @Sendable () -> Date = { Date() }) throws -> (status: String, id: String, summary: String) {
        let prepared = try prepare(settings: settings, inputs: inputs, path: path)
        let outbox = OutboxStore.directory(workspace: workspace, flowID: flowID)
        try write(prepared, into: outbox, now: now())
        return (prepared.status, prepared.id, prepared.summary)
    }

    static func write(_ p: Prepared, into outbox: URL, now: Date) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: outbox, withIntermediateDirectories: true)
        let entryURL = outbox.appendingPathComponent("\(p.id).json")
        guard !fm.fileExists(atPath: entryURL.path) else { return }
        let filesDir = outbox.appendingPathComponent("files", isDirectory: true)
        try fm.createDirectory(at: filesDir, withIntermediateDirectories: true)
        let stored = "\(p.id)\(p.ext)"
        try p.data.write(to: filesDir.appendingPathComponent(stored))
        let j = FlowParity.asciiJSON
        let json = """
        {
          "id": \(j(p.id)),
          "kind": "print",
          "destination": \(j(destination)),
          "text": \(j(p.name)),
          "file_name": \(j(p.name)),
          "mime": \(j(p.mime)),
          "sha256": \(j(p.digest)),
          "stored_file": \(j(stored)),
          "status": "pending",
          "staged_at": \(j(OutboxStore.timestamp(now)))
        }
        """
        try json.write(to: entryURL, atomically: true, encoding: .utf8)
    }
}

// MARK: - Approval

/// What the person did at the print panel.
nonisolated enum PrintPanelOutcome: Sendable, Equatable {
    case completed
    case cancelled
    case failed(String)
}

/// The seam between the outbox and the system print panel. The app's one conformer opens the standard panel (`SystemPrintPresenter`);
/// a test injects a fake that answers `completed` / `cancelled` / `failed` without opening anything.
protocol PrintPresenting {
    @MainActor func present(fileURL: URL, mime: String, title: String) async -> PrintPanelOutcome
}

nonisolated enum PrintApprovalResult: Equatable {
    case committed
    /// The person cancelled the panel: the entry stays pending, with no error recorded.
    case cancelled
    /// The print failed (or the entry could not be printed): the entry stays pending and records `last_error`.
    case failed(String)
    case notFound
}

/// Approving a `print` entry. The entry is marked committed ONLY when the presenter reports `.completed`.
enum PrintOutbox {

    /// The entry's own stored copy, recomputed under `<outbox>/files/` from a bare file name, so a hand-edited entry can never
    /// point the print anywhere else. `nil` when the name is not bare or the copy is gone.
    static func storedCopy(of entry: [String: Any], in outbox: URL) -> URL? {
        guard let name = entry["stored_file"] as? String, name == (name as NSString).lastPathComponent,
              !name.isEmpty, name != ".", name != ".." else { return nil }
        let url = outbox.appendingPathComponent("files", isDirectory: true).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    @MainActor
    static func approve(id: String, outbox: URL, presenter: any PrintPresenting,
                        now: @Sendable () -> Date = { Date() }) async -> PrintApprovalResult {
        let entryURL = outbox.appendingPathComponent("\(id).json")
        guard id == (id as NSString).lastPathComponent, let data = try? Data(contentsOf: entryURL),
              var entry = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              entry["kind"] as? String == "print", entry["status"] as? String == "pending" else { return .notFound }
        let attempted = OutboxStore.timestamp(now())
        guard let stored = storedCopy(of: entry, in: outbox) else {
            let message = "the entry's stored copy of the file is missing -- discard this entry and stage it again"
            record(&entry, error: message, at: attempted, to: entryURL)
            return .failed(message)
        }
        let title = (entry["file_name"] as? String) ?? stored.lastPathComponent
        switch await presenter.present(fileURL: stored, mime: (entry["mime"] as? String) ?? "", title: title) {
        case .cancelled:
            return .cancelled
        case .failed(let message):
            record(&entry, error: message, at: attempted, to: entryURL)
            return .failed(message)
        case .completed:
            entry["status"] = "committed"
            entry["committed_at"] = attempted
            entry["last_error"] = nil
            let committedDir = outbox.appendingPathComponent("committed", isDirectory: true)
            do {
                try FileManager.default.createDirectory(at: committedDir, withIntermediateDirectories: true)
                try encode(entry).write(to: committedDir.appendingPathComponent("\(id).json"), options: .atomic)
                try FileManager.default.removeItem(at: entryURL)
            } catch {
                // The paper is out; failing to file the entry must not reprint it as a surprise, so say so and keep it pending.
                let message = "printed, but the entry could not be filed (\(error.localizedDescription))"
                record(&entry, error: message, at: attempted, to: entryURL)
                return .failed(message)
            }
            return .committed
        }
    }

    /// A failed attempt: still `pending`, with `last_error`, `last_attempt_at` and `attempts` in its JSON.
    private static func record(_ entry: inout [String: Any], error: String, at stamp: String, to url: URL) {
        entry["last_error"] = error
        entry["last_attempt_at"] = stamp
        entry["attempts"] = ((entry["attempts"] as? Int) ?? 0) + 1
        try? encode(entry).write(to: url, options: .atomic)
    }

    private static func encode(_ entry: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: entry, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
