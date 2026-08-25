import Foundation
import CryptoKit

/// CFM-R12-8 — staged effects (`Stage Send` / `Stage Post`), the `outbox.py` port.
///
/// **Nothing here sends.** Per Spec §14.1 ("no task sends, posts, … without trash"), a
/// staged row queues one JSON entry into the flow's visible outbox directory and stops —
/// there is no transport anywhere in the app (or the reference runtime). The entry schema,
/// content-digest id (SPEC-Q103), and status sentence match the Python byte for byte.
nonisolated enum OutboxStore {

    /// One pending entry (what the flow detail's Outbox disclosure lists).
    struct Entry: Identifiable, Hashable {
        let id: String
        let kind: String
        let destination: String
        let text: String
        let stagedAt: String
    }

    /// The per-flow outbox directory: `flows/<flow-id>/outbox/`.
    static func directory(workspace: FlowWorkspace, flowID: String) -> URL {
        workspace.directory(for: flowID).appendingPathComponent("outbox", isDirectory: true)
    }

    /// Stage one entry. Returns `(statusSentence, id, summary)` — the status is the row's
    /// output, the id + summary feed the interpreter's `effect_staged` event.
    static func stage(row: Row, inputs: [Asset], kind: String,
                      workspace: FlowWorkspace, flowID: String,
                      now: @Sendable () -> Date = { Date() }) throws
        -> (status: String, id: String, summary: String) {
        let fm = FileManager.default
        let outboxDir = directory(workspace: workspace, flowID: flowID)
        try fm.createDirectory(at: outboxDir, withIntermediateDirectories: true)

        let destination = FlowSettings(row.settings).firstBare() ?? "outbox"
        let text = inputs.first?.items.first?.value ?? ""
        let id = freeID(in: outboxDir, basis: "\(row.task ?? "")|\(row.settings ?? "")|\(text)")
        let summary = "to \(destination): \(preview(text))"

        let date = now()
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var stagedAt = formatter.string(from: date)
        if stagedAt.hasSuffix("+00:00") { stagedAt = String(stagedAt.dropLast(6)) + "Z" }

        let json = entryJSON(id: id, kind: kind, destination: destination, text: text,
                             stagedAt: stagedAt)
        try json.write(to: outboxDir.appendingPathComponent("\(id).json"),
                       atomically: true, encoding: .utf8)
        return ("queued to \(destination) (\(id))", id, summary)
    }

    /// The pending entries, oldest first (the Outbox disclosure's source).
    static func entries(workspace: FlowWorkspace, flowID: String) -> [Entry] {
        let fm = FileManager.default
        let dir = directory(workspace: workspace, flowID: flowID)
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil,
                                                      options: [.skipsHiddenFiles]) else { return [] }
        return files
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { url -> Entry? in
                guard let data = try? Data(contentsOf: url),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
                guard let id = json["id"] as? String, let kind = json["kind"] as? String,
                      let destination = json["destination"] as? String,
                      let text = json["text"] as? String, let at = json["staged_at"] as? String else { return nil }
                return Entry(id: id, kind: kind, destination: destination, text: text, stagedAt: at)
            }
    }

    // MARK: - Helpers

    /// SPEC-Q103: content-digest id — sha256 of the staging basis, first 10 hex chars, with a
    /// numeric suffix only on an actual on-disk collision.
    private static func freeID(in dir: URL, basis: String) -> String {
        let digest = SHA256.hash(data: Data(basis.utf8)).map { String(format: "%02x", $0) }.joined()
        let base = String(digest.prefix(10))
        var candidate = base
        var n = 2
        while FileManager.default.fileExists(atPath: dir.appendingPathComponent("\(candidate).json").path) {
            candidate = "\(base)-\(n)"
            n += 1
        }
        return candidate
    }

    private static func preview(_ text: String) -> String {
        // FIX-11: Python's `text[:60]` counts *code points*, not graphemes.
        let scalars = Array(text.unicodeScalars)
        return scalars.count <= 60 ? text : String(String.UnicodeScalarView(scalars.prefix(60))) + "…"
    }

    /// The Python `json.dumps(entry, indent=2)` shape, field order and all.
    private static func entryJSON(id: String, kind: String, destination: String,
                                  text: String, stagedAt: String) -> String {
        let jid = jstr(id), jkind = jstr(kind), jdest = jstr(destination),
            jtext = jstr(text), jat = jstr(stagedAt)
        return """
        {
          "id": \(jid),
          "kind": \(jkind),
          "destination": \(jdest),
          "text": \(jtext),
          "status": "pending",
          "staged_at": \(jat)
        }
        """
    }

    /// A JSON-escaped, double-quoted string literal — the Python's `json.dumps(str,
    /// ensure_ascii=True)` (CFM-R12-FIX-11), so a non-ASCII post still matches byte for byte.
    private static func jstr(_ s: String) -> String {
        FlowParity.asciiJSON(s)
    }
}
