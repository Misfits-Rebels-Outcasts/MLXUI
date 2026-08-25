import Foundation

/// CFM-R12-FIX-11 — small parity helpers shared across the tools.
nonisolated enum FlowParity {
    /// Python's `json.dumps(str, ensure_ascii=True)`: a JSON-escaped, double-quoted string
    /// with every non-ASCII character written as `\uXXXX` (Swift's JSONSerialization emits
    /// raw UTF-8, so "byte-identical" claims about files the Python writes were wrong).
    static func asciiJSON(_ s: String) -> String {
        let base = String(data: try! JSONSerialization.data(withJSONObject: s,
                                                            options: [.fragmentsAllowed]),
                          encoding: .utf8)!
        var out = ""
        for scalar in base.unicodeScalars {
            if scalar.value < 128 {
                out.unicodeScalars.append(scalar)
            } else {
                out += String(format: "\\u%04x", scalar.value)
            }
        }
        return out
    }

    /// The Python `_trash`: move a file to a sibling `.trash/` with a timestamp instead of
    /// deleting it (Spec §14.1 — "no task … deletes without trash"). Used before any
    /// overwrite the `Save *`/`Store Index` tools perform.
    static func moveToTrash(_ url: URL) throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return }
        let trash = url.deletingLastPathComponent().appendingPathComponent(".trash", isDirectory: true)
        try fm.createDirectory(at: trash, withIntermediateDirectories: true)
        let stamp = String(Int(Date().timeIntervalSince1970 * 1000))
        let dest = trash.appendingPathComponent("\(url.lastPathComponent).\(stamp)")
        try fm.moveItem(at: url, to: dest)
    }
}
