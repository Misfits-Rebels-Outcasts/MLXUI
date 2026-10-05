//
//  TextEdit.swift
//  MLXUI — shared code (ships in BOTH editions).
//
//  The exact-string edit engine behind `edit_file`. Same reasoning as `PathPolicy`: the
//  logic both editions share has to live where `MLXUITests` can reach it, because that
//  target cannot compile `Apps/Direct/`. The two tools (`Apps/Direct/FileTools.swift`,
//  `MLXUI/Core/Agent/ScopedFileTools.swift`) are thin wrappers around this.
//
//  Two layers:
//
//    * `replace(in:old:new:replaceAll:)` — pure `String` → `Outcome`, no I/O.
//    * `editFile(at:old:new:replaceAll:)` — the file round-trip: existence / size /
//      encoding checks, BOM handling, CRLF adaptation, atomic write, permission
//      preservation, and the model-facing result string.
//
//  Matching is exact and literal (D2): no Unicode canonical equivalence, no case or
//  whitespace folding. The implementation searches **UTF-8 bytes**, which is exactly
//  what the model sent, so a precomposed "é" never matches a decomposed "e\u{301}".
//  (Swift's `String ==` and `String.range(of:)` compare by canonical equivalence, so
//  they are deliberately not used here.)
//
//  Nothing here touches the network, and `editFile` never throws: every failure is a
//  returned string the model can read and retry, like the other agent tools.
//

import Foundation

nonisolated enum TextEdit {

    /// Refuse anything larger than this instead of loading it into memory (D7).
    static let maxFileBytes = 1_024 * 1_024

    /// How many numbered lines of context the D8 result shows, and the hard byte cap on
    /// that snippet.
    static let maxSnippetLines = 7
    static let maxSnippetBytes = 1_024

    /// How many replacement line numbers the D8 sentence itself lists. `replace_all` on
    /// a one-character `old_string` can match thousands of times, and the result goes
    /// back into the model's context, so the list is summarised rather than clipped.
    static let maxReportedLines = 10

    /// How many match lines the D4 "matches N places" error names: the first five.
    static let maxAmbiguousLines = 5

    /// Layer-1 result. `lines` is 1-based and refers to the **original** text. `lines` holds
    /// at most `maxReportedLines` entries — the first replacements, which is all the D8
    /// sentence prints — while `count` is the true number of replacements (EF-A F1).
    enum Outcome: Equatable, Sendable {
        /// `newText` is the whole text after the change.
        case edited(newText: String, count: Int, lines: [Int])
        /// Nothing matched. When a whitespace-tolerant comparison would have matched,
        /// this is the 1-based line where it starts (D3).
        case notFound(indentTolerantLine: Int?)
        /// More than one match and `replace_all` was false. `lines` holds the first five
        /// match lines (D4), `count` the true total.
        case ambiguous(count: Int, lines: [Int])
        /// Degenerate input the caller can never apply (D5). The string is model-facing.
        case invalid(String)
    }

    // MARK: - Layer 1: pure replacement

    /// Replace `old` with `new` in `text`. Non-overlapping, left to right (so `"aa"` in
    /// `"aaa"` is one match, not two). Pure and deterministic.
    static func replace(in text: String, old: String, new: String, replaceAll: Bool) -> Outcome {
        let textBytes = Array(text.utf8)
        let oldBytes = Array(old.utf8)
        let newBytes = Array(new.utf8)

        guard !oldBytes.isEmpty else {
            return .invalid("Error: 'old_string' is empty. Use write_file to create a file.")
        }
        // Byte-identical, not `old == new`: Swift's `String ==` folds canonical
        // equivalence, which would call a precomposed/decomposed pair "identical".
        guard oldBytes != newBytes else {
            return .invalid("Error: 'old_string' and 'new_string' are identical; nothing to change.")
        }

        var offsets: [Int] = []
        if oldBytes.count <= textBytes.count {
            var index = 0
            let last = textBytes.count - oldBytes.count
            while index <= last {
                if textBytes[index..<(index + oldBytes.count)].elementsEqual(oldBytes) {
                    offsets.append(index)
                    index += oldBytes.count
                } else {
                    index += 1
                }
            }
        }

        guard !offsets.isEmpty else {
            return .notFound(indentTolerantLine: indentTolerantLine(in: text, old: old))
        }
        guard offsets.count == 1 || replaceAll else {
            return .ambiguous(
                count: offsets.count,
                lines: lineNumbers(for: offsets.prefix(maxAmbiguousLines), in: textBytes))
        }

        let chosen = replaceAll ? offsets : [offsets[0]]
        var output: [UInt8] = []
        output.reserveCapacity(textBytes.count + chosen.count * max(0, newBytes.count - oldBytes.count))
        var cursor = 0
        for offset in chosen {
            output.append(contentsOf: textBytes[cursor..<offset])
            output.append(contentsOf: newBytes)
            cursor = offset + oldBytes.count
        }
        output.append(contentsOf: textBytes[cursor...])
        return .edited(
            newText: String(decoding: output, as: UTF8.self),
            count: offsets.count,
            lines: lineNumbers(for: offsets.prefix(maxReportedLines), in: textBytes))
    }

    // MARK: - Layer 2: file round-trip

    /// Edit the UTF-8 file at `url`. Returns the model-facing result: the D8 success
    /// sentence plus a numbered context snippet, or an `"Error: …"` sentence. Never
    /// throws and never modifies the file on a non-`.edited` outcome.
    ///
    /// A symbolic link is followed once, at the top: the **target** is the file that gets
    /// edited and the link stays a link (N1). `PathPolicy`/`FolderGrants` have already
    /// resolved the same link before allowing the path, so this changes no policy.
    static func editFile(at url: URL, old: String, new: String, replaceAll: Bool) -> String {
        let url = url.resolvingSymlinksInPath()

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return "Error: \(url.path) does not exist. Use write_file to create it."
        }
        guard !isDirectory.boolValue else {
            return "Error: \(url.path) is a directory, not a file."
        }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              attributes[.type] as? FileAttributeType == .typeRegular else {
            return "Error: \(url.path) is not a regular file."
        }
        let size = (attributes[.size] as? NSNumber)?.intValue ?? 0
        guard size <= maxFileBytes else {
            return "Error: \(url.lastPathComponent) is \(size) bytes; edit_file refuses files over "
                + "1 MB. Use read_file and write_file instead."
        }

        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            return "Error: could not read \(url.path): \(error.localizedDescription)"
        }

        // D6: a leading UTF-8 BOM is remembered and re-attached byte-for-byte.
        let bom: [UInt8] = [0xEF, 0xBB, 0xBF]
        let hasBOM = data.count >= bom.count && data.prefix(bom.count).elementsEqual(bom)
        let body = hasBOM ? Data(data.dropFirst(bom.count)) : data
        guard let text = String(data: body, encoding: .utf8) else {
            return "Error: \(url.lastPathComponent) is not UTF-8 text (\(data.count) bytes)."
        }

        // D6: the text the model sent wins, exactly as sent. Only when that matches nothing,
        // the file contains a CRLF pair and `old` carries no CR of its own do we retry with
        // both strings adapted to CRLF. So a file with mixed endings still takes an exact LF
        // match, and a pure-CRLF file still takes LF-only input (F2).
        var outcome = replace(in: text, old: old, new: new, replaceAll: replaceAll)
        if case .notFound = outcome, text.contains("\r\n"), !old.contains("\r") {
            // A `new` that already carries a CR is left alone, so the fallback can never
            // write "\r\r\n".
            let adaptedOld = old.replacingOccurrences(of: "\n", with: "\r\n")
            let adaptedNew = new.contains("\r")
                ? new
                : new.replacingOccurrences(of: "\n", with: "\r\n")
            outcome = replace(in: text, old: adaptedOld, new: adaptedNew, replaceAll: replaceAll)
        }

        switch outcome {
        case .edited(let newText, let count, let lines):
            var output = Data()
            if hasBOM { output.append(contentsOf: bom) }
            output.append(Data(newText.utf8))
            do {
                try output.write(to: url, options: .atomic)
            } catch {
                return "Error: could not write \(url.path): \(error.localizedDescription)"
            }
            // D12: the atomic write swaps in a new file, so re-apply the mode we read.
            if let permissions = attributes[.posixPermissions] as? NSNumber {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: permissions],
                    ofItemAtPath: url.path)
            }
            return resultText(for: newText, count: count, lines: lines, path: url.path)

        case .notFound(let indentTolerantLine):
            if let indentTolerantLine {
                return "Error: old_string not found. A match ignoring indentation exists at line "
                    + "\(indentTolerantLine). Copy the exact text from read_file."
            }
            return "Error: old_string not found in \(url.path). Copy the exact text from read_file."

        case .ambiguous(let count, let lines):
            let list = lines.map(String.init).joined(separator: ", ")
            return "Error: old_string matches \(count) places (lines \(list)). Add surrounding "
                + "lines to make it unique, or set replace_all."

        case .invalid(let message):
            return message
        }
    }

    // MARK: - Helpers

    /// D3 hint: the first 1-based line where the file matches `old` once every line is
    /// trimmed. `nil` when even that would not match. Compared as UTF-8 bytes for the
    /// same literal reason as `replace`: a canonical-equivalence difference is not an
    /// indentation difference, and the message must not claim otherwise.
    private static func indentTolerantLine(in text: String, old: String) -> Int? {
        func trimmedLines(of value: String) -> [[UInt8]] {
            value.components(separatedBy: "\n")
                .map { Array($0.trimmingCharacters(in: .whitespacesAndNewlines).utf8) }
        }
        // N2: models often end `old_string` with the line's newline. A trailing empty line is
        // not something the file has to contain, so it is dropped before matching.
        var oldLines = trimmedLines(of: old)
        while oldLines.last?.isEmpty == true { oldLines.removeLast() }
        guard !oldLines.isEmpty else { return nil }
        let fileLines = trimmedLines(of: text)
        guard fileLines.count >= oldLines.count else { return nil }
        for start in 0...(fileLines.count - oldLines.count) {
            if Array(fileLines[start..<(start + oldLines.count)]) == oldLines { return start + 1 }
        }
        return nil
    }

    /// 1-based lines for `offsets` (ascending, inside `bytes`) in ONE forward pass: the cursor
    /// only moves forward, so the call is O(bytes + offsets) however many offsets there are.
    /// Callers pass just the offsets they report; the old per-offset rescan from byte 0 made
    /// `replace_all` quadratic (EF-A F1). LF bytes are counted in the same view that was
    /// searched, so a `\r\n` pair counts once (the `Character` view would not).
    private static func lineNumbers<S: Sequence>(for offsets: S, in bytes: [UInt8]) -> [Int]
    where S.Element == Int {
        var lines: [Int] = []
        var line = 1
        var cursor = 0
        for offset in offsets {
            while cursor < offset {
                if bytes[cursor] == 0x0A { line += 1 }
                cursor += 1
            }
            lines.append(line)
        }
        return lines
    }

    /// D8: the success sentence, then the numbered lines around the first edit as they
    /// read *after* the change. The sentence's count is the true replacement count; `lines`
    /// only carries the first `maxReportedLines` of them.
    private static func resultText(for text: String, count: Int, lines: [Int], path: String) -> String {
        let header: String
        if count == 1 {
            header = "Edited \(path): replaced 1 occurrence (line \(lines.first ?? 1))."
        } else if count <= maxReportedLines {
            let list = lines.map(String.init).joined(separator: ", ")
            header = "Edited \(path): replaced \(count) occurrences (lines \(list))."
        } else {
            let list = lines.map(String.init).joined(separator: ", ")
            header = "Edited \(path): replaced \(count) occurrences (lines \(list); "
                + "\(count - maxReportedLines) more)."
        }
        return header + "\n" + snippet(of: text, around: lines.first ?? 1)
    }

    private static let clipNotice = "\n… (snippet clipped at 1 KB)"

    /// Up to `maxSnippetLines` numbered lines around `line`, capped at `maxSnippetBytes`
    /// UTF-8 bytes with an explicit notice when the cap bites (never clip silently). A
    /// trailing CR is dropped from display only; the file on disk keeps it.
    private static func snippet(of text: String, around line: Int) -> String {
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }   // a trailing newline is not a line
        guard !lines.isEmpty else { return "" }

        let editIndex = min(max(line - 1, 0), lines.count - 1)
        var start = max(0, editIndex - maxSnippetLines / 2)
        start = min(start, max(0, lines.count - maxSnippetLines))
        let end = min(lines.count, start + maxSnippetLines)

        let numbered = (start..<end).map { index -> String in
            let raw = lines[index]
            let display = raw.hasSuffix("\r") ? String(raw.dropLast()) : raw
            return "\(index + 1): \(display)"
        }.joined(separator: "\n")

        let clipped = maxSnippetBytes - clipNotice.utf8.count - 4
        guard numbered.utf8.count > maxSnippetBytes else { return numbered }
        return String(decoding: Array(numbered.utf8.prefix(clipped)), as: UTF8.self) + clipNotice
    }
}
