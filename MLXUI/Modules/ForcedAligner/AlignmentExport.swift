import Foundation

/// FA-5 for issue #13: the pure half of forced alignment — the `text,start,end` table the Run screen shows and exports,
/// the CSV a flow row writes, and the SRT/VTT cues. No MLX, no UI; every function is deterministic and tested on its own.
///
/// The output contract is owner decision 4: one row per aligned unit in transcript order, times in seconds to three
/// decimals (the millisecond). CSV's header is exactly `text,start,end`; SRT uses `00:00:01,240` and VTT
/// `00:00:01.240`.
nonisolated enum AlignmentExport {
    static let columns = ["text", "start", "end"]
    static let csvHeader = "text,start,end"

    /// A cue's maximum length in characters and in seconds (decision 7): a cue ends at sentence-ending punctuation, or
    /// before adding the next unit would pass either limit.
    static let cueMaxCharacters = 42
    static let cueMaxSeconds = 6.0

    /// Seconds to three decimals with a `.` separator whatever the locale.
    static func seconds(_ value: Double) -> String {
        String(format: "%.3f", max(0, value))
    }

    // MARK: - CSV (the table, and a flow row's output)

    /// `text,start,end` with a header row, one line per unit, `\n` line endings including the last. The inverse is
    /// ``parseCSV(_:)``; a unit's text never contains a comma (the library's splitter keeps letters, numbers and
    /// apostrophes only), so a plain split is exact.
    static func csv(_ units: [AlignedUnit]) -> String {
        var lines = [csvHeader]
        for unit in units {
            lines.append("\(unit.text),\(seconds(unit.start)),\(seconds(unit.end))")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    enum CSVError: Error, Equatable {
        case badHeader
        case badRow(Int)
    }

    /// The inverse of ``csv(_:)`` — how the `Align` flow row reads a stage's output back into a table. Rejects a missing
    /// or different header, a row without three fields, a non-numeric or non-finite time, an empty text, or a unit that
    /// ends before it starts.
    static func parseCSV(_ csv: String) throws -> [AlignedUnit] {
        let lines = csv.split(whereSeparator: \.isNewline).map(String.init)
        guard lines.first == csvHeader else { throw CSVError.badHeader }
        var units: [AlignedUnit] = []
        for (offset, line) in lines.dropFirst().enumerated() {
            let fields = line.split(separator: ",", omittingEmptySubsequences: false)
            guard fields.count == 3,
                  !fields[0].isEmpty,
                  let start = Double(fields[1]), let end = Double(fields[2]),
                  start.isFinite, end.isFinite, end >= start else {
                throw CSVError.badRow(offset + 2)
            }
            units.append(AlignedUnit(text: String(fields[0]), start: start, end: end))
        }
        return units
    }

    /// Rows for `TableTool.tableItem(columns:rows:in:)` — the flow's file-backed `.table`, one row per unit. `start` and
    /// `end` are numbers (seconds), so a downstream `Table to Text`/`Read CSV` sees them as numbers, not strings.
    static func tableRows(_ units: [AlignedUnit]) -> [[Any?]] {
        units.map { unit in [unit.text as Any?, unit.start as Any?, unit.end as Any?] }
    }

    // MARK: - SRT / VTT (decision 7's cue rule)

    /// One subtitle cue: consecutive units grouped by ``cues(_:maxCharacters:maxSeconds:)``.
    struct Cue: Sendable, Equatable, Hashable {
        let start: Double
        let end: Double
        let text: String
    }

    /// Groups consecutive units into cues: a cue ends at sentence-ending punctuation, or **before** adding the next unit
    /// would push the cue past 42 characters or 6 seconds (decision 7). The cue's start is the first unit's start and
    /// its end the last unit's end. An empty result is no cues.
    ///
    /// The pinned library's splitter drops punctuation, so the punctuation branch is a no-op on its own output today —
    /// it is kept because the rule is part of the contract, and it is what a future unit text with punctuation needs.
    static func cues(_ units: [AlignedUnit],
                     maxCharacters: Int = cueMaxCharacters,
                     maxSeconds: Double = cueMaxSeconds) -> [Cue] {
        guard !units.isEmpty else { return [] }
        var cues: [Cue] = []
        var current: [AlignedUnit] = []
        func flush() {
            guard let first = current.first, let last = current.last else { return }
            cues.append(Cue(start: first.start, end: last.end, text: cueText(current)))
            current = []
        }
        for unit in units {
            if current.isEmpty {
                current = [unit]
                if endsASentence(unit.text) { flush() }
                continue
            }
            let candidate = cueText(current + [unit])
            let duration = unit.end - current[0].start
            if candidate.count > maxCharacters || duration > maxSeconds {
                flush()
                current = [unit]
            } else {
                current.append(unit)
            }
            if endsASentence(unit.text) { flush() }
        }
        flush()
        return cues
    }

    /// A unit's text ends a sentence: the last character is `.`, `!`, `?`, `…` or one of their full-width forms.
    static func endsASentence(_ text: String) -> Bool {
        guard let last = text.last else { return false }
        return ".!?…。！？".contains(last)
    }

    /// Units joined as one line: no space between CJK characters, a single space everywhere else.
    static func cueText(_ units: [AlignedUnit]) -> String {
        var out = ""
        for unit in units {
            if out.isEmpty {
                out = unit.text
            } else if let lastCharacter = out.last, let firstCharacter = unit.text.first,
                      isCJK(lastCharacter) || isCJK(firstCharacter) {
                out += unit.text
            } else {
                out += " " + unit.text
            }
        }
        return out
    }

    /// CJK ideographs, kana (Japanese writes without spaces) and Hangul syllables — no space is inserted beside these.
    static func isCJK(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first else { return false }
        let code = scalar.value
        return (0x3000...0x303F).contains(code)      // CJK symbols and punctuation
            || (0x3040...0x30FF).contains(code)      // Hiragana + Katakana
            || (0x3400...0x4DBF).contains(code)      // CJK Extension A
            || (0x4E00...0x9FFF).contains(code)      // CJK Unified Ideographs
            || (0xAC00...0xD7AF).contains(code)      // Hangul syllables
            || (0xF900...0xFAFF).contains(code)      // CJK Compatibility Ideographs
    }

    /// `00:00:01,240` — hours, minutes, seconds, milliseconds with the SRT comma.
    static func srtTimestamp(_ value: Double) -> String {
        let totalMs = Int((max(0, value) * 1000).rounded())
        let (ms, totalSeconds) = (totalMs % 1000, totalMs / 1000)
        let (s, m, h) = (totalSeconds % 60, (totalSeconds / 60) % 60, totalSeconds / 3600)
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, ms)
    }

    /// `00:00:01.240` — the same with a dot, for WebVTT.
    static func vttTimestamp(_ value: Double) -> String {
        let totalMs = Int((max(0, value) * 1000).rounded())
        let (ms, totalSeconds) = (totalMs % 1000, totalMs / 1000)
        let (s, m, h) = (totalSeconds % 60, (totalSeconds / 60) % 60, totalSeconds / 3600)
        return String(format: "%02d:%02d:%02d.%03d", h, m, s, ms)
    }

    /// SRT: numbered cues, `start --> end`, the cue's text, a blank line between cues. No units is the empty string.
    static func srt(_ units: [AlignedUnit]) -> String {
        let cues = cues(units)
        guard !cues.isEmpty else { return "" }
        var blocks: [String] = []
        for (index, cue) in cues.enumerated() {
            blocks.append("\(index + 1)\n\(srtTimestamp(cue.start)) --> \(srtTimestamp(cue.end))\n\(cue.text)")
        }
        return blocks.joined(separator: "\n\n") + "\n"
    }

    /// WebVTT: the `WEBVTT` header, then cues with `start --> end` and the text, a blank line between cues. No units is
    /// just the header (a valid, empty VTT file).
    static func vtt(_ units: [AlignedUnit]) -> String {
        let cues = cues(units)
        var blocks = ["WEBVTT"]
        for cue in cues {
            blocks.append("\(vttTimestamp(cue.start)) --> \(vttTimestamp(cue.end))\n\(cue.text)")
        }
        return blocks.joined(separator: "\n\n") + "\n"
    }
}
