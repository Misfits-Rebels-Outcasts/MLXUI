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
    /// The pinned library's splitter drops punctuation, so on its output the sentence ends come from `sentenceEnds`
    /// (one mark per unit, from ``sentenceEnds(in:for:)``; a missing mark is false). A unit whose own text ends with the
    /// punctuation ends a cue too. A cue's text is the units joined by ``cueText(_:)``; the transcript's punctuation is
    /// not put back (implementer's call, pending owner confirmation).
    static func cues(_ units: [AlignedUnit],
                     sentenceEnds: [Bool] = [],
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
        for (index, unit) in units.enumerated() {
            let endsHere = endsASentence(unit.text) || (index < sentenceEnds.count && sentenceEnds[index])
            if current.isEmpty {
                current = [unit]
                if endsHere { flush() }
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
            if endsHere { flush() }
        }
        flush()
        return cues
    }

    /// How far past the expected place ``sentenceEnds(in:for:)`` looks for a unit, in letters and digits, so one unit that
    /// isn't in the transcript can't pull the walk to a far-away match (implementer's call, pending owner confirmation).
    static let unitSearchWindow = 32

    /// Per unit, whether the transcript has sentence-ending punctuation after it and before the next word or character.
    ///
    /// The library's unit is a transcript word with everything but letters, digits and apostrophes removed (CJK text one
    /// character at a time), so the units are looked for, in order, in the transcript reduced the same way. A unit that
    /// can't be found gets no mark and the walk carries on. A sentence end *inside* one unit gets no mark either (the
    /// space splitter joins `mill.Every`, or Japanese kana across a `。`, into one unit).
    ///
    /// A `.` between two digits is a decimal point, not a sentence end (the Chinese splitter makes `3.5` two units;
    /// implementer's call, pending owner confirmation). **Known limitation:** an abbreviation's period (`Dr. Smith`) ends
    /// a cue; there is no abbreviation list.
    static func sentenceEnds(in transcript: String, for units: [AlignedUnit]) -> [Bool] {
        let characters = Array(transcript)
        let kept = characters.indices.filter { isUnitCharacter(characters[$0]) }
        var cursor = 0   // the first of `kept` no earlier unit has used
        return units.map { unit in
            let target = unit.text.filter(isUnitCharacter).map { $0 }
            let lastStart = min(cursor + unitSearchWindow, kept.count - target.count)
            guard !target.isEmpty, cursor <= lastStart,
                  let start = (cursor...lastStart).first(where: { start in
                      target.indices.allSatisfy { characters[kept[start + $0]] == target[$0] }
                  }) else { return false }
            cursor = start + target.count
            let afterUnit = kept[cursor - 1] + 1
            let nextWord = cursor < kept.count ? kept[cursor] : characters.count
            return (afterUnit..<nextWord).contains { index in
                guard sentenceEnders.contains(characters[index]) else { return false }
                let isDecimalPoint = characters[index] == "." && characters[index - 1].isNumber
                    && index + 1 < characters.count && characters[index + 1].isNumber
                return !isDecimalPoint
            }
        }
    }

    /// The characters the library's splitter keeps (`ForceAlignProcessor.isKeptChar`).
    private static func isUnitCharacter(_ character: Character) -> Bool {
        character == "'" || character.isLetter || character.isNumber
    }

    /// `.`, `!`, `?`, `…` and their full-width forms.
    static let sentenceEnders = ".!?…。！？"

    /// A unit's text ends a sentence: its last character is one of ``sentenceEnders``.
    static func endsASentence(_ text: String) -> Bool {
        guard let last = text.last else { return false }
        return sentenceEnders.contains(last)
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

    /// The cues of a run: with its `transcript`, sentences end cues; without one, only a unit's own punctuation does.
    private static func cues(_ units: [AlignedUnit], transcript: String?) -> [Cue] {
        cues(units, sentenceEnds: transcript.map { sentenceEnds(in: $0, for: units) } ?? [])
    }

    /// SRT: numbered cues, `start --> end`, the cue's text, a blank line between cues. No units is the empty string.
    /// `transcript` is the text that was aligned (see ``sentenceEnds(in:for:)``).
    static func srt(_ units: [AlignedUnit], transcript: String? = nil) -> String {
        let cues = cues(units, transcript: transcript)
        guard !cues.isEmpty else { return "" }
        var blocks: [String] = []
        for (index, cue) in cues.enumerated() {
            blocks.append("\(index + 1)\n\(srtTimestamp(cue.start)) --> \(srtTimestamp(cue.end))\n\(cue.text)")
        }
        return blocks.joined(separator: "\n\n") + "\n"
    }

    /// WebVTT: the `WEBVTT` header, then cues with `start --> end` and the text, a blank line between cues. No units is
    /// just the header (a valid, empty VTT file).
    static func vtt(_ units: [AlignedUnit], transcript: String? = nil) -> String {
        let cues = cues(units, transcript: transcript)
        var blocks = ["WEBVTT"]
        for cue in cues {
            blocks.append("\(vttTimestamp(cue.start)) --> \(vttTimestamp(cue.end))\n\(cue.text)")
        }
        return blocks.joined(separator: "\n\n") + "\n"
    }
}
