import Foundation

/// One stretch of one speaker talking. `speaker` is a 0-based number; the displayed label and the CSV column are 1-based
/// ("Speaker 1"; implementer's call, pending owner confirmation). Straight from the model it is one of 4 internal slots, and
/// **the slots are not numbered in order of appearance** (the real run on the two-speaker clip used slots 0 and 2), so
/// ``SpeakerSegments/relabeledByFirstAppearance(_:)`` renumbers them 0, 1, 2, 3 before anything is shown or compared.
nonisolated struct SpeakerSegment: Sendable, Equatable, Hashable {
    let speaker: Int
    let start: Double
    let end: Double

    var duration: Double { end - start }
    var label: String { "Speaker \(speaker + 1)" }
}

/// The pure half of the diarization feature: deterministic clean-up, the text and CSV renderings the Run UI copies, the CSV
/// reader that brings the stage's output back into segments, and the timeline's lanes and axis. No MLX, no UI.
nonisolated enum SpeakerSegments {

    // MARK: Clean-up

    /// Drops segments that are not finite or have no length, clamps the rest to `0...duration` when a duration is given,
    /// merges same-speaker segments that touch or overlap (or sit within `mergeGap` seconds), and sorts by start, then end,
    /// then speaker. Deterministic whatever order the input arrives in. With `mergeGap == 0` the library's own output is
    /// unchanged (it only ever emits a speaker's segments separated by an inactive frame).
    static func normalized(_ segments: [SpeakerSegment], duration: Double? = nil, mergeGap: Double = 0) -> [SpeakerSegment] {
        var cleaned: [SpeakerSegment] = []
        for s in segments where s.start.isFinite && s.end.isFinite && s.speaker >= 0 {
            let start = max(0, s.start)
            var end = s.end
            if let duration { end = min(end, duration) }
            guard end > start else { continue }
            cleaned.append(SpeakerSegment(speaker: s.speaker, start: start, end: end))
        }
        var merged: [SpeakerSegment] = []
        let bySpeaker = Dictionary(grouping: cleaned, by: \.speaker)
        for speaker in bySpeaker.keys.sorted() {
            var run: SpeakerSegment?
            for s in (bySpeaker[speaker] ?? []).sorted(by: { ($0.start, $0.end) < ($1.start, $1.end) }) {
                if let current = run, s.start - current.end <= max(0, mergeGap) {
                    run = SpeakerSegment(speaker: speaker, start: current.start, end: max(current.end, s.end))
                } else {
                    if let current = run { merged.append(current) }
                    run = s
                }
            }
            if let current = run { merged.append(current) }
        }
        return merged.sorted { ($0.start, $0.end, $0.speaker) < ($1.start, $1.end, $1.speaker) }
    }

    /// Renumbers speakers 0, 1, 2, ... in order of first appearance (the earliest-starting segment of each speaker decides),
    /// keeping everything else. `segments` should already be sorted (``normalized(_:duration:mergeGap:)``); the first time a
    /// speaker number is met in the given order fixes its new number. Lead's call, pending owner confirmation.
    static func relabeledByFirstAppearance(_ segments: [SpeakerSegment]) -> [SpeakerSegment] {
        var mapping: [Int: Int] = [:]
        return segments.map { s in
            if mapping[s.speaker] == nil { mapping[s.speaker] = mapping.count }
            return SpeakerSegment(speaker: mapping[s.speaker] ?? s.speaker, start: s.start, end: s.end)
        }
    }

    /// Two segments of one speaker closer than this (seconds) are one stretch: the pieces a streaming chunk edge leaves touch
    /// exactly, give or take float rounding. Far below the model's 80 ms frame, so no real pause is ever joined.
    static let joinGap = 0.001

    /// What the Run UI shows, the CSV carries and the golden holds: ``normalized(_:duration:mergeGap:)`` (joining pieces that
    /// touch), then ``relabeledByFirstAppearance(_:)``. Idempotent, except that two different speakers with exactly the same start and end
    /// (a tie the sort breaks by speaker number) can swap places on a second pass; the app cleans once.
    static func cleaned(_ segments: [SpeakerSegment], duration: Double? = nil) -> [SpeakerSegment] {
        relabeledByFirstAppearance(normalized(segments, duration: duration, mergeGap: joinGap))
    }

    /// The speakers that have at least one segment, in number order.
    static func speakers(in segments: [SpeakerSegment]) -> [Int] {
        Set(segments.map(\.speaker)).sorted()
    }

    /// One lane per speaker that spoke, in slot order, each lane's segments sorted by start.
    static func lanes(_ segments: [SpeakerSegment]) -> [(speaker: Int, segments: [SpeakerSegment])] {
        speakers(in: segments).map { speaker in
            (speaker, segments.filter { $0.speaker == speaker }.sorted { $0.start < $1.start })
        }
    }

    // MARK: Renderings

    /// `m:ss.mmm` (or `h:mm:ss.mmm` from one hour), the same rounding as the CSV's milliseconds.
    static func clock(_ seconds: Double) -> String {
        let totalMs = Int((max(0, seconds) * 1000).rounded())
        let ms = totalMs % 1000
        let totalSeconds = totalMs / 1000
        let s = totalSeconds % 60
        let m = (totalSeconds / 60) % 60
        let h = totalSeconds / 3600
        return h > 0 ? String(format: "%d:%02d:%02d.%03d", h, m, s, ms) : String(format: "%d:%02d.%03d", m, s, ms)
    }

    /// One line per segment: `Speaker 1  0:00.500 – 0:06.297`. Empty input is the empty string.
    static func text(_ segments: [SpeakerSegment]) -> String {
        segments.map { "\($0.label)  \(clock($0.start)) – \(clock($0.end))" }.joined(separator: "\n")
    }

    static let csvHeader = "speaker,start,end"

    /// `speaker,start,end` with a header row; speaker is the 1-based number, start and end are seconds with three decimals and
    /// a `.` separator whatever the locale. Lines end in `\n`, including the last.
    static func csv(_ segments: [SpeakerSegment]) -> String {
        var lines = [csvHeader]
        for s in segments {
            lines.append("\(s.speaker + 1),\(seconds(s.start)),\(seconds(s.end))")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Seconds to three decimals, always with a `.` (`String(format:)` without a locale is not localized).
    static func seconds(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    /// The inverse of ``csv(_:)`` for what this app writes. Rejects a missing or different header, a row without three
    /// fields, a non-numeric field, a speaker below 1, or a segment that does not end after it starts.
    static func parseCSV(_ csv: String) throws -> [SpeakerSegment] {
        let lines = csv.split(whereSeparator: \.isNewline).map(String.init)
        guard lines.first == csvHeader else { throw CSVError.badHeader }
        var out: [SpeakerSegment] = []
        for (offset, line) in lines.dropFirst().enumerated() {
            let fields = line.split(separator: ",", omittingEmptySubsequences: false)
            guard fields.count == 3, let speaker = Int(fields[0]), speaker >= 1,
                  let start = Double(fields[1]), let end = Double(fields[2]),
                  start.isFinite, end.isFinite, end > start else { throw CSVError.badRow(offset + 2) }
            out.append(SpeakerSegment(speaker: speaker - 1, start: start, end: end))
        }
        return out
    }

    enum CSVError: Error, Equatable { case badHeader, badRow(Int) }

    // MARK: Timeline axis

    /// Tick positions (seconds) for a time axis over `0...duration`: the smallest "nice" step (1, 2, 5, 10, 15, 30, 60, 120,
    /// 300, 600, 1800, 3600 s) that gives at most `maxTicks` ticks, starting at 0. A non-positive or non-finite duration
    /// gives just `[0]`.
    static func axisTicks(duration: Double, maxTicks: Int = 8) -> [Double] {
        guard duration.isFinite, duration > 0, maxTicks > 1 else { return [0] }
        let steps: [Double] = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600]
        let step = steps.first { floor(duration / $0) + 1 <= Double(maxTicks) } ?? steps[steps.count - 1]
        return stride(from: 0.0, through: duration, by: step).map { $0 }
    }
}
