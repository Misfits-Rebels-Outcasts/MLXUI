import Testing
import Foundation
@testable import MLXUI

/// MP4-3a: the parity scaffold for the Sortformer diarization golden. Everything here is TEST TARGET ONLY: the clip, its
/// ground truth and the golden live in `MLXUITests/ModelPack/` and are read through `#filePath`, never from the app bundle.
///
/// The golden (MP4-3b) is a real run of the installed model, recorded after `SpeakerSegments.cleaned` (pieces joined, speakers
/// renumbered by first appearance), so the comparison below is on cleaned segments. While `pending` is true the model-backed
/// comparison reports "golden pending" and does not fail. The pure tests (ground truth, comparison logic, golden file shape,
/// golden against ground truth) always run.
enum SortformerParity {
    struct Turn: Decodable, Equatable { let speaker: Int; let start: Double; let end: Double; let voice: String }
    struct Truth: Decodable { let clip: String; let sampleRate: Int; let durationSeconds: Double; let turns: [Turn] }
    struct GoldenSegment: Decodable, Equatable { let speaker: Int; let start: Double; let end: Double }
    struct Golden: Decodable {
        let pending: Bool
        let clip: String
        let threshold: Double
        let toleranceSeconds: Double?
        let notes: String
        let segments: [GoldenSegment]
    }

    nonisolated static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("ModelPack")
    nonisolated static var clipURL: URL { directory.appendingPathComponent("two_speakers.wav") }

    nonisolated static func truth() throws -> Truth {
        try JSONDecoder().decode(Truth.self, from: Data(contentsOf: directory.appendingPathComponent("two_speakers_truth.json")))
    }

    nonisolated static func golden() throws -> Golden {
        try JSONDecoder().decode(Golden.self, from: Data(contentsOf: directory.appendingPathComponent("two_speakers_golden.json")))
    }

    /// Differences between `actual` and the golden, one sentence each; empty = match. `actual` is compared in order after
    /// `SpeakerSegments.normalized` (callers pass `SpeakerSegments.cleaned` output, which this leaves unchanged); speaker
    /// numbers must be equal, starts and ends within `tolerance` seconds.
    nonisolated static func differences(actual: [SpeakerSegment], golden: [GoldenSegment], tolerance: Double) -> [String] {
        let sorted = SpeakerSegments.normalized(actual)
        var out: [String] = []
        if sorted.count != golden.count { out.append("segment count \(sorted.count), golden \(golden.count)") }
        for (i, (a, g)) in zip(sorted, golden).enumerated() {
            if a.speaker != g.speaker { out.append("segment \(i): speaker \(a.speaker), golden \(g.speaker)") }
            if abs(a.start - g.start) > tolerance { out.append(String(format: "segment %d: start %.3f, golden %.3f (tolerance %.3f)", i, a.start, g.start, tolerance)) }
            if abs(a.end - g.end) > tolerance { out.append(String(format: "segment %d: end %.3f, golden %.3f (tolerance %.3f)", i, a.end, g.end, tolerance)) }
        }
        return out
    }
}

struct SortformerParityTests {

    // MARK: ground truth (pure, no model)

    @Test func groundTruthIsAlternatingSortedAndInsideTheClip() throws {
        let truth = try SortformerParity.truth()
        #expect(truth.sampleRate == 16_000)
        #expect(truth.turns.count == 8)
        for (i, turn) in truth.turns.enumerated() {
            #expect(turn.speaker == i % 2, "turn \(i) alternates the voices")
            #expect(turn.start >= 0 && turn.end > turn.start && turn.end <= truth.durationSeconds)
            if i > 0 {
                let previous = truth.turns[i - 1]
                #expect(turn.start > previous.end, "turn \(i) starts after turn \(i - 1) ends (a gap, no overlap)")
                #expect(turn.speaker != previous.speaker)
            }
        }
        #expect(truth.turns.map(\.voice) == ["Samantha", "Daniel", "Samantha", "Daniel", "Samantha", "Daniel", "Samantha", "Daniel"])
        #expect((30...60).contains(truth.durationSeconds), "the owner asked for a 30 to 60 s clip: \(truth.durationSeconds)")
    }

    @Test func theClipFileMatchesTheGroundTruthDuration() throws {
        let truth = try SortformerParity.truth()
        let buffer = try AudioFileReader.read(SortformerParity.clipURL)
        #expect(buffer.sampleRate == 16_000)
        #expect(abs(Double(buffer.samples.count) / 16_000 - truth.durationSeconds) < 0.001)
        // Every turn is audible and every gap is silent (the builder pads gaps with exact zeros).
        func rms(_ a: Double, _ b: Double) -> Float {
            let range = Int(a * 16_000)..<Int(b * 16_000)
            let slice = buffer.samples[range]
            return (slice.reduce(0) { $0 + $1 * $1 } / Float(slice.count)).squareRoot()
        }
        for (i, turn) in truth.turns.enumerated() {
            #expect(rms(turn.start, turn.end) > 0.01, "turn \(i) carries speech")
            if i > 0 { #expect(rms(truth.turns[i - 1].end, turn.start) == 0, "gap before turn \(i) is silent") }
        }
    }

    // MARK: comparison logic (pure)

    @Test func differencesAcceptsWithinToleranceAndNamesEveryMismatch() {
        let golden = [SortformerParity.GoldenSegment(speaker: 0, start: 0.5, end: 6.3),
                      SortformerParity.GoldenSegment(speaker: 1, start: 7.3, end: 13.7)]
        let close = [SpeakerSegment(speaker: 1, start: 7.34, end: 13.66), SpeakerSegment(speaker: 0, start: 0.52, end: 6.27)]
        #expect(SortformerParity.differences(actual: close, golden: golden, tolerance: 0.1).isEmpty)
        #expect(SortformerParity.differences(actual: close, golden: golden, tolerance: 0.01).count == 4)
        let wrongSpeaker = [SpeakerSegment(speaker: 1, start: 0.5, end: 6.3), SpeakerSegment(speaker: 1, start: 7.3, end: 13.7)]
        #expect(SortformerParity.differences(actual: wrongSpeaker, golden: golden, tolerance: 0.1) == ["segment 0: speaker 1, golden 0"])
        #expect(SortformerParity.differences(actual: [], golden: golden, tolerance: 0.1) == ["segment count 0, golden 2"])
    }

    // MARK: golden file

    @Test func goldenFileDecodesAndIsConsistent() throws {
        let golden = try SortformerParity.golden()
        #expect(golden.clip == "two_speakers.wav")
        #expect(golden.threshold == 0.5)
        if golden.pending {
            #expect(golden.segments.isEmpty && golden.toleranceSeconds == nil)
        } else {
            #expect(!golden.segments.isEmpty)
            #expect((golden.toleranceSeconds ?? 0) > 0, "a filled golden names its tolerance")
            #expect(golden.notes.contains("toleranceSeconds"), "and documents it in the notes")
            #expect(golden.notes.contains("confirmed by the owner 2026-10-25"))
        }
    }

    /// The golden is the model's own output, not the truth: but on this clip it must agree with the truth. Every truth turn is
    /// covered by golden segments of ONE speaker (within 0.25 s at the edges), and that speaker is the truth's speaker (the golden
    /// is renumbered by first appearance, so the first voice is 0). Also: the golden is sorted, non-overlapping, compact-numbered.
    @Test func goldenAgreesWithTheGroundTruth() throws {
        let golden = try SortformerParity.golden()
        try #require(!golden.pending)
        let truth = try SortformerParity.truth()
        let edge = 0.25
        #expect(Set(golden.segments.map(\.speaker)) == Set(truth.turns.map(\.speaker)))
        for (a, b) in zip(golden.segments, golden.segments.dropFirst()) { #expect(b.start >= a.end && a.start < a.end) }
        for (i, turn) in truth.turns.enumerated() {
            let inside = golden.segments.filter { $0.end > turn.start + edge && $0.start < turn.end - edge }
            #expect(!inside.isEmpty, "turn \(i) has golden segments")
            #expect(inside.allSatisfy { $0.speaker == turn.speaker }, "turn \(i): one speaker, the truth's")
            let first = try #require(inside.first), last = try #require(inside.last)
            #expect(first.start <= turn.start + edge && last.end >= turn.end - edge, "turn \(i) is covered edge to edge")
            // Gaps inside a turn are the model's pauses; they must be short (a handful of frames).
            for (a, b) in zip(inside, inside.dropFirst()) { #expect(b.start - a.end < 0.25, "turn \(i): gap \(b.start - a.end)") }
        }
    }

    // MARK: real model (owner-run; skipped when the weights are not installed)

    /// Diarizes the clip at threshold 0.5 and compares with the golden. `golden pending` while the golden is a placeholder.
    @Test(.enabled(if: SortformerRealLoad.directory != nil))
    func realModelMatchesTheGolden() async throws {
        let golden = try SortformerParity.golden()
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let samples = try SortformerRealLoad.clipSamples()
        let raw = try await SortformerEngine.diarize(samples, threshold: Float(golden.threshold), modelDirectory: copy.url)
        let segments = SpeakerSegments.cleaned(raw, duration: Double(samples.count) / 16_000)
        print("SORTFORMERPARITY segments: \(segments.map { String(format: "(%d, %.3f, %.3f)", $0.speaker, $0.start, $0.end) }.joined(separator: " "))")
        guard !golden.pending else {
            print("SORTFORMERPARITY golden pending: \(segments.count) segments recorded above, nothing compared")
            return
        }
        let tolerance = try #require(golden.toleranceSeconds)
        let differences = SortformerParity.differences(actual: segments, golden: golden.segments, tolerance: tolerance)
        #expect(differences.isEmpty, "\(differences)")
    }
}
