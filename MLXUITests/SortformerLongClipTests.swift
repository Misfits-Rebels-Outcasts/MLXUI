import Testing
import Foundation
@testable import MLXUI

/// MP4-3b: the long-clip policy (offline up to 60 s, streaming to 60 min, a sentence past that), the speaker renumbering and
/// the streaming chunking. Pure: no model. The model-backed halves are `SortformerSyntheticRunTests` (tiny model, streaming
/// branch, limit before load) and `SortformerRealLoadTests` (the real checkpoint, both paths).
struct SortformerLongClipTests {

    // MARK: path choice and limit

    @Test func shortClipsGoOfflineAndLongOnesStream() {
        #expect(SortformerEngine.path(forSeconds: 52.7) == .offline)
        #expect(SortformerEngine.path(forSeconds: SortformerEngine.offlineMaxSeconds) == .offline)
        #expect(SortformerEngine.path(forSeconds: SortformerEngine.offlineMaxSeconds + 0.1) == .streaming)
        #expect(SortformerEngine.path(forSeconds: SortformerEngine.maxSeconds) == .streaming)
        #expect(SortformerEngine.path(forSeconds: 52.7, offlineLimit: 1) == .streaming)
    }

    @Test func theLimitIsInclusiveAndTheMessageIsASentence() throws {
        try SortformerEngine.requireWithinLimit(seconds: SortformerEngine.maxSeconds)
        #expect(throws: SortformerEngine.LimitError.self) { try SortformerEngine.requireWithinLimit(seconds: SortformerEngine.maxSeconds + 1) }
        let message = try #require(SortformerEngine.LimitError(seconds: 4_500).errorDescription)
        #expect(message.contains("75 minutes") && message.contains("up to 60 minutes"))
        #expect(message.contains("split the recording"))
        #expect(MLXErrorGuard.sentence(for: SortformerEngine.LimitError(seconds: 4_500)) == message)
    }

    @Test func theStageRefusesALongRecordingBeforeCallingTheDiarizer() async throws {
        let stage = SortformerStage(id: "t", name: "t") { _ in
            Issue.record("the diarizer must not run past the limit")
            return []
        }
        let samples = [Float](repeating: 0, count: (Int(SortformerEngine.maxSeconds) + 60) * 16_000)
        await #expect(throws: SortformerEngine.LimitError.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: samples, sampleRate: 16_000)), progress: { _ in })
        }
    }

    @Test func theRunUIExplainsTheLimit() {
        #expect(SortformerRunView.lengthNote.contains("60 minutes"))
    }

    // MARK: streaming chunks

    @Test func chunksCoverEverySampleOnceAndFoldASliver() {
        let chunk = 80_000
        #expect(SortformerEngine.streamingChunks(sampleCount: 0, chunkSamples: chunk).isEmpty)
        #expect(SortformerEngine.streamingChunks(sampleCount: 1_000, chunkSamples: chunk) == [0..<1_000])
        #expect(SortformerEngine.streamingChunks(sampleCount: 160_000, chunkSamples: chunk) == [0..<80_000, 80_000..<160_000])
        // 10.4 s: the 0.4 s remainder is under half a chunk, so it joins the second chunk.
        #expect(SortformerEngine.streamingChunks(sampleCount: 166_400, chunkSamples: chunk) == [0..<80_000, 80_000..<166_400])
        // 13 s: a 3 s remainder (over half) stays its own chunk.
        let ranges = SortformerEngine.streamingChunks(sampleCount: 208_000, chunkSamples: chunk)
        #expect(ranges.count == 3 && ranges.last == 160_000..<208_000)
        for n in [80_001, 123_456, 3_600 * 16_000] {
            let r = SortformerEngine.streamingChunks(sampleCount: n, chunkSamples: chunk)
            #expect(r.first?.lowerBound == 0 && r.last?.upperBound == n)
            for (a, b) in zip(r, r.dropFirst()) { #expect(a.upperBound == b.lowerBound) }
        }
    }

    // MARK: speaker numbering

    @Test func speakersAreRenumberedByFirstAppearance() {
        let raw = [
            SpeakerSegment(speaker: 2, start: 0, end: 1), SpeakerSegment(speaker: 0, start: 1, end: 2),
            SpeakerSegment(speaker: 3, start: 2, end: 3), SpeakerSegment(speaker: 2, start: 3, end: 4),
        ]
        #expect(SpeakerSegments.relabeledByFirstAppearance(raw).map(\.speaker) == [0, 1, 2, 0])
        #expect(SpeakerSegments.relabeledByFirstAppearance([]).isEmpty)
        // The case the real clip showed: slots 0 and 2 become 0 and 1, first-speaking voice first.
        let real = [SpeakerSegment(speaker: 0, start: 0.5, end: 6), SpeakerSegment(speaker: 2, start: 7, end: 13)]
        #expect(SpeakerSegments.cleaned(real).map(\.speaker) == [0, 1])
        // First appearance means earliest start, not the order the library listed them in.
        let unsorted = [SpeakerSegment(speaker: 1, start: 5, end: 6), SpeakerSegment(speaker: 3, start: 1, end: 2)]
        #expect(SpeakerSegments.cleaned(unsorted).map(\.speaker) == [0, 1])
        #expect(SpeakerSegments.cleaned(unsorted).map(\.start) == [1, 5])
    }

    @Test func cleanedJoinsTouchingPiecesKeepsRealPausesAndIsIdempotent() {
        let pieces = [
            SpeakerSegment(speaker: 2, start: 0, end: 5), SpeakerSegment(speaker: 2, start: 5.0004, end: 10),   // chunk edge: joined
            SpeakerSegment(speaker: 2, start: 10.08, end: 12),                                                  // one frame: kept
            SpeakerSegment(speaker: 1, start: 12, end: 14),
        ]
        let cleaned = SpeakerSegments.cleaned(pieces, duration: 13)
        #expect(cleaned == [
            SpeakerSegment(speaker: 0, start: 0, end: 10), SpeakerSegment(speaker: 0, start: 10.08, end: 12),
            SpeakerSegment(speaker: 1, start: 12, end: 13),
        ])
        #expect(SpeakerSegments.cleaned(cleaned, duration: 13) == cleaned)
    }
}
