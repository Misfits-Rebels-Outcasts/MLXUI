import Testing
import Foundation
@testable import MLXUI

/// MP4-3a: the pure half of the diarization feature, no model needed.
struct SpeakerSegmentsTests {

    private func seg(_ speaker: Int, _ start: Double, _ end: Double) -> SpeakerSegment {
        SpeakerSegment(speaker: speaker, start: start, end: end)
    }

    // MARK: normalized

    @Test func normalizedSortsByStartThenEndThenSpeakerWhateverTheInputOrder() {
        let a = [seg(1, 5, 8), seg(0, 0, 4), seg(2, 0, 4), seg(0, 9, 10)]
        let expected = [seg(0, 0, 4), seg(2, 0, 4), seg(1, 5, 8), seg(0, 9, 10)]
        #expect(SpeakerSegments.normalized(a) == expected)
        #expect(SpeakerSegments.normalized(a.reversed()) == expected)
        #expect(SpeakerSegments.normalized(a.shuffled()) == expected)
    }

    @Test func normalizedDropsInvalidSegmentsAndClampsToTheClip() {
        let a = [seg(0, -1, 2), seg(0, 3, 3), seg(0, 5, 4), seg(1, 8, 12), seg(1, 20, 25),
                 SpeakerSegment(speaker: 0, start: .nan, end: 1), SpeakerSegment(speaker: -1, start: 0, end: 1)]
        #expect(SpeakerSegments.normalized(a, duration: 10) == [seg(0, 0, 2), seg(1, 8, 10)])
        #expect(SpeakerSegments.normalized(a) == [seg(0, 0, 2), seg(1, 8, 12), seg(1, 20, 25)])
    }

    @Test func normalizedMergesTouchingOrOverlappingSameSpeakerOnly() {
        let a = [seg(0, 0, 2), seg(0, 2, 4), seg(0, 3, 5), seg(1, 4, 6), seg(0, 5.5, 6)]
        #expect(SpeakerSegments.normalized(a) == [seg(0, 0, 5), seg(1, 4, 6), seg(0, 5.5, 6)])
        #expect(SpeakerSegments.normalized(a, mergeGap: 0.5) == [seg(0, 0, 6), seg(1, 4, 6)])
        #expect(SpeakerSegments.normalized([]).isEmpty)
    }

    @Test func lanesAreOneForEachSpeakerWhoSpokeInSlotOrder() {
        let a = [seg(2, 0, 1), seg(0, 4, 5), seg(0, 2, 3)]
        let lanes = SpeakerSegments.lanes(a)
        #expect(lanes.map(\.speaker) == [0, 2])
        #expect(lanes[0].segments == [seg(0, 2, 3), seg(0, 4, 5)])
        #expect(SpeakerSegments.speakers(in: a) == [0, 2])
        #expect(SpeakerSegments.lanes([]).isEmpty)
    }

    // MARK: renderings

    @Test func clockFormatsMinutesSecondsAndMilliseconds() {
        #expect(SpeakerSegments.clock(0) == "0:00.000")
        #expect(SpeakerSegments.clock(6.2966875) == "0:06.297")
        #expect(SpeakerSegments.clock(59.9996) == "1:00.000")
        #expect(SpeakerSegments.clock(125.5) == "2:05.500")
        #expect(SpeakerSegments.clock(3_725.25) == "1:02:05.250")
        #expect(SpeakerSegments.clock(-3) == "0:00.000")
    }

    @Test func textIsOneLinePerSegmentWithOneBasedSpeakers() {
        let text = SpeakerSegments.text([seg(0, 0.5, 6.2966875), seg(1, 7.5, 13.25)])
        #expect(text == "Speaker 1  0:00.500 – 0:06.297\nSpeaker 2  0:07.500 – 0:13.250")
        #expect(SpeakerSegments.text([]) == "")
    }

    @Test func csvHasAHeaderOneBasedSpeakersThreeDecimalsAndATrailingNewline() {
        let csv = SpeakerSegments.csv([seg(0, 0.5, 6.2966875), seg(1, 7.5, 13.25), seg(3, 62, 70)])
        #expect(csv == "speaker,start,end\n1,0.500,6.297\n2,7.500,13.250\n4,62.000,70.000\n")
        #expect(SpeakerSegments.csv([]) == "speaker,start,end\n")
    }

    /// `String(format:)` without a locale must not follow the user's decimal separator (a German Mac writes "0,500").
    @Test func csvUsesADotWhateverTheCurrentLocale() {
        let csv = SpeakerSegments.csv([seg(0, 1.5, 2.25)])
        #expect(csv.contains("1.500") && csv.contains("2.250"))
        #expect(!csv.contains("1,500"))
        let fields = csv.split(separator: "\n")[1].split(separator: ",", omittingEmptySubsequences: false)
        #expect(fields.count == 3, "exactly three comma-separated fields: \(fields)")
    }

    @Test func parseCSVRoundTripsWhatCSVWrites() throws {
        let original = [seg(0, 0.5, 6.297), seg(1, 7.5, 13.25), seg(3, 62, 70)]
        #expect(try SpeakerSegments.parseCSV(SpeakerSegments.csv(original)) == original)
        #expect(try SpeakerSegments.parseCSV(SpeakerSegments.csv([])).isEmpty)
    }

    @Test func parseCSVRejectsWhatItDidNotWrite() {
        #expect(throws: SpeakerSegments.CSVError.badHeader) { try SpeakerSegments.parseCSV("") }
        #expect(throws: SpeakerSegments.CSVError.badHeader) { try SpeakerSegments.parseCSV("a,b,c\n1,0,1\n") }
        #expect(throws: SpeakerSegments.CSVError.badRow(2)) { try SpeakerSegments.parseCSV("speaker,start,end\n0,0,1\n") }
        #expect(throws: SpeakerSegments.CSVError.badRow(2)) { try SpeakerSegments.parseCSV("speaker,start,end\n1,2,1\n") }
        #expect(throws: SpeakerSegments.CSVError.badRow(3)) { try SpeakerSegments.parseCSV("speaker,start,end\n1,0,1\n1,x,2\n") }
        #expect(throws: SpeakerSegments.CSVError.badRow(2)) { try SpeakerSegments.parseCSV("speaker,start,end\n1,0\n") }
    }

    @Test func labelsAreOneBased() {
        #expect(seg(0, 0, 1).label == "Speaker 1")
        #expect(seg(3, 0, 1).label == "Speaker 4")
        #expect(seg(0, 1, 3.5).duration == 2.5)
    }

    // MARK: axis

    @Test func axisTicksPickTheSmallestNiceStepWithinTheTickBudget() {
        #expect(SpeakerSegments.axisTicks(duration: 5) == [0, 1, 2, 3, 4, 5])
        #expect(SpeakerSegments.axisTicks(duration: 52.7) == [0, 10, 20, 30, 40, 50])
        #expect(SpeakerSegments.axisTicks(duration: 600) == [0, 120, 240, 360, 480, 600])
        #expect(SpeakerSegments.axisTicks(duration: 0) == [0])
        #expect(SpeakerSegments.axisTicks(duration: .nan) == [0])
        #expect(SpeakerSegments.axisTicks(duration: 40_000).count <= 12)
        #expect(SpeakerSegments.axisTicks(duration: 52.7).count <= 8)
        #expect(SpeakerTimelineView.tickLabel(60) == "1:00")
    }
}
