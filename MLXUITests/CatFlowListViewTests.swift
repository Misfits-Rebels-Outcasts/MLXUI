import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R6-2: the flow list renders the serializer's canonical lines ("rows read like
/// the file") — the English one-line descriptions and the hand-computed reference labels are
/// gone, and `CatSerializer.serializeLines` gives each row its lines. See
/// `RSI/DelegateMergeBacklog.md` CFM-R6-1/R6-2.
struct CatFlowListViewTests {

    // MARK: - Fixtures

    private func decode(_ flowID: String) throws -> FlowDocument {
        let filePath = #filePath
        let repoRoot = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).cat")
        return try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
    }

    // MARK: - R6-2: Spoken Summary renders exactly the six canonical lines

    @Test func spokenSummaryRendersTheSixCanonicalLines() throws {
        let doc = try decode("01-SpokenSummary")
        let (lines, ranges) = CatSerializer.serializeLines(doc)
        // The serializer emits the version header as lines[0]; the view slices per row, so
        // assert on the row ranges' slices — the thing the view actually renders (FIX-1).
        #expect(lines.first == "catflow 0.8")
        let rowSlices = doc.rows.compactMap { ranges[$0.id].map { Array(lines[$0]) } }
        #expect(rowSlices == [
            ["1. Read Audio         memo.m4a"],
            ["2. Transcribe         Whisper Large v3; lang=en"],
            ["3. Summarize          Qwen3 8B; \"TL;DR in 3 bullets\""],
            ["4. Speak              Kokoro 82M; af_heart"],
            ["5. Save Audio         memo-tldr.wav"],
            ["6. Save Text    (2)   memo-transcript.txt"],
        ])
        // Every row's range is non-empty and contiguous; the header belongs to no row.
        #expect(ranges.count == doc.rows.count)
        let covered = doc.rows.compactMap { ranges[$0.id] }.reduce(0, { $0 + $1.count })
        #expect(covered == lines.count - 1)   // the 6 row lines, not the header
    }

    // MARK: - Row 6's ref column comes from the serializer, not a hand label

    @Test func spokenSummaryRow6LineHasParen2Ref() throws {
        let doc = try decode("01-SpokenSummary")
        let row6 = try #require(doc.rows.last)
        let (_, ranges) = CatSerializer.serializeLines(doc)
        let range = try #require(ranges[row6.id])
        // The serializer puts `(2)` in the ref column of row 6's canonical line.
        #expect(CatSerializer.serialize(doc).contains("6. Save Text    (2)   memo-transcript.txt"))
    }

    @Test func spokenSummaryHasNoChainBreak() throws {
        let doc = try decode("01-SpokenSummary")
        #expect(doc.rows.allSatisfy { !FlowRowSummary.hasChainBreakBefore($0) })
    }

    // MARK: - Policy Diff

    @Test func policyDiffReportsChainBreakBeforeRow3() throws {
        let doc = try decode("08-PolicyDiff")
        let row3 = try #require(doc.rows.dropFirst(2).first)
        #expect(FlowRowSummary.hasChainBreakBefore(row3))
        #expect(!FlowRowSummary.hasChainBreakBefore(doc.rows[1]))
    }

    @Test func policyDiffRow3LineHasParen12Ref() throws {
        let doc = try decode("08-PolicyDiff")
        let serialized = CatSerializer.serialize(doc)
        #expect(serialized.contains("3. Diff        (1,2)   format=unified"))
    }

    // MARK: - Photo Web Prep (blocks)

    @Test func photoWebPrepBlockTaskName() throws {
        let doc = try decode("21-PhotoWebPrep")
        let block = try #require(doc.rows.dropFirst(1).first)
        #expect(FlowRowSummary.taskName(for: block) == "<each>")
        #expect(FlowRowSummary.isBlock(block))
    }

    // MARK: - Display numbers

    @Test func displayNumberIsOneBased() {
        let a = Row(task: "Read Text")
        let b = Row(task: "Diff")
        #expect(FlowRowSummary.displayNumber(forID: b.id, in: [a, b]) == 2)
        #expect(FlowRowSummary.displayNumber(forID: UUID(), in: [a, b]) == nil)
    }
}
