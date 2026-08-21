import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R1-5's view-model logic: the pure `FlowRowSummary` functions the read-only
/// flow list renders from. See `RSI/DelegateMergeBacklog.md` CFM-R1-5.
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

    // MARK: - Spoken Summary

    @Test func spokenSummaryRow6ReferenceLabelIsParen2() throws {
        let doc = try decode("01-SpokenSummary")
        let row6 = try #require(doc.rows.last)
        #expect(FlowRowSummary.referenceLabel(for: row6, in: doc.rows) == "(2)")
    }

    @Test func spokenSummaryRow2SubtitleIsWhisperLargeV3() throws {
        let doc = try decode("01-SpokenSummary")
        let row2 = try #require(doc.rows.dropFirst(1).first)
        #expect(FlowRowSummary.subtitle(for: row2) == "Whisper Large v3")
    }

    @Test func spokenSummaryRow1SubtitleIsTaskDescription() throws {
        // A row with no model gets a one-line task description, not the raw settings line.
        let doc = try decode("01-SpokenSummary")
        let row1 = try #require(doc.rows.first)
        #expect(FlowRowSummary.subtitle(for: row1) == "Reads an audio file")
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

    @Test func policyDiffRow3ReferenceLabelIsParen12() throws {
        let doc = try decode("08-PolicyDiff")
        let row3 = try #require(doc.rows.dropFirst(2).first)
        #expect(FlowRowSummary.referenceLabel(for: row3, in: doc.rows) == "(1,2)")
    }

    // MARK: - Photo Web Prep (blocks)

    @Test func photoWebPrepBlockTaskName() throws {
        let doc = try decode("21-PhotoWebPrep")
        let block = try #require(doc.rows.dropFirst(1).first)
        #expect(FlowRowSummary.taskName(for: block) == "<each>")
        #expect(FlowRowSummary.isBlock(block))
    }

    @Test func photoWebPrepChildRefIsNotARowLabel() throws {
        // The block's first child references its block input (`(input:1)`), which is not a
        // `(N)` row reference — no reference label is produced.
        let doc = try decode("21-PhotoWebPrep")
        let block = try #require(doc.rows.dropFirst(1).first)
        let child = try #require(block.children.first)
        #expect(FlowRowSummary.referenceLabel(for: child, in: block.children) == nil)
    }

    // MARK: - Display numbers

    @Test func displayNumberIsOneBased() {
        let a = Row(task: "Read Text")
        let b = Row(task: "Diff")
        #expect(FlowRowSummary.displayNumber(forID: b.id, in: [a, b]) == 2)
        #expect(FlowRowSummary.displayNumber(forID: UUID(), in: [a, b]) == nil)
    }
}
