import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R1-3 / the R5 parser retirement: `FlowDocument` construction from the
/// parse tree. The pre-parsed-JSON resources are retired — the app now parses each
/// gallery `.cat` at runtime via `CatParser` (pinned against the Python corpus by
/// `CatFlowParserTests`), which mints a UUID per row and resolves numeric refs to ids
/// **during parse**, so no number-shaped reference survives. The JSON decode/encode
/// round-trip and the unresolved-reference failure are covered here against inline JSON
/// (the wire shape `tests/conftest.py::flow_to_dict`).
struct CatFlowDocumentTests {

    // MARK: - Fixtures (loaded from the repo, mirroring how the app loads browser.json)

    private func loadCatText(_ flowID: String) throws -> String {
        let filePath = #filePath                      // .../MLXUITests/CatFlowDocumentTests.swift
        let repoRoot = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).cat")
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func parse(_ flowID: String) throws -> FlowDocument {
        try CatParser.parse(try loadCatText(flowID))
    }

    // MARK: - All bundled flows parse

    @Test func spokenSummaryParses() throws {
        let doc = try parse("01-SpokenSummary")
        #expect(doc.version == "0.8")
        #expect(doc.rows.count == 6)
        #expect(doc.rows.map(\.task) == [
            "Read Audio", "Transcribe", "Summarize", "Speak", "Save Audio", "Save Text",
        ])
        #expect(doc.rows.map(\.chainBreak) == [false, false, false, false, false, false])
        // Row 6's ref resolves to row 2's id.
        let row2 = try #require(doc.rows.dropFirst(1).first)
        let row6Refs = doc.rows.last?.refs
        #expect(row6Refs == [.rowRef(row2.id)])
    }

    @Test func policyDiffParsesWithChainBreak() throws {
        let doc = try parse("08-PolicyDiff")
        #expect(doc.rows.count == 6)
        #expect(doc.rows.map(\.task) == [
            "Read Text", "Read Text", "Diff", "Summarize", "Save Text", "Save Text",
        ])
        // The blank line sits before row 3.
        #expect(doc.rows[2].chainBreak == true)
        #expect(doc.rows.map(\.chainBreak) == [false, false, true, false, false, false])
        // Row 3 bundles (1,2); row 6 references row 3.
        let row1 = try #require(doc.rows.dropFirst(0).first)
        let row2 = try #require(doc.rows.dropFirst(1).first)
        let row3 = try #require(doc.rows.dropFirst(2).first)
        #expect(doc.rows[2].refs == [.rowRef(row1.id), .rowRef(row2.id)])
        #expect(doc.rows[5].refs == [.rowRef(row3.id)])
    }

    @Test func photoWebPrepParsesWithBlockChildren() throws {
        let doc = try parse("21-PhotoWebPrep")
        #expect(doc.rows.count == 3)
        #expect(doc.rows.map(\.task) == ["Read Images", nil, "Save Images"])
        let each = try #require(doc.rows.dropFirst(1).first)
        #expect(each.blockKind == .each)
        #expect(each.blockName == "prep_photo")
        #expect(each.declaredSignature == "[image] -> [image]")
        #expect(each.children.count == 2)
        // The block's first child references its own block input, not a row.
        let child = try #require(each.children.first)
        #expect(child.task == "Resize")
        #expect(child.refs == [.inputRef(1)])
        // The block's children carry their own scope: `(input:1)` is not a row ref.
        let childIDs = Set(each.children.map(\.id))
        #expect(childIDs.count == 2)
    }

    // MARK: - Round-trip identity

    @Test func encodeDecodeRoundTripIsIdentity() throws {
        for flowID in ["01-SpokenSummary", "08-PolicyDiff", "21-PhotoWebPrep"] {
            let original = try parse(flowID)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]   // canonical bytes across encodes
            let data = try encoder.encode(original)
            let roundTripped = try JSONDecoder().decode(FlowDocument.self, from: data)
            #expect(roundTripped.version == original.version)
            #expect(roundTripped.rows.count == original.rows.count)
            #expect(roundTripped.rows.map(\.task) == original.rows.map(\.task))
            #expect(roundTripped.rows.map(\.chainBreak) == original.rows.map(\.chainBreak))
            // UUIDs are re-minted per decode, so compare refs by the *position* they
            // resolve to, not by id.
            #expect(positions(of: roundTripped.rows, refs: roundTripped.rows.last?.refs ?? [])
                    == positions(of: original.rows, refs: original.rows.last?.refs ?? []))
            // Canonical re-encode is byte-stable across a second round (no drift).
            let data2 = try encoder.encode(roundTripped)
            #expect(data == data2)
        }
    }

    /// The 1-based row numbers a set of rowRefs resolve to, in order.
    private func positions(of rows: [Row], refs: [Ref]) -> [Int] {
        let byID = Dictionary(uniqueKeysWithValues: rows.enumerated().map { ($1.id, $0 + 1) })
        return refs.compactMap { ref in
            if case .rowRef(let id) = ref { return byID[id] }
            return nil
        }
    }

    @Test func rowIDsAreReMintedOnEachDecode() throws {
        // UUIDs are minted at decode time — two decodes of the same file differ in ids,
        // which is what makes references id-based rather than number-based.
        let a = try parse("01-SpokenSummary")
        let b = try parse("01-SpokenSummary")
        #expect(a.rows.first?.id != b.rows.first?.id)
    }

    // MARK: - Nonexistent row reference fails loudly

    @Test func referenceToNonexistentRowFails() throws {
        // A JSON tree whose row 6 references row 99 (which doesn't exist).
        let json = """
        {
          "version": "0.8",
          "rows": [
            { "task": "Read Audio", "model": null, "settings": "a.m4a",
              "refs": [], "chain_break": false, "block_kind": null,
              "block_name": null, "children": [] },
            { "task": "Save Text", "model": null, "settings": "a.txt",
              "refs": [ { "kind": "row", "number": 99 } ], "chain_break": false,
              "block_kind": null, "block_name": null, "children": [] }
          ]
        }
        """
        #expect(throws: (any Error).self) {
            _ = try JSONDecoder().decode(FlowDocument.self, from: Data(json.utf8))
        }
    }
}
