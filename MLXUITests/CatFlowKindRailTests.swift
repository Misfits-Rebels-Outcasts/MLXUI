import Testing
import Foundation
import AppKit
@testable import MLXUI

/// KR-1 (`RSI/DelegateKindRailBacklog.md`) — the kind rail's pure parts: `KindStyle` covers every
/// kind with a symbol that exists on macOS, `ShapeChipModel` reads every `Shape` case, and
/// `FlowEditorModel.rowSignature` hands the rail the editor's own signatures.
struct CatFlowKindRailTests {

    @Test func kindStyleCoversEveryKindWithARealSymbol() {
        #expect(Kind.allCases.count == 14)
        for kind in Kind.allCases {
            let style = KindStyle.style(for: kind)
            #expect(style.word == kind.rawValue)
            #expect(NSImage(systemSymbolName: style.symbol, accessibilityDescription: nil) != nil,
                    "\(kind): no SF Symbol \(style.symbol)")
        }
        #expect(Set(Kind.allCases.map { KindStyle.style(for: $0).symbol }).count == 14)
    }

    @Test func shapeChipReadsEveryShapeCase() {
        #expect(ShapeChipModel(.single(.audio)).text == "audio")
        #expect(ShapeChipModel(.listOf(.image)).text == "[image]")
        #expect(ShapeChipModel(.tupleOf([.text, .vector])).text == "[text, vector]")
        #expect(ShapeChipModel(.unionOf([.audio, .video])).text == "audio|video")
        #expect(ShapeChipModel(.anyKind).text == "any")
        #expect(ShapeChipModel(.anyKind).kinds.isEmpty)
        // `same` with nothing known above stays "same"; with a single/list above it reads that kind.
        #expect(ShapeChipModel(.sameAsInput).text == "same")
        #expect(ShapeChipModel(.sameAsInput, incoming: .unionOf([.audio, .video])).text == "same")
        #expect(ShapeChipModel(.sameAsInput, incoming: .single(.video)).text == "video")
        #expect(ShapeChipModel(.sameAsInput, incoming: .listOf(.audio)).text == "[audio]")
        // For every Shape the text agrees with Shape.signatureText, except any→"any".
        for shape in [Shape.single(.text), .listOf(.table), .tupleOf([.text, .vector]), .unionOf([.audio, .video])] {
            #expect(ShapeChipModel(shape).text == shape.signatureText)
        }
    }

    @MainActor private func editor(forGallery name: String) throws -> FlowEditorModel {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/Gallery/\(name).cat")
        let doc = try CatParser.parse(String(contentsOf: url, encoding: .utf8))
        return FlowEditorModel(name: name, document: doc,
                               workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
    }

    @Test @MainActor func rowSignatureOnSpokenSummaryMatchesTheCatalog() throws {
        let e = try editor(forGallery: "01-SpokenSummary")
        let rows = e.document.rows
        let expected: [(String, Shape, Shape)] = [
            ("Read Audio", .single(.file), .single(.audio)),
            ("Transcribe", .single(.audio), .single(.text)),
            ("Summarize", .single(.text), .single(.text)),
        ]
        for (i, want) in expected.enumerated() {
            let sig = try #require(e.rowSignature(rows[i].id))
            #expect(rows[i].task == want.0)
            #expect(sig.accepts == want.1 && sig.gives == want.2, "\(want.0): \(sig)")
        }
        let save = try #require(e.rowSignature(rows[4].id))
        #expect(save.accepts == .anyKind)
        #expect(save.gives == .single(.status))
    }

    @Test @MainActor func rowSignatureOnMeetingMinutesParallelBlockAndChildren() throws {
        let e = try editor(forGallery: "02-MeetingMinutes")
        let block = try #require(e.document.rows.first { $0.blockKind == .parallel })
        let sig = try #require(e.rowSignature(block.id))
        // KR-1-FIX-1: the validator's answer (one slot per chain), also the row's own declared one.
        #expect(sig.accepts == .single(.text))
        #expect(sig.gives == .tupleOf([.text, .text, .text]))
        #expect(block.declaredSignature == "text -> [text, text, text]")
        let child = try #require(block.children.first)
        let childSig = try #require(e.rowSignature(child.id))
        #expect(childSig.accepts == .single(.text) && childSig.gives == .single(.text))
    }

    @Test @MainActor func undeclaredParallelBlockBuildsTheTupleFromItsChainTails() throws {
        let text = """
        mlxflow 0.8
        1. Read Text   a.txt
        2. <parallel p>
             1. Summarize   (input:1)   Qwen3 8B; "a"

             2. Summarize   (input:1)   Qwen3 8B; "b"
             3. Rewrite                 Ministral 3B; "c"
        3. Save Text   out.txt
        """
        let doc = try CatParser.parse(text)
        let e = FlowEditorModel(name: "p", document: doc,
                                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        let block = try #require(e.document.rows.first { $0.blockKind == .parallel })
        #expect(block.declaredSignature == nil)
        // two chains: [1] and [2, 3] — tails Summarize (text) and Rewrite (text)
        #expect(e.rowSignature(block.id)?.gives == .tupleOf([.text, .text]))
        // a plain row is untouched, and so is the shared inference
        let read = try #require(e.document.rows.first)
        #expect(e.rowSignature(read.id)?.gives == .single(.text))
        #expect(e.outputShape(for: block.id) == .single(.text))
    }

    @Test @MainActor func parallelWithAnUnknownTailGivesAnyKind() throws {
        let text = """
        mlxflow 0.8
        1. Read Text   a.txt
        2. <parallel p>
             1. Summarize   (input:1)   Qwen3 8B; "a"

             2. Frobnicate  (input:1)
        """
        let doc = try CatParser.parse(text)
        let e = FlowEditorModel(name: "p", document: doc,
                                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        let block = try #require(e.document.rows.first { $0.blockKind == .parallel })
        #expect(e.rowSignature(block.id)?.gives == .anyKind)
    }

    @Test @MainActor func rowSignatureIsNilForAnUnknownRowID() throws {
        let e = try editor(forGallery: "01-SpokenSummary")
        #expect(e.rowSignature(UUID()) == nil)
    }
}
