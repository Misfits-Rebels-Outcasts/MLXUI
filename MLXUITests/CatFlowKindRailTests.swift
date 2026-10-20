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

    // MARK: - KR-2: input sources

    @MainActor private func parsedEditor(_ text: String) throws -> FlowEditorModel {
        FlowEditorModel(name: "kr2", document: try CatParser.parse(text),
                        workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
    }

    @Test @MainActor func aRowFedByTheRowAboveIsAbove() throws {
        let e = try parsedEditor("mlxflow 0.8\n1. Read Audio   memo.m4a\n2. Transcribe   Whisper Large v3; lang=en\n")
        #expect(e.railInputs(for: e.document.rows[1].id).map(\.source) == [.above])
    }

    @Test @MainActor func aStartingRowWithItsOwnValueIsOwnNotMissing() throws {
        let e = try parsedEditor("mlxflow 0.8\n1. Read Audio   memo.m4a\n")
        #expect(e.railInputs(for: e.document.rows[0].id).map(\.source) == [.own("file")])
        // Generate Image as row 1 with the default prompt the editor seeds.
        let g = try parsedEditor("mlxflow 0.8")
        g.add(task: "Generate Image")
        let row = try #require(g.document.rows.first)
        #expect(g.railInputs(for: row.id).map(\.source) == [.own("prompt")])
    }

    @Test @MainActor func anUnfedRowWithNoValueIsMissing() throws {
        let e = try parsedEditor("mlxflow 0.8\n1. Summarize   Qwen3 8B; \"TL;DR\"\n")
        #expect(e.railInputs(for: e.document.rows[0].id).map(\.source) == [.missing])
        // A starting row whose own value was cleared is missing too.
        let g = try parsedEditor("mlxflow 0.8")
        g.add(task: "Generate Image")
        let row = try #require(g.document.rows.first)
        g.setInstruction(nil, for: row.id); g.setSetting(key: "seed", value: "1", for: row.id)
        #expect(g.railInputs(for: row.id).map(\.source) == [.missing])
    }

    @Test @MainActor func inputRefsResolveToRowNAndPointAtTheBlocksSource() throws {
        let e = try editor(forGallery: "02-MeetingMinutes")
        let block = try #require(e.document.rows.first { $0.blockKind == .parallel })
        let summarize = try #require(block.children.first)
        let inputs = e.railInputs(for: summarize.id)
        // KR-2-FIX-1: `(input:1)` is the block's first input — here the row above the block (2).
        #expect(inputs.map(\.source) == [.row(2)])
        #expect(inputs.first?.source.label == "from 2")
        #expect(inputs.first?.tooltip == "block input 1 — row 2's output")
        #expect(inputs.first?.target == e.document.rows[1].id)
        // Rewrite has no ref and no blank line above it: fed by the row above in its chain.
        let rewrite = block.children[2]
        #expect(e.railInputs(for: rewrite.id).map(\.source) == [.above])
    }

    @Test @MainActor func everyParallelChildInMeetingMinutesReadsFrom2() throws {
        let e = try editor(forGallery: "02-MeetingMinutes")
        let block = try #require(e.document.rows.first { $0.blockKind == .parallel })
        for child in block.children where child.refs == [.inputRef(1)] {
            #expect(e.railInputs(for: child.id).map(\.source.label) == ["from 2"])
        }
    }

    @Test @MainActor func aTopLevelInputRefKeepsItsNumber() throws {
        let e = try parsedEditor("mlxflow 0.8\n1. Read Text   a.txt\n2. Summarize   (input:1)   Qwen3 8B; \"x\"\n")
        let inputs = e.railInputs(for: e.document.rows[1].id)
        #expect(inputs.map(\.source) == [.row(1)])
        #expect(inputs.first?.tooltip == nil)
        #expect(inputs.first?.target == nil)
    }

    @Test @MainActor func aBundleBlocksInputsResolveEachToItsOwnRowOrFallBack() throws {
        let e = try parsedEditor("""
        mlxflow 0.8
        1. Read Text   a.txt
        2. Read Text   b.txt
        3. <list pair>   (1,2)
             1. Template   (input:2,input:1)   "{1} {2}"
             2. Template   (input:3)   "{1}"
        """)
        let block = e.document.rows[2]
        let first = e.railInputs(for: block.children[0].id)
        #expect(first.map(\.source) == [.row(2), .row(1)])
        #expect(first.map(\.target) == [e.document.rows[1].id, e.document.rows[0].id])
        // input 3 of a 2-ref bundle isn't one row: "from block input 3", same tooltip pattern.
        let beyond = e.railInputs(for: block.children[1].id)
        #expect(beyond.map(\.source) == [.blockInput(3)])
        #expect(beyond.first?.source.label == "from block input 3")
        #expect(beyond.first?.tooltip == "block input 3 — not from a single row")
        #expect(beyond.first?.target == nil)
    }

    @Test @MainActor func aNestedBlocksRefFollowsTheChainOutward() throws {
        let e = try parsedEditor("""
        mlxflow 0.8
        1. Read Text   a.txt
        2. Read Text   b.txt
        3. <list outer>   (1)
             1. <list inner>   (input:1)
                  1. Summarize   (input:1)   Qwen3 8B; "x"
        """)
        let inner = e.document.rows[2].children[0]
        let leaf = inner.children[0]
        // leaf (input:1) → inner's input 1 → inner ref (input:1) → outer's input 1 → outer ref (1) → row 1
        #expect(e.railInputs(for: leaf.id).map(\.source) == [.row(1)])
        #expect(e.railInputs(for: leaf.id).first?.target == e.document.rows[0].id)
    }

    @Test @MainActor func twoRefsKeepBothInOrderWithTheirTargets() throws {
        let e = try parsedEditor("mlxflow 0.8\n1. Read Text   a.txt\n2. Read Text   b.txt\n\n3. Template (2,1)   \"{1} {2}\"\n")
        let rows = e.document.rows
        let inputs = e.railInputs(for: rows[2].id)
        #expect(inputs.map(\.source) == [.row(2), .row(1)])
        #expect(inputs.map(\.target) == [rows[1].id, rows[0].id])
    }

    @Test @MainActor func pureSourceFunctionCoversTheFourStates() throws {
        func row(_ line: String) throws -> Row {
            try #require(try CatParser.parse("mlxflow 0.8\n" + line + "\n").rows.first)
        }
        let prev = try row("1. Read Audio   memo.m4a")
        let fed = try row("2. Transcribe   Whisper Large v3")
        #expect(RailInputSources.sources(of: fed, previous: prev, number: { _ in nil }) == [.above])
        let own = try row("1. Read Audio   memo.m4a")
        #expect(RailInputSources.sources(of: own, previous: nil, number: { _ in nil }) == [.own("file")])
        let none = try row("1. Summarize   Qwen3 8B; \"x\"")
        #expect(RailInputSources.sources(of: none, previous: nil, number: { _ in nil }) == [.missing])
        #expect(RailInputSource.row(3).label == "from 3")
    }

    // MARK: - KR-2: the mismatch outline

    @Test @MainActor func mismatchIsTheEditorsOwnSentenceAndOnlyForInputProblems() throws {
        // A text row cannot feed Transcribe: the editor's input sentence becomes the tooltip.
        let bad = try parsedEditor("mlxflow 0.8\n1. Read Text   a.txt\n2. Transcribe   Whisper Large v3\n")
        let id = bad.document.rows[1].id
        let warning = try #require(bad.warning(for: id))
        #expect(bad.inputMismatch(for: id) == warning)
        // A model problem is a warning but not an input problem: no outline.
        let model = try parsedEditor("mlxflow 0.8\n1. Read Text   a.txt\n2. Summarize   Nonexistent 1B; \"x\"\n")
        let mid = model.document.rows[1].id
        #expect(model.warning(for: mid) != nil)
        #expect(model.inputMismatch(for: mid) == nil)
        // A healthy row has neither.
        #expect(bad.inputMismatch(for: bad.document.rows[0].id) == nil)
    }
}
