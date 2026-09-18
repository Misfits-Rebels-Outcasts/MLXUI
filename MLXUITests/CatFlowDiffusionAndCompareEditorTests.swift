import Testing
import Foundation
@testable import MLXUI

/// RT-3 (`RSI/DelegateRowTextBacklog.md`) — the remaining Tier-1 rows, first two sub-commits:
/// the diffusion prompt (`Generate Image`/`Generate Sound`/`Generate Video`, all read the same
/// way — `RealExecutor.runModel`'s no-input fallback, `firstBare()`, `RealExecutor.swift:
/// 469-479`) and `Compare`'s criterion (`CompareTool.run`, `firstBare()`,
/// `CalcTools.swift:374`). Both write through `setInstruction` (first quoted token), same as
/// RT-2 — neither value is the row's *whole* settings string the way `Template`'s is.
///
/// `Embed`'s query and `Improvise`'s goal are **not** built in this cycle. `Embed`: §6 Q3 is
/// confirmed real, not just a candidate — `RealExecutor.runModel`'s no-input fallback only
/// covers `engines.diffusion.generate_*` (`RealExecutor.swift:469`); a no-input `Embed` row
/// falls to the `else` branch and throws `badInputCardinality`, where the reference falls back
/// to the settings string. That's a parity bug already live in 3 bundled flows
/// (`17-AskYourDocs`, `19-CorrectiveRag`, `44-FrontierEscalate`), logged as `catflow-mlx
/// /SPEC_QUESTIONS.md` Q228, fixed in its own cycle per the backlog's own §9 — building a
/// Query box now would edit a value this runtime cannot currently read. `Improvise`: §6 Q6 was
/// put to the owner and ruled "agent goal" (`SPEC_QUESTIONS.md` Q227), but the fix is
/// `FencedRunner` actually driving an agentic loop instead of executing the token as a shell
/// command — a rewrite the backlog's own §9 keeps out of this stream regardless of the
/// ruling's direction. Shipping a "Goal" label before the runtime behaves like one would be
/// worse than the current gap.
struct CatFlowDiffusionAndCompareEditorTests {

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-rt3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "RT3", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    /// Like `editor(_:)`, but keeps the loaded document's own header flags — `55
    /// -ReceiptsLedger.cat` declares `; events` (its `On File` trigger row needs it), and
    /// `editor(_:)`'s bare `FlowDocument(version:rows:)` would drop it, failing `save()`'s own
    /// E-check for a trigger row with no `events` flag on line one.
    private func editorForDocument(_ doc: FlowDocument) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-rt3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "RT3", document: doc, workspace: FlowWorkspace(root: root))
    }

    // MARK: - Diffusion prompts: the edited bare token is what RealExecutor would read

    @Test func editingGenerateImagePromptChangesWhatRealExecutorWouldRead() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "60-GenerateProductShot")
        let row0 = doc.rows[0]
        #expect(row0.task == "Generate Image")
        let original = row0.settings ?? ""

        let model = try editor(doc.rows)
        // No-op edit: byte-identical (QR9).
        model.setInstruction(FlowSettings(original).firstBare(), for: row0.id)
        #expect(model.row(withID: row0.id)?.settings == original)

        // A real edit changes only the quoted span; seed=/width=/height= survive untouched,
        // and RealExecutor.runModel's no-input fallback (`FlowSettings(settings)
        // .firstBare()`, RealExecutor.swift:469-479) reads the new text.
        model.setInstruction("a bright red ceramic mug on a wooden table, studio lighting", for: row0.id)
        let edited = model.row(withID: row0.id)?.settings ?? ""
        #expect(FlowSettings(edited).value(for: "seed") == "7")
        #expect(FlowSettings(edited).value(for: "width") == "1024")
        #expect(FlowSettings(edited).value(for: "height") == "1024")
        #expect(FlowSettings(edited).firstBare() == "a bright red ceramic mug on a wooden table, studio lighting")

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        #expect(reparsed.rows[0].settings == edited)
        #expect(FlowSettings(reparsed.rows[0].settings).firstBare()
                == "a bright red ceramic mug on a wooden table, studio lighting")
    }

    @Test func editingGenerateSoundPromptChangesWhatRealExecutorWouldRead() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "65-VoiceoverBed")
        let row0 = doc.rows[0]
        #expect(row0.task == "Generate Sound")
        let original = row0.settings ?? ""

        let model = try editor(doc.rows)
        model.setInstruction("a tense low synth drone, no melody", for: row0.id)
        let edited = model.row(withID: row0.id)?.settings ?? ""
        #expect(edited != original)
        #expect(FlowSettings(edited).value(for: "seed") == "7")
        #expect(FlowSettings(edited).firstBare() == "a tense low synth drone, no melody")

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        #expect(FlowSettings(reparsed.rows[0].settings).firstBare() == "a tense low synth drone, no melody")
    }

    /// No bundled row exercises `Generate Video`'s bare prompt today, but the same
    /// `engines.diffusion.generate_*` prefix drives it — cover it directly rather than leave
    /// it untested until a gallery flow happens to use it.
    @Test func generateVideoRoutesThroughTheSamePromptReading() throws {
        let r = Row(id: UUID(), task: "Generate Video", blockKind: nil, model: "Wan2.2 TI2V-5B",
                    settings: "seed=3; \"a lighthouse beam sweeping across a dark sea\"",
                    refs: [], children: [], clause: nil)
        let model = try editor([r])
        model.setInstruction("a paper boat drifting down a rain-slicked street at night", for: r.id)
        let edited = model.row(withID: r.id)?.settings ?? ""
        #expect(FlowSettings(edited).value(for: "seed") == "3")
        #expect(FlowSettings(edited).firstBare() == "a paper boat drifting down a rain-slicked street at night")
    }

    // MARK: - Compare's criterion: the edited token changes which tag actually fires

    @Test func editingCompareCriterionChangesWhichTagCompareToolFires() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "55-ReceiptsLedger")
        let compareRow = doc.rows[8]
        #expect(compareRow.task == "Compare")
        #expect(compareRow.tags == ["escalate", "normal"])

        let model = try editorForDocument(doc)
        let sixty = Asset(items: [Item(kind: .text, value: "60", path: nil, sourceText: nil)])

        // Original criterion "> 50": 60 > 50 holds -> the first declared tag fires.
        let before = try CompareTool.run(row: model.row(withID: compareRow.id)!, inputs: [sixty], path: "9")
        #expect(before.firedTag == "escalate")

        // Edit to "> 100": 60 > 100 does not hold -> the second tag fires instead. Same
        // input, same declared tags/edges -- only the criterion changed the outcome, which is
        // the proof this is the token CompareTool actually reads, not just what's displayed.
        model.setInstruction("> 100", for: compareRow.id)
        let edited = model.row(withID: compareRow.id)?.settings ?? ""
        #expect(FlowSettings(edited).firstBare() == "> 100")
        #expect(model.row(withID: compareRow.id)?.tags == ["escalate", "normal"])

        let after = try CompareTool.run(row: model.row(withID: compareRow.id)!, inputs: [sixty], path: "9")
        #expect(after.firedTag == "normal")

        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        let reparsedRow = reparsed.rows[8]
        #expect(reparsedRow.settings == edited)
        #expect(reparsedRow.tags == ["escalate", "normal"])
        #expect(reparsedRow.clause != nil)   // the decide clause (-> { escalate: 10 | normal: done }) survives
    }
}
