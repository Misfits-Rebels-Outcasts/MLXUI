import Testing
import Foundation
@testable import MLXUI

/// CFM-R8 + CFM-R8-FIX — the editing engine's identity contract and the step picker. The
/// QR8 attack surface is delete and reorder in every order: a reference must never be
/// silently re-aimed, only ever render `(?N)` + yellow until the user fixes it or undoes.
/// The validity contract (FIX-1) is the second half: the save gate refuses every file
/// `catflow check` refuses (forward/deleted refs, arity, `(input:N)` out of range).
struct CatFlowEditingTests {

    // MARK: - Helpers

    /// A model rooted at a temp dir so `save()` never touches real user data.
    private func editor(name: String = "My Flow",
                        rows: [Row] = [],
                        workspaceRoot: URL? = nil) throws -> FlowEditorModel {
        let root = workspaceRoot ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-editor-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        // A flow started in the editor is `mlxflow`-headed (CFM-R18-5), same as
        // `FlowEditorModel`'s own nil-document fallback.
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: rows)
        return FlowEditorModel(name: name, document: doc,
                               workspace: FlowWorkspace(root: root))
    }

    /// An editor with the real catalog + claim table wired, so a model row's runnability is
    /// judged against the derived pool (CFM-R14-FIX-6: an empty catalog is "no verdict", so
    /// tests that assert a model row is green must supply one). `@MainActor` for the registry.
    @MainActor
    private func wiredEditor(rows: [Row]) throws -> FlowEditorModel {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let model = try editor(rows: rows)
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        return model
    }

    private func row(_ task: String?, model: String? = nil, settings: String? = nil,
                     refs: [Ref] = [], children: [Row] = [], blockKind: BlockKind? = nil,
                     blockName: String? = nil, clause: Clause? = nil) -> Row {
        Row(id: UUID(), task: task, blockKind: blockKind, blockName: blockName,
            model: model, settings: settings, refs: refs, children: children, clause: clause)
    }

    /// The bundled catalog + the registry's own claim answer — the derived pool's inputs
    /// (CFM-R14-2). `@MainActor` because `ModelRegistry` is.
    @MainActor
    private func loadedCatalogAndClaimable() throws -> (catalog: [ModelEntry], claimable: Set<String>) {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
            .domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        return (catalog, claimable)
    }

    // MARK: - CFM-R8-1: the step picker

    @Test func stepsAcceptingTextOutputListsTextConsumers() {
        let steps = FlowEditorModel.stepsAccepting(.single(.text))
        let names = Set(steps.map(\.name))
        #expect(names.contains("Summarize"))
        #expect(names.contains("Translate"))
        #expect(names.contains("Speak"))
        #expect(names.contains("Save Text"))
        // FIX-2: a tupleOf-accepts task whose first slot fits is offerable (Diff).
        #expect(names.contains("Diff"))
        #expect(!names.contains("Transcribe"))     // accepts audio
        #expect(!names.contains("Resize"))         // accepts image
        #expect(!names.contains("Read Text"))      // accepts file
        #expect(!names.contains("Retrieve"))       // [index, vector] — neither fits text
    }

    @Test func stepsAcceptingAudioOutputListsTranscribe() {
        let steps = FlowEditorModel.stepsAccepting(.single(.audio))
        let names = Set(steps.map(\.name))
        #expect(names.contains("Transcribe"))
        #expect(!names.contains("Summarize"))
    }

    @Test func stepsHideAdvancedUnlessFullCatalog() {
        let defaulted = Set(FlowEditorModel.stepsAccepting(.single(.text)).map(\.name))
        let full = Set(FlowEditorModel.stepsAccepting(.single(.text), includeAdvanced: true).map(\.name))
        #expect(!defaulted.contains("Generate"))
        #expect(!defaulted.contains("Decide"))
        #expect(full.contains("Generate"))
        #expect(full.contains("Decide"))
        // FIX-2: Full Catalog also relaxes the shape filter — a mismatched task like
        // Transcribe (audio) is offerable after a text row (answer l: it turns yellow until
        // the user points its input at a source).
        #expect(!defaulted.contains("Transcribe"))
        #expect(full.contains("Transcribe"))
    }

    @Test func startingNodesAreSourcesCanRunAccepts() {
        let starts = Set(FlowEditorModel.startingNodes().map(\.name))
        #expect(starts.contains("Read Text"))
        #expect(starts.contains("Read Audio"))
        #expect(starts.contains("Read Images"))
        #expect(starts.contains("Template"))
        #expect(starts.contains("Save Text"))       // anyKind, instant
        // FIX-7: the hidden primitives, the triggers, and the things `canRun` refuses are out.
        #expect(!starts.contains("Decide"))
        #expect(!starts.contains("Generate"))
        #expect(!starts.contains("On File"))
        #expect(!starts.contains("On Schedule"))
        #expect(!starts.contains("Ask Human"))
        // Things that need an upstream asset are not starting nodes.
        #expect(!starts.contains("Summarize"))
        #expect(!starts.contains("Transcribe"))
        #expect(!starts.contains("Speak"))
    }

    @Test @MainActor func defaultModelIsDerivedRunnable() throws {
        // CFM-R14-2 + CFM-R14-FIX-2: the default is the first pool-named entry of the
        // **derived** pool (the registry's own claim answer + the executor genuinely serving
        // the task), never a dead one. Transcribe seeds the first pool entry — Whisper Tiny
        // (0.11 GB) — matching the Python's `_ASR_MODELS` order. Segment seeds **SAM Base**
        // since CFM-R15-1: the owner ruled hazard H2 option (1) 2026-08-27, the executor
        // serves `engines.diffusion.segment`, and `SAM Base` bridges onto the derived sam3
        // entry (`.substitute`). Read Text is an instant tool with no model.
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        #expect(FlowEditorModel.defaultModel(forTask: "Transcribe", catalog: catalog, claimableModelIDs: claimable) == "Whisper Tiny")
        #expect(FlowEditorModel.defaultModel(forTask: "Summarize", catalog: catalog, claimableModelIDs: claimable) == "Ministral 3B")
        #expect(FlowEditorModel.defaultModel(forTask: "Speak", catalog: catalog, claimableModelIDs: claimable) == "Kokoro 82M")
        #expect(FlowEditorModel.defaultModel(forTask: "Segment", catalog: catalog, claimableModelIDs: claimable) == "SAM Base")
        #expect(FlowEditorModel.defaultModel(forTask: "Read Text", catalog: catalog, claimableModelIDs: claimable) == nil)
    }

    @Test @MainActor func everySeededDefaultIsInTheDerivedPool() throws {
        // FIX-3's invariant, CFM-R14-2 edition: whatever a task seeds as its default, the
        // derived pool can run it (the registry claims it). This is the "one list, cross-
        // checked" rule the tools already have, applied to the model side.
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        for entry in FlowEditorModel.allDefaultModels(for: catalog, claimableModelIDs: claimable) {
            if let model = entry.model {
                let derived = TaskModels.derivedModels(for: entry.task, catalog: catalog,
                                                       claimableModelIDs: claimable)
                #expect(derived.contains { $0.modelEntry?.hfModelId == model || $0.displayName == model },
                        "\(entry.task) seeds \(model), which isn't a derived-pool model")
            }
        }
    }

    // MARK: - MoC-2-4: appending "Qwen3.5 9B" to llmModels must not move the seed

    /// Decision D as ruled 2026-09-05: "Qwen3.5 9B" joins `llmModels` **last**, so every
    /// llmModels-keyed task must still seed "Ministral 3B" — a test that would fail the moment
    /// someone "helpfully" moves the new name toward the front of that array.
    @Test @MainActor func llmModelsSeedIsUnchangedByQwen35NineB() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let llmModelsKeyedTasks = ["Generate", "Summarize", "Translate", "Answer", "Rewrite",
                                   "Draft", "Ask", "Title", "Critique", "Verify", "Revise",
                                   "Merge", "Text to Table"]
        for task in llmModelsKeyedTasks {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Ministral 3B",
                    "\(task) should still seed Ministral 3B")
        }
    }

    /// The new entry is menu-reachable even though it never seeds — CFM-R14-2's "pools order,
    /// never filter" applied to a freshly-appended name.
    @Test @MainActor func qwen35NineBIsReachableFromTheGenerateMenu() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let derived = TaskModels.derivedModels(for: "Generate", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.contains { $0.displayName == "Qwen3.5 9B" })
    }

    // MARK: - CFM-R8-2: add / remove / reorder

    @Test func addInsertsBelowSelection() throws {
        let model = try editor(rows: [row("Read Text", settings: "memo.txt"), row("Summarize")])
        model.selectedRowID = model.document.rows[0].id
        model.add(task: "Title")
        #expect(model.document.rows.count == 3)
        #expect(model.document.rows[1].task == "Title")
        #expect(model.document.rows[0].task == "Read Text")
        #expect(model.document.rows[2].task == "Summarize")
    }

    @Test func addAppendsWhenNothingSelected() throws {
        let model = try editor(rows: [row("Read Text")])
        model.add(task: "Summarize")
        #expect(model.document.rows.count == 2)
        #expect(model.document.rows[1].task == "Summarize")
    }

    @Test @MainActor func addGivesModelClassRowADefaultModel() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let model = try editor()
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        model.add(task: "Transcribe")
        // CFM-R14-1/2: the seed is the first pool-named derived entry — Whisper Tiny now.
        #expect(model.document.rows[0].model == "Whisper Tiny")
        model.add(task: "Read Text")
        #expect(model.document.rows[1].model == nil)
    }

    @Test func removeByIdKeepsOtherRows() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let r2 = row("Summarize")
        let r3 = row("Save Text", settings: "out.txt")
        let model = try editor(rows: [r1, r2, r3])
        model.remove(r2.id)
        #expect(model.document.rows.map(\.id) == [r1.id, r3.id])
    }

    @Test func moveReordersTopLevel() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let r2 = row("Summarize")
        let r3 = row("Save Text", settings: "out.txt")
        let model = try editor(rows: [r1, r2, r3])
        model.move(from: IndexSet(integer: 2), to: 0)
        #expect(model.document.rows.map(\.task) == ["Save Text", "Read Text", "Summarize"])
    }

    // MARK: - CFM-R8-3 + FIX-1: the identity and validity contract

    @Test func deletingARowNeverSilentlyReAimsItsReferences() throws {
        let r1 = row("Read Audio", settings: "memo.m4a")
        let r2 = row("Transcribe", model: "Whisper Large v3")
        let r3 = row("Summarize", model: "Ministral 3B")
        let r4 = row("Save Text", refs: [.rowRef(r2.id)])
        let model = try editor(rows: [r1, r2, r3, r4])

        model.remove(r2.id)

        // The reference is NOT re-aimed at the new row 2 (r3) — it stays broken.
        let saveRow = model.document.rows.last!
        #expect(saveRow.refs == [.rowRef(r2.id)])
        #expect(saveRow.refs != [.rowRef(r3.id)])
        // It renders `(?2)` — the number the deleted row held.
        #expect(model.referenceLabel(for: saveRow.id) == "(?2)")
        // The row is yellow and save is refused.
        #expect(model.warning(for: saveRow.id) != nil)
        #expect(model.saveBlockReason != nil)
        #expect(!model.canSave)
    }

    @Test func reorderingMakesAForwardReferenceYellowNotReAimed() throws {
        // FIX-1: after moving the referenced row past its referencer, the reference is
        // still pointed at the same row (not re-aimed) but is now a *forward* reference —
        // `catflow check` rejects forward references (E203), so it is yellow and un-saveable.
        let r1 = row("Read Audio", settings: "memo.m4a")
        let r2 = row("Transcribe", model: "Whisper Large v3")
        let r3 = row("Summarize", model: "Ministral 3B")
        let r4 = row("Save Text", refs: [.rowRef(r2.id)])
        let model = try editor(rows: [r1, r2, r3, r4])

        model.move(from: IndexSet(integer: 1), to: 4)   // move row 2 to the end

        let saveRow = model.document.rows.first { $0.id == r4.id }!
        // Still pointed at r2 (never at the new row 2, r3)…
        #expect(saveRow.refs == [.rowRef(r2.id)])
        #expect(saveRow.refs != [.rowRef(r3.id)])
        // …but r4 now sits *before* r2 → a forward reference, yellow + un-saveable.
        #expect(model.warning(for: saveRow.id) != nil)
        #expect(model.saveBlockReason != nil)
        #expect(!model.canSave)
        // Undo restores the valid backward reference.
        model.undo()
        #expect(model.warning(for: r4.id) == nil)
        #expect(model.canSave)
    }

    @Test func forwardReferenceIsRefused() throws {
        // The reviewer's two-click reproduction: [Read Text, Summarize, Title], pointing
        // Summarize at Title (a forward reference) must refuse, not silently save.
        let r1 = row("Read Text", settings: "memo.txt")
        let r2 = row("Summarize")
        let r3 = row("Title")
        let model = try editor(rows: [r1, r2, r3])

        // The input picker does not even offer the forward row.
        let inputs = model.validInputs(for: r2.id)
        #expect(!inputs.contains { $0.rowID == r3.id })
        // Forcing it yields a yellow row and a save refusal.
        model.setReference(to: r3.id, for: r2.id)
        #expect(model.warning(for: r2.id) != nil)
        #expect(model.saveBlockReason != nil)
        #expect(!model.canSave)
        #expect(throws: (any Error).self) { try model.save() }
    }

    @Test func referenceLabelFollowsReorder() throws {
        let r1 = row("Read Audio", settings: "memo.m4a")
        let r2 = row("Transcribe", model: "Whisper Large v3")
        let r4 = row("Save Text", refs: [.rowRef(r2.id)])
        let model = try editor(rows: [r1, r2, r4])
        #expect(model.referenceLabel(for: r4.id) == "(2)")
        // Move r1 to the end: r2 shifts to position 1, the ref label follows its target.
        model.move(from: IndexSet(integer: 0), to: 3)
        #expect(model.referenceLabel(for: r4.id) == "(1)")
        // r4 (position 2) still references r2 (position 1) — a valid backward reference.
        #expect(model.warning(for: r4.id) == nil)
        #expect(model.canSave)
    }

    @Test @MainActor func settingAReferenceFixesAYellowRow() throws {
        // [Read Audio, Save Text, Transcribe]: Transcribe sits below Save Text (status),
        // so it can't auto-chain — yellow until it's pointed at the audio source (answer l).
        let audio = row("Read Audio", settings: "memo.m4a")
        let save = row("Save Text", settings: "out.txt")
        let transcribe = row("Transcribe", model: "Whisper Large v3")
        let model = try wiredEditor(rows: [audio, save, transcribe])
        #expect(model.warning(for: transcribe.id) != nil)
        model.setReference(to: audio.id, for: transcribe.id)
        #expect(model.warning(for: transcribe.id) == nil)
        #expect(model.canSave)
    }

    @Test func shapeIncompatibleReferenceIsYellowAndBlocksSave() throws {
        let text = row("Read Text", settings: "memo.txt")
        let incompatible = row("Transcribe", model: "Whisper Tiny", refs: [.rowRef(text.id)])
        let model = try editor(rows: [text, incompatible])
        #expect(model.warning(for: incompatible.id) != nil)
        #expect(model.saveBlockReason != nil)
        #expect(!model.canSave)
    }

    @Test @MainActor func firstRowNeedingInputIsYellow() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let model = try editor()
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        model.add(task: "Summarize")
        #expect(model.warning(for: model.document.rows[0].id) != nil)   // nothing feeds it
        model.reset()
        model.add(task: "Read Text")
        #expect(model.warning(for: model.document.rows[0].id) == nil)
        model.add(task: "Summarize")
        #expect(model.warning(for: model.document.rows[1].id) == nil)   // text → text
    }

    @Test @MainActor func rowWithNoRunnableModelIsYellow() throws {
        // FIX-3 + CFM-R14-2: a model-class row seeded with no runnable model warns. Upscale
        // has no catalog model at all → seeds nothing and warns; a row whose model name is
        // neither a bridge display nor a catalog hfModelId warns too.
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let model = try editor()
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        model.add(task: "Upscale")
        #expect(model.document.rows[0].model == nil)
        #expect(model.warning(for: model.document.rows[0].id) != nil)
        // A row whose model isn't in the derived pool warns too.
        let model2 = try editor(rows: [row("Transcribe", model: "Totally Missing")])
        model2.modelCatalog = catalog
        model2.claimableModelIDs = claimable
        #expect(model2.warning(for: model2.document.rows[0].id) != nil)
    }

    // MARK: - CFM-R8-FIX-2: bundles (Diff)

    @Test func diffCanBeGivenTwoOrderedInputs() throws {
        let a = row("Read Text", settings: "a.txt")
        let b = row("Read Text", settings: "b.txt")
        let diff = row("Diff")
        let model = try editor(rows: [a, b, diff])

        // Diff has two input slots.
        #expect(model.inputSlotCount(for: diff.id) == 2)
        // Both earlier text rows are valid inputs.
        let slot1 = model.validInputs(for: diff.id, slot: 1)
        #expect(slot1.map(\.rowID).contains(a.id))
        #expect(slot1.map(\.rowID).contains(b.id))
        #expect(slot1.map(\.rowID).contains(diff.id) == false)   // not itself
        // Assigning both slots makes the row green.
        model.setReference(to: a.id, slot: 1, for: diff.id)
        #expect(model.warning(for: diff.id) != nil)              // still needs slot 2
        model.setReference(to: b.id, slot: 2, for: diff.id)
        #expect(model.warning(for: diff.id) == nil)
        #expect(model.canSave)
        // The serialized row reads `(1,2)`.
        let text = CatSerializer.serialize(model.document)
        #expect(text.contains("(1,2)"))
    }

    @Test func loadedPolicyDiffStaysGreenAndEditable() throws {
        // 08-PolicyDiff's `Diff (1,2)` is a shipped flow the editor must load without yellow.
        let r1 = row("Read Text", settings: "policy-2025.md")
        let r2 = row("Read Text", settings: "policy-2026.md")
        let diff = row("Diff", settings: "format=unified", refs: [.rowRef(r1.id), .rowRef(r2.id)])
        let model = try editor(rows: [r1, r2, diff])
        #expect(model.warning(for: diff.id) == nil)
        #expect(model.canSave)
        // An edit round-trips without breaking it.
        model.add(task: "Summarize")
        #expect(model.warning(for: diff.id) == nil)
        #expect(model.canSave)
    }

    // MARK: - CFM-R8-FIX-5: dotted display paths

    @Test func displayNumberIsDottedForNestedRows() throws {
        let child1 = row("Draft")
        let child2 = row("Draft")
        let block = row(nil, children: [child1, child2], blockKind: .each)
        let model = try editor(rows: [row("Read Text", settings: "memo.txt"), block])
        #expect(model.displayNumber(of: child1.id) == "2.1")
        #expect(model.displayNumber(of: child2.id) == "2.2")
        #expect(model.displayNumber(of: block.id) == "2")
    }

    @Test func nestedRowWarningsNameTheDottedPath() throws {
        let child = row("Summarize")   // needs an input, is a block child at 2.1
        let block = row(nil, children: [child], blockKind: .each)
        let model = try editor(rows: [row("Read Text", settings: "memo.txt"), block])
        let warning = model.warning(for: child.id)
        #expect(warning?.contains("2.1") == true)
    }

    // MARK: - CFM-R8-FIX-6: blocks, chain breaks, duplicate

    @Test func insertBlockCreatesAnEnterableBlock() throws {
        let model = try editor(rows: [row("Read Text", settings: "memo.txt")])
        model.insertBlock(kind: .each, name: "loop", after: model.document.rows[0].id)
        let block = model.document.rows[1]
        #expect(block.blockKind == .each)
        #expect(block.blockName == "loop")
        #expect(block.children.count == 1)
        // "Entered": adding with the child selected inserts into the block, not the top level.
        model.selectedRowID = block.children[0].id
        model.add(task: "Title")
        #expect(model.document.rows.count == 2)            // top level unchanged
        #expect(model.document.rows[1].children.count == 2)
        #expect(model.document.rows[1].children[1].task == "Title")
    }

    @Test func blockChildrenCanReorderAndRemove() throws {
        let childA = row("Draft")
        let childB = row("Draft")
        let block = row(nil, children: [childA, childB], blockKind: .each)
        let model = try editor(rows: [block])
        model.moveInside(blockID: block.id, from: IndexSet(integer: 1), to: 0)
        #expect(model.document.rows[0].children.map(\.id) == [childB.id, childA.id])
        // A grandchild removal works (childIndex is depth-aware).
        model.remove(childA.id)
        #expect(model.document.rows[0].children.count == 1)
        #expect(model.document.rows[0].children[0].id == childB.id)
    }

    @Test @MainActor func chainBreakSetAndCleared() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let r2 = row("Summarize", model: "Ministral 3B")
        let model = try wiredEditor(rows: [r1, r2])
        #expect(model.warning(for: r2.id) == nil)   // auto-chains from r1
        model.setChainBreak(true, for: r1.id)
        #expect(model.warning(for: r2.id) != nil)   // blank line breaks the chain
        model.setChainBreak(false, for: r1.id)
        #expect(model.warning(for: r2.id) == nil)
    }

    @Test func duplicateMintsFreshIdsAndNeverTraps() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let model = try editor(rows: [r1])
        model.duplicate(r1.id)
        #expect(model.document.rows.count == 2)
        // Fresh ids — no duplicate-key trap anywhere downstream.
        #expect(model.document.rows[0].id != model.document.rows[1].id)
        #expect(model.displayNumber(of: model.document.rows[1].id) == "2")
        #expect(model.saveBlockReason == nil)
        try model.save()
    }

    @Test func wrapInBlockWrapsARow() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let r2 = row("Summarize")
        let model = try editor(rows: [r1, r2])
        model.wrapInBlock(r2.id, kind: .parallel, name: "group")
        #expect(model.document.rows.count == 2)
        let block = model.document.rows[1]
        #expect(block.blockKind == .parallel)
        #expect(block.children.map(\.task) == ["Summarize"])
    }

    // MARK: - CFM-R8-FIX-8: undo hygiene

    @Test func noOpEditDoesNotPushUndoOrClearRedo() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let model = try editor(rows: [r1])
        model.add(task: "Summarize")
        model.undo()                       // [r1]
        #expect(model.undoStack.isEmpty)
        model.redo()                       // [r1, Summarize]
        #expect(model.undoStack.count == 1)
        // A no-op move must not clear redo or push undo.
        model.move(from: IndexSet(integer: 0), to: 0)
        #expect(model.undoStack.count == 1)
        #expect(model.redoStack.isEmpty)   // redo was cleared by the earlier add; a no-op must not matter
    }

    @Test func selectionSurvivesUndo() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let model = try editor(rows: [r1])
        model.selectedRowID = r1.id
        model.add(task: "Summarize")       // selects the new row
        let newID = model.document.rows[1].id
        #expect(model.selectedRowID == newID)
        model.undo()
        // Selection is restored to the pre-edit row, not dangling.
        #expect(model.selectedRowID == r1.id)
    }

    @Test func undoRestoresTheWholeDocumentValue() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let model = try editor(rows: [r1])
        model.add(task: "Summarize")
        #expect(model.document.rows.count == 2)
        model.undo()
        #expect(model.document.rows.count == 1)
        #expect(model.document.rows[0].id == r1.id)
        model.redo()
        #expect(model.document.rows.count == 2)
    }

    @Test func undoRestoresADeletedRowAndClearsTheBrokenRef() throws {
        let r1 = row("Read Audio", settings: "memo.m4a")
        let r2 = row("Transcribe", model: "Whisper Large v3")
        let r4 = row("Save Text", refs: [.rowRef(r2.id)])
        let model = try editor(rows: [r1, r2, r4])
        model.remove(r2.id)
        #expect(model.warning(for: r4.id) != nil)
        #expect(!model.canSave)
        model.undo()
        #expect(model.document.rows.count == 3)
        #expect(model.warning(for: r4.id) == nil)
        #expect(model.canSave)
    }

    @Test func undoRedoSequencesInOrder() throws {
        let model = try editor()
        model.add(task: "Read Text")
        model.add(task: "Summarize")
        model.add(task: "Save Text")
        model.undo()
        #expect(model.document.rows.map(\.task) == ["Read Text", "Summarize"])
        model.undo()
        #expect(model.document.rows.map(\.task) == ["Read Text"])
        model.redo()
        #expect(model.document.rows.map(\.task) == ["Read Text", "Summarize"])
    }

    // MARK: - CFM-R8-5: save

    @Test func saveWritesTheCanonicalCatIntoTheFlowFolder() throws {
        let model = try editor(name: "My Flow", rows: [row("Read Text", settings: "memo.txt")])
        try model.save()
        let url = try #require(model.savedURL)
        #expect(url.lastPathComponent == "My Flow.cat")
        let text = try String(contentsOf: url, encoding: .utf8)
        #expect(text.contains("mlxflow 0.8"))
        #expect(text.contains("1. Read Text"))
        #expect(model.isDirty == false)
        model.add(task: "Summarize")
        #expect(model.isDirty == true)
    }

    @Test func renamingLeavesExactlyOneFlowFile() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-rename-\(UUID().uuidString)")
        let model = try editor(name: "Untitled Flow",
                               rows: [row("Read Image", settings: "budget.png")],
                               workspaceRoot: root)
        try model.save()
        let dir = model.savedURL!.deletingLastPathComponent()
        func flowFiles() throws -> [String] {
            try FileManager.default.contentsOfDirectory(atPath: dir.path)
                .filter { $0.hasSuffix(".cat") || $0.hasSuffix(".catpipeline") }.sorted()
        }
        #expect(try flowFiles() == ["Untitled Flow.cat"])

        // A rename in the same session (the tracked `savedURL` path) …
        model.name = "Extract table data from image"
        try model.save()
        #expect(try flowFiles() == ["Extract table data from image.cat"])

        // … and a rename after the editor was closed and reopened (fresh model, no
        // `savedURL`) still ends with one file, not two.
        let reopened = FlowEditorModel(name: "Extract table data from image",
                                       flowID: model.flowID,
                                       document: model.document,
                                       workspace: FlowWorkspace(root: root),
                                       savedText: model.savedText)
        reopened.name = "EXTRACT TABLE DATA FROM IMAGE"
        try reopened.save()
        #expect(try flowFiles() == ["EXTRACT TABLE DATA FROM IMAGE.cat"])
    }

    @Test func saveRefusesWhileAReferenceIsBroken() throws {
        let r1 = row("Read Audio", settings: "memo.m4a")
        let r2 = row("Transcribe", model: "Whisper Large v3")
        let r4 = row("Save Text", refs: [.rowRef(r2.id)])
        let model = try editor(rows: [r1, r2, r4])
        model.remove(r2.id)
        #expect(throws: (any Error).self) { try model.save() }
        do {
            try model.save()
            Issue.record("expected save refusal")
        } catch let error as FlowEditingError {
            guard case .refusingToSave = error else {
                Issue.record("unexpected error \(error)")
                return
            }
        }
    }

    @Test func sanitizedFileNameProducesAValidStem() {
        #expect(FlowEditorModel.sanitizedFileName("My/Flow: v1") == "My-Flow- v1")
        #expect(FlowEditorModel.sanitizedFileName("") == "Untitled")
    }

    // MARK: - The Spoken-Summary-shaped exit build

    @Test @MainActor func buildingSpokenSummaryShapeStaysGreenEndToEnd() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let model = try editor(name: "Spoken Summary")
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        model.add(task: "Read Audio")          // source
        model.add(task: "Transcribe")          // audio → text
        model.add(task: "Summarize")           // text → text
        model.add(task: "Speak")               // text → audio
        model.add(task: "Save Audio")          // anyKind
        for row in model.document.rows {
            #expect(model.warning(for: row.id) == nil,
                    "\(row.task ?? "?") should be green")
        }
        // The file round-trips: save, then re-parse → same structure **including refs**.
        try model.save()
        let text = try String(contentsOf: model.savedURL!, encoding: .utf8)
        let reparsed = try CatParser.parse(text)
        #expect(reparsed.rows.count == model.document.rows.count)
        #expect(reparsed.rows.map(\.task) == model.document.rows.map(\.task))
        #expect(reparsed.rows.map(\.model) == model.document.rows.map(\.model))
        #expect(reparsed.rows.map(\.refs) == model.document.rows.map(\.refs))
        #expect(model.runnability == .runnable)
    }

    @Test func outOfRangeInputRefIsRefusedByTheSaveGate() throws {
        // FIX-6: an `(input:3)` in a two-input block is E401 — the validator (via the save
        // gate) catches it, even though the row has no rowRef the local checks would see.
        let child = row("Draft", refs: [.inputRef(3)])
        let block = row(nil, children: [child], blockKind: .each)
        let model = try editor(rows: [row("Read Text", settings: "memo.txt"), block])
        #expect(model.saveBlockReason != nil)
        #expect(!model.canSave)
    }

    @Test func serializerRendersDeletedRefAsQuestionMarkNumber() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let r2 = row("Transcribe", model: "Whisper Tiny")
        let r4 = row("Save Text", refs: [.rowRef(r2.id)])
        let model = try editor(rows: [r1, r2, r4])
        model.remove(r2.id)
        let lines = CatSerializer.serializeLines(model.document, deadRefNumbers: model.tombstones).lines
        #expect(lines.contains { $0.contains("(?2)") })
        let plain = CatSerializer.serializeLines(model.document).lines
        #expect(plain.contains { $0.contains("(-1)") })
    }

    // MARK: - CFM-R11-0: the routes into the editor (Duplicate & Edit / Edit a Copy)

    @Test func duplicateAndEditCopiesFlowAndAssetsIntoTheUsersFolder() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-edit-route-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // A temp "bundle" that mirrors the flattened app bundle: memo.m4a at the flat root.
        let sourceDir = root.appendingPathComponent("bundle")
        try FileManager.default.createDirectory(at: sourceDir, withIntermediateDirectories: true)
        let audio = sourceDir.appendingPathComponent("memo.m4a")
        try AudioWriter.writeWAV(AudioBuffer(samples: [0.1, 0.2], sampleRate: 24_000), to: audio)

        // A Meeting-Minutes-shaped document that reads memo.m4a.
        let doc = FlowDocument(version: "0.8", rows: [
            Row(id: UUID(), task: "Read Audio", settings: "memo.m4a"),
            Row(id: UUID(), task: "Transcribe", model: "Whisper Tiny"),
        ])
        let ws = FlowWorkspace(root: root.appendingPathComponent("flows"))

        let target = try FlowEditRoute.duplicateAndEdit(
            flowID: "01-SpokenSummary", title: "Spoken Summary", document: doc,
            workspace: ws, sourceDir: sourceDir)

        // The copy lands in its own flow folder with the .cat and the bundled asset.
        let dir = ws.directory(for: target.flowID)
        #expect(target.flowID != "01-SpokenSummary")
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("Spoken Summary.cat").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("memo.m4a").path))

        // The editor built from the route opens the copy clean and saves to the real folder.
        let model = FlowEditorModel(name: target.name, flowID: target.flowID,
                                    document: target.document, workspace: ws,
                                    savedText: target.savedText)
        #expect(!model.isDirty)
        #expect(model.runnability == .runnable)
        try model.save()
        #expect(model.savedURL?.deletingLastPathComponent() == dir)
    }

    @Test func editOpenedCopyWritesTheFileIntoTheFlowFolder() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-edit-opened-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let source = """
        mlxflow 0.8
        1. Read Text   memo.txt
        2. Summarize   Qwen3 8B
        3. Save Text   out.md
        """
        let parsed = try CatParser.parseForValidation(source)
        let ws = FlowWorkspace(root: root.appendingPathComponent("flows"))

        let target = try FlowEditRoute.editOpenedCopy(displayName: "My Notes", parsed: parsed,
                                                      workspace: ws)
        let dir = ws.directory(for: target.flowID)
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("My Notes.cat").path))

        // The editor opens it clean (canonical text == on-disk text) and can save.
        let model = FlowEditorModel(name: target.name, flowID: target.flowID,
                                    document: target.document, workspace: ws,
                                    savedText: target.savedText)
        #expect(!model.isDirty)
        #expect(model.canSave)
        #expect(model.document.rows.count == 3)
    }

    @Test func newFlowRouteOpensWithNilDocument() {
        // The gallery's New Flow badge sets editingFlow with a nil document — the editor
        // starts blank (the CFM-R8 route, now driven through the same destination).
        let target = FlowEditTarget(flowID: UUID().uuidString, name: "Untitled Flow",
                                    document: nil, savedText: nil)
        let model = FlowEditorModel(name: target.name, flowID: target.flowID,
                                    document: target.document)
        #expect(model.document.rows.isEmpty)
    }

    // MARK: - CFM-R11-3: .catpipeline saves back as its own kind

    @Test func catpipelineDocumentSavesBackAsCatpipeline() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-pipe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let doc = FlowDocument(version: "0.8", fileKind: .catpipeline, rows: [
            Row(id: UUID(), task: "Generate Image", model: "Z-Image Turbo", settings: "a tree"),
        ])
        let model = FlowEditorModel(name: "Look", flowID: "pipe-flow", document: doc,
                                    workspace: FlowWorkspace(root: root), savedText: nil)
        try model.save()
        #expect(model.savedURL?.pathExtension == "catpipeline")
        let text = try String(contentsOf: try #require(model.savedURL), encoding: .utf8)
        // The header round-trips the kind (the serializer writes it).
        #expect(text.hasPrefix("catpipeline"))
        // Re-opening it keeps the kind.
        let reparsed = try CatParser.parse(text)
        #expect(reparsed.fileKind == .catpipeline)
    }

    @Test func catflowDocumentSavesBackAsCat() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-pipe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = FlowEditorModel(name: "Plain", flowID: "plain-flow",
                                    workspace: FlowWorkspace(root: root))
        model.add(task: "Read Text")
        try model.save()
        #expect(model.savedURL?.pathExtension == "cat")
    }

    // MARK: - CFM-R12-3: rows inside a block are rows

    @Test func flattenListsEveryChildWithItsDepth() throws {
        // 02-MeetingMinutes: a top-level `<parallel three_reads>` with four children.
        let doc = try GalleryLoader.loadDocument(flowID: "02-MeetingMinutes")
        let entries = FlowRowFlatten.flatten(doc.rows, collapsed: [])
        let block = try #require(doc.rows.first { $0.blockKind != nil })
        #expect(entries.contains { $0.row.id == block.id && $0.depth == 0 })
        for child in block.children {
            #expect(entries.contains { $0.row.id == child.id && $0.depth == 1 },
                    "child \(child.id) should appear flattened at depth 1")
        }
        // Round-trip: flattening never changes the serialized bytes of an untouched flow.
        let roundTrip = try CatParser.parse(CatSerializer.serialize(doc))
        #expect(CatSerializer.serialize(roundTrip) == CatSerializer.serialize(doc))
    }

    @Test func flattenHonorsCollapsedBlocks() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "02-MeetingMinutes")
        let block = try #require(doc.rows.first { $0.blockKind != nil })
        let collapsed = FlowRowFlatten.flatten(doc.rows, collapsed: [block.id])
        #expect(collapsed.contains { $0.row.id == block.id })
        #expect(collapsed.filter { $0.row.id != block.id }.allSatisfy { $0.depth == 0 })
        for child in block.children {
            #expect(!collapsed.contains { $0.row.id == child.id })
        }
    }

    @Test func unwrapBlockKeepsTheSteps() throws {
        let r1 = row("Read Text", settings: "memo.txt")
        let child1 = row("Summarize", model: "Qwen3 8B")
        let child2 = row("Save Text", settings: "out.md")
        let block = row(nil, children: [child1, child2], blockKind: .parallel)
        let model = try editor(rows: [r1, block])

        model.unwrapBlock(block.id)

        // The children land in the block's position, in order.
        #expect(model.document.rows.map(\.task) == ["Read Text", "Summarize", "Save Text"])
        #expect(!model.document.rows.contains { $0.id == block.id })
    }

    @Test func emptyBlockBlocksSave() throws {
        let block = row(nil, children: [], blockKind: .parallel)
        let model = try editor(rows: [block])
        #expect(!model.canSave)
        #expect(model.saveBlockReason?.contains("no steps") == true)
    }

    @Test func enclosingBlockNameNamesTheTarget() throws {
        let child1 = row("Summarize", model: "Qwen3 8B")
        let block = row(nil, children: [child1], blockKind: .parallel, blockName: "three_reads")
        let model = try editor(rows: [block])
        #expect(model.enclosingBlockName(for: child1.id) == "three_reads")
        let (blockID, index) = try #require(model.childIndexOf(child1.id))
        #expect(blockID == block.id)
        #expect(index == 0)
        #expect(model.childCount(of: block.id) == 1)
    }

    // MARK: - CFM-R12-FIX-2/4: the block-editing trio

    @Test func flattenedRenderCoversEachLineExactlyOnce() throws {
        // CFM-R12-FIX-2 + QR12R2-1: a block's range is its header only; the flattened list
        // renders each *row* line exactly once (the double-body bug is gone), and a block's
        // clause line (emitted after its children) is drawn too — nothing in the file stops
        // appearing. Runs over **every** gallery flow, not one.
        let flows = GalleryLoader.loadMetadata().map(\.flowID)
        var failures: [String] = []
        for flowID in flows {
            guard let doc = try? GalleryLoader.loadDocument(flowID: flowID) else { continue }
            let serialized = CatSerializer.serializeLines(doc)
            let entries = FlowRowFlatten.flatten(doc.rows, collapsed: [])
            let rendered = entries.flatMap { entry -> [String] in
                var out: [String] = []
                if let range = serialized.lineRanges[entry.row.id] {
                    out.append(contentsOf: serialized.lines[range])
                }
                if let clause = serialized.clauseRanges[entry.row.id] {
                    out.append(contentsOf: serialized.lines[clause])
                }
                return out
            }
            // Every covered line index (a row's range or its clause) renders **exactly once** —
            // no double-body, no dropped line. Order is deliberately not compared: a block's
            // clause line sits after its children in the file but the block cell renders
            // header+clause before the children, and two sibling blocks can carry identical
            // child text at different indices. Structural sections (definitions:/uses:/…)
            // belong to no row and render nowhere (the list is rows-only).
            var counts: [Int: Int] = [:]
            for entry in entries {
                if let r = serialized.lineRanges[entry.row.id] {
                    for idx in r { counts[idx, default: 0] += 1 }
                }
                if let c = serialized.clauseRanges[entry.row.id] {
                    for idx in c { counts[idx, default: 0] += 1 }
                }
            }
            if rendered.count != counts.count {
                failures.append("\(flowID): rendered \(rendered.count) but \(counts.count) covered lines")
            }
            let bad = counts.first { $0.value != 1 }
            if bad != nil {
                failures.append("\(flowID): line \(bad!.key) rendered \(bad!.value) times")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "; ")))

        // QR12R2-1's named case: 45-RouterDesk's block clause (`-> 7`) must render.
        let doc = try GalleryLoader.loadDocument(flowID: "45-RouterDesk")
        let serialized = CatSerializer.serializeLines(doc)
        let rendered = FlowRowFlatten.flatten(doc.rows, collapsed: []).flatMap { entry -> [String] in
            var out: [String] = []
            if let range = serialized.lineRanges[entry.row.id] {
                out.append(contentsOf: serialized.lines[range])
            }
            if let clause = serialized.clauseRanges[entry.row.id] {
                out.append(contentsOf: serialized.lines[clause])
            }
            return out
        }
        #expect(rendered.contains { $0.contains("-> 7") }, "45-RouterDesk's block clause line must render")
    }

    @Test func addingWithABlockHeaderSelectedInsertsInside() throws {
        // CFM-R12-FIX-4: "Add step inside" with the block *header* selected appends as the
        // block's last child — the toolbar label and the action now agree, and an emptied
        // block can be refilled without undo.
        let child1 = row("Summarize", model: "Qwen3 8B")
        let block = row(nil, children: [child1], blockKind: .parallel, blockName: "grp")
        let model = try editor(rows: [row("Read Text", settings: "m.txt"), block])
        model.selectedRowID = block.id
        model.add(task: "Save Text")
        #expect(model.childCount(of: block.id) == 2)
        #expect(model.row(withID: block.id)?.children.last?.task == "Save Text")
        #expect(model.document.rows.count == 2)   // still one top-level block, no stray insert
    }

    @Test func moveInsideKeepsClauseTargetsPointingAtTheSameRow() throws {
        // CFM-R12-FIX-4: reordering a block's children runs the same identity reaim as the
        // top-level move — a decider's edge target never silently re-aims (QR9).
        let gate = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 2)),
        ]))
        let b = row("Summarize", model: "Qwen3 8B")
        let block = row(nil, children: [gate, b], blockKind: .parallel, blockName: "grp")
        let model = try editor(rows: [block])
        model.moveInside(blockID: block.id, from: [1], to: 0)
        let edge = try #require(model.row(withID: gate.id)?.clause?.edges?.first)
        #expect(edge.target == .row(number: 2))
    }
}
