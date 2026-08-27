import Testing
import Foundation
@testable import MLXUI

/// CFM-R9 — the row inspector's editing, and the QR9 gate ("settings must survive a parse →
/// edit → serialize round-trip unchanged"). Covers the byte-preserving settings editor, the
/// decider frame renderers + F004 tag extraction, and the model-layer ops the inspector sits
/// on.
struct CatFlowR9Tests {

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-r9-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "R9", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String?, model: String? = nil, settings: String? = nil,
                     refs: [Ref] = [], children: [Row] = [], blockKind: BlockKind? = nil,
                     clause: Clause? = nil) -> Row {
        Row(id: UUID(), task: task, blockKind: blockKind, model: model, settings: settings,
            refs: refs, children: children, clause: clause)
    }

    // MARK: - QR9: the settings round-trip (CFM-R9-1)

    @Test func editingOneSettingLeavesTheRestByteIdentical() throws {
        let raw = "lang=en; \"TL;DR in 3 bullets\"; timestamps=true"
        let edited = FlowSettingsEditor.replace(key: "lang", value: "fr", in: raw)
        #expect(edited == "lang=fr; \"TL;DR in 3 bullets\"; timestamps=true")
    }

    @Test func editingInsideQuotesDoesNotTouchOtherTokens() throws {
        // The value contains a quoted span that must not be matched by a key scan.
        let raw = "prompt=\"chunk_size=500\"; lang=en"
        let edited = FlowSettingsEditor.replace(key: "lang", value: "de", in: raw)
        #expect(edited == "prompt=\"chunk_size=500\"; lang=de")
    }

    @Test func replacingTheInstructionPreservesSettings() throws {
        let raw = "lang=en; \"TL;DR in 3 bullets\""
        let edited = FlowSettingsEditor.replaceInstruction("one page", in: raw)
        #expect(edited == "lang=en; \"one page\"")
    }

    @Test func removingAKeyLeavesNoDanglingSeparator() throws {
        let raw = "lang=en; timestamps=true"
        let edited = FlowSettingsEditor.replace(key: "lang", value: nil, in: raw)
        #expect(edited == "timestamps=true")
    }

    @Test func addingAKeyAppends() throws {
        #expect(FlowSettingsEditor.replace(key: "lang", value: "en", in: "timestamps=true")
            == "timestamps=true; lang=en")
        #expect(FlowSettingsEditor.replace(key: "lang", value: "en", in: nil) == "lang=en")
    }

    @Test func valuesWithSpacesAreQuoted() throws {
        let edited = FlowSettingsEditor.replace(key: "voice", value: "af heart", in: nil)
        #expect(edited == "voice=\"af heart\"")
    }

    @Test func settingsSurviveParseEditSerializeRoundTrip() throws {
        // The QR9 gate: a row's settings edited through the inspector serialize back with
        // every untouched token byte-identical, and the whole doc re-parses cleanly.
        let r1 = row("Transcribe", model: "Whisper Large v3", settings: "lang=en; timestamps=true")
        let model = try editor([r1])
        model.setSetting(key: "lang", value: "de", for: r1.id)
        #expect(model.row(withID: r1.id)?.settings == "lang=de; timestamps=true")
        model.setInstruction("captions, timestamped", for: r1.id)
        #expect(model.row(withID: r1.id)?.settings == "lang=de; timestamps=true; \"captions, timestamped\"")

        // Save → reparse → the settings string survives.
        try model.save()
        let text = try String(contentsOf: model.savedURL!, encoding: .utf8)
        let reparsed = try CatParser.parse(text)
        #expect(reparsed.rows[0].settings == "lang=de; timestamps=true; \"captions, timestamped\"")
    }

    // MARK: - CFM-R9-2: the model ops

    @Test func modelSettingAndClearing() throws {
        let r1 = row("Transcribe")
        let model = try editor([r1])
        model.setModel("Whisper Large v3", for: r1.id)
        #expect(model.row(withID: r1.id)?.model == "Whisper Large v3")
        model.setModel(nil, for: r1.id)
        #expect(model.row(withID: r1.id)?.model == nil)
    }

    // MARK: - CFM-R14-4 + FIX-7: unbridged picks pin into the models: block

    @Test @MainActor func unbridgedPickWritesIntoModelsBlock() throws {
        // An R14-2 derived pick with no bridge entry is displayed by its catalog `displayName`
        // and pinned `displayName = hfModelId` (FIX-7: display and id are two different
        // strings — a readable row plus a real pin). These need the catalog wired: the pin
        // resolves the displayName → id via the model catalog.
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let r1 = row("Describe Image")
        let model = try editor([r1])
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        let display = "Qwen3-VL-4B-Instruct"
        let unbridged = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
        model.setModel(display, for: r1.id)
        #expect(model.row(withID: r1.id)?.model == display)
        #expect(model.document.models[display] == unbridged)
        #expect(model.document.modelsOrder.contains(display))
        // The serialized file carries the pin and re-parses with the block intact.
        let reparsed = try CatParser.parse(model.catText)
        #expect(reparsed.models[display] == unbridged)
    }

    @Test func bridgedPickStaysOutOfModelsBlock() throws {
        // A bridge-resolvable display never needs a pin — the bridge resolves it at runtime.
        let r1 = row("Transcribe")
        let model = try editor([r1])
        model.setModel("Whisper Large v3", for: r1.id)
        #expect(model.document.models.isEmpty)
    }

    @Test @MainActor func pinDropsWhenLastRowChangesModel() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let r1 = row("Describe Image")
        let r2 = row("Describe Image")
        let model = try editor([r1, r2])
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        let display = "Qwen3-VL-4B-Instruct"
        let unbridged = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
        model.setModel(display, for: r1.id)
        model.setModel(display, for: r2.id)
        #expect(model.document.models[display] == unbridged)
        // Both rows off it → the pin is dropped (nothing uses it anymore).
        model.setModel("LFM2-VL 1.6B", for: r1.id)
        model.setModel("LFM2-VL 1.6B", for: r2.id)
        #expect(model.document.models[display] == nil)
    }

    @Test @MainActor func pinRoundTripsThroughFmt() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let r1 = row("Read Image", settings: "photo.png")
        let r2 = row("Describe Image", refs: [.rowRef(r1.id)])
        let model = try editor([r1, r2])
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        let display = "Qwen3-VL-4B-Instruct"
        model.setModel(display, for: r2.id)
        // fmt idempotence: serialize(parse(serialize(parse(x)))) == serialize(parse(x)).
        let once = model.catText
        let reparsed = try CatParser.parse(once)
        let twice = CatSerializer.serialize(reparsed)
        #expect(twice == once)
        // Reloads clean: no E104, no yellow warning.
        let validation = try CatParser.parseForValidation(once)
        let issues = FlowValidator.checkFlow(validation, workspace: model.workspace, flowID: model.flowID)
        #expect(!issues.contains { $0.code == "E104" })
        #expect(model.warning(for: r2.id) == nil)
    }

    @Test @MainActor func candidateModelsForTaskIncludeThePool() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let transcribe = FlowEditorModel.candidateModels(for: "Transcribe", catalog: catalog,
                                                         claimableModelIDs: claimable)
        // CFM-R14-1 bridged all three ASR gaps + R14-2 derives the pool from the registry:
        // every Transcribe-capable ASR model is offered (Whisper Tiny, Small, Large v3,
        // Voxtral) by its bridge display name.
        #expect(transcribe.map(\.display).contains("Whisper Large v3"))
        #expect(transcribe.map(\.display).contains("Whisper Tiny"))
        #expect(transcribe.map(\.display).contains("Whisper Small"))
        #expect(transcribe.map(\.display).contains("Voxtral Mini 4B Realtime"))
        // RAM-sorted ascending.
        let ram = transcribe.map { $0.model.ramGB }
        #expect(ram == ram.sorted())
    }

    // MARK: - CFM-R14-3: install state sections the Model menu

    @Test @MainActor func installedModelsSectionFirst() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        // Say Whisper Large v3 is installed: it must section under Installed, everything else
        // under Available, and the header total must equal the sum of the rest. The set holds
        // `ModelEntry.id` (the `--` form), exactly what AppState.installedModelIDs stores.
        let largeID = "mlx-community--whisper-large-v3-asr-fp16"
        let sections = FlowEditorModel.sectionedModelCandidates(
            for: "Transcribe", catalog: catalog, installedModelIDs: [largeID],
            claimableModelIDs: claimable)
        #expect(sections.installed.map(\.display) == ["Whisper Large v3"])
        #expect(sections.available.map(\.display).contains("Whisper Tiny"))
        #expect(sections.available.map(\.display).contains("Whisper Small"))
        #expect(sections.available.map(\.display).contains("Voxtral Mini 4B Realtime"))
        let expectedTotal = sections.available.reduce(0.0) { $0 + $1.model.downloadSizeGB }
        #expect(abs(sections.availableTotalGB - expectedTotal) < 0.0001)
    }

    @Test @MainActor func everyCandidateSectionsIntoExactlyOneBucket() throws {
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let all = FlowEditorModel.candidateModels(for: "Transcribe", catalog: catalog,
                                                  claimableModelIDs: claimable)
        let sections = FlowEditorModel.sectionedModelCandidates(
            for: "Transcribe", catalog: catalog, installedModelIDs: [], claimableModelIDs: claimable)
        #expect(sections.installed.isEmpty)
        #expect(sections.available.map { $0.model.id } == all.map { $0.model.id })
        // Installed + available is a partition: no candidate appears in both, none is dropped.
        let ids = Set(all.map(\.model.id))
        #expect(Set(sections.available.map(\.model.id)) == ids)
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
    @Test func suggestedInstructionsPerTask() {
        #expect(FlowEditorModel.suggestedInstructions(for: "Summarize").contains("TL;DR in 3 bullets"))
        #expect(FlowEditorModel.suggestedInstructions(for: "Read Text").isEmpty)
    }

    // MARK: - CFM-R9-5: decisions & budget

    @Test func editingTagsVisitsAndBudget() throws {
        let r1 = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 2)),
            ClauseEdge(tag: "bad", target: .row(number: 3)),
        ]))
        let model = try editor([r1])
        model.setTags(["ok", "bad"], for: r1.id)
        #expect(model.row(withID: r1.id)?.tags == ["ok", "bad"])
        model.setVisitsLeq(3, for: r1.id)
        model.setOnBudget("bad", for: r1.id)
        #expect(model.row(withID: r1.id)?.visitsLeq == 3)
        #expect(model.row(withID: r1.id)?.onBudget == "bad")
        model.setClauseEdge(tag: "ok", target: 4, at: 0, for: r1.id)
        let edges = model.deciderEdges(for: r1.id)
        #expect(edges.map(\.tag) == ["ok", "bad"])
        #expect(edges.map(\.target) == [4, 3])
    }

    // MARK: - CFM-R9-5: the decider frame renderers + F004

    @Test func renderFrameSubstitutesPlaceholders() {
        let asset = Asset(items: [Item(kind: .text, value: "hello", path: nil, sourceText: nil)])
        let prompt = DeciderFrame.renderFrame(
            frameText: "Criterion: {settings}\nAllowed: {tags}\nInput:\n{asset}",
            settings: "keep it short", inputs: [asset], tags: ["yes", "no"])
        #expect(prompt.contains("Criterion: keep it short"))
        #expect(prompt.contains("Allowed: yes, no"))
        #expect(prompt.contains("Input:\nhello"))
    }

    @Test func renderJudgeFrameLabelsCandidatesByTag() {
        let a = Asset(items: [Item(kind: .text, value: "first", path: nil, sourceText: nil)])
        let b = Asset(items: [Item(kind: .text, value: "second", path: nil, sourceText: nil)])
        let prompt = DeciderFrame.renderJudgeFrame(
            frameText: "Pick:\n{candidates}", settings: nil,
            inputs: [a, b], tags: ["a", "b"])
        #expect(prompt.contains("a: first"))
        #expect(prompt.contains("b: second"))
        #expect(prompt.contains("Candidates:"))
    }

    @Test func renderThinkFrameAndTranscript() {
        let asset = Asset(items: [Item(kind: .text, value: "task", path: nil, sourceText: nil)])
        let edges = [ClauseEdge(tag: "search", target: .call(number: 3)),
                     ClauseEdge(tag: "final", target: .resume)]
        let tools = DeciderFrame.renderThinkTools(edges: edges)
        let transcript = DeciderFrame.renderTranscript(entries: [
            FlowInterpreter.TranscriptEntry(tool: "search", input: "q", observation: "found"),
        ])
        let prompt = DeciderFrame.renderThinkFrame(
            frameText: "Task: {asset}\nTools:\n{tools}\nSo far:\n{transcript}\n{tags}",
            settings: nil, inputs: [asset], tags: ["search", "final"],
            tools: tools, transcript: transcript)
        #expect(prompt.contains("- search: calls row 3."))
        #expect(prompt.contains("- final: finish and give your answer."))
        #expect(prompt.contains("[1] search(\"q\")  -> \"found\""))
        #expect(prompt.contains("search | final"))
        #expect(DeciderFrame.renderTranscript(entries: []) == "(nothing yet)")
    }

    @Test func extractTagIsWholeWordCaseInsensitiveLongestFirst() {
        #expect(DeciderFrame.extractTag(from: "The answer is OK.", tags: ["ok", "okay"]) == "ok")
        #expect(DeciderFrame.extractTag(from: "yes please", tags: ["yes", "no"]) == "yes")
        #expect(DeciderFrame.extractTag(from: "I choose the first one.", tags: ["a", "b"]) == nil)
        // Longest-first: "okay" must not be shadowed by "ok" when "okay" is present.
        #expect(DeciderFrame.extractTag(from: "okay then", tags: ["ok", "okay"]) == "okay")
        #expect(DeciderFrame.extractTag(from: "NO", tags: ["no", "yes"]) == "no")
    }

    // MARK: - The QR9 gate, driven off the corpus (CFM-R9-FIX-1/2)

    /// Every distinct settings string across the gallery, conformance, and traces corpora.
    private func corpusSettings() throws -> [(flow: String, settings: String)] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
        let dirs = [
            root.appendingPathComponent("MLXUI/Resources/Gallery"),
            root.appendingPathComponent("Fixtures/CatFlow/conformance"),
            root.appendingPathComponent("Fixtures/CatFlow/traces"),
        ]
        var out: [(String, String)] = []
        var seen = Set<String>()
        for dir in dirs {
            for file in try FileManager.default.contentsOfDirectory(atPath: dir.path)
                .filter({ $0.hasSuffix(".cat") }).sorted() {
                let doc = try CatParser.parse(try String(contentsOf: dir.appendingPathComponent(file), encoding: .utf8))
                for row in allRows(doc.rows) {
                    guard let settings = row.settings, !settings.isEmpty, !seen.contains(settings) else { continue }
                    seen.insert(settings)
                    out.append((file, settings))
                }
            }
        }
        return out
    }

    private func allRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + allRows($0.children) }
    }

    /// FIX-1's gate: a no-op edit is byte-identical for every corpus settings string.
    @Test func noOpEditsAreByteIdenticalAcrossTheCorpus() throws {
        let corpus = try corpusSettings()
        #expect(corpus.count >= 300, "expected the corpus to be substantial (got \(corpus.count))")
        var edited = 0
        for (flow, settings) in corpus {
            let s = FlowSettings(settings)
            // No-op key write (first key → its own value).
            if let key = s.testKeys.sorted().first, let value = s.value(for: key) {
                let round = FlowSettingsEditor.replace(key: key, value: value, in: settings)
                #expect(round == settings, "\(flow): no-op key edit changed '\(settings)' → '\(round ?? "nil")'")
                edited += 1
            }
            // No-op instruction write (first quoted span → its own content).
            if let token = FlowSettingsEditor.firstQuotedToken(in: settings) {
                let content = FlowSettings.unquote(token.token)
                let round = FlowSettingsEditor.replaceInstruction(content, in: settings)
                #expect(round == settings, "\(flow): no-op instruction edit changed '\(settings)' → '\(round ?? "nil")'")
                edited += 1
            }
        }
        #expect(edited > 0)
    }

    /// FIX-2's property: quote is the exact inverse of unquote for every corpus token value
    /// (including the escaped ones like `separator="\n"`).
    @Test func quoteUnquoteRoundTripsAcrossTheCorpus() throws {
        let corpus = try corpusSettings()
        for (flow, settings) in corpus {
            let s = FlowSettings(settings)
            for token in s.testBare {
                #expect(FlowSettings.unquote(FlowSettingsEditor.quote(token)) == token,
                        "\(flow): quote/unquote round-trip failed on bare '\(token)'")
            }
            for key in s.testKeys {
                if let value = s.value(for: key) {
                    #expect(FlowSettings.unquote(FlowSettingsEditor.quote(value)) == value,
                            "\(flow): quote/unquote round-trip failed on \(key)='\(value)'")
                }
            }
        }
    }

    // MARK: - CFM-R9-FIX-1: the quoted-value corruption is gone

    @Test func editingAKeyWithAQuotedValueDoesNotCorrupt() {
        // The reviewer's reproduction: a `key="quoted value"` token's true extent must be
        // spliced, never left with a stale tail.
        let raw = #"contains="retrieval: miss"; on_error=skip"#
        // The edit splices the whole token; the existing quoting style is preserved.
        #expect(FlowSettingsEditor.replace(key: "contains", value: "ZZZ", in: raw)
            == #"contains="ZZZ"; on_error=skip"#)
        // A no-op edit (same value) is byte-identical.
        #expect(FlowSettingsEditor.replace(key: "contains", value: "retrieval: miss", in: raw) == raw)
        // The escaped separator case.
        let sep = #"separator="\n""#
        #expect(FlowSettingsEditor.replace(key: "separator", value: "\n", in: sep) == sep)
    }

    // MARK: - CFM-R9-FIX-2: control-character escapes round-trip

    @Test func quoteEscapesNewlinesTabsAndQuotes() {
        let x = "Context:\n{input}\n\nQ: \"what\""
        let quoted = FlowSettingsEditor.quote(x)
        #expect(quoted == "\"Context:\\n{input}\\n\\nQ: \\\"what\\\"\"")
        #expect(FlowSettings.unquote(quoted) == x)
    }

    // MARK: - CFM-R9-FIX-3: the save gate fails closed

    @Test func saveRefusesAnUnparseableDocument() throws {
        // A settings string the serializer passes through but the parser can't read back — an
        // unterminated quoted span — must fail the save gate *closed*, not open.
        let model = try editor([row("Read Text", settings: "\"unterminated")])
        #expect(!model.canSave)
        #expect(model.saveBlockReason != nil)
        #expect(throws: (any Error).self) { try model.save() }
    }

    // MARK: - CFM-R9-FIX-4: tags and edges stay in sync

    @Test func tagOperationsKeepBothHalvesInSync() throws {
        let r1 = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 2)),
            ClauseEdge(tag: "bad", target: .row(number: 3)),
        ]))
        let r2 = row("Save Text", settings: "out.md")
        let r3 = row("Save Text", settings: "other.md")
        let model = try editor([r1, r2, r3])
        // Rename a tag in the clause → the declared tags follow.
        model.setClauseEdge(tag: "yes", target: 2, at: 0, for: r1.id)
        #expect(model.row(withID: r1.id)?.tags == ["yes", "bad"])
        #expect(model.structuralIssues().isEmpty)
        // Delete an edge → both halves shrink together.
        model.setClauseEdge(tag: nil, target: nil, at: 1, for: r1.id)
        #expect(model.row(withID: r1.id)?.tags == ["yes"])
        #expect(model.row(withID: r1.id)?.clause?.edges?.count == 1)
        #expect(model.structuralIssues().isEmpty)
        // "Add tag" writes an edge AND the tags list.
        model.setClauseEdge(tag: "no", target: 3, at: 1, for: r1.id)
        #expect(model.row(withID: r1.id)?.tags == ["yes", "no"])
        #expect(model.structuralIssues().isEmpty)
    }

    // MARK: - CFM-R9-FIX-5: per-slot targets, done/resume, nested excluded

    @Test func editingTheSecondEdgeTargetKeepsItsOwnTag() throws {
        let r1 = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "yes", target: .row(number: 6)),
            ClauseEdge(tag: "no", target: .row(number: 5)),
        ]))
        let model = try editor([r1])
        model.setClauseEdge(tag: "no", target: 7, at: 1, for: r1.id)
        let edges = model.row(withID: r1.id)?.clause?.edges ?? []
        #expect(edges.count == 2)
        #expect(edges[0].tag == "yes")
        #expect(edges[1].tag == "no")
        #expect(edges[1].target == .row(number: 7))
    }

    @Test func doneAndResumeProduceRealClauseTargets() throws {
        let r1 = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "yes", target: .row(number: 2)),
            ClauseEdge(tag: "no", target: .row(number: 3)),
        ]))
        let model = try editor([r1])
        model.setClauseEdge(tag: "no", target: -1, at: 1, for: r1.id)
        #expect(model.row(withID: r1.id)?.clause?.edges?[1].target == .resume)
        model.setClauseEdge(tag: "yes", target: 0, at: 0, for: r1.id)
        #expect(model.row(withID: r1.id)?.clause?.edges?[0].target == .done)
        // Serializes to valid `-> { yes: done | no: resume }`-style clauses the parser reads back.
        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        #expect(reparsed.rows[0].clause?.edges?.count == 2)
    }

    // MARK: - QR9 (ClauseTarget ruling): reordering never re-aims a decide edge

    @Test func movingRowsRenumbersClauseTargetsByIdentity() throws {
        // [A, Gate(edge→B), B, C] — move A to the end: [Gate, B, C, A], B is now row 2.
        let a = row("Read Text")
        let gate = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 3)),
        ]))
        let b = row("Summarize", model: "Qwen3 8B")
        let c = row("Save Text")
        let model = try editor([a, gate, b, c])

        model.move(from: [0], to: 4)

        let edge = try #require(model.row(withID: gate.id)?.clause?.edges?.first)
        guard case .row(let number) = edge.target else {
            Issue.record("not a row target")
            return
        }
        // B sits at row 2 now; the edge still names B, not whatever is at slot 3.
        #expect(number == 2)
    }

    @Test func insertingRowsRenumbersClauseTargetsBeyondTheGap() throws {
        // [Gate(edge→C), B, C] — insert D after B: [Gate, B, D, C], C is now row 4.
        let gate = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 3)),
        ]))
        let b = row("Summarize", model: "Qwen3 8B")
        let c = row("Save Text")
        let model = try editor([gate, b, c])

        model.selectedRowID = b.id
        model.add(task: "Join Text")

        let edge = try #require(model.row(withID: gate.id)?.clause?.edges?.first)
        guard case .row(let number) = edge.target else {
            Issue.record("not a row target")
            return
        }
        #expect(number == 4)
    }

    @Test func removingANonTargetedRowRenumbersOtherTargets() throws {
        // [Gate(edge→D), B, C, D] — remove B: [Gate, C, D], D is now row 3.
        let gate = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 4)),
        ]))
        let b = row("Summarize", model: "Qwen3 8B")
        let c = row("Save Text", settings: "c.md")
        let d = row("Save Text", settings: "d.md")
        let model = try editor([gate, b, c, d])

        model.remove(b.id)

        let edge = try #require(model.row(withID: gate.id)?.clause?.edges?.first)
        guard case .row(let number) = edge.target else {
            Issue.record("not a row target")
            return
        }
        #expect(number == 3)
    }

    @Test func removingTheTargetedRowGoesStaleAndBlocksSave() throws {
        let gate = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 2)),
        ]))
        let b = row("Summarize", model: "Qwen3 8B")
        let model = try editor([gate, b])

        model.remove(b.id)

        #expect(model.staleClauseSlots(for: gate.id) == [0])
        #expect(!model.canSave)
        #expect(model.saveBlockReason?.contains("deleted") == true)
    }

    @Test func repointingAStaleEdgeClearsIt() throws {
        let gate = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 2)),
        ]))
        let b = row("Summarize", model: "Qwen3 8B")
        let c = row("Save Text", settings: "c.md")
        let model = try editor([gate, b, c])

        model.remove(b.id)
        #expect(!model.canSave)

        // Re-point the edge at the live row that now occupies slot 2.
        model.setClauseEdge(tag: "ok", target: 2, at: 0, for: gate.id)
        #expect(model.staleClauseSlots(for: gate.id).isEmpty)
        #expect(model.canSave)
    }

    @Test func undoingTheRemovalRestoresTheEdge() throws {
        let gate = row("Gate", model: "Ministral 3B", clause: .decide(edges: [
            ClauseEdge(tag: "ok", target: .row(number: 2)),
        ]))
        let b = row("Summarize", model: "Qwen3 8B")
        let model = try editor([gate, b])

        model.remove(b.id)
        #expect(!model.canSave)
        model.undo()
        #expect(model.staleClauseSlots(for: gate.id).isEmpty)
        #expect(model.canSave)
        let edge = try #require(model.row(withID: gate.id)?.clause?.edges?.first)
        guard case .row(let number) = edge.target else {
            Issue.record("not a row target")
            return
        }
        #expect(number == 2)
    }

    // MARK: - CFM-R9-FIX-6: the instruction box reads and writes the same token

    @Test func instructionReadsTheFirstQuotedSpan() throws {
        let r1 = row("Transcribe", model: "Whisper Large v3", settings: "lang=en; \"captions, timestamped\"")
        let model = try editor([r1])
        let s = FlowSettings(r1.settings)
        // The box reads the first QUOTED span, not the first bare token (lang).
        let quoted = FlowSettingsEditor.firstQuotedToken(in: r1.settings ?? "")
        #expect(quoted != nil)
        #expect(FlowSettings.unquote(quoted!.token) == "captions, timestamped")
        // Writing replaces that same span.
        let edited = FlowSettingsEditor.replaceInstruction("one page", in: r1.settings)
        #expect(edited == "lang=en; \"one page\"")
        #expect(s.value(for: "lang") == "en")   // lang untouched
    }

    // MARK: - The R9 exit: 02-MeetingMinutes editable from the inspector

    @Test @MainActor func loadedMeetingMinutesStaysGreenAndEditable() throws {
        // The real gallery flow loads into the editor without going yellow, and the
        // inspector's edits round-trip. Needs the catalog wired: Meeting Minutes has model
        // rows (Transcribe, Summarize), and FIX-6's empty-catalog is "no verdict".
        let (catalog, claimable) = try loadedCatalogAndClaimable()
        let doc = try GalleryLoader.loadDocument(flowID: "02-MeetingMinutes")
        let model = try editor(doc.rows)
        model.modelCatalog = catalog
        model.claimableModelIDs = claimable
        for row in doc.rows {
            #expect(model.warning(for: row.id) == nil, "\(row.task ?? "<block>") should be green")
        }
        #expect(model.canSave)
        // An inspector edit (e.g. the block's declared signature stays, a child's model
        // changes) keeps it saveable.
        let transcribe = doc.rows[1]
        model.setSetting(key: "lang", value: "de", for: transcribe.id)
        #expect(model.canSave)
        try model.save()
        let reparsed = try CatParser.parse(try String(contentsOf: model.savedURL!, encoding: .utf8))
        #expect(reparsed.rows.count == doc.rows.count)
    }
}
