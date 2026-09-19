import Foundation

/// The real executor — dispatches a row to its instant tool or its model stage (through
/// the app's registry, injected as a closure so FlowKit stays free of model modules).
/// Mirrors `engines/real.py::_dispatch`: instant tool → frame-backed model → plain model.
///
/// **Frame-backed model rows** (Summarize, Rewrite, …) render the task's frame (settings +
/// asset substituted) and feed that rendered prompt to the LLM stage — `run_framed`'s
/// contract: the runtime-owned frame is the prompt, handed to the model verbatim.
///
/// **Sequential load/release** is correct by construction: each row builds a **fresh** stage
/// from the resolver and holds no reference past its row, so one engine is released before
/// the next loads (there is no engine cache in R2).
nonisolated struct RealExecutor: FlowExecutor {
    let workspace: FlowWorkspace
    let flowID: String
    let blobDirectory: URL
    /// Builds a `PipelineStage` for an installed model. Captures the app's `ModelRegistry`
    /// (constructed on the main actor); the closure hops to the main actor itself.
    let makeModelStage: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage
    /// OCP-2-3: how the row's named model's stage varies with `StageConfig.prompt` — resolved
    /// through the app registry (`AppFlowExecutorFactory`), injected as a closure so FlowKit
    /// stays module-free. `.none` default keeps mock executors / `stageConfig` tests working.
    var promptSupport: @Sendable (ModelEntry) async -> PromptSupport = { _ in .none }
    /// Installed catalog ids, so a not-installed model refuses with the right sentence.
    let installedModelIDs: Set<String>
    /// The flat `browser.json` entries, for `CatalogBridge.resolve`.
    let catalog: [ModelEntry]
    /// CFM-R10-Direct: the flow's declared `transforms:` — a row naming one runs its script
    /// fenced (never in the App Store build; `canRun` refuses there).
    var transforms: [String: TransformDef] = [:]
    /// CFM-R9-5: the decider's fired tag, readable after `execute` (reference box so a value
    /// type can report it through the `FlowExecutor` existential, like `MockExecutor`).
    private let tagBox = TagBox()
    /// CFM-R10-Human: the F002 timeout disclosure from a human row's default, same box.
    private let flagBox = FlagBox()
    /// CFM-R12-8: a staged row's effect, read by the interpreter for `effect_staged`.
    private let stagedBox = StagedBox()

    var lastTag: String? { tagBox.tag }
    var lastTimeoutFlag: (code: String, message: String)? { flagBox.timeout }
    /// RM-2 — the F010 disclosure from a provider decider's strict-parse tag, same box
    /// shape as `lastTimeoutFlag`. `nil` for every non-provider row.
    var lastProviderDeciderFlag: (code: String, message: String)? { flagBox.providerDecider }
    var lastStaged: (id: String, kind: String, summary: String)? { stagedBox.staged }

    /// A reference box for the decider's fired tag.
    private nonisolated final class TagBox: @unchecked Sendable {
        var tag: String?
    }

    /// A reference box for the F002 timeout disclosure and RM-2's F010 disclosure.
    private nonisolated final class FlagBox: @unchecked Sendable {
        var timeout: (code: String, message: String)?
        var providerDecider: (code: String, message: String)?
    }

    /// CFM-R12-8: a staged row's effect, read by the interpreter for `effect_staged`.
    private nonisolated final class StagedBox: @unchecked Sendable {
        var staged: (id: String, kind: String, summary: String)?
    }

    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        let task = row.task ?? ""
        guard let desc = TaskCatalog.get(task) else {
            throw FlowError.unknownTask(row: task)
        }
        tagBox.tag = nil
        flagBox.timeout = nil
        flagBox.providerDecider = nil

        // CFM-R10-Direct (FIX-2): a row naming a `transforms:` entry runs that script fenced —
        // in the Direct build only. The App Store build compiles the fence out and refuses
        // (canRun refuses these flows earlier; this is defense in depth).
        if let transform = transforms[row.task ?? ""] {
            #if DIRECT_BUILD
            // FIX-1 belt-and-suspenders: even if the compile-time guard were ever removed,
            // the runtime channel gate still refuses.
            if CapabilityGate.isAppStoreBuild {
                throw FlowError.stageFailure(row: path,
                                             message: "the App Store build refuses \(row.task ?? "a transform") — distribute it directly instead.")
            }
            return try FencedRunner.runTransform(transform, row: row, inputs: inputs,
                                                 workspace: workspace, flowID: flowID,
                                                 blobDirectory: blobDirectory)
            #else
            throw FlowError.stageFailure(row: path,
                                         message: "the App Store build refuses \(row.task ?? "a transform") — distribute it directly instead.")
            #endif
        }
        // CFM-R10-Direct: `Improvise` runs its instruction as a fenced shell command in the
        // flow's improvise workdir, snapshotted for undo — Direct build only.
        if row.task == "Improvise" {
            #if DIRECT_BUILD
            // FIX-1 belt-and-suspenders: same runtime channel gate.
            if CapabilityGate.isAppStoreBuild {
                throw FlowError.stageFailure(row: path,
                                             message: "the App Store build refuses Improvise — distribute it directly instead.")
            }
            let output = try FencedRunner.runImprovise(settings: row.settings,
                                                       workspace: workspace, flowID: flowID)
            return Asset(items: [Item(kind: .text, value: output, path: nil, sourceText: nil)])
            #else
            throw FlowError.stageFailure(row: path,
                                         message: "the App Store build refuses Improvise — distribute it directly instead.")
            #endif
        }
        if row.task == "Ask Human" || row.task == "Human Input" {
            let s = FlowSettings(row.settings)
            let timeout = s.value(for: "timeout") ?? ""
            let dflt = s.value(for: "default") ?? ""
            let output = inputs.first ?? Asset(items: [])
            if row.task == "Ask Human" {
                tagBox.tag = dflt.isEmpty ? nil : dflt
                if !timeout.isEmpty && !dflt.isEmpty {
                    flagBox.timeout = ("F002", "Nobody answered by \(timeout) — proceeded as `\(dflt)`, unreviewed.")
                }
            } else if !timeout.isEmpty {
                flagBox.timeout = ("F002", "Nobody answered by \(timeout) — proceeded as `unchanged`, unreviewed.")
            }
            return output
        }

        switch desc.taskClass {
        case .instant:
            return try await runInstant(desc, row: row, inputs: inputs, path: path)
        case .model:
            // AFM-2: a row naming the system slot dispatches to Apple Foundation Models
            // entirely, bypassing `resolveModel`/`makeModelStage` (there is no `ModelEntry`
            // to resolve) — checked before the MLX dispatch below, which is otherwise
            // completely untouched.
            if case .runnable(.system(let systemRef), _, _) = CatalogBridge.resolve(row.model ?? "", catalog: catalog) {
                if TaskCatalog.deciderTasks[task] != nil {
                    return try await runAppleFoundationDecider(desc, row: row, inputs: inputs, path: path,
                                                               transcript: transcript, context: context, ref: systemRef)
                }
                return try await runAppleFoundationFrame(desc, row: row, inputs: inputs, path: path, ref: systemRef)
            }
            // RM-2: a row naming a provider slot dispatches to `ProviderStage` (HTTP)
            // entirely, the same bypass-`resolveModel` shape as `.system` above — there is
            // no `ModelEntry`/local file for a remote row either.
            if case .runnable(.provider(let providerRef), _, _) = CatalogBridge.resolve(row.model ?? "", catalog: catalog) {
                if TaskCatalog.deciderTasks[task] != nil {
                    return try await runProviderDecider(desc, row: row, inputs: inputs, path: path,
                                                        transcript: transcript, context: context, ref: providerRef)
                }
                return try await runProviderFrame(desc, row: row, inputs: inputs, path: path, ref: providerRef)
            }
            // CFM-R9-5: deciders render their own frame and fire a declared tag (F004
            // strict-parse), instead of returning the raw reply as content.
            if TaskCatalog.deciderTasks[task] != nil {
                return try await runDecider(desc, row: row, inputs: inputs, path: path,
                                            transcript: transcript, context: context)
            }
            return try await runModel(desc, row: row, inputs: inputs, path: path)
        case .human, .trigger, .agent:
            // Human rows return their default earlier; trigger fires carry their occurrence
            // via the arming session; agent is channel-refused by `canRun`. Keep a truthful
            // sentence as defense in depth.
            throw FlowError.unsupportedTask(row: path, task: row.task ?? "?")
        case .staged:
            // CFM-R12-8: Stage Send / Stage Post queue a visible outbox entry — never send.
            return try await runStaged(row: row, inputs: inputs, path: path)
        case .net:
            // CFM-R12-9 (approved scope) + WS-2: five real tools now — Web Search joined
            // the four GET-only ones once §0 ruling 2 named its providers.
            return try await runNet(row: row, inputs: inputs, path: path)
        }
    }

    /// CFM-R12-9 + WS-2: dispatch the ported networked tools.
    private func runNet(row: Row, inputs: [Asset], path: String) async throws -> Asset {
        switch row.task {
        case "Web Fetch":
            return try await WebFetchTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        case "HTTP Get":
            return try await HTTPGetTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        case "Fetch Feed":
            return try await FetchFeedTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        case "Download File":
            return try await DownloadFileTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        case "Web Search":
            return try await WebSearchTool(settings: row.settings ?? "").run(inputs: inputs)
        default:
            throw FlowError.unsupportedTask(row: path, task: row.task ?? "?")
        }
    }

    /// CFM-R12-8: stage a Send/Post row into the flow's outbox (the only staged tasks). The
    /// interpreter reads `lastStaged` for the `effect_staged` event.
    private func runStaged(row: Row, inputs: [Asset], path: String) async throws -> Asset {
        let kind: String
        switch row.task {
        case "Stage Send": kind = "send"
        case "Stage Post": kind = "post"
        default:
            throw FlowError.unsupportedTask(row: path, task: row.task ?? "?")
        }
        let result = try OutboxStore.stage(row: row, inputs: inputs, kind: kind,
                                           workspace: workspace, flowID: flowID)
        stagedBox.staged = (result.id, kind, result.summary)
        return Asset(items: [Item(kind: .status, value: result.status, path: nil, sourceText: nil)])
    }

    // MARK: - Instant tools

    private func runInstant(_ desc: TaskDescriptor, row: Row, inputs: [Asset], path: String) async throws -> Asset {
        switch desc.name {
        case "Read Audio":
            return try await ReadAudioTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Text":
            return try await ReadTextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Image":
            return try await ReadImageTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Images":
            return try await ReadImagesTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Files":
            return try await ReadFilesTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read PDF":
            return try await ReadPDFTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Index":
            return try await ReadIndexTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Store Index":
            return try await StoreIndexTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        case "Retrieve":
            return try await RetrieveTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        case "Keyword Search":
            return try await KeywordSearchTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        case "Save Audio":
            return try await SaveAudioTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Save Image":
            return try await SaveImageTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Save Images":
            return try await SaveImagesTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Save Video":
            return try await SaveVideoTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Save Text":
            return try await SaveTextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Split":
            return try await SplitTool(settings: row.settings ?? "")
                .run(bundle(inputs)) { _ in }
        case "Filter":
            return try await FilterTool(settings: row.settings ?? "")
                .run(bundle(inputs)) { _ in }
        case "Dedupe":
            return try await DedupeTool(settings: row.settings ?? "")
                .run(bundle(inputs)) { _ in }
        case "Sort":
            return try await SortTool(settings: row.settings ?? "")
                .run(bundle(inputs)) { _ in }
        case "Extract":
            return try await ExtractTool(settings: row.settings ?? "")
                .run(bundle(inputs)) { _ in }
        case "Count":
            return try await CountTool(settings: row.settings ?? "")
                .run(bundle(inputs)) { _ in }
        case "Join Text":
            return try await JoinTextTool(settings: row.settings ?? "")
                .run(bundle(inputs)) { _ in }
        case "Template":
            return try await TemplateTool(settings: row.settings ?? "")
                .run(bundle(inputs)) { _ in }
        case "Read CSV":
            return try TableTool.readCSV(settings: row.settings, from: try resolveFile(row: row, path: path, inputs: inputs))
        case "Read Video":
            return try await ReadVideoTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Extract Frame":
            return try await ExtractFrameTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Extract Audio":
            return try await ExtractAudioTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Trim":
            return try await TrimTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Mux":
            return try await MuxTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        case "Join Video":
            return try await JoinVideoTool(workspace: workspace, flowID: flowID)
                .run(inputs: inputs)
        case "Detect Edges":
            return try await DetectEdgesTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Detect Pose":
            return try await DetectPoseTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read JSON":
            return try TableTool.readJSON(settings: row.settings, from: try resolveFile(row: row, path: path, inputs: inputs))
        case "Read Context":
            return try await ReadContextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "", path: path)
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Save Context":
            return try await SaveContextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Count Context":
            return try await CountContextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Query Table":
            return try TableTool.queryTable(settings: row.settings, from: try tableInput(row: row, path: path, inputs: inputs))
        case "Set Field":
            return try TableTool.setField(settings: row.settings, from: try tableInput(row: row, path: path, inputs: inputs))
        case "Append Row":
            return try TableTool.appendRow(settings: row.settings, from: try tableInput(row: row, path: path, inputs: inputs))
        case "Merge Record":
            return try TableTool.mergeRecord(settings: row.settings, inputs: inputs)
        case "Table to Text":
            return try TableTool.tableToText(settings: row.settings, from: try tableInput(row: row, path: path, inputs: inputs))
        case "Store Query":
            return try StoreTool.storeQuery(inputs: inputs, settings: row.settings, workspace: workspace, flowID: flowID)
        case "Store Read":
            return try StoreTool.storeRead(inputs: inputs, settings: row.settings, workspace: workspace, flowID: flowID)
        case "Store Write":
            return try StoreTool.storeWrite(inputs: inputs, settings: row.settings, workspace: workspace, flowID: flowID)
        case "Diff":
            // Diff reads inputs[0].items[0] / inputs[1].items[0] — NOT the flattened bundle —
            // so a list-shaped ref is read by its first item only, as in the Python (CFM-FIX-1).
            return try await DiffTool(settings: row.settings ?? "").run(inputs: inputs)
        case "Calculate":
            return try await CalculateTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Range":
            return try await RangeTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Chart":
            return try await ChartTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Compare":
            // CFM-R12-7 group a: a deterministic branch — the fired tag drives the row's
            // `-> { tag: N }` edge, exactly like a decider (the interpreter reads lastTag).
            let result = try CompareTool.run(row: row, inputs: inputs, path: path)
            tagBox.tag = result.firedTag
            return result.output
        case "Resize":
            return try await ResizeTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Crop":
            return try await CropTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Convert":
            return try await ConvertTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Watermark":
            return try await WatermarkTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Overlay Text":
            return try await OverlayTextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Contact Sheet":
            return try await ContactSheetTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs: inputs)
        default:
            // A catalog task with no Swift tool implementation yet. Name the task, not a
            // misleading kind — the flow declines rather than approximating (rule 5).
            throw FlowError.unsupportedTask(row: path, task: desc.name)
        }
    }

    /// The §2 text tools consume the **bundled flat list** across all refs (the Python's
    /// `_flat_texts(inputs)`) — `Template`'s `{1}`/`{2}` and `Diff`'s `(1,2)` read
    /// positions across refs, not just the first input.
    private func bundle(_ inputs: [Asset]) -> Asset {
        Asset(items: inputs.flatMap { $0.items })
    }

    // MARK: - Model rows

    private func runModel(_ desc: TaskDescriptor, row: Row, inputs: [Asset], path: String) async throws -> Asset {
        // Text to Table (DA-6, SPEC-Q112) — must run **before** `resolveModel`: a model-less
        // row is legitimate (its deterministic fast path inverts CSV a sibling
        // `Table to Text format=csv` row produced), where `resolveModel(display: nil)` throws.
        if desc.refName == "engines.llm.text_to_table" {
            return try await runTextToTable(row: row, inputs: inputs, path: path)
        }

        let modelEntry = try resolveModel(display: row.model, path: path)

        // Extract Structured (DA-3a) — its own branch, ahead of the generic tail. A plain
        // LLM stage would return whatever prose the model emits, never a `.table` (backlog
        // §0b). `ExtractStructuredStage` owns the table's structure; the model is only ever
        // asked "another record?" (a yes/no tag via `fireTag`) and one bounded field value
        // at a time.
        if desc.refName == "engines.llm.extract_structured" {
            let stage = try await makeExtractionStage(modelEntry)
            let (gate, field) = Self.llmExtractionCalls(stage: stage, path: path)
            return try await ExtractStructuredStage.extract(
                inputs: inputs, settings: row.settings, gate: gate, field: field,
                in: blobDirectory)
        }

        // Frame-backed: render the frame into a prompt, then run the LLM stage on it. The
        // full reference bundle feeds the renderer (B1) — `Rewrite (1,2)` sees both refs.
        // FV-2: goes through `FramePreview.render`, the same function the Properties-tab
        // preview calls, so a run and its preview cannot render different prompts.
        if desc.refKind == .frame {
            guard !inputs.isEmpty else {
                throw FlowError.badInputCardinality(row: "\(path)", expected: "a text input", got: 0)
            }
            let prompt = try FramePreview.render(task: desc.name, refName: desc.refName,
                                                 settings: row.settings, assets: inputs, tags: [])
            // SET-1 / RA-04: a framed generation row (Summarize, Rewrite, Revise, …) receives
            // its manifest's sampler defaults — before this it always ran at StageConfig's
            // hardcoded 512 tokens whatever the manifest or a `max_tokens=` row setting said.
            let config = try llmRunConfig(.default, model: row.model, rowSettings: row.settings, path: "\(path)")
            let stage = try await makeModelStage(modelEntry, config)
            let result = try await stage.run(.text(prompt), progress: { _ in })
            return try persist(result, rowLabel: "\(path)")
        }

        // Rerank: list-shaped ([text] + query → [text], reordered), never a
        // `SingleMediaStage` — that adapter requires exactly one item and must not be
        // widened (MoC-3-3, RSI/DelegateMoCBacklog.md). `stageConfig` already refused a
        // missing query with a named error; `config.query` is guaranteed here.
        if desc.refName.hasPrefix("engines.rerank.") {
            guard let input = inputs.first else {
                throw FlowError.badInputCardinality(row: "\(path)", expected: "a list of text items", got: 0)
            }
            let config = try stageConfig(for: desc, row: row, path: "\(path)")
            guard let query = config.query else {
                throw FlowError.missingRerankQuery(row: "\(path)")
            }
            let stage = try await makeModelStage(modelEntry, config)
            // The seam MoC-4's RerankSDK conforms to: its `PipelineStage` is `.text → .text`,
            // called once per candidate (never batched — right-padding a batch corrupts the
            // last-token read for every candidate shorter than the longest), with the query
            // and candidate joined by a newline; the returned text parses as the score.
            let scorer: @Sendable (String, String) async throws -> Double = { query, candidate in
                let result = try await stage.run(.text("\(query)\n\(candidate)")) { _ in }
                guard case .text(let scoreText) = result,
                      let score = Double(scoreText.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw FlowError.stageFailure(row: "\(path)", message: "the rerank model returned a non-numeric score")
                }
                return score
            }
            let rerank = RerankStage(id: modelEntry.id, name: display(row), rowLabel: "\(path)",
                                     query: query, topK: config.topK, scorer: scorer)
            return try await rerank.run(input) { _ in }
        }

        // Plain model: wrap the registry stage in SingleMediaStage (unwraps one Item,
        // runs, persists heavy outputs file-backed). A row with no upstream input falls
        // back to the settings' first bare token as the text prompt for the prompt-driven
        // diffusion engines — the Python `_prompt` rule (row 1 of `65-VoiceoverBed`'s
        // `Generate Sound` carries its prompt in the settings).
        let media: Asset
        if let input = inputs.first {
            media = input
        } else if desc.refName.hasPrefix("engines.diffusion.generate_") {
            guard let prompt = FlowSettings(row.settings).firstBare() else {
                throw FlowError.missingInlineValue(row: "\(path)", kind: .text)
            }
            media = Asset(items: [Item(kind: .text, value: prompt, path: nil, sourceText: nil)])
        } else {
            // SPEC-Q228: the reference's `engines/embed.py` falls back to the settings string
            // here too (`engines.embed.embed`), the same way the diffusion branch above does —
            // this `else` throws instead. A no-input `Embed` row fails on this runtime where
            // the reference succeeds; confirmed live in 3 bundled flows (17-AskYourDocs,
            // 19-CorrectiveRag, 44-FrontierEscalate). Not fixed here — its own cycle.
            throw FlowError.badInputCardinality(row: "\(path)", expected: "an input", got: 0)
        }
        let config = try await sanitizedPrompt(
            stageConfig(for: desc, row: row, path: "\(path)"), for: modelEntry)
        let stage = try await makeModelStage(modelEntry, config)
        let adapter = SingleMediaStage(id: modelEntry.id, name: display(row),
                                       inner: stage, rowLabel: "\(path)",
                                       blobDirectory: blobDirectory)
        return try await adapter.run(media) { _ in }
    }

    /// Text to Table (DA-6, SPEC-Q112). The asymmetry, verbatim from
    /// `engines/llm.py::text_to_table`: a row with **no** model tries `parseDelimitedTable`
    /// and raises only if it genuinely fails; a row that **names** one skips the fast path
    /// entirely and runs the extraction loop (`ExtractStructuredStage.extractRows`, shared per
    /// SPEC-Q112). Explicit beats implicit.
    private func runTextToTable(row: Row, inputs: [Asset], path: String) async throws -> Asset {
        guard row.model != nil else {
            return try await TextToTableStage.toTable(
                inputs: inputs, settings: row.settings, model: nil, in: blobDirectory)
        }
        let modelEntry = try resolveModel(display: row.model, path: path)
        let stage = try await makeExtractionStage(modelEntry)
        let (gate, field) = Self.llmExtractionCalls(stage: stage, path: path)
        return try await TextToTableStage.toTable(
            inputs: inputs, settings: row.settings, model: (gate, field), in: blobDirectory)
    }

    /// The one LLM stage both `Extract Structured` and a model-named `Text to Table` build:
    /// `maxTokens` = `DEFAULT_MAX_FIELD_TOKENS` (the yes/no gate needs only a word — a shared
    /// cap, a noted divergence from the Python whose `decide` path isn't token-bounded), and
    /// **`temperature: 0`** (DA-5 — the Python pins `temp=0.0` on both calls; a sampled gate
    /// makes the loop's *termination* non-deterministic).
    private func makeExtractionStage(_ model: ModelEntry) async throws -> any PipelineStage {
        try await makeModelStage(
            model, StageConfig(maxTokens: ExtractStructuredStage.maxFieldTokens, temperature: 0))
    }

    /// `_extract_rows`'s two model calls, bridged onto a built stage: the yes/no gate
    /// (`decide(…, tags=("yes","no"))` → `fireTag`) and one bounded line
    /// (`_mlx_complete_line` → the stage's first output line).
    private static func llmExtractionCalls(stage: any PipelineStage, path: String)
        -> (gate: @Sendable (String) async throws -> String,
            field: @Sendable (String) async throws -> String) {
        let gate: @Sendable (String) async throws -> String = { prompt in
            let (tag, _) = try await Self.fireTag(
                stage: stage, prompt: prompt,
                tags: ExtractStructuredStage.continueTags, rowLabel: path)
            return tag
        }
        let field: @Sendable (String) async throws -> String = { prompt in
            let result = try await stage.run(.text(prompt), progress: { _ in })
            guard case .text(let text) = result else { return "" }
            return text.split(separator: "\n", maxSplits: 1,
                              omittingEmptySubsequences: false).first.map(String.init) ?? text
        }
        return (gate, field)
    }

    private func display(_ row: Row) -> String { row.model ?? "" }

    /// A `.table` item from the first input (the table tools' input contract).
    private func tableInput(row: Row, path: String, inputs: [Asset]) throws -> Item {
        guard let item = inputs.first?.items.first, item.kind == .table else {
            throw FlowError.badInputCardinality(row: "\(path)", expected: "a table input", got: 0)
        }
        return item
    }

    /// Resolve a Read tool's source file: an upstream `.file` item, else the settings path.
    /// FIP-3: routed through the shared `ReadPath.resolve`, so `Read CSV`/`Read JSON` report a
    /// missing file the same house-voice way as the other nine `Read *` tasks, instead of
    /// letting `TableTool.readCSV`/`readJSON`'s raw `Data(contentsOf:)` failure escape uncaught.
    private func resolveFile(row: Row, path: String, inputs: [Asset]) throws -> URL {
        do {
            return try ReadPath.resolve(workspace: workspace, flowID: flowID, path: path,
                                        settings: row.settings ?? "", inputs: inputs,
                                        kind: .file, row: row.task ?? "?")
        } catch FlowError.missingInlineValue {
            throw FlowError.badInputCardinality(row: row.task ?? "?", expected: "a file path", got: 0)
        }
    }

    private func resolveModel(display: String?, path: String) throws -> ModelEntry {
        guard let display else {
            throw FlowError.missingInlineValue(row: "\(path)", kind: .text)
        }
        let modelEntry: ModelEntry
        switch CatalogBridge.resolve(display, catalog: catalog) {
        case .runnable(let slot, _, _):
            guard let entry = slot.modelEntry else {
                // MS: a non-cataloged slot (system/provider) has no MLX `ModelEntry` to load.
                // AFM/RM each add their own stage before this path is reachable for real —
                // `resolve` never produces one of these yet (their registries are empty).
                throw FlowError.modelNotRunnable(row: "\(path)", display: slot.displayName,
                                                 reason: "\(slot.displayName) doesn't run through the MLX executor.")
            }
            modelEntry = entry
        case .notRunnable(let name, let reason, _):
            throw FlowError.modelNotRunnable(row: "\(path)", display: name, reason: reason)
        }
        guard installedModelIDs.contains(modelEntry.id) else {
            throw StageError.modelNotInstalled(id: modelEntry.id)
        }
        return modelEntry
    }

    // MARK: - CFM-R9-5: deciders

    /// CFM-R9-5: render the decider's own frame (Judge's candidate labeling, Think's
    /// tool/transcript loop, the rest the shared frame), run the model, and fire a declared
    /// tag via F004 strict-parse + one retry (the `mlx-swift-lm`-portable stand-in for the
    /// Python's token-trie logits mask — see `DeciderFrame`'s SPEC-Q note). Never guesses.
    private func runDecider(_ desc: TaskDescriptor, row: Row, inputs: [Asset], path: String,
                            transcript: [FlowInterpreter.TranscriptEntry]?,
                            context: [(label: String, content: String)]?) async throws -> Asset {
        let modelEntry = try resolveModel(display: row.model, path: path)
        let task = desc.name
        let (basePrompt, tags) = try Self.deciderPrompt(desc, row: row, inputs: inputs,
                                                        transcript: transcript, context: context)

        let stage = try await makeModelStage(modelEntry, .default)
        let (tag, rawReply) = try await Self.fireTag(stage: stage, prompt: basePrompt, tags: tags, rowLabel: path)
        tagBox.tag = tag

        // The payload follows Spec §7.4: Judge delivers the winning candidate (R2); Think
        // the model's own tool/final line (R3); the rest pass through (R1).
        if task == "Judge" {
            let (_, candidates) = DeciderFrame.splitJudgeBundle(inputs, tags: tags)
            if let index = tags.firstIndex(of: tag), index < candidates.count {
                return Asset(items: candidates[index].items)
            }
            return inputs.first ?? Asset(items: [])
        }
        if task == "Think" {
            return Asset(items: [Item(kind: .text, value: rawReply, path: nil, sourceText: nil)])
        }
        return inputs.first ?? Asset(items: [])
    }

    /// The decider's prompt + declared tags, extracted from `runDecider` (AFM-2) so
    /// `runAppleFoundationDecider` builds the identical prompt a real MLX decider would, and
    /// only the tag-acquisition step differs (F004 strict-parse-retry vs. AFM's guided
    /// generation). `static` — reads only its arguments, same reasoning as `fireTag`.
    private static func deciderPrompt(_ desc: TaskDescriptor, row: Row, inputs: [Asset],
                                      transcript: [FlowInterpreter.TranscriptEntry]?,
                                      context: [(label: String, content: String)]?) throws -> (prompt: String, tags: [String]) {
        let task = desc.name
        let settings = row.settings
        let ctxText = context?.map { "\($0.label): \($0.content)" }.joined(separator: "\n")
        let tags = Self.declaredTags(row)

        let basePrompt: String
        if desc.refName.hasPrefix("frames/") {
            // FV-2: goes through `FramePreview.render`, the same function the Properties-tab
            // preview calls (fact 7's Judge/Think/else split now lives there, once, instead
            // of here). `toolLines`/`soFar` are only read by the Think branch inside it;
            // computing them for every decider is cheap (empty edges/transcript otherwise).
            let toolLines = DeciderFrame.renderThinkTools(edges: row.clause?.edges ?? [])
            let soFar = DeciderFrame.renderTranscript(entries: transcript ?? [])
            basePrompt = try FramePreview.render(task: task, refName: desc.refName, settings: settings,
                                                 assets: inputs, tags: tags, tools: toolLines,
                                                 transcript: soFar, context: ctxText)
        } else {
            // `Decide` (the engine-backed decider) has no frame file — the Python's `decide`
            // builds the ask from the asset directly.
            // SPEC-Q230 (does this get a Properties-tab prompt box, and if so what does it
            // say when `text` isn't empty and this fallback never runs?) / SPEC-Q231 (`settings`
            // here is `row.settings` raw — quote marks and escapes included, never unquoted —
            // matching the reference's `_resolve_prompt`; any future box over this value
            // should still display the *unquoted* text, per every other editor in this app).
            let text = DeciderFrame.flatTexts(inputs).joined(separator: "\n")
            basePrompt = text.isEmpty ? settings ?? "" : text
        }
        return (basePrompt, tags)
    }

    // MARK: - AFM-2: Apple Foundation Models dispatch

    /// A frame-backed, non-decider row (Summarize, Rewrite, Translate, …) run against Apple
    /// Foundation Models instead of an MLX stage. Renders the identical frame `runModel`
    /// would (same `FrameRenderer` call), then hands the rendered prompt to the AFM executor
    /// with no system instructions — the frame's own text already carries the whole ask, same
    /// as the MLX path.
    private func runAppleFoundationFrame(_ desc: TaskDescriptor, row: Row, inputs: [Asset],
                                         path: String, ref: SystemModelRef) async throws -> Asset {
        guard case .ready = ref.readiness else {
            throw FlowError.stageFailure(row: path, message: Self.appleFoundationUnreadyMessage(ref))
        }
        guard let executor = AppleFoundationAvailability.makeExecutor() else {
            throw FlowError.stageFailure(row: path,
                                         message: "Apple Intelligence isn't available on this Mac right now.")
        }
        guard desc.refKind == .frame else {
            // AFM-2's curated task list is frame-backed text tasks plus the deciders above;
            // anything else naming the system slot is a manifest/task mismatch, not a row
            // the user can fix by editing settings.
            throw FlowError.stageFailure(row: path,
                                         message: "\(ref.displayName) doesn't serve \(desc.name) yet.")
        }
        guard !inputs.isEmpty else {
            throw FlowError.badInputCardinality(row: "\(path)", expected: "a text input", got: 0)
        }
        let frameName = desc.refName
            .replacingOccurrences(of: "frames/", with: "")
            .replacingOccurrences(of: ".frame.txt", with: "")
        let frame = try FrameRenderer.loadFrame(named: frameName)
        let prompt = try FrameRenderer.render(frameText: frame, settings: row.settings, assets: inputs)
        let text = try await executor.generate(instructions: "", prompt: prompt)
        return try persist(.text(text), rowLabel: "\(path)")
    }

    /// A decider row (Classify, Gate, Score, Judge, Decide) run against Apple Foundation
    /// Models' guided generation — never Spec §12.2's strict-parse-plus-one-retry path, and
    /// never **F010** (AFM-2). Builds the identical prompt `runDecider` would
    /// (`deciderPrompt`), then constrains the model's answer to the row's declared tags.
    /// `Think`'s tool/transcript loop is out of AFM's curated scope (not in AFM-2's decider
    /// list) — reaching here for `Think` means the row named the system slot by hand; refused
    /// the same way an unserved task is above.
    private func runAppleFoundationDecider(_ desc: TaskDescriptor, row: Row, inputs: [Asset], path: String,
                                           transcript: [FlowInterpreter.TranscriptEntry]?,
                                           context: [(label: String, content: String)]?,
                                           ref: SystemModelRef) async throws -> Asset {
        guard case .ready = ref.readiness else {
            throw FlowError.stageFailure(row: path, message: Self.appleFoundationUnreadyMessage(ref))
        }
        guard let executor = AppleFoundationAvailability.makeExecutor() else {
            throw FlowError.stageFailure(row: path,
                                         message: "Apple Intelligence isn't available on this Mac right now.")
        }
        let task = desc.name
        guard task != "Think" else {
            throw FlowError.stageFailure(row: path, message: "\(ref.displayName) doesn't serve Think yet.")
        }
        let (basePrompt, tags) = try Self.deciderPrompt(desc, row: row, inputs: inputs,
                                                        transcript: transcript, context: context)
        guard !tags.isEmpty else {
            throw FlowError.stageFailure(row: path, message: "Row \(path) declares no tags to decide between.")
        }
        let tag = try await executor.generateTag(instructions: "", prompt: basePrompt, tags: tags)
        guard tags.contains(tag) else {
            // Guided generation is supposed to make this unreachable — refuse rather than
            // guess if it ever isn't (the same "never a plausible wrong answer" rule F004
            // enforces on the MLX path).
            throw FlowError.stageFailure(row: path,
                                         message: "Apple Intelligence returned a tag outside the declared set (\(tags.joined(separator: ", "))).")
        }
        tagBox.tag = tag

        if task == "Judge" {
            let (_, candidates) = DeciderFrame.splitJudgeBundle(inputs, tags: tags)
            if let index = tags.firstIndex(of: tag), index < candidates.count {
                return Asset(items: candidates[index].items)
            }
            return inputs.first ?? Asset(items: [])
        }
        return inputs.first ?? Asset(items: [])
    }

    private static func appleFoundationUnreadyMessage(_ ref: SystemModelRef) -> String {
        switch ref.readiness {
        case .needsSetup(let reason, _): return reason
        case .unavailable(let reason): return reason
        case .ready, .needsDownload: return "\(ref.displayName) isn't ready."
        }
    }

    // MARK: - RM-2: provider dispatch

    private static func providerUnreadyMessage(_ ref: ProviderModelRef) -> String {
        switch ref.readiness {
        case .needsSetup(let reason, _): return reason
        case .unavailable(let reason): return reason
        case .ready, .needsDownload: return "\(ref.displayName) isn't ready."
        }
    }

    /// A provider's `max_tokens`, resolved through the manifest's own `resolveEngineSettings`
    /// (SET-1) — the same `maps_to`/default machinery every MLX manifest already uses, so a
    /// row's `max_tokens=` setting (or the manifest's own default) reaches the request body,
    /// never a hardcoded number. Falls back to `512` (the reference's own default) only when
    /// the manifest declares no `max_tokens` setting at all.
    private static func providerMaxTokens(_ ref: ProviderModelRef, row: Row, path: String) throws -> Int {
        let resolved = try ref.manifest.resolveEngineSettings(row.settings, rowLabel: path)
        return resolved["max_tokens"].flatMap(Int.init) ?? 512
    }

    /// A frame-backed row (Answer, Summarize, Rewrite, …) or a plain `.engine` text task
    /// (Generate) run against a remote provider instead of an MLX stage. Mirrors
    /// `runAppleFoundationFrame`'s shape; the frame-vs-plain branch mirrors `runModel`'s own
    /// (a provider manifest's `tasks` list is advisory for the picker, same as AFM's, not a
    /// hard per-task allow-list at dispatch time — `44-FrontierEscalate.cat`'s own `Answer`
    /// row against `claude-sonnet @ anthropic` relies on exactly this: the manifest's
    /// `tasks: ["Generate", "Decide"]` doesn't list `Answer`, and the row still runs).
    private func runProviderFrame(_ desc: TaskDescriptor, row: Row, inputs: [Asset],
                                  path: String, ref: ProviderModelRef) async throws -> Asset {
        guard case .ready = ref.readiness else {
            throw FlowError.stageFailure(row: path, message: Self.providerUnreadyMessage(ref))
        }
        guard !inputs.isEmpty else {
            throw FlowError.badInputCardinality(row: "\(path)", expected: "a text input", got: 0)
        }
        let prompt: String
        if desc.refKind == .frame {
            let frameName = desc.refName
                .replacingOccurrences(of: "frames/", with: "")
                .replacingOccurrences(of: ".frame.txt", with: "")
            let frame = try FrameRenderer.loadFrame(named: frameName)
            prompt = try FrameRenderer.render(frameText: frame, settings: row.settings, assets: inputs)
        } else {
            prompt = DeciderFrame.flatTexts(inputs).joined(separator: "\n")
        }
        let maxTokens = try Self.providerMaxTokens(ref, row: row, path: path)
        let executor = ProviderAvailability.makeExecutor(for: ref.manifest)
        do {
            let text = try await executor.generate(instructions: "", prompt: prompt, maxTokens: maxTokens, temperature: 0)
            return try persist(.text(text), rowLabel: "\(path)")
        } catch let error as ProviderRequestError {
            throw FlowError.stageFailure(row: path, message: (try? ErrorCatalog.fill(
                code: "R904", values: ["provider": ref.displayName, "status": error.status], isV08: true)) ?? "")
        }
    }

    /// A decider row (Classify, Gate, Score, Judge, Decide) run against a remote provider.
    /// Every provider manifest declares `capabilities.constrained_decoding: false`, so this
    /// is Spec §12.2's strict-parse-plus-one-retry path via `fireTag` — the same F004
    /// mechanism the MLX path uses, wrapped through `ProviderStage` (`PipelineStage`) rather
    /// than reimplemented — and, unlike AFM's guided generation, **F010 is disclosed**: the
    /// row's tag is parsed, not guaranteed.
    private func runProviderDecider(_ desc: TaskDescriptor, row: Row, inputs: [Asset], path: String,
                                    transcript: [FlowInterpreter.TranscriptEntry]?,
                                    context: [(label: String, content: String)]?,
                                    ref: ProviderModelRef) async throws -> Asset {
        guard case .ready = ref.readiness else {
            throw FlowError.stageFailure(row: path, message: Self.providerUnreadyMessage(ref))
        }
        let task = desc.name
        // Same scope line AFM-2 drew: Think's tool/transcript loop is a much larger feature
        // than "call the provider and parse a tag," and no ported manifest's `tasks` names
        // it. A row that names Think against a provider directly is refused, not guessed.
        guard task != "Think" else {
            throw FlowError.stageFailure(row: path, message: "\(ref.displayName) doesn't serve Think yet.")
        }
        let (basePrompt, tags) = try Self.deciderPrompt(desc, row: row, inputs: inputs,
                                                        transcript: transcript, context: context)
        guard !tags.isEmpty else {
            throw FlowError.stageFailure(row: path, message: "Row \(path) declares no tags to decide between.")
        }
        let maxTokens = try Self.providerMaxTokens(ref, row: row, path: path)
        let executor = ProviderAvailability.makeExecutor(for: ref.manifest)
        let stage = ProviderStage(id: ref.id, name: ref.displayName, executor: executor,
                                  maxTokens: maxTokens, temperature: 0)
        let tag: String
        do {
            (tag, _) = try await Self.fireTag(stage: stage, prompt: basePrompt, tags: tags, rowLabel: path)
        } catch let error as ProviderRequestError {
            throw FlowError.stageFailure(row: path, message: (try? ErrorCatalog.fill(
                code: "R904", values: ["provider": ref.displayName, "status": error.status], isV08: true)) ?? "")
        }
        tagBox.tag = tag
        flagBox.providerDecider = ("F010", (try? ErrorCatalog.fill(
            code: "F010", values: ["n": path, "task": task, "provider": ref.displayName], isV08: true)) ?? "")

        if task == "Judge" {
            let (_, candidates) = DeciderFrame.splitJudgeBundle(inputs, tags: tags)
            if let index = tags.firstIndex(of: tag), index < candidates.count {
                return Asset(items: candidates[index].items)
            }
            return inputs.first ?? Asset(items: [])
        }
        return inputs.first ?? Asset(items: [])
    }

    /// F004: run the prompt, strict-parse the tag (whole-word, case-insensitive, longest
    /// first); one retry with a stricter ask; then raise rather than guess. Returns the
    /// fired tag and the raw reply (Think's R3 payload).
    ///
    /// `static` (DA-3a): the body reads only its arguments — `ExtractStructuredStage`'s
    /// yes/no gate reuses it as `decide(…, tags=("yes","no"))` without capturing `self`.
    /// (`stageConfig` above is likewise non-`private` for the same in-target reuse reason.)
    static func fireTag(stage: any PipelineStage, prompt: String, tags: [String],
                        rowLabel: String) async throws -> (String, String) {
        func reply(for ask: String) async throws -> String {
            let result = try await stage.run(.text(ask), progress: { _ in })
            if case .text(let s) = result { return s }
            return ""
        }
        let ask = "\(prompt)\n\nAnswer with exactly one of these words, nothing else: \(tags.joined(separator: ", "))"
        var text = try await reply(for: ask)
        if let tag = DeciderFrame.extractTag(from: text, tags: tags) { return (tag, text) }
        let retry = "\(prompt)\n\nAnswer with exactly one word, one of: \(tags.joined(separator: ", ")). Nothing else -- no punctuation, no explanation."
        text = try await reply(for: retry)
        if let tag = DeciderFrame.extractTag(from: text, tags: tags) { return (tag, text) }
        throw FlowError.stageFailure(row: rowLabel,
                                     message: "the model didn't answer with any of the declared tags (\(tags.joined(separator: ", "))) — F004 gave up rather than guess.")
    }

    /// A decider's declared tags: `row.tags` if declared, else the decide clause's edge tags.
    /// `static` (AFM-2) — `deciderPrompt` needs it and reads only its argument. Widened from
    /// `private` for FV-2 (`RSI/DelegateFrameViewBacklog.md` fact 13): the Properties-tab
    /// frame preview needs the same rule for `{tags}`, and re-deriving "row.tags, else the
    /// decide clause's edge tags" in a View would be exactly the second copy §6 warns against.
    static func declaredTags(_ row: Row) -> [String] {
        if let tags = row.tags, !tags.isEmpty { return tags }
        if let edges = row.clause?.edges, !edges.isEmpty { return edges.map(\.tag) }
        return []
    }

    /// Build the `StageConfig` for a model row. For TTS rows the settings' bare token (or
    /// `voice=`) is the voice — matching the Python's `voice = s.get("voice") or
    /// s.first_bare() or "af_heart"` — so `Speak Kokoro 82M; af_heart` uses `af_heart`.
    /// For ASR rows the settings' `lang=` becomes the language — the Python's
    /// `lang = s.get("lang")` — but **defaulted to "en"**: the app's `MLXAudioSTT` has no real
    /// auto-detection, and a `nil` language makes a weak model (e.g. Whisper Small) loop
    /// ("mother mother mother…"). The standalone run sheet already defaults to "en" for the
    /// same reason; this makes the flow behave identically. The default is a deliberate
    /// divergence from the Python (which leaves `language` unset and lets its own `mlx_whisper`
    /// detect) — recorded, not silent.
    /// For diffusion rows (CFM-R16-1) the settings' `seed`/`width`/`height`/`steps` become
    /// the config. The `{item}`/`{index}` substitution (an enclosing `<each>`) and the
    /// `_with_seed` derivation (`CachingExecutor`) have **already** run by the time dispatch
    /// reaches here, so `seed` is a concrete number when the row is stochastic — the stage
    /// sees exactly what the row settled on, never the `seed=auto`/`{item}` token.
    /// Internal (not `private`) so the settings→voice/language/seed mapping is unit-testable.
    func stageConfig(for desc: TaskDescriptor, row: Row, path: String = "?") throws -> StageConfig {
        let settings = FlowSettings(row.settings)
        if desc.refName == "engines.tts.speak" {
            let voice = settings.value(for: "voice") ?? settings.firstBare()
            let speed = Float(settings.value(for: "speed") ?? "") ?? 1.0
            return StageConfig(voice: voice, speed: speed)
        }
        if desc.refName == "engines.asr.transcribe" {
            let language = settings.value(for: "lang") ?? "en"
            return StageConfig(language: language)
        }
        if desc.refName.hasPrefix("engines.diffusion.") {
            let number = { (key: String) -> String? in
                settings.value(for: key)?.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return StageConfig(
                seed: number("seed").flatMap(UInt64.init),
                width: number("width").flatMap(Int.init),
                height: number("height").flatMap(Int.init),
                steps: number("steps").flatMap(Int.init)
            )
        }
        if desc.refName.hasPrefix("engines.rerank.") {
            // `query=` first, else the settings' first bare token — the same fallback
            // `firstBare()` already gives the diffusion prompt above. A row with neither
            // must fail loudly: MoC-3-2 (RSI/DelegateMoCBacklog.md), the CFM-R16-1 lesson
            // applied here — a wrong output nobody notices is worse than an error.
            guard let query = settings.value(for: "query") ?? settings.firstBare() else {
                throw FlowError.missingRerankQuery(row: path)
            }
            // MoC-FIX-2: `query="line one\nline two"` unescapes to a real newline
            // (`FlowSettings.unquote` processes `\n` inside quoted values — verified this
            // applies here too, not just in Swift: the Python `_settings.py::_unquote` does
            // the identical `\\n` → `\n` replacement). `RerankSDK.makeStage`'s stage packs
            // `"\(query)\n\(candidate)"` and splits on the first newline to recover the two
            // halves — a query containing one would silently steal the first line of the
            // first candidate. Refuse it here, at the seam, rather than let it corrupt
            // scores with no error and no crash.
            guard !query.contains("\n") else {
                throw FlowError.invalidSettings(
                    row: path, setting: "query",
                    detail: "can't contain a newline — it's packed with the candidate text on one line internally")
            }
            let topK = settings.value(for: "top_k")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .flatMap(Int.init)
            return StageConfig(query: query, topK: topK)
        }
        if desc.refName == "engines.llm.generate" {
            return try llmRunConfig(.default, model: row.model, rowSettings: row.settings, path: path)
        }
        if desc.refName.hasPrefix("engines.vlm.") {
            // OCP-2-1 (`RSI/DelegateOCRPromptBacklog.md` §4), ported from
            // `catflow-mlx/src/catflow/engines/vlm.py:68-100`: `describe_image` uses
            // `s.first_bare() or "Describe this image."` and model-routed `ocr` uses
            // `s.first_bare() or "Extract all text from this image, verbatim."`. Passing the
            // bare token (or `nil`) is the whole fix — a `Describe Image` row's quoted
            // instruction never reached the model before. `nil` means **the SDK's own
            // default applies** (`VLMSDK.defaultPrompt` / `OCRSDK.ocrPrompt` / a PaddleOCR-VL
            // mode) — never a literal here. For a `.modes` model the token is a recognition
            // mode; `runModel` drops an out-of-set value to `nil` before the stage (OCP-2-3).
            // `lang=` folding (the Python's ` The text is in {lang}.` suffix on `ocr`) is out
            // of scope — a known remaining divergence, recorded in journal `2026-233`.
            return StageConfig(prompt: settings.firstBare())
        }
        return .default
    }

    /// SET-1 — apply the row's serving-manifest `engines.llm.*` settings schema to `base`:
    /// a row's bound `max_tokens=` / `temperature=`, else the manifest's declared defaults,
    /// over `StageConfig`'s hardcoded fallback (Registry §3, `catflow-mlx` `catalog/registry.py`
    /// + `engines/real.py::_resolve_for_run` — RA-04: "a framed row must receive its manifest's
    /// sampler defaults").
    ///
    /// **Framed generation rows' prose will change:** a manifest's `temperature: 0.3` replaces
    /// `StageConfig`'s 0.7 on every row naming that model (Qwen3 8B) — less varied, more
    /// repeatable, aligned with the reference (owner-approved 2026-09-10, journal `2026-251`).
    ///
    /// No model / no manifest / a `settings: {}` manifest → `base` unchanged (SPEC-Q76).
    /// Scope is `engines.llm.*` generation only: the decider gate and Extract Structured /
    /// Text to Table keep their task pins (`makeExtractionStage`, `runDecider`), never reached
    /// from here.
    func llmRunConfig(_ base: StageConfig, model display: String?, rowSettings: String?,
                      path: String) throws -> StageConfig {
        guard let display,
              let entry = CatalogBridge.entry(for: display),
              let manifest = CuratedManifest.load(manifestFile: entry.manifestFile),
              !manifest.settings.isEmpty else { return base }
        let resolved = try manifest.resolveEngineSettings(rowSettings, rowLabel: path)
        var config = base
        if let raw = resolved["max_tokens"], let value = Double(raw), value >= 1 {
            config.maxTokens = Int(value)
        }
        if let raw = resolved["temp"], let value = Float(raw) {
            config.temperature = value
        }
        return config
    }

    /// OCP-2-3 (RULED 2026-09-08: **validator warns, runtime ignores**). For a `.modes` model
    /// an out-of-set `prompt` (a recognition mode that isn't one of the model's) is dropped to
    /// `nil` so the SDK's own default applies. The SDK stays strict — `PaddleOCRSDK.makeStage`
    /// still throws `StageError.unsupportedSetting`, which the Run UI's picker can never
    /// trigger — so the leniency lives here, at the flow boundary the ruling is about. A
    /// `.freeText` / `.none` model's prompt is untouched (a `.none` model ignores it in its
    /// own `makeStage`; the row inspector warns that it is stranded).
    func sanitizedPrompt(_ config: StageConfig, for model: ModelEntry) async -> StageConfig {
        guard case let .modes(values, _) = await promptSupport(model),
              let prompt = config.prompt, !prompt.isEmpty, !values.contains(prompt)
        else { return config }
        var sanitized = config
        sanitized.prompt = nil
        return sanitized
    }

    private func persist(_ media: Media, rowLabel: String) throws -> Asset {
        switch media {
        case .text(let s):
            return Asset(items: [Item(kind: .text, value: s, path: nil, sourceText: nil)])
        case .audio(let buffer):
            let fm = FileManager.default
            try? fm.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
            let url = blobDirectory.appendingPathComponent("\(rowLabel).\(UUID().uuidString).wav")
            try AudioWriter.writeWAV(buffer, to: url)
            return Asset(items: [Item(kind: .audio, value: nil, path: url, sourceText: nil)])
        case .image(let image):
            let fm = FileManager.default
            try? fm.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
            let url = blobDirectory.appendingPathComponent("\(rowLabel).\(UUID().uuidString).png")
            guard let data = PNGEncoder.pngData(from: image.cgImage) else {
                throw FlowError.writeFailed(row: rowLabel, path: url.lastPathComponent)
            }
            try data.write(to: url)
            return Asset(items: [Item(kind: .image, value: nil, path: url, sourceText: nil)])
        case .embedding:
            throw FlowError.unsupportedKind(row: rowLabel, kind: .vector)
        }
    }
}
