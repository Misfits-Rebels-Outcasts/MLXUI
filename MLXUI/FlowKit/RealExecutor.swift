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
    var lastStaged: (id: String, kind: String, summary: String)? { stagedBox.staged }

    /// A reference box for the decider's fired tag.
    private final class TagBox: @unchecked Sendable {
        var tag: String?
    }

    /// A reference box for the F002 timeout disclosure.
    private final class FlagBox: @unchecked Sendable {
        var timeout: (code: String, message: String)?
    }

    /// CFM-R12-8: a staged row's effect, read by the interpreter for `effect_staged`.
    private final class StagedBox: @unchecked Sendable {
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
            // CFM-R9-5: deciders render their own frame and fire a declared tag (F004
            // strict-parse), instead of returning the raw reply as content.
            if TaskCatalog.deciderTasks[task] != nil {
                return try await runDecider(desc, row: row, inputs: inputs, path: path,
                                            transcript: transcript, context: context)
            }
            return try await runModel(desc, row: row, inputs: inputs, path: path)
        case .human, .trigger, .net, .agent:
            // Human rows return their default earlier; trigger fires carry their occurrence
            // via the arming session; net/agent are channel-refused by `canRun`. Keep a
            // truthful sentence as defense in depth.
            throw FlowError.unsupportedTask(row: path, task: row.task ?? "?")
        case .staged:
            // CFM-R12-8: Stage Send / Stage Post queue a visible outbox entry — never send.
            return try await runStaged(row: row, inputs: inputs, path: path)
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
            return try await ReadAudioTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Text":
            return try await ReadTextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Image":
            return try await ReadImageTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Images":
            return try await ReadImagesTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Files":
            return try await ReadFilesTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read PDF":
            return try await ReadPDFTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Index":
            return try await ReadIndexTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
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
            return try TableTool.readCSV(settings: row.settings, from: try resolveFile(row: row, inputs: inputs))
        case "Read JSON":
            return try TableTool.readJSON(settings: row.settings, from: try resolveFile(row: row, inputs: inputs))
        case "Read Context":
            return try await ReadContextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
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
        let modelEntry = try resolveModel(display: row.model, path: path)
        // Frame-backed: render the frame into a prompt, then run the LLM stage on it. The
        // full reference bundle feeds the renderer (B1) — `Rewrite (1,2)` sees both refs.
        if desc.refKind == .frame {
            guard !inputs.isEmpty else {
                throw FlowError.badInputCardinality(row: "\(path)", expected: "a text input", got: 0)
            }
            let frameName = desc.refName
                .replacingOccurrences(of: "frames/", with: "")
                .replacingOccurrences(of: ".frame.txt", with: "")
            let frame = try FrameRenderer.loadFrame(named: frameName)
            let prompt = try FrameRenderer.render(frameText: frame, settings: row.settings, assets: inputs)
            let stage = try await makeModelStage(modelEntry, .default)
            let result = try await stage.run(.text(prompt), progress: { _ in })
            return try persist(result, rowLabel: "\(path)")
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
            throw FlowError.badInputCardinality(row: "\(path)", expected: "an input", got: 0)
        }
        let stage = try await makeModelStage(modelEntry, stageConfig(for: desc, row: row))
        let adapter = SingleMediaStage(id: modelEntry.id, name: display(row),
                                       inner: stage, rowLabel: "\(path)",
                                       blobDirectory: blobDirectory)
        return try await adapter.run(media) { _ in }
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
    private func resolveFile(row: Row, inputs: [Asset]) throws -> URL {
        if let first = inputs.first?.items.first, first.kind == .file, let filePath = first.path {
            return filePath
        }
        guard let raw = FlowSettings(row.settings).pathValue() else {
            throw FlowError.badInputCardinality(row: row.task ?? "?", expected: "a file path", got: 0)
        }
        return try workspace.resolve(raw, flowID: flowID)
    }

    private func resolveModel(display: String?, path: String) throws -> ModelEntry {
        guard let display else {
            throw FlowError.missingInlineValue(row: "\(path)", kind: .text)
        }
        let modelEntry: ModelEntry
        switch CatalogBridge.resolve(display, catalog: catalog) {
        case .runnable(let model, _, _):
            modelEntry = model
        case .notRunnable(let name, let reason):
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
        let settings = row.settings
        let ctxText = context?.map { "\($0.label): \($0.content)" }.joined(separator: "\n")
        let tags = declaredTags(row)

        let basePrompt: String
        if desc.refName.hasPrefix("frames/") {
            let frameName = desc.refName
                .replacingOccurrences(of: "frames/", with: "")
                .replacingOccurrences(of: ".frame.txt", with: "")
            let frameText = try FrameRenderer.loadFrame(named: frameName)
            if task == "Judge" {
                basePrompt = DeciderFrame.renderJudgeFrame(frameText: frameText, settings: settings,
                                                           inputs: inputs, tags: tags, context: ctxText)
            } else if task == "Think" {
                let toolLines = DeciderFrame.renderThinkTools(edges: row.clause?.edges ?? [])
                let soFar = DeciderFrame.renderTranscript(entries: transcript ?? [])
                basePrompt = DeciderFrame.renderThinkFrame(frameText: frameText, settings: settings,
                                                           inputs: inputs, tags: tags, tools: toolLines,
                                                           transcript: soFar, context: ctxText)
            } else {
                basePrompt = DeciderFrame.renderFrame(frameText: frameText, settings: settings,
                                                      inputs: inputs, tags: tags, context: ctxText)
            }
        } else {
            // `Decide` (the engine-backed decider) has no frame file — the Python's `decide`
            // builds the ask from the asset directly.
            let text = DeciderFrame.flatTexts(inputs).joined(separator: "\n")
            basePrompt = text.isEmpty ? settings ?? "" : text
        }

        let stage = try await makeModelStage(modelEntry, .default)
        let (tag, rawReply) = try await fireTag(stage: stage, prompt: basePrompt, tags: tags, rowLabel: path)
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

    /// F004: run the prompt, strict-parse the tag (whole-word, case-insensitive, longest
    /// first); one retry with a stricter ask; then raise rather than guess. Returns the
    /// fired tag and the raw reply (Think's R3 payload).
    private func fireTag(stage: any PipelineStage, prompt: String, tags: [String],
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
    private func declaredTags(_ row: Row) -> [String] {
        if let tags = row.tags, !tags.isEmpty { return tags }
        if let edges = row.clause?.edges, !edges.isEmpty { return edges.map(\.tag) }
        return []
    }

    /// Build the `StageConfig` for a model row. For TTS rows the settings' bare token (or
    /// `voice=`) is the voice — matching the Python's `voice = s.get("voice") or
    /// s.first_bare() or "af_heart"` — so `Speak Kokoro 82M; af_heart` uses `af_heart`.
    /// Internal (not `private`) so the settings→voice mapping is unit-testable.
    func stageConfig(for desc: TaskDescriptor, row: Row) -> StageConfig {
        let settings = FlowSettings(row.settings)
        if desc.refName == "engines.tts.speak" {
            let voice = settings.value(for: "voice") ?? settings.firstBare()
            let speed = Float(settings.value(for: "speed") ?? "") ?? 1.0
            return StageConfig(voice: voice, speed: speed)
        }
        return .default
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
