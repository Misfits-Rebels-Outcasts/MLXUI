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

    func execute(path: String, row: Row, inputs: [Asset]) async throws -> Asset {
        let task = row.task ?? ""
        guard let desc = TaskCatalog.get(task) else {
            throw FlowError.unknownTask(row: task)
        }

        switch desc.taskClass {
        case .instant:
            return try await runInstant(desc, row: row, inputs: inputs)
        case .model:
            return try await runModel(desc, row: row, inputs: inputs, path: path)
        case .human, .trigger, .staged, .net, .agent:
            throw FlowError.unsupportedKind(row: String(path), kind: .file)   // refused by canRun earlier
        }
    }

    // MARK: - Instant tools

    private func runInstant(_ desc: TaskDescriptor, row: Row, inputs: [Asset]) async throws -> Asset {
        switch desc.name {
        case "Read Audio":
            return try await ReadAudioTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Read Text":
            return try await ReadTextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Save Audio":
            return try await SaveAudioTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        case "Save Text":
            return try await SaveTextTool(workspace: workspace, flowID: flowID, settings: row.settings ?? "")
                .run(inputs.first ?? Asset(items: [])) { _ in }
        default:
            throw FlowError.unsupportedKind(row: row.task ?? "?", kind: .file)
        }
    }

    // MARK: - Model rows

    private func runModel(_ desc: TaskDescriptor, row: Row, inputs: [Asset], path: String) async throws -> Asset {
        guard let display = row.model else {
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

        // Frame-backed: render the frame into a prompt, then run the LLM stage on it.
        if desc.refKind == .frame {
            guard let input = inputs.first else {
                throw FlowError.badInputCardinality(row: "\(path)", expected: "a text input", got: 0)
            }
            // refName is "frames/Summarize.frame.txt" — derive the bundle name "Summarize"
            // (loadFrame appends ".frame.txt").
            let frameName = desc.refName
                .replacingOccurrences(of: "frames/", with: "")
                .replacingOccurrences(of: ".frame.txt", with: "")
            let frame = try FrameRenderer.loadFrame(named: frameName)
            let prompt = try FrameRenderer.render(frameText: frame, settings: row.settings, asset: input)
            let stage = try await makeModelStage(modelEntry, .default)
            let result = try await stage.run(.text(prompt), progress: { _ in })
            return try persist(result, rowLabel: "\(path)")
        }

        // Plain model: wrap the registry stage in SingleMediaStage (unwraps one Item,
        // runs, persists heavy outputs file-backed).
        guard let input = inputs.first else {
            throw FlowError.badInputCardinality(row: "\(path)", expected: "an input", got: 0)
        }
        let stage = try await makeModelStage(modelEntry, stageConfig(for: desc, row: row))
        let adapter = SingleMediaStage(id: modelEntry.id, name: display,
                                       inner: stage, rowLabel: "\(path)",
                                       blobDirectory: blobDirectory)
        return try await adapter.run(input) { _ in }
    }

    /// Build the `StageConfig` for a model row. For TTS rows the settings' bare token (or
    /// `voice=`) is the voice — matching the Python's `voice = s.get("voice") or
    /// s.first_bare() or "af_heart"` — so `Speak Kokoro 82M; af_heart` uses `af_heart`.
    /// Internal (not `private`) so the settings→voice mapping is unit-testable.
    func stageConfig(for desc: TaskDescriptor, row: Row) -> StageConfig {
        let settings = FlowSettings(row.settings)
        if desc.refName == "engines.tts.speak" {
            let voice = settings.value(for: "voice") ?? settings.pathValue()
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
