//
//  ModelToolRunner.swift
//  MLXUI — Model Tools, MT1-3 (plan §3.7, steps 1–6, on the path MT1-0 chose).
//
//  A model tool call is a **one-row flow**, run by calling `RealExecutor.execute` directly (never
//  `FlowRunner.run`, which cannot bind an input to a one-row document) on a fresh executor per
//  call. The row text is built through the shared settings escaper, the model is chosen by
//  `CapabilityResolver` only, and the result is a plain sentence (no `ModelToolHost`: flows, the
//  Local Server, tests) or the one-line JSON of plan §3.7 step 6 (a host is bound).
//

import Foundation
import MLXLMCommon
import ImageIO

/// Everything a tool call needs from the app, as a Sendable snapshot: the catalog and what is installed
/// (the same tables the flows use) plus a factory for a fresh executor. Built on the main actor
/// (`ModelToolRuntime.live`); used from any context. `ModelToolRuntime.current` is bound per dispatch.
nonisolated struct ModelToolRuntime: Sendable {
    let catalog: [ModelEntry]
    let installedModelIDs: Set<String>
    let claimableModelIDs: Set<String>
    let totalRAMGB: Double
    /// A **fresh** executor per call (it keeps reference boxes, so one instance is not safe for concurrent rows).
    /// `blobDirectory` is where the executor writes; `conversationID` is `nil` outside a chat.
    let makeExecutor: @Sendable (_ blobDirectory: URL, _ conversationID: UUID?) async -> any FlowExecutor
    /// Tests only: the roots a `path` input may come from, in place of the live grant store. The first one stands in for
    /// the app container when the file is copied out of its security scope.
    var pathRoots: [URL]? = nil

    @TaskLocal static var current: ModelToolRuntime?
}

nonisolated struct ModelToolRunner: Sendable {
    let spec: ModelToolSpec
    let runtime: ModelToolRuntime

    // MARK: - Arguments (validated against the spec)

    enum Value: Sendable, Equatable {
        case string(String)
        case int(Int)
    }

    /// The input arguments a tool takes besides its spec params. They are not in `ModelToolSpec` (data about rows);
    /// they say where the input asset comes from.
    static func inputArgumentNames(for spec: ModelToolSpec) -> [String] {
        switch spec.input {
        case .none: return []
        case .some(.text): return spec.param(named: "asset") != nil ? [] : ["text", "asset"]
        case .some: return ["asset", "path", "url"]
        }
    }

    /// Every argument name the tool accepts, in the order a corrective message lists them.
    static func validArgumentNames(for spec: ModelToolSpec) -> [String] {
        inputArgumentNames(for: spec) + spec.paramNames
    }

    /// Checks `arguments` against the spec. Success carries the typed spec params; failure is the sentence for the router.
    static func validate(_ arguments: [String: JSONValue], for spec: ModelToolSpec) -> Result<[String: Value], ToolFailure> {
        let given = arguments.filter { if case .null = $0.value { return false } else { return true } }
        let order = validArgumentNames(for: spec)
        let valid = Set(order)
        let unknown = given.keys.filter { !valid.contains($0) }.sorted()
        if !unknown.isEmpty {
            let names = unknown.map { "\"\($0)\"" }.joined(separator: ", ")
            let list = order.isEmpty ? "none" : order.joined(separator: ", ")
            return .failure(ToolFailure("unknown argument\(unknown.count == 1 ? "" : "s") \(names) for \(spec.name). Valid arguments: \(list)."))
        }
        var typed: [String: Value] = [:]
        for param in spec.params {
            guard let raw = given[param.name] else { continue }
            switch param.type {
            case .string:
                guard case .string(let s) = raw else {
                    return .failure(ToolFailure("\(param.name) must be text."))
                }
                let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { typed[param.name] = .string(trimmed) }
            case .integer(let min, let max):
                guard let n = integer(raw) else {
                    return .failure(ToolFailure("\(param.name) must be a whole number."))
                }
                if let min, n < min { return .failure(ToolFailure("\(param.name) must be at least \(min) (got \(n)).")) }
                if let max, n > max { return .failure(ToolFailure("\(param.name) must be at most \(max) (got \(n)).")) }
                typed[param.name] = .int(n)
            }
        }
        if let missing = spec.requiredParams.first(where: { typed[$0.name] == nil && $0.placement != .tool }) {
            return .failure(ToolFailure("\(spec.name) needs the \"\(missing.name)\" argument: \(missing.description)"))
        }
        return .success(typed)
    }

    private static func integer(_ value: JSONValue) -> Int? {
        switch value {
        case .int(let i): return i
        case .double(let d): return d.rounded() == d && abs(d) < 1e15 ? Int(d) : nil
        case .string(let s): return Int(s.trimmingCharacters(in: .whitespaces))
        default: return nil
        }
    }

    // MARK: - The one row

    /// The row's settings tokens: the prompt-placed param first (a bare quoted token), then `key=value` settings in spec
    /// order. Every string goes through `FlowSettingsEscaper`; integers are digits. An absent optional sends nothing.
    static func settingsTokens(spec: ModelToolSpec, values: [String: Value]) -> [String] {
        var tokens: [String] = []
        for param in spec.params {
            guard case .rowPrompt = param.placement, case .string(let s)? = values[param.name] else { continue }
            tokens.append(FlowSettingsEscaper.quoted(s))
        }
        for param in spec.params {
            guard case .setting(let key) = param.placement, let value = values[param.name] else { continue }
            switch value {
            case .string(let s): tokens.append(FlowSettingsEscaper.pair(key, s))
            case .int(let n): tokens.append("\(key)=\(n)")
            }
        }
        return tokens
    }

    /// The row text for `spec` on `modelDisplay`.
    static func rowText(spec: ModelToolSpec, modelDisplay: String, values: [String: Value]) -> String {
        guard let task = spec.task else { return "" }
        let tokens = settingsTokens(spec: spec, values: values)
        let tail = tokens.isEmpty ? "" : "; " + tokens.joined(separator: "; ")
        return "1. \(task)   \(modelDisplay)\(tail)"
    }

    /// The one-row document (header + row), through the real parser.
    static func document(rowText: String) throws -> FlowDocument {
        try CatParser.parse("mlxflow 0.8\n\n\(rowText)\n")
    }

    /// The one-row document for a tool call. The text goes through `CatParser`; but a few display names cannot be written in
    /// a `.cat` (the parser reads a lowercase-leading or dotted name such as `dots.ocr` / `olmOCR-2 7B` as settings, DA-8), and
    /// the executor does not care how a `Row` was made, so then the row is built directly with the same model and settings.
    static func oneRowDocument(spec: ModelToolSpec, modelDisplay: String, values: [String: Value]) -> (doc: FlowDocument, row: Row)? {
        let text = rowText(spec: spec, modelDisplay: modelDisplay, values: values)
        guard var doc = try? document(rowText: text), let parsed = doc.rows.first else { return nil }
        if parsed.model == modelDisplay { return (doc, parsed) }
        guard let task = spec.task else { return nil }
        let tokens = settingsTokens(spec: spec, values: values)
        let row = Row(task: task, model: modelDisplay, settings: tokens.isEmpty ? nil : tokens.joined(separator: "; "))
        doc.rows = [row]
        return (doc, row)
    }

    // MARK: - Run

    /// What one call came to, before it is worded for the router.
    enum Outcome: Sendable {
        case failure(ToolFailure)
        /// A download the user declined: the sentence goes back as-is.
        case declined(String)
        case output(Item, model: String, rowText: String, inputHandles: [String], milliseconds: Int)
    }

    /// The result string for the router. Failures are `Error: …` sentences (they never throw).
    func run(arguments: [String: JSONValue]) async -> String {
        let host = ModelToolHost.current
        switch await perform(arguments: arguments, fixedInput: nil, host: host) {
        case .failure(let failure): return failure.text
        case .declined(let sentence): return sentence
        case .output(let item, let model, let rowText, let handles, let ms):
            return await finish(item: item, display: model, rowText: rowText, inputHandles: handles,
                                milliseconds: ms, host: host)
        }
    }

    /// CW-3's entry point (attachment OCR): the text of a local `file`, or `nil` when no model is *installed* for the tool
    /// (it never offers a download), the row is refused, or the run fails. Same resolver, same row, same executor.
    func recognize(file: URL) async -> String? {
        guard case .output(let item, _, _, _, _) = await perform(arguments: [:], fixedInput: file, host: nil),
              item.kind == .text else { return nil }
        return item.value
    }

    private func perform(arguments: [String: JSONValue], fixedInput: URL?, host: ModelToolHost?) async -> Outcome {
        host?.progress(ModelToolProgress(toolName: spec.name, phase: .preparing))

        // 1. arguments
        let values: [String: Value]
        switch Self.validate(arguments, for: spec) {
        case .failure(let failure): return .failure(failure)
        case .success(let typed): values = typed
        }
        guard spec.task != nil else { return .failure(ToolFailure("\(spec.name) runs no model.")) }

        // 2. the model — CapabilityResolver only
        let status = CapabilityResolver.status(
            for: spec, catalog: runtime.catalog, installedModelIDs: runtime.installedModelIDs,
            claimableModelIDs: runtime.claimableModelIDs, totalRAMGB: runtime.totalRAMGB)
        let entry: ModelEntry
        switch status {
        case .ready(let e): entry = e
        case .needsDownload(let e, _): entry = e
        case .cantRun(let reason): return .failure(ToolFailure(reason))
        case .none: return .failure(ToolFailure(Self.noModelSentence(for: spec)))
        }
        let display = TaskModels.displayName(for: entry)

        // 3. the row, and the same gates a flow passes
        let text = Self.rowText(spec: spec, modelDisplay: display, values: values)
        guard let (doc, row) = Self.oneRowDocument(spec: spec, modelDisplay: display, values: values) else {
            return .failure(ToolFailure("couldn't build the request for \(spec.name)."))
        }
        if case .notRunnable(let reason) = FlowRunner.canRun(doc) { return .failure(ToolFailure(reason)) }

        let preflight = FlowPreflight.run(doc, catalog: runtime.catalog, installedModelIDs: runtime.installedModelIDs,
                                          totalRAMGB: runtime.totalRAMGB, claimableModelIDs: runtime.claimableModelIDs)
        if let reason = FlowPreflight.blockedReason(preflight, totalRAMGB: runtime.totalRAMGB) {
            return .failure(ToolFailure(reason))
        }
        if let need = preflight.downloadSet.first {
            // Nothing downloads here. A host asks the user (MT1: always declines); with no host it is today's sentence.
            guard let host else { return .failure(ToolFailure(Self.noModelSentence(for: spec))) }
            let ask = ModelInstallNeed(toolName: spec.name, modelDisplayName: display, modelID: need.id,
                                       downloadGB: preflight.totalDownloadGB)
            guard await host.installRequest(ask) else { return .declined(ModelToolHost.declineSentence(for: display)) }
        }

        // 4. the input
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-tool-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let inputs: [Asset]
        var inputHandles: [String] = []
        var cleanup: [URL] = []
        defer { for url in cleanup { try? FileManager.default.removeItem(at: url) } }
        if let fixedInput, let kind = spec.input {
            inputs = [Asset(items: [Item(kind: kind, path: fixedInput)])]
        } else {
            switch await resolveInput(arguments: arguments, host: host, scratch: scratch) {
            case .failure(let failure): return .failure(failure)
            case .success(let input):
                inputs = input.asset.map { [$0] } ?? []
                inputHandles = input.handles
                cleanup = input.temporaryFiles
            }
        }

        // 5. run — a fresh executor, directly (RealExecutor.execute keeps the error guard, the 1 GiB buffer scope and the lease)
        host?.progress(ModelToolProgress(toolName: spec.name, phase: .running))
        let blobDirectory = host?.assets.blobDirectory ?? scratch.appendingPathComponent("blobs", isDirectory: true)
        try? FileManager.default.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
        let executor = await runtime.makeExecutor(blobDirectory, host?.assets.conversationID)
        let started = Date()
        let output: Asset
        do {
            // MT1-3b: with a host, the row can report progress through `RowProgress.report`; without one, no sink is bound.
            let toolName = spec.name
            let sink: RowProgress.Sink? = host.map { host in
                { @Sendable (update: RowProgress) in
                    host.progress(ModelToolProgress(toolName: toolName, phase: .running,
                                                    detail: update.message, fraction: update.fraction))
                }
            }
            output = try await executor.execute(path: "1", row: row, inputs: inputs, transcript: nil, context: nil,
                                                usedFlowContent: nil, progress: sink)
        } catch {
            if error is CancellationError { return .failure(ToolFailure("the request was stopped.")) }
            return .failure(ToolFailure(FlowErrorDisplay.sentence(for: error)))
        }
        guard let item = output.items.first else { return .failure(ToolFailure("\(spec.name) produced nothing.")) }
        return .output(item, model: display, rowText: text, inputHandles: inputHandles,
                       milliseconds: Int(Date().timeIntervalSince(started) * 1000))
    }

    // MARK: - Result

    private func finish(item: Item, display: String, rowText: String, inputHandles: [String],
                        milliseconds: Int, host: ModelToolHost?) async -> String {
        guard let host else {
            // No host: exactly the plain text the tools returned before Model Tools.
            guard item.kind == .text, let value = item.value else {
                return ToolFailure("\(spec.name) makes a file, and needs a chat to keep it.").text
            }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? Self.emptyResult(for: spec) : trimmed
        }
        host.progress(ModelToolProgress(toolName: spec.name, phase: .saving))
        let provenance = ChatAssetProvenance(rowText: rowText, model: display, inputs: inputHandles, milliseconds: milliseconds)
        do {
            if item.kind == .text, let value = item.value {
                let record = try host.assets.addText(value, name: "\(spec.name) result", provenance: provenance)
                return Self.textResultJSON(record: record, model: display, text: value, budgetTokens: host.inlineBudgetTokens)
            }
            guard let path = item.path else { return ToolFailure("\(spec.name) produced no file.").text }
            let record = try host.assets.adopt(file: path, kind: item.kind.rawValue, name: "\(spec.name) result",
                                               provenance: provenance)
            return Self.mediaResultJSON(record: record, model: display, pixelSize: item.kind == .image ? Self.pixelSize(of: host.assets.url(for: record)) : nil)
        } catch {
            return ToolFailure("couldn't keep the result of \(spec.name).").text
        }
    }

    /// Plan §3.7 step 6. A text within the inline budget comes back in full; a longer one as a 400-character preview and a note.
    static func textResultJSON(record: ChatAssetRecord, model: String, text: String, budgetTokens: Int) -> String {
        let words = record.words ?? ChatAssetStore.wordCount(text)
        var parts = [
            "\"asset\":\(json(record.handle))", "\"kind\":\"text\"", "\"model\":\(json(model))", "\"words\":\(words)",
        ]
        if estimatedTokens(text) <= budgetTokens {
            parts.append("\"text\":\(json(text))")
        } else {
            parts.append("\"preview\":\(json(String(text.prefix(previewCharacters)) + "…"))")
            parts.append("\"note\":\(json("Long text: pass \"\(record.handle)\" to another tool, or call read_asset."))")
        }
        return "{" + parts.joined(separator: ",") + "}"
    }

    /// A media result is a handle, plus the produced size for an image (read from the file's header, no pixels decoded).
    static func mediaResultJSON(record: ChatAssetRecord, model: String, pixelSize: (width: Int, height: Int)? = nil) -> String {
        var parts = ["\"asset\":\(json(record.handle))", "\"kind\":\(json(record.kind))", "\"model\":\(json(model))"]
        if let pixelSize { parts += ["\"width\":\(pixelSize.width)", "\"height\":\(pixelSize.height)"] }
        parts.append("\"note\":\(json("Pass \"\(record.handle)\" to another tool. The user can see it in the chat."))")
        return "{" + parts.joined(separator: ",") + "}"
    }

    /// Image dimensions from the header (ImageIO), without decoding the pixels.
    static func pixelSize(of file: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int, let h = props[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        return (w, h)
    }

    static let previewCharacters = 400

    /// A cheap token estimate (4 UTF-8 bytes per token) — the budget is a guard, not a billing figure.
    static func estimatedTokens(_ text: String) -> Int { (text.utf8.count + 3) / 4 }

    static func json(_ string: String) -> String {
        guard let data = try? JSONEncoder().encode(string), let text = String(data: data, encoding: .utf8) else { return "\"\"" }
        return text
    }

    // MARK: - Sentences

    /// The sentence when nothing can serve the tool (today's wording for the two tools that existed).
    static func noModelSentence(for spec: ModelToolSpec) -> String {
        switch spec.name {
        case "transcribe_audio": return "no speech-to-text model is installed. Install one from the Browse tab first."
        case "ocr_image": return "no OCR or vision model is installed. Install one from the Browse tab first."
        default: return "no model for \(spec.name) is installed. Install one from the Browse tab first."
        }
    }

    static func emptyResult(for spec: ModelToolSpec) -> String {
        switch spec.name {
        case "transcribe_audio": return "(no speech detected in the audio)"
        case "ocr_image": return "(no text found in the image)"
        default: return "(\(spec.name) returned nothing)"
        }
    }
}

/// A failure sentence; renders as `Error: …`.
nonisolated struct ToolFailure: Error, Sendable, Equatable {
    let message: String
    init(_ message: String) { self.message = message }
    var text: String { "Error: \(message)" }
}
