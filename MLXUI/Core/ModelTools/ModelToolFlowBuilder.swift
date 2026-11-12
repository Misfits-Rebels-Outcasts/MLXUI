//
//  ModelToolFlowBuilder.swift
//  MLXUI
//
//  MT5-1: a `ModelToolTrace` → a `.cat` the chat's work can be rerun from. Pure: no files are copied here (the `copies`
//  list says which to copy), nothing is read from disk, and nothing leaves the machine.
//
//  What is a row (owner rulings F-1…F-12, 2026-11-12): the model-tool steps that ran, the attachments they read, the
//  router's answers (a `Template` carrying the context and the user's question, then an `Answer`), and the outputs
//  (`Save` rows, last, with explicit refs). Every other tool is named in the banner and never becomes a row, and no
//  capability flag is ever added. A step whose model name a `.cat` cannot hold is left out (F-5, SPEC-Q249).
//

import Foundation

nonisolated enum ModelToolFlowBuilder {

    /// `Answer`'s row model for Apple Intelligence (the system slot; no `models:` pin, no `offdevice`).
    static let appleDisplay = "apple-foundation @ system"
    /// `ChatMessage.modelID` of Apple Intelligence (`AppleIntelligenceChat.modelID`).
    static let appleModelID = "apple-intelligence"

    /// `nil` for a trace with no step at all and nothing to say; otherwise the flow (possibly with no rows, for a trace
    /// whose every step was left out: the banner says why).
    static func build(_ trace: ModelToolTrace, catalog: ModelToolFlowCatalog = ModelToolFlowCatalog()) -> ModelToolFlow {
        var builder = Builder(trace: trace, catalog: catalog)
        builder.run()
        return builder.finish()
    }

    /// A Template placeholder: `{input}`, `{input:2}`, `{1}`.
    static func hasPlaceholder(_ text: String) -> Bool {
        text.range(of: #"\{(input(:\d+)?|\d+)\}"#, options: .regularExpression) != nil
    }

    /// What the Template under an Answer must say: the producing turn's question and one placeholder per context row.
    struct AnswerExpectation: Sendable, Equatable {
        let question: String
        let contextCount: Int
    }

    static func answerBody(question: String, contextCount: Int) -> String {
        switch contextCount {
        case 0: return question
        case 1: return "Context:\n{input}" + (question.isEmpty ? "" : "\n\nQuestion: " + question)
        default:
            let slots = (1...contextCount).map { "{\($0)}" }.joined(separator: "\n\n")
            return "Context:\n" + slots + (question.isEmpty ? "" : "\n\nQuestion: " + question)
        }
    }

    /// How many Answer rows are not fed exactly the Template their producing turn calls for (its question, its context
    /// count): catches a mis-bound turn as well as a dropped placeholder.
    static func answersMisfed(rows: [Row], expected: [UUID: AnswerExpectation]) -> Int {
        var bad = 0
        for (i, row) in rows.enumerated() {
            guard let want = expected[row.id] else { continue }
            guard i > 0, rows[i - 1].task == "Template",
                  rows[i - 1].settings == FlowSettingsEscaper.quoted(answerBody(question: want.question, contextCount: want.contextCount))
            else { bad += 1; continue }
        }
        return bad
    }

    /// A Template pattern that reads its input: `{input}`, `{input:N}` or `{N}`.
    static func carriesContext(_ pattern: String) -> Bool { hasPlaceholder(pattern) }

    /// How many of the `contextful` Answer rows are fed by a Template with no placeholder (the invariant check).
    static func answersMissingContext(rows: [Row], contextful: Set<UUID>) -> Int {
        var bad = 0
        for (i, row) in rows.enumerated() where row.task == "Answer" && contextful.contains(row.id) {
            guard i > 0, rows[i - 1].task == "Template", carriesContext(rows[i - 1].settings ?? "") else { bad += 1; continue }
        }
        return bad
    }

    /// A file name a `.cat` row can hold: lowercase (an uppercase-leading word reads as a model), `[a-z0-9_-]` plus one extension.
    static func fileName(_ raw: String, kind: String, fallbackBase: String) -> String {
        let ext = kind == "text" ? "txt" : cleanExtension(raw, kind: kind)
        let stem = (raw as NSString).deletingPathExtension.lowercased()
        var base = ""
        var lastDash = false
        for scalar in stem.unicodeScalars {
            let ok = (scalar.value >= 97 && scalar.value <= 122) || (scalar.value >= 48 && scalar.value <= 57) || scalar == "_"
            if ok { base.unicodeScalars.append(scalar); lastDash = false }
            else if !lastDash { base += "-"; lastDash = true }
        }
        base = base.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        if base.isEmpty { base = fallbackBase }
        return base + "." + ext
    }

    private static func cleanExtension(_ raw: String, kind: String) -> String {
        let ext = (raw as NSString).pathExtension.lowercased()
        let valid = !ext.isEmpty && ext.count <= 4 && ext.unicodeScalars.allSatisfy {
            ($0.value >= 97 && $0.value <= 122) || ($0.value >= 48 && $0.value <= 57)
        }
        if valid { return ext }
        switch kind {
        case "audio": return "wav"
        case "video": return "mp4"
        default: return "png"
        }
    }

    // MARK: - The walk

    private struct Turn {
        var userText = ""
        var attachments: [String] = []
        var modelID: String?
        var events: [ModelToolTrace.Event] = []
    }

    private struct Builder {
        let trace: ModelToolTrace
        let catalog: ModelToolFlowCatalog
        var assets: [String: ModelToolTrace.Asset] = [:]
        var rows: [Row] = []
        var saveRows: Set<UUID> = []
        var rowOfHandle: [String: UUID] = [:]
        var consumed: Set<String> = []
        var literalRows: [String: UUID] = [:]
        var answerRows: [String: UUID] = [:]
        var readRows: [String: UUID] = [:]
        var copies: [ModelToolFlow.Copy] = []
        var usedNames: Set<String> = []
        var pins: [String: String] = [:]
        var pinOrder: [String] = []
        var turns: [Turn] = []
        var textOutputs: [Int: [String]] = [:]
        var consumedRouterTexts: Set<String> = []
        /// Answer rows whose turn had text context (attachments or earlier text outputs): each must be fed a Template
        /// that carries it (MT5-2-FIX-1 invariant).
        var contextfulAnswers: Set<UUID> = []
        /// The Answer row a turn's tools consumed router text from (the latest one).
        var routerAnswerByConsumingTurn: [Int: UUID] = [:]
        var contextLost = false
        /// Per Answer row: the question and context count its Template must carry (the producing turn's).
        var answerExpectations: [UUID: AnswerExpectation] = [:]

        // Banner counters.
        var leftOutTools: [(name: String, count: Int)] = []
        var failedCalls = 0
        var unwritable: [String] = []
        var unresolved: [String] = []
        var noRouter = 0
        var missing: [String] = []
        var braceTokens = 0
        var dependsOnLeftOut = 0
        var emptyText = 0
        var contextTrimmed = false
        var unsaved: [String] = []
        var stepsNeedingMissingFile: [String] = []

        init(trace: ModelToolTrace, catalog: ModelToolFlowCatalog) {
            self.trace = trace
            self.catalog = catalog
            for asset in trace.assets { assets[asset.handle] = asset }
        }

        // MARK: Walk

        mutating func run() {
            var current: Turn?
            for event in trace.events {
                if event.kind == .user {
                    if let c = current { turns.append(c) }
                    var t = Turn()
                    t.userText = event.text ?? ""
                    t.attachments = event.attachments ?? []
                    current = t
                } else {
                    if current == nil { current = Turn() }
                    if event.kind == .assistant, current?.modelID == nil { current?.modelID = event.modelID }
                    current?.events.append(event)
                }
            }
            if let c = current { turns.append(c) }

            for index in turns.indices {
                for (position, event) in turns[index].events.enumerated() where event.kind == .tool {
                    handleTool(event, turnIndex: index, position: position)
                }
            }
            emitOutputs()
        }

        // MARK: Rows

        @discardableResult
        mutating func emit(_ row: Row) -> UUID {
            rows.append(row)
            return row.id
        }

        func refs(for producer: UUID) -> [Ref] {
            if let last = rows.last, last.id == producer, !saveRows.contains(producer) { return [] }
            return [.rowRef(producer)]
        }

        /// The display needs a `models:` pin unless the bridge names it (or it is the system slot).
        mutating func pin(_ display: String) -> Bool {
            if display == ModelToolFlowBuilder.appleDisplay || CatalogBridge.entry(for: display) != nil { return true }
            if pins[display] != nil { return true }
            guard let hf = catalog.hfModelIDs[display], !hf.isEmpty else {
                if !unresolved.contains(display) { unresolved.append(display) }
                return false
            }
            pins[display] = hf
            pinOrder.append(display)
            return true
        }

        func routerDisplay(for turn: Turn) -> String? {
            guard let id = turn.modelID else { return nil }
            if id == ModelToolFlowBuilder.appleModelID { return ModelToolFlowBuilder.appleDisplay }
            return catalog.routerDisplays[id]
        }

        /// The row text a tool ran, parsed back: nil unless the model name survives the round trip (F-5).
        func parseStep(_ asset: ModelToolTrace.Asset) -> Row? {
            guard let rowText = asset.rowText, let model = asset.model,
                  let doc = try? CatParser.parse("mlxflow 0.8\n\n\(rowText)\n"),
                  let row = doc.rows.first, doc.rows.count == 1, row.model == model, row.task != nil else { return nil }
            return row
        }

        /// The row that holds `handle`: a step's own row, or the `Read` row of an attachment (emitted at first use).
        mutating func producer(forHandle handle: String) -> UUID? {
            if let id = rowOfHandle[handle] { consumed.insert(handle); return id }
            guard let asset = assets[handle], asset.rowText == nil else { dependsOnLeftOut += 1; return nil }
            if asset.missing == true {
                if !missing.contains(asset.name) { missing.append(asset.name) }
                return nil
            }
            guard let id = emitRead(asset) else { dependsOnLeftOut += 1; return nil }
            consumed.insert(handle)
            return id
        }

        mutating func emitRead(_ asset: ModelToolTrace.Asset) -> UUID? {
            if let id = readRows[asset.handle] { return id }
            let task: String
            switch asset.kind {
            case "image": task = "Read Image"
            case "audio": task = "Read Audio"
            case "text": task = "Read Text"
            default: return nil
            }
            var name = unique(ModelToolFlowBuilder.fileName(asset.name, kind: asset.kind, fallbackBase: "file"))
            if !readsAsFile(task: task, name: name) {
                usedNames.remove(name)
                name = unique("file-" + name)
                guard readsAsFile(task: task, name: name) else { return nil }
            }
            copies.append(ModelToolFlow.Copy(handle: asset.handle, source: asset.fileName, destination: name))
            let id = emit(Row(task: task, settings: name))
            readRows[asset.handle] = id
            rowOfHandle[asset.handle] = id
            return id
        }

        func readsAsFile(task: String, name: String) -> Bool {
            guard let doc = try? CatParser.parse("mlxflow 0.8\n\n1. \(task)   \(name)\n"), let row = doc.rows.first else { return false }
            return (row.model ?? "").isEmpty && row.settings == name
        }

        mutating func unique(_ name: String) -> String {
            var candidate = name
            if usedNames.contains(candidate) {
                let ext = (name as NSString).pathExtension
                let stem = (name as NSString).deletingPathExtension
                var n = 2
                repeat {
                    candidate = "\(stem)-\(n).\(ext)"
                    n += 1
                } while usedNames.contains(candidate)
            }
            usedNames.insert(candidate)
            return candidate
        }

        // MARK: A tool call

        mutating func handleTool(_ event: ModelToolTrace.Event, turnIndex: Int, position: Int) {
            guard let name = event.tool, name != "list_capabilities", name != "read_asset" else { return }
            guard let spec = ModelToolSpec.spec(named: name), spec.task != nil else {
                if event.ok == false { failedCalls += 1 } else { countLeftOut(name) }
                return
            }
            guard event.ok != false, let out = event.output, let outAsset = assets[out], outAsset.rowText != nil else {
                failedCalls += 1
                return
            }
            guard let parsed = parseStep(outAsset), let display = parsed.model else {
                unwritable.append(outAsset.model ?? name)
                return
            }
            guard pin(display) else { return }

            let source: UUID?
            if let handle = event.assetArgument, !handle.isEmpty {
                source = producer(forHandle: handle)
            } else if let text = event.text {
                source = inlineSource(text, turnIndex: turnIndex, position: position)
            } else {
                emptyText += 1
                source = nil
            }
            guard let source else {
                if let handle = event.assetArgument, let a = assets[ModelToolTrace.normalizedHandle(handle)], a.rowText == nil, a.missing == true {
                    stepsNeedingMissingFile.append("\(parsed.task ?? name) with \(display) (its file \(a.name) is gone)")
                }
                return
            }

            let id = emit(Row(task: parsed.task, model: display, settings: parsed.settings, refs: refs(for: source)))
            rowOfHandle[out] = id
            if outAsset.kind == "text" { textOutputs[turnIndex, default: []].append(out) }
        }

        mutating func countLeftOut(_ name: String) {
            if let i = leftOutTools.firstIndex(where: { $0.name == name }) { leftOutTools[i].count += 1 }
            else { leftOutTools.append((name, 1)) }
        }

        /// A tool's inline `text` (F-4): a step's output it repeats, a literal source when it is the user's own text,
        /// otherwise the router's answer.
        mutating func inlineSource(_ text: String, turnIndex: Int, position: Int) -> UUID? {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { emptyText += 1; return nil }
            if let asset = trace.assets.first(where: {
                $0.kind == "text" && rowOfHandle[$0.handle] != nil && $0.rowText != nil
                    && ($0.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines) == trimmed
            }) {
                return producer(forHandle: asset.handle)
            }
            guard !ModelToolFlowBuilder.hasPlaceholder(text) else { braceTokens += 1; return nil }
            let turn = turns[turnIndex]
            let inUser = turn.userText.contains(text) || turn.userText.contains(trimmed)
            let inAttachment = turn.attachments.contains { (assets[$0]?.text ?? "").contains(trimmed) }
            if inUser || inAttachment { return literal(text) }
            // The router wrote this text in some turn, maybe an earlier one: that turn's question and context feed it.
            let producing = producingTurn(of: trimmed, consumedIn: turnIndex, before: position)
            guard let id = answerPair(turnIndex: producing, key: trimmed) else { return nil }
            routerAnswerByConsumingTurn[turnIndex] = id
            return id
        }

        /// The turn whose assistant reply the router wrote `text` as, by transcript order: only a reply that comes BEFORE
        /// the call counts (an earlier turn, or an earlier event of the same turn). Among the preceding replies the
        /// EARLIEST exact match wins, then the earliest containing it: a later reply that merely repeats the text (the
        /// router echoing it after a `speak`) is not where it was written, and the first one carries the real question
        /// and context. A reply after the call never counts. With no preceding reply the consuming turn is the producer.
        func producingTurn(of text: String, consumedIn turn: Int, before position: Int) -> Int {
            var earlier: [(turn: Int, text: String)] = []
            for t in 0...turn {
                for (i, event) in turns[t].events.enumerated() where event.kind == .assistant {
                    if t == turn && i >= position { break }
                    let reply = (event.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    if !reply.isEmpty { earlier.append((t, reply)) }
                }
            }
            if let hit = earlier.first(where: { $0.text == text }) { return hit.turn }
            if let hit = earlier.first(where: { $0.text.contains(text) }) { return hit.turn }
            return turn
        }

        /// Text a turn's answer can draw on: its readable text attachments and the text its steps produced so far.
        func hasOwnContext(turn: Int) -> Bool {
            if !(textOutputs[turn] ?? []).isEmpty { return true }
            return turns[turn].attachments.contains { assets[$0]?.kind == "text" && assets[$0]?.rowText == nil && assets[$0]?.missing != true }
        }

        mutating func literal(_ text: String) -> UUID {
            if let id = literalRows[text] { return id }
            let id = emit(Row(task: "Template", settings: FlowSettingsEscaper.quoted(text)))
            literalRows[text] = id
            return id
        }

        /// `Template` (the context the router read, then the user's question) followed by `Answer` on the router's model.
        mutating func answerPair(turnIndex: Int, key: String) -> UUID? {
            let cacheKey = "\(turnIndex)|\(key)"
            if let id = answerRows[cacheKey] { consumedRouterTexts.insert(key); return id }
            let turn = turns[turnIndex]
            guard let router = routerDisplay(for: turn) else { noRouter += 1; return nil }
            let question = turn.userText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !ModelToolFlowBuilder.hasPlaceholder(question) else { braceTokens += 1; return nil }
            guard pin(router) else { return nil }

            // The context: this turn's text attachments, then the text each earlier step of the turn produced.
            var context: [UUID] = []
            for handle in turn.attachments {
                guard let asset = assets[handle], asset.kind == "text", asset.rowText == nil else { continue }
                if asset.missing == true {
                    if !missing.contains(asset.name) { missing.append(asset.name) }
                    continue
                }
                if let id = emitRead(asset) { consumed.insert(handle); context.append(id) }
            }
            for handle in textOutputs[turnIndex] ?? [] {
                if let id = rowOfHandle[handle] { consumed.insert(handle); context.append(id) }
            }
            if context.count > 4 { context = Array(context.suffix(4)); contextTrimmed = true }

            let body = ModelToolFlowBuilder.answerBody(question: question, contextCount: context.count)
            guard !body.isEmpty else { emptyText += 1; return nil }
            // A question with no {input} must never feed an Answer that had context to read (MT5-2-FIX-1).
            if !context.isEmpty && !ModelToolFlowBuilder.carriesContext(body) { contextLost = true; return nil }

            var refs: [Ref] = []
            if context.count == 1 { refs = self.refs(for: context[0]) }
            else if context.count > 1 { refs = context.map { Ref.rowRef($0) } }
            emit(Row(task: "Template", settings: FlowSettingsEscaper.quoted(body), refs: refs))
            let answer = emit(Row(task: "Answer", model: router))
            if !context.isEmpty { contextfulAnswers.insert(answer) }
            answerExpectations[answer] = AnswerExpectation(question: question, contextCount: context.count)
            answerRows[cacheKey] = answer
            consumedRouterTexts.insert(key)
            return answer
        }

        // MARK: The outputs

        static func saveName(for row: Row?, kind: String) -> String {
            switch row?.task ?? "" {
            case "Speak": return "speech.wav"
            case "Generate Sound": return "music.wav"
            case "Generate Image": return "image.png"
            case "Upscale": return "upscaled.png"
            case "Segment": return "mask.png"
            case "Estimate Depth": return "depth.png"
            case "Translate": return "translation.txt"
            case "Transcribe": return "transcript.txt"
            case "OCR": return "ocr.txt"
            case "Describe Image": return "description.txt"
            default: return kind == "audio" ? "audio.wav" : (kind == "text" ? "output.txt" : "image.png")
            }
        }

        mutating func save(task: String, name: String, producer: UUID) {
            let refs = self.refs(for: producer)
            let id = emit(Row(task: task, settings: unique(name), refs: refs))
            saveRows.insert(id)
        }

        mutating func emitOutputs() {
            // The chat's last reply, answered by the router that wrote it.
            var finalReply: (turn: Int, text: String)?
            for (i, turn) in turns.enumerated() {
                for event in turn.events where event.kind == .assistant {
                    let t = (event.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    if !t.isEmpty { finalReply = (i, t) }
                }
            }
            var replyAnswer: UUID?
            if let reply = finalReply,
               turns[reply.turn].events.contains(where: { $0.kind == .tool })
                || turns[reply.turn].attachments.contains(where: { assets[$0]?.kind == "text" }) {
                if !hasOwnContext(turn: reply.turn), let earlier = routerAnswerByConsumingTurn[reply.turn] {
                    // The reply only restates router text an earlier Answer already produced: save that one.
                    replyAnswer = earlier
                } else {
                    replyAnswer = answerPair(turnIndex: reply.turn, key: reply.text)
                }
            }

            // Media nothing downstream used.
            for asset in trace.assets where asset.rowText != nil && !consumed.contains(asset.handle) {
                guard let id = rowOfHandle[asset.handle] else { continue }
                let row = rows.first { $0.id == id }
                switch asset.kind {
                case "audio": save(task: "Save Audio", name: Builder.saveName(for: row, kind: "audio"), producer: id)
                case "image": save(task: "Save Image", name: Builder.saveName(for: row, kind: "image"), producer: id)
                case "text": continue
                default: unsaved.append(asset.name)
                }
            }

            if let replyAnswer {
                save(task: "Save Text", name: "answer.txt", producer: replyAnswer)
            } else if finalReply == nil,
                      let last = trace.assets.last(where: { $0.kind == "text" && $0.rowText != nil && rowOfHandle[$0.handle] != nil }),
                      let id = rowOfHandle[last.handle] {
                // No reply to save: the chat's last text result is the deliverable (Q-MT5-M).
                let row = rows.first { $0.id == id }
                save(task: "Save Text", name: Builder.saveName(for: row, kind: "text"), producer: id)
            }
        }

        // MARK: Result

        func banner() -> [ModelToolFlowBannerEntry] {
            var out: [ModelToolFlowBannerEntry] = []
            func add(_ kind: ModelToolFlowBannerEntry.Kind, _ count: Int, _ items: [String] = []) {
                if count > 0 { out.append(ModelToolFlowBannerEntry(kind: kind, count: count, items: items)) }
            }
            add(.leftOutTools, leftOutTools.reduce(0) { $0 + $1.count },
                leftOutTools.map { $0.count > 1 ? "\($0.name) x\($0.count)" : $0.name })
            add(.failedCalls, failedCalls)

            var assistantTexts: [String] = []
            for turn in turns {
                for event in turn.events where event.kind == .assistant {
                    let t = (event.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    if !t.isEmpty { assistantTexts.append(t) }
                }
            }
            let used = Set(answerRows.keys.compactMap { $0.split(separator: "|", maxSplits: 1).last.map(String.init) })
            let dropped = assistantTexts.filter { !used.contains($0) }.count
            add(.droppedReplies, dropped)
            let userTurns = trace.events.filter { $0.kind == .user }.count
            add(.history, userTurns > 1 ? userTurns - 1 : 0)

            add(.unwritableModel, unwritable.count, unwritable)
            add(.unresolvedModel, unresolved.count, unresolved)
            add(.noRouter, noRouter)
            add(.braceToken, braceTokens)
            add(.emptyText, emptyText)
            add(.dependsOnLeftOut, dependsOnLeftOut)

            var missingNames = missing
            var idle: [String] = []
            for event in trace.events where event.kind == .user {
                for handle in event.attachments ?? [] {
                    guard let asset = assets[handle], asset.rowText == nil, readRows[handle] == nil else { continue }
                    if asset.missing == true { if !missingNames.contains(asset.name) { missingNames.append(asset.name) } }
                    else if !idle.contains(asset.name) { idle.append(asset.name) }
                }
            }
            add(.missingAttachment, missingNames.count, missingNames)
            add(.stepNeedsMissingFile, stepsNeedingMissingFile.count, stepsNeedingMissingFile)
            add(.unconsumedAttachment, idle.count, idle)
            add(.contextTrimmed, contextTrimmed ? 1 : 0)
            add(.unsavedOutput, unsaved.count, unsaved)
            add(.contextLost, (contextLost || ModelToolFlowBuilder.answersMissingContext(rows: rows, contextful: contextfulAnswers) > 0
                || ModelToolFlowBuilder.answersMisfed(rows: rows, expected: answerExpectations) > 0) ? 1 : 0)
            return out
        }

        func finish() -> ModelToolFlow {
            var document: FlowDocument
            if let seed = try? CatParser.parse("mlxflow 0.8\n\n1. Template   \"x\"\n") {
                document = seed
            } else {
                document = FlowDocument(version: "0.8", headerKeyword: "mlxflow")
            }
            document.rows = rows
            document.models = pins
            document.modelsOrder = pinOrder
            let text = CatSerializer.serialize(document)
            return ModelToolFlow(document: document, text: text, copies: copies, banner: banner(),
                                 roundTrips: Builder.roundTrips(document, text: text)
                                    && ModelToolFlowBuilder.answersMissingContext(rows: rows, contextful: contextfulAnswers) == 0
                                    && ModelToolFlowBuilder.answersMisfed(rows: rows, expected: answerExpectations) == 0)
        }

        static func roundTrips(_ document: FlowDocument, text: String) -> Bool {
            guard !document.rows.isEmpty, let back = try? CatParser.parse(text),
                  back.rows.count == document.rows.count, back.models == document.models else { return false }
            for (a, b) in zip(document.rows, back.rows) {
                guard a.task == b.task, (a.model ?? "") == (b.model ?? ""),
                      (a.settings ?? "") == (b.settings ?? ""), a.refs.count == b.refs.count else { return false }
            }
            return true
        }
    }
}
