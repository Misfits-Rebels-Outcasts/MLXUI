import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Token → text pipeline (pure; tested without a model)

/// Turns a stream of **raw token ids** into text deltas.
///
/// TCP-1 (journal `2026-230`, readme T1): the `.chunk` text path runs through mlx-swift-lm's
/// `ToolCallProcessor`, which drops the text before a `<` that partially matches `<tool_call>` —
/// `<div><span>` arrives as `<div<span>`. So the server never consumes `.chunk`; it consumes
/// `.token(id)` and detokenizes here. (S2 must keep this regression test when it adds tools.)
///
/// Windowed like Hugging Face's text-generation-inference: decode `ids[prefix...]`, compare with
/// the already-decoded `ids[prefix..<read]`, and hold the delta back while it ends in an
/// incomplete UTF-8 sequence (U+FFFD). Keeps leading-space handling right and stays O(n).
nonisolated struct IncrementalDetokenizer {
    private let decode: ([Int]) -> String
    private var ids: [Int] = []
    private var prefixOffset = 0
    private var readOffset = 0

    init(decode: @escaping ([Int]) -> String) { self.decode = decode }

    mutating func push(_ id: Int) -> String {
        ids.append(id)
        return delta(holdingIncomplete: true)
    }

    /// Whatever is still held back (an incomplete trailing sequence), at end of stream.
    mutating func flush() -> String { delta(holdingIncomplete: false) }

    private mutating func delta(holdingIncomplete: Bool) -> String {
        guard readOffset < ids.count else { return "" }
        let prefixText = decode(Array(ids[prefixOffset..<readOffset]))
        let newText = decode(Array(ids[prefixOffset...]))
        let prefixCount = prefixText.unicodeScalars.count
        guard newText.unicodeScalars.count > prefixCount,
              !(holdingIncomplete && newText.hasSuffix("\u{FFFD}")) else { return "" }
        let tail = String(String.UnicodeScalarView(newText.unicodeScalars.dropFirst(prefixCount)))
        prefixOffset = readOffset
        readOffset = ids.count
        return tail
    }
}

/// Streaming equivalent of `LLMEngine.stripThinkBlock`: drop one leading, **closed**
/// `<think>…</think>` block (and the whitespace after it). An unclosed block is left as-is.
nonisolated struct LeadingThinkStripper {
    private enum State { case deciding, inThink, trimming, passthrough }
    private var state = State.deciding
    private var buffer = ""

    mutating func push(_ text: String) -> String {
        switch state {
        case .passthrough:
            return text
        case .trimming:
            let rest = String(text.drop(while: \.isWhitespace))
            if !rest.isEmpty { state = .passthrough }
            return rest
        case .deciding:
            buffer += text
            let trimmed = String(buffer.drop(while: \.isWhitespace))
            if trimmed.isEmpty || "<think>".hasPrefix(trimmed) { return "" }     // could still become <think>
            guard trimmed.hasPrefix("<think>") else {
                state = .passthrough
                defer { buffer = "" }
                return buffer
            }
            state = .inThink
            return closeIfComplete()
        case .inThink:
            buffer += text
            return closeIfComplete()
        }
    }

    /// End of stream: anything still held (undecided or an unclosed think block) goes out as-is.
    mutating func finish() -> String {
        defer { buffer = ""; state = .passthrough }
        return buffer
    }

    private mutating func closeIfComplete() -> String {
        guard let end = buffer.range(of: "</think>") else { return "" }
        let rest = String(buffer[end.upperBound...].drop(while: \.isWhitespace))
        buffer = ""
        state = rest.isEmpty ? .trimming : .passthrough
        return rest
    }
}

/// `stop` sequences: cut the text at the first match, and hold back the last `longest − 1`
/// characters so a stop string split across two deltas is still caught.
nonisolated struct StopMatcher {
    private let stops: [String]
    private let holdback: Int
    private var held = ""

    init(_ stops: [String]) {
        self.stops = stops
        self.holdback = max(0, (stops.map(\.count).max() ?? 0) - 1)
    }

    mutating func push(_ text: String) -> (emit: String, stopped: Bool) {
        guard !stops.isEmpty else { return (text, false) }
        held += text
        let hits = stops.compactMap { held.range(of: $0)?.lowerBound }
        if let cut = hits.min() {
            let emit = String(held[..<cut])
            held = ""
            return (emit, true)
        }
        guard held.count > holdback else { return ("", false) }
        let emit = String(held.dropLast(holdback))
        held = String(held.suffix(holdback))
        return (emit, false)
    }

    mutating func finish() -> String { defer { held = "" }; return held }
}

/// detokenize → strip a leading think block → apply `stop`.
nonisolated struct ServeTextPipeline {
    private var detokenizer: IncrementalDetokenizer
    private var think = LeadingThinkStripper()
    private var stop: StopMatcher
    private(set) var stopped = false

    init(decode: @escaping ([Int]) -> String, stop: [String]) {
        self.detokenizer = IncrementalDetokenizer(decode: decode)
        self.stop = StopMatcher(stop)
    }

    mutating func push(_ id: Int) -> (text: String, stopped: Bool) {
        guard !stopped else { return ("", true) }
        return process(detokenizer.push(id))
    }

    /// The tail still held anywhere in the pipeline.
    mutating func finish() -> String {
        var out = ""
        if !stopped { out += process(detokenizer.flush()).text }
        if !stopped { out += apply(think.finish()).text }
        return out + stop.finish()
    }

    private mutating func process(_ piece: String) -> (text: String, stopped: Bool) {
        piece.isEmpty ? ("", false) : apply(think.push(piece))
    }

    private mutating func apply(_ text: String) -> (text: String, stopped: Bool) {
        let result = stop.push(text)
        if result.stopped { stopped = true }
        return (result.emit, result.stopped)
    }
}

// MARK: - Sampling defaults

/// What a request that omits `temperature` / `top_p` samples with (owner ruling R20, 2026-10-02):
/// the checkpoint's own `generation_config.json` when it has the key, else **0.7 / 1.0**. An
/// explicit request value always wins (the backend only consults this for the missing ones).
nonisolated struct SamplingDefaults: Equatable {
    var temperature: Double
    var topP: Double

    static let fallback = SamplingDefaults(temperature: 0.7, topP: 1.0)

    /// Each key falls back independently, so a config with only `top_p` still gets 0.7.
    static func parse(_ data: Data?) -> SamplingDefaults {
        guard let data, let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return fallback
        }
        func value(_ key: String) -> Double? {
            guard let number = object[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
            let double = number.doubleValue
            return double.isFinite ? double : nil
        }
        return SamplingDefaults(temperature: value("temperature").flatMap { $0 >= 0 ? $0 : nil } ?? fallback.temperature,
                                topP: value("top_p").flatMap { $0 > 0 && $0 <= 1 ? $0 : nil } ?? fallback.topP)
    }

    /// The values to sample with: the request's own when given, these defaults otherwise.
    func applying(temperature: Double?, topP: Double?) -> SamplingDefaults {
        SamplingDefaults(temperature: temperature ?? self.temperature, topP: topP ?? self.topP)
    }

    static func load(directory: URL) -> SamplingDefaults {
        parse(try? Data(contentsOf: directory.appendingPathComponent("generation_config.json")))
    }
}

// MARK: - Backend

/// Serves an installed MLX chat model: container from `ModelContainerPool`, `UserInput` from the
/// messages (structured when the history carries tool traffic — S2-2), `enable_thinking=false` as
/// `LLMEngine` passes it (reasoning passthrough is S2-3), raw token ids through `ServeTextPipeline`
/// and `ServeToolCallScanner`. Never logs request or response content.
nonisolated struct MLXChatBackend: ServeBackend {
    let directory: URL
    /// The catalog `ramGB` in bytes when known; `nil` falls back to the directory's on-disk size.
    let footprintBytes: Int64?
    let architecture: String?
    let pool: ModelContainerPool

    /// The checkpoint's `model_type`, read once when the backend is built. It picks the tool-call
    /// format, the `system` placement, and — through `ServeBackend.modelType` — whether the route
    /// honours a client's `tools`.
    let checkpointModelType: String

    /// The context window the checkpoint's own `config.json` declares (S2-6, R23) — the number the
    /// pre-count compares against. `nil` when it declares none, in which case nothing is pre-counted
    /// (`ServeBackend.countPromptTokens` answers `nil` too).
    let contextLimit: Int?

    var modelType: String? { checkpointModelType.isEmpty ? nil : checkpointModelType }

    /// S3-5a: the memory admission facts (R47), read once when the backend is built — the weights' bytes
    /// on disk, the KV cost from `config.json`, and the shared budget's capacity as the room.
    let memoryProfile: ServeMemoryProfile?

    /// PC-2: told what each request did with the prompt cache (counts only, never content). Tests
    /// set it; PC-3 turns `reusedTokens` into `usage.cached_tokens` and the request log.
    var promptCacheObserver: (@Sendable (PromptCacheOutcome) -> Void)? = nil

    init(directory: URL, footprintBytes: Int64? = nil, architecture: String? = nil,
         pool: ModelContainerPool = .shared, roomBytes: Int64 = ServeMemoryAdmission.room()) {
        self.directory = directory
        self.footprintBytes = footprintBytes
        self.architecture = architecture
        self.pool = pool
        // One read of `config.json` for both facts.
        let configData = ModelContainerPool.configData(directory: directory)
        self.checkpointModelType = ModelContainerPool.modelTypeString(configData: configData)
            ?? architecture ?? ""
        self.contextLimit = ModelContextWindow.limit(configData: configData)
        self.memoryProfile = ServeMemoryAdmission.profile(
            configData: configData, weightsBytes: Self.weightsBytes(in: directory), roomBytes: roomBytes)
    }

    /// The weights on disk: the sum of the `*.safetensors` files (not the whole directory, so tokenizer and
    /// config files don't count). `0` when there are none.
    nonisolated static func weightsBytes(in directory: URL) -> Int64 {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles])) ?? []
        return files.filter { $0.pathExtension == "safetensors" }
            .reduce(Int64(0)) { $0 + Int64((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
    }

    /// Whether this checkpoint's family can take tools at all (S2-1's table).
    var acceptsTools: Bool { ToolCallFormatTable.acceptsTools(forModelType: checkpointModelType) }

    // MARK: - What the prompt will be (one source for the count and the generation — S2-6)

    /// The tools that will be rendered for this request (S2-2's gate and `tool_choice`).
    nonisolated func effectiveTools(_ request: ServeRequest) -> [[String: any Sendable]] {
        acceptsTools ? request.renderedTools.compactMap(\.object) : []
    }

    /// The template's render context (S2-3's thinking switch).
    nonisolated func templateContext(_ request: ServeRequest) -> [String: any Sendable] {
        ["enable_thinking": ReasoningTemplateTable.thinkingOn(requested: request.reasoning,
                                                              forModelType: checkpointModelType)]
    }

    /// Whether the history carries tool traffic, which decides the `UserInput` shape — and so which
    /// message conversion the template sees.
    nonisolated static func hasToolHistory(_ messages: [ServeMessage]) -> Bool {
        messages.contains { $0.role == .tool || !$0.toolCalls.isEmpty }
    }

    /// The messages as the **template** receives them (S2-6): exactly what generation sends, so the
    /// pre-count counts the real prompt. With tool history the raw dictionaries are used (that path
    /// bypasses mlx-swift-lm's `MessageGenerator` anyway); without it, the role/content pair is what
    /// `DefaultMessageGenerator` produces — and the one generator that would differ
    /// (`NoSystemMessageGenerator`, for a family with no system role) has had its input already
    /// folded away by S2-1's assembly.
    nonisolated static func templateMessages(_ messages: [ServeMessage]) -> [[String: any Sendable]] {
        if hasToolHistory(messages) { return rawMessages(messages) }
        return messages.map { ["role": $0.role.rawValue, "content": $0.text] }
    }

    /// R23 (S2-6): the real prompt's token count, rendered with the **tokenizer alone** — the same
    /// chat template the generation uses, without loading weights, so the head never waits for a
    /// model load. `nil` when there is no declared limit to compare against, when the tokenizer
    /// can't be read, or when the checkpoint has no chat template; the request then simply runs and
    /// the runtime's own overflow handling applies (S1's behaviour).
    ///
    /// The tokenizer comes from `ServePromptCounter`, which caches it: measured cold at 1,043 ms
    /// (reading `tokenizer.json` for a 151k vocabulary) and warm at a few milliseconds, which is the
    /// difference between a usable head and a one-second stall.
    func countPromptTokens(_ request: ServeRequest) async throws -> Int? {
        // No `contextLimit` early-out: the render is also how a template *refusal* is found (S2-8),
        // and a refused prompt must be a 400 whether or not the checkpoint declares a window.
        let tools = effectiveTools(request)
        do {
            return try await ServePromptCounter.shared.count(directory: directory,
                                                            messages: Self.templateMessages(request.messages),
                                                            tools: tools.isEmpty ? nil : tools,
                                                            context: templateContext(request))
        } catch {
            // S2-8: the checkpoint refusing the request is a 400 the route can send before the head. S3-5b:
            // so is a template that cannot be *run* on this request (a tool schema it can't serialise): a
            // clear `400` naming the cause, never S2-6's fail-open. Only a checkpoint with no chat template at
            // all is "can't count" (the route then answers `nil`, and `count_tokens` a `501`).
            switch ChatTemplateFailure.classify(error) {
            case .refusal(let refusal): throw refusal
            case .missingTemplate: return nil
            case .renderFailure:
                throw ChatTemplateFailure.renderFailure(
                    toolNames: await ServePromptCounter.shared.failingToolNames(
                        directory: directory, tools: tools, context: templateContext(request)))
            }
        }
    }

    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await run(request, into: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            // The consumer going away (client disconnect, S1-0) cancels the generation.
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(_ request: ServeRequest,
                     into continuation: AsyncThrowingStream<ServeEvent, Error>.Continuation) async throws {
        // S3-5c: MLX's buffer pool is capped while a server generation runs and restored after it.
        ServeMLXMemory.shared.begin()
        defer { ServeMLXMemory.shared.end() }
        try await runInner(request, into: continuation)
    }

    private func runInner(_ request: ServeRequest,
                          into continuation: AsyncThrowingStream<ServeEvent, Error>.Continuation) async throws {
        let runStart = ContinuousClock.now
        let container = try await pool.container(directory: directory, footprintBytes: footprintBytes,
                                                 architecture: architecture)
        // S2-2: S2-1's `system` hoisting and the flatten-or-keep decision for tool traffic now
        // happen in the route, which reads this backend's `modelType` and so has the one fact both
        // need. The backend stays a renderer: messages in, prompt out.
        let messages = request.messages
        // S2-2 gate: `tools` are honoured only for a family whose template renders them *and* whose
        // output a parser reads. `tool_choice: "none"` renders none at all; a named function
        // renders just that tool when the client sent it (`renderedTools`), since a model can't be
        // forced — best effort, implementer's call, pending owner confirmation.
        let tools = effectiveTools(request)
        let format = ToolCallFormatTable.resolved(forModelType: checkpointModelType,
                                                 configData: ModelContainerPool.configData(directory: directory)) ?? .json
        let hasToolHistory = Self.hasToolHistory(messages)
        // S2-3: thinking is OFF unless the request asked for it *and* the family's template reads
        // `enable_thinking` — the same shape as the tools gate above, and the same default S1 had.
        // The client's `reasoning_effort` string is deliberately **not** forwarded: the Qwen3.8 and
        // Ternary-Bonsai templates raise on anything but `xhigh`/`medium`/`low`, and the OpenAI
        // vocabulary (`minimal`) is not in their list, so a forwarded value would kill the render
        // and silently turn thinking off. The templates have no budget control to set anyway.
        // Implementer's call, pending owner confirmation.
        let thinking = ReasoningTemplateTable.thinkingOn(requested: request.reasoning,
                                                         forModelType: checkpointModelType)
        let renderContext: [String: any Sendable] = ["enable_thinking": thinking]
        let containerSeconds = Self.seconds(ContinuousClock.now - runStart)
        try await container.perform { context in
            let performStart = ContinuousClock.now
            let input: LMInput
            // PC-2 (T5): which render attempt succeeded — the probe renders must use the same one.
            var usedContext = true
            if hasToolHistory {
                // `.messages` carries the raw template dictionaries, which is the only way to hand
                // mlx-swift-lm an assistant `tool_calls` array or a `tool_call_id`: `Chat.Message`
                // has no field for either (3.31.3).
                let raw = Self.rawMessages(messages)
                do {
                    input = try await context.processor.prepare(
                        input: UserInput(prompt: .messages(raw), tools: tools.isEmpty ? nil : tools,
                                         additionalContext: renderContext))
                } catch {
                    usedContext = false
                    // As below: the optional render context is retried without, and a refusal that
                    // survives the retry is the template's own 400 (S2-8).
                    do {
                        input = try await context.processor.prepare(
                            input: UserInput(prompt: .messages(raw), tools: tools.isEmpty ? nil : tools))
                    } catch {
                        // A refusal goes out on the stream's **error channel** rather than being
                        // thrown: the route turns that into the error the client sees, and throwing
                        // would replace it with S2's opaque 500 (which exists so a backend failure
                        // never leaks its internals to a client). Not a refusal → rethrow, as before.
                        switch ChatTemplateFailure.classify(error) {
                        case .refusal(let refusal):
                            continuation.yield(.error(refusal))
                            return
                        case .missingTemplate:
                            throw error
                        case .renderFailure:
                            // S3-5b: the same clear 400 as the pre-count, not a generic 500.
                            continuation.yield(.error(ChatTemplateFailure.renderFailure(
                                toolNames: await ServePromptCounter.shared.failingToolNames(
                                    directory: directory, tools: tools, context: renderContext))))
                            return
                        }
                    }
                }
            } else {
                let chat = messages.map { Chat.Message(role: Self.role($0.role), content: $0.text) }
                do {
                    input = try await context.processor.prepare(
                        input: UserInput(chat: chat, tools: tools.isEmpty ? nil : tools,
                                         additionalContext: renderContext))
                } catch {
                    usedContext = false
                    // The render context (`enable_thinking`) is optional by design — a template that
                    // doesn't read it may raise, so retry without it. S2-8: if the *retry* is refused
                    // too, the checkpoint is declining the request: report that as a 400, not a 500.
                    do {
                        input = try await context.processor.prepare(
                            input: UserInput(chat: chat, tools: tools.isEmpty ? nil : tools))
                    } catch {
                        // A refusal goes out on the stream's **error channel** rather than being
                        // thrown: the route turns that into the error the client sees, and throwing
                        // would replace it with S2's opaque 500 (which exists so a backend failure
                        // never leaks its internals to a client). Not a refusal → rethrow, as before.
                        switch ChatTemplateFailure.classify(error) {
                        case .refusal(let refusal):
                            continuation.yield(.error(refusal))
                            return
                        case .missingTemplate:
                            throw error
                        case .renderFailure:
                            // S3-5b: the same clear 400 as the pre-count, not a generic 500.
                            continuation.yield(.error(ChatTemplateFailure.renderFailure(
                                toolNames: await ServePromptCounter.shared.failingToolNames(
                                    directory: directory, tools: tools, context: renderContext))))
                            return
                        }
                    }
                }
            }
            let promptTokens = input.text.tokens.size
            let renderSeconds = Self.seconds(ContinuousClock.now - performStart)

            // S3-8B — make the room admission assumed. Every other resident model counts against the budget: evict
            // them (LRU first) until this request's whole need fits beside the rest, or refuse. Prompt-cache
            // snapshots are neither counted nor evicted (see `MemoryBudget.reserveWorkingSet`).
            var workingSet: MemoryBudget.Reservation?
            if let profile = memoryProfile {
                let need = profile.needBytes(promptTokens: promptTokens, replyTokens: request.effectiveMaxTokens)
                workingSet = await pool.reserveWorkingSet(totalNeedBytes: need, directory: directory)
                if workingSet == nil {
                    continuation.yield(.error(ServeMemoryAdmission.refusal(
                        needBytes: need, roomBytes: pool.budget.capacityBytes)))
                    return
                }
            }
            defer { if let workingSet { pool.budget.release(workingSet) } }

            if let seed = request.seed { MLXRandom.seed(UInt64(truncatingIfNeeded: seed)) }
            let sampling = SamplingDefaults.load(directory: directory)
                .applying(temperature: request.temperature, topP: request.topP)
            let parameters = GenerateParameters(
                maxTokens: request.effectiveMaxTokens,
                temperature: Float(sampling.temperature),
                topP: Float(sampling.topP))

            // PC-2 — the one seam. Everything below this block is unchanged: the stream is just
            // generated from a prefilled cache and the tail instead of from the whole prompt.
            let prepared = await Self.preparePromptCache(
                request: request, messages: messages, input: input, context: context, parameters: parameters,
                directory: directory, modelType: checkpointModelType, tools: tools,
                hasToolHistory: hasToolHistory, renderContext: usedContext ? renderContext : nil,
                observer: promptCacheObserver, container: containerSeconds, render: renderSeconds)
            if Task.isCancelled { throw CancellationError() }
            let stream = try MLXLMCommon.generateTokens(
                input: prepared.cache == nil ? input : LMInput(tokens: MLXArray(prepared.tailTokens)),
                cache: prepared.cache, parameters: parameters, context: context)

            // The scanner exists only when a tool call could actually be produced; `tools` empty
            // means the family can't use them or the client asked for `tool_choice: "none"`.
            var scanner: ServeToolCallScanner? = tools.isEmpty
                ? nil : ServeToolCallScanner(format: format, tools: tools)
            var callIndex = 0

            /// The answer half — after the split when thinking is on — goes through the scanner
            /// (when there is one) and out; any call it completed is yielded as its own event.
            /// A `<tool_call>` written inside a think block is *reasoning*, and never reaches here
            /// (S2-3 decision 4).
            func routeContent(_ text: String) {
                guard !text.isEmpty else { return }
                if scanner != nil {
                    let safe = scanner?.push(text) ?? text
                    if !safe.isEmpty { continuation.yield(.textDelta(safe)) }
                    Self.yieldCalls(from: &scanner, into: continuation, index: &callIndex)
                } else {
                    continuation.yield(.textDelta(text))
                }
            }

            var completionTokens = 0
            var libraryReason: GenerateStopReason?
            var stoppedByClient = false

            if thinking {
                // S2-3: the raw text is split into its thinking and its answer, and only the answer
                // is scanned for tool calls, matched against `stop`, or emitted as `content`.
                // `ServeTextPipeline` is deliberately not used here: its leading-`<think>` strip is
                // the *thinking-off* behaviour, and it would eat the block we are meant to forward.
                var pipeline = ServeThinkingTextPipeline(
                    decode: { context.tokenizer.decode(tokenIds: $0) },
                    stop: request.stop,
                    // S2-3b: the *prompt* decides whether the output starts inside the block, not
                    // a guess from the first characters (`qwen3_5*` opens it; `qwen3` does not).
                    startsInsideThink: Self.promptOpensThinkBlock(context: context, tokens: input.text.tokens))
                func emitHalves(_ halves: (reasoning: String, content: String)) {
                    if !halves.reasoning.isEmpty { continuation.yield(.reasoningDelta(halves.reasoning)) }
                    routeContent(halves.content)
                }
                for await generation in stream {
                    if Task.isCancelled { break }
                    switch generation {
                    case .token(let id):
                        completionTokens += 1
                        emitHalves(pipeline.push(id))
                    case .info(let info):
                        libraryReason = info.stopReason
                    }
                    if pipeline.stopped { break }
                }
                if Task.isCancelled { throw CancellationError() }
                // End of stream: an unclosed block runs to EOS (a `max_tokens` cut mid-think) — and
                // nothing at all is flushed once a stop sequence has matched.
                emitHalves(pipeline.finish())
                stoppedByClient = pipeline.stopped
            } else {
                // S1's path, unchanged: detokenize → strip a leading closed `<think>` → `stop`.
                var pipeline = ServeTextPipeline(decode: { context.tokenizer.decode(tokenIds: $0) },
                                                 stop: request.stop)
                for await generation in stream {
                    if Task.isCancelled { break }
                    switch generation {
                    case .token(let id):
                        completionTokens += 1
                        routeContent(pipeline.push(id).text)
                    case .info(let info):
                        libraryReason = info.stopReason
                    }
                    if pipeline.stopped { break }
                }
                if Task.isCancelled { throw CancellationError() }
                stoppedByClient = pipeline.stopped
                routeContent(pipeline.finish())
            }
            // Whatever the scanner still held: a call the stream ended inside, or plain text.
            routeContent(scanner?.finish() ?? "")
            Self.yieldCalls(from: &scanner, into: continuation, index: &callIndex)

            let hitLimit = libraryReason == .length
                || (libraryReason == nil && !stoppedByClient && completionTokens >= request.effectiveMaxTokens)
            continuation.yield(.usage(ServeUsage(
                promptTokens: promptTokens, completionTokens: completionTokens, outcome: prepared.outcome)))
            // S2-2: a turn that produced calls ends as `tool_calls`, whatever the sampler said.
            let finish: ServeFinishReason = callIndex > 0 ? .toolCalls
                : (hitLimit && !stoppedByClient ? .length : .stop)
            continuation.yield(.finish(finish))
        }
    }

    nonisolated static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }

    /// PC-2: the cache work for one request, or `nil` cache when it is bypassed or fails. A failure
    /// of the cache path is never a failure of the request: it runs as if the cache were off.
    /// Cancellation is the one thing that propagates (the caller checks `Task.isCancelled`).
    private static func preparePromptCache(
        request: ServeRequest, messages: [ServeMessage], input: LMInput, context: ModelContext,
        parameters: GenerateParameters, directory: URL, modelType: String,
        tools: [[String: any Sendable]], hasToolHistory: Bool, renderContext: [String: any Sendable]?,
        observer: (@Sendable (PromptCacheOutcome) -> Void)?, container: Double, render: Double
    ) async -> PreparedPrompt {
        let promptTokens = input.text.tokens.size
        var outcome = PromptCacheOutcome(
            bypass: PromptCachePolicy.bypass(enabled: PromptCache.isEnabled, modelType: modelType,
                                             seed: request.seed, kvBits: parameters.kvBits,
                                             maxKVSize: parameters.maxKVSize),
            promptTokens: promptTokens)
        var prepared = PreparedPrompt(cache: nil, tailTokens: [], outcome: outcome)
        if outcome.bypass == nil {
            let key = PromptCacheModelKey(
                directory: directory,
                parameters: "kvBits=nil;maxKVSize=nil;prefillStepSize=\(parameters.prefillStepSize)")
            let toolsArgument: [[String: any Sendable]]? = tools.isEmpty ? nil : tools
            // Probe renders: the same shape (raw `.messages` vs `Chat.Message`), the same `tools`
            // and the same context attempt as the real prompt (T5).
            let render: ([ServeMessage]) async throws -> [Int] = { probe in
                let probeInput: LMInput
                if hasToolHistory {
                    probeInput = try await context.processor.prepare(
                        input: UserInput(prompt: .messages(Self.rawMessages(probe)), tools: toolsArgument,
                                         additionalContext: renderContext))
                } else {
                    let chat = probe.map { Chat.Message(role: Self.role($0.role), content: $0.text) }
                    probeInput = try await context.processor.prepare(
                        input: UserInput(chat: chat, tools: toolsArgument, additionalContext: renderContext))
                }
                return probeInput.text.tokens.asArray(Int.self)
            }
            do {
                if PromptCache.injectFailureForTesting { throw PromptCacheFault.injected }
                prepared = try await PromptCheckpointing.prepare(
                    store: PromptCacheShared.store, key: key, messages: messages,
                    tokens: input.text.tokens.asArray(Int.self), context: context,
                    parameters: parameters, render: render)
                outcome = prepared.outcome
            } catch {
                // Cancellation is re-raised by the caller's `Task.isCancelled` check; anything else
                // falls back to the uncached path.
                if !(error is CancellationError) { outcome.bypass = .failed }
                prepared = PreparedPrompt(cache: nil, tailTokens: [], outcome: outcome)
            }
        }
        outcome.timings.container = container
        outcome.timings.render = render
        prepared.outcome = outcome
        PromptCacheCounters.shared.record(outcome)
        observer?(outcome)
        return prepared
    }

    /// Whether the **rendered prompt** already opened the think block (S2-3b, review call 2).
    ///
    /// Read from the prompt the tokenizer actually produced rather than guessed from the family: the
    /// `qwen3_5*` templates end their generation prompt with `<think>\n`, and `qwen3`'s ends at the
    /// assistant header. Trailing whitespace is trimmed so the newline can't hide the tag.
    nonisolated static func promptOpensThinkBlock(context: ModelContext, tokens: MLXArray) -> Bool {
        let ids = tokens.asArray(Int.self)
        guard !ids.isEmpty else { return false }
        // A window is enough — only the tail matters — and it keeps the decode cheap per request.
        return promptOpensThinkBlock(decodedTail: context.tokenizer.decode(tokenIds: Array(ids.suffix(promptTailTokens))))
    }

    /// How many prompt tokens the tail probe decodes. One is enough in principle (`<think>` is a
    /// single token in these tokenizers); sixteen costs nothing and survives a tokenizer that
    /// splits it.
    static let promptTailTokens = 16

    /// The decision itself, on text — so a tokenizer-only test can make it without a model.
    nonisolated static func promptOpensThinkBlock(decodedTail: String) -> Bool {
        decodedTail.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("<think>")
    }

    /// Emit every call the scanner has completed since the last drain.
    private static func yieldCalls(from scanner: inout ServeToolCallScanner?,
                                   into continuation: AsyncThrowingStream<ServeEvent, Error>.Continuation,
                                   index: inout Int) {
        for call in scanner?.takeCalls() ?? [] {
            continuation.yield(.toolCallDelta(ServeToolCallDelta(
                index: index, id: OpenAIChatDialect.makeToolCallID(), name: call.function.name,
                argumentsFragment: argumentsJSON(call.function.arguments))))
            index += 1
        }
    }

    /// The parser's argument dictionary as the JSON **string** the wire carries.
    nonisolated static func argumentsJSON(_ arguments: [String: JSONValue]) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(arguments), let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    }

    /// The raw template dictionaries for a history that carries tool traffic (S2-2).
    ///
    /// `Chat.Message` can express a role and text and nothing else, so an assistant `tool_calls`
    /// array and a `tool` turn's `tool_call_id` have to go in as dictionaries —
    /// `UserInput.Prompt.messages` passes them straight through to the Jinja template, which is
    /// where every family reads them.
    nonisolated static func rawMessages(_ messages: [ServeMessage]) -> [[String: any Sendable]] {
        messages.map { message in
            var dictionary: [String: any Sendable] = ["role": message.role.rawValue, "content": message.text]
            if !message.toolCalls.isEmpty {
                dictionary["tool_calls"] = message.toolCalls.map { call -> [String: any Sendable] in
                    var function: [String: any Sendable] = ["name": call.name]
                    function["arguments"] = argumentsValue(call.argumentsJSON)
                    var entry: [String: any Sendable] = ["type": "function", "function": function]
                    if let id = call.id { entry["id"] = id }
                    return entry
                }
            }
            if let id = message.toolCallID { dictionary["tool_call_id"] = id }
            if let name = message.toolName { dictionary["name"] = name }
            return dictionary
        }
    }

    /// `function.arguments` for the template: the JSON **object** when the string parses as one,
    /// else the string. Qwen3.5's and Llama 3.1's templates iterate it (`arguments|items`), so a
    /// string there would raise; Qwen3's and Mistral's accept either (`is string` → as-is, else
    /// `|tojson`).
    nonisolated static func argumentsValue(_ json: String) -> any Sendable {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: any Sendable]
        else { return json }
        return object
    }

    nonisolated private static func role(_ role: ServeMessage.Role) -> Chat.Message.Role {
        switch role {
        case .system: return .system
        case .user: return .user
        case .assistant: return .assistant
        case .tool: return .tool
        }
    }
}
