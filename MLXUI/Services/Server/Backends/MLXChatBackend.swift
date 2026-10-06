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

    var modelType: String? { checkpointModelType.isEmpty ? nil : checkpointModelType }

    init(directory: URL, footprintBytes: Int64? = nil, architecture: String? = nil,
         pool: ModelContainerPool = .shared) {
        self.directory = directory
        self.footprintBytes = footprintBytes
        self.architecture = architecture
        self.pool = pool
        self.checkpointModelType = ModelContainerPool.modelTypeString(directory: directory,
                                                                     architecture: architecture)
    }

    /// Whether this checkpoint's family can take tools at all (S2-1's table).
    var acceptsTools: Bool { ToolCallFormatTable.acceptsTools(forModelType: checkpointModelType) }

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
        let container = try await pool.container(directory: directory, footprintBytes: footprintBytes,
                                                 architecture: architecture)
        // S2-2: S2-1's `system` hoisting and the flatten-or-keep decision for tool traffic now
        // happen in the route, which reads this backend's `modelType` and so has the one fact both
        // need. The backend stays a renderer: messages in, prompt out.
        let acceptsTools = self.acceptsTools
        let messages = request.messages
        // S2-2 gate: `tools` are honoured only for a family whose template renders them *and* whose
        // output a parser reads. `tool_choice: "none"` renders none at all; a named function
        // renders just that tool when the client sent it (`renderedTools`), since a model can't be
        // forced — best effort, implementer's call, pending owner confirmation.
        let tools: [[String: any Sendable]] = acceptsTools
            ? request.renderedTools.compactMap(\.object) : []
        let format = ToolCallFormatTable.resolved(forModelType: checkpointModelType,
                                                 configData: ModelContainerPool.configData(directory: directory)) ?? .json
        let hasToolHistory = messages.contains { $0.role == .tool || !$0.toolCalls.isEmpty }
        let noThinking = LLMEngine.noThinkingContext
        try await container.perform { context in
            let input: LMInput
            if hasToolHistory {
                // `.messages` carries the raw template dictionaries, which is the only way to hand
                // mlx-swift-lm an assistant `tool_calls` array or a `tool_call_id`: `Chat.Message`
                // has no field for either (3.31.3).
                let raw = Self.rawMessages(messages)
                do {
                    input = try await context.processor.prepare(
                        input: UserInput(prompt: .messages(raw), tools: tools.isEmpty ? nil : tools,
                                         additionalContext: noThinking))
                } catch {
                    input = try await context.processor.prepare(
                        input: UserInput(prompt: .messages(raw), tools: tools.isEmpty ? nil : tools))
                }
            } else {
                let chat = messages.map { Chat.Message(role: Self.role($0.role), content: $0.text) }
                do {
                    input = try await context.processor.prepare(
                        input: UserInput(chat: chat, tools: tools.isEmpty ? nil : tools,
                                         additionalContext: noThinking))
                } catch {
                    input = try await context.processor.prepare(
                        input: UserInput(chat: chat, tools: tools.isEmpty ? nil : tools))
                }
            }
            let promptTokens = input.text.tokens.size

            if let seed = request.seed { MLXRandom.seed(UInt64(truncatingIfNeeded: seed)) }
            let sampling = SamplingDefaults.load(directory: directory)
                .applying(temperature: request.temperature, topP: request.topP)
            let parameters = GenerateParameters(
                maxTokens: request.effectiveMaxTokens,
                temperature: Float(sampling.temperature),
                topP: Float(sampling.topP))
            let stream = try MLXLMCommon.generateTokens(input: input, parameters: parameters, context: context)

            var pipeline = ServeTextPipeline(decode: { context.tokenizer.decode(tokenIds: $0) }, stop: request.stop)
            // The scanner exists only when a tool call could actually be produced; `tools` empty
            // means the family can't use them or the client asked for `tool_choice: "none"`.
            var scanner: ServeToolCallScanner? = tools.isEmpty
                ? nil : ServeToolCallScanner(format: format, tools: tools)
            var callIndex = 0

            /// Text goes through the scanner (when there is one) and out; any call it completed is
            /// yielded as its own event.
            func route(_ text: String) {
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
            for await generation in stream {
                if Task.isCancelled { break }
                switch generation {
                case .token(let id):
                    completionTokens += 1
                    route(pipeline.push(id).text)
                case .info(let info):
                    libraryReason = info.stopReason
                }
                if pipeline.stopped { break }
            }
            if Task.isCancelled { throw CancellationError() }

            route(pipeline.finish())
            // Whatever the scanner still held: a call the stream ended inside, or plain text.
            route(scanner?.finish() ?? "")
            Self.yieldCalls(from: &scanner, into: continuation, index: &callIndex)

            let hitLimit = libraryReason == .length
                || (libraryReason == nil && !pipeline.stopped && completionTokens >= request.effectiveMaxTokens)
            continuation.yield(.usage(ServeUsage(promptTokens: promptTokens, completionTokens: completionTokens)))
            // S2-2: a turn that produced calls ends as `tool_calls`, whatever the sampler said.
            let finish: ServeFinishReason = callIndex > 0 ? .toolCalls
                : (hitLimit && !pipeline.stopped ? .length : .stop)
            continuation.yield(.finish(finish))
        }
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
