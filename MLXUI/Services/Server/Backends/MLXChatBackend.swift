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

// MARK: - Backend

/// Serves an installed MLX chat model: container from `ModelContainerPool`, `UserInput(chat:)`
/// from the messages, `enable_thinking=false` as `LLMEngine` passes it (reasoning passthrough is
/// S2), raw token ids through `ServeTextPipeline`. Never logs request or response content.
nonisolated struct MLXChatBackend: ServeBackend {
    let directory: URL
    /// The catalog `ramGB` in bytes when known; `nil` falls back to the directory's on-disk size.
    let footprintBytes: Int64?
    let architecture: String?
    let pool: ModelContainerPool

    init(directory: URL, footprintBytes: Int64? = nil, architecture: String? = nil,
         pool: ModelContainerPool = .shared) {
        self.directory = directory
        self.footprintBytes = footprintBytes
        self.architecture = architecture
        self.pool = pool
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
        let container = try await pool.container(directory: directory, footprintBytes: footprintBytes,
                                                 architecture: architecture)
        try await container.perform { context in
            let chat = request.messages.map { message in
                Chat.Message(role: Self.role(message.role), content: message.text)
            }
            let input: LMInput
            do {
                input = try await context.processor.prepare(
                    input: UserInput(chat: chat, additionalContext: LLMEngine.noThinkingContext))
            } catch {
                input = try await context.processor.prepare(input: UserInput(chat: chat))
            }
            let promptTokens = input.text.tokens.size

            if let seed = request.seed { MLXRandom.seed(UInt64(truncatingIfNeeded: seed)) }
            let parameters = GenerateParameters(
                maxTokens: request.effectiveMaxTokens,
                temperature: Float(request.temperature ?? Self.defaultTemperature),
                topP: Float(request.topP ?? 1.0))
            let stream = try MLXLMCommon.generateTokens(input: input, parameters: parameters, context: context)

            var pipeline = ServeTextPipeline(decode: { context.tokenizer.decode(tokenIds: $0) }, stop: request.stop)
            var completionTokens = 0
            var libraryReason: GenerateStopReason?
            for await generation in stream {
                if Task.isCancelled { break }
                switch generation {
                case .token(let id):
                    completionTokens += 1
                    let output = pipeline.push(id)
                    if !output.text.isEmpty { continuation.yield(.textDelta(output.text)) }
                case .info(let info):
                    libraryReason = info.stopReason
                }
                if pipeline.stopped { break }
            }
            if Task.isCancelled { throw CancellationError() }

            let tail = pipeline.finish()
            if !tail.isEmpty { continuation.yield(.textDelta(tail)) }
            let hitLimit = libraryReason == .length
                || (libraryReason == nil && !pipeline.stopped && completionTokens >= request.effectiveMaxTokens)
            continuation.yield(.usage(ServeUsage(promptTokens: promptTokens, completionTokens: completionTokens)))
            continuation.yield(.finish(hitLimit && !pipeline.stopped ? .length : .stop))
        }
    }

    /// OpenAI's documented default is 1.0. Implementer's call, pending owner confirmation: follow it
    /// rather than the 0.7 `LLMEngine` uses for flow rows — a client that wants less sets it.
    nonisolated static let defaultTemperature = 1.0

    nonisolated private static func role(_ role: ServeMessage.Role) -> Chat.Message.Role {
        switch role {
        case .system: return .system
        case .user: return .user
        case .assistant: return .assistant
        case .tool: return .tool
        }
    }
}
