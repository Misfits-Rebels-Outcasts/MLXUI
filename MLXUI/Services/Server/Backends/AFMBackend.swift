import Foundation

/// Apple Foundation Models streams **snapshots** — the whole reply so far — not deltas (readme T7).
/// This turns them into the deltas the neutral core carries: each snapshot is diffed against what
/// has already been emitted. A snapshot that doesn't extend what was emitted (AFM revising earlier
/// text, which a stream of deltas cannot retract) is held until a later one does, so a client never
/// sees text repeated or rewound.
nonisolated struct SnapshotDeltaConverter {
    private var emitted = ""

    mutating func push(_ snapshot: String) -> String {
        guard snapshot.hasPrefix(emitted), snapshot.count > emitted.count else { return "" }
        let delta = String(snapshot.dropFirst(emitted.count))
        emitted = snapshot
        return delta
    }
}

/// The readiness question, as the one sentence a `503` carries (backlog S1-4: "not ready → 503 with
/// the `Readiness` reason sentence").
nonisolated enum ReadinessSentence {
    static func text(for readiness: Readiness) -> String {
        func sentence(_ text: String) -> String { text.hasSuffix(".") ? text : text + "." }
        switch readiness {
        case .ready: return "Apple Foundation Models is ready."
        case .needsDownload(let gb): return "The model needs to be downloaded first (about \(String(format: "%.1f", gb)) GB)."
        case .needsSetup(let reason, _): return sentence(reason)
        case .unavailable(let reason): return sentence(reason)
        }
    }

    /// The `503` for a model that is served but not ready.
    static func error(for readiness: Readiness) -> ServeError {
        ServeError(status: 503, message: text(for: readiness), type: "server_error", code: "model_not_ready")
    }
}

/// Serves Apple Foundation Models (on-device — nothing leaves the Mac). Sees only the
/// non-availability-tagged seam in `FlowKit/AppleFoundationExecutor.swift`; readiness comes through
/// the **injected** checker, never `SystemLanguageModel` (readme T8, AFM-FOLLOWUP-1).
///
/// Mapping: `system` messages → the session's instructions; earlier turns → folded into the prompt
/// text (a constructed `Transcript` is ignored by the model — see the seam's doc); the last user
/// turn is the message to answer. `temperature` and `max_tokens` map to `GenerationOptions`;
/// `top_p`, `seed` and `tools` are ignored (R9) — `stop` is honoured here, on the deltas.
/// Never logs request or response content.
nonisolated struct AFMBackend: ServeBackend {
    let streamer: any AppleFoundationChatStreaming
    let readiness: @Sendable () -> Readiness?

    init(streamer: any AppleFoundationChatStreaming,
         readiness: @escaping @Sendable () -> Readiness? = { AppleFoundationAvailability.currentReadiness() }) {
        self.streamer = streamer
        self.readiness = readiness
    }

    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                await run(request, into: continuation)
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(_ request: ServeRequest, into continuation: AsyncThrowingStream<ServeEvent, Error>.Continuation) async {
        switch readiness() {
        case .ready?: break
        case let state?: continuation.yield(.error(ReadinessSentence.error(for: state))); return
        case nil:
            continuation.yield(.error(ServeError(status: 404, message: "Apple Foundation Models isn't available on this Mac.",
                                                 code: "model_not_found")))
            return
        }

        let instructions = request.messages.filter { $0.role == .system }.map(\.text)
            .filter { !$0.isEmpty }.joined(separator: "\n\n")
        let conversation = request.messages.filter { $0.role != .system }
        guard let last = conversation.last, last.role == .user else {
            continuation.yield(.error(ServeError(status: 400, message: "The last message must come from the user.",
                                                 code: "invalid_messages")))
            return
        }
        let history = Array(conversation.dropLast())
        let prompt = Self.prompt(history: history, last: last.text)

        var converter = SnapshotDeltaConverter()
        var stop = StopMatcher(request.stop)
        var stopped = false
        var reply = ""
        var counted: (prompt: Int, completion: Int)?

        do {
            let stream = streamer.streamChat(instructions: instructions, prompt: prompt,
                                             temperature: request.temperature,
                                             maximumResponseTokens: request.effectiveMaxTokens)
            for try await event in stream {
                switch event {
                case .snapshot(let snapshot):
                    let delta = converter.push(snapshot)
                    guard !delta.isEmpty else { continue }
                    let output = stop.push(delta)
                    if !output.emit.isEmpty { reply += output.emit; continuation.yield(.textDelta(output.emit)) }
                    if output.stopped { stopped = true }
                case .usage(let prompt, let completion):
                    counted = (prompt, completion)
                }
                if stopped { break }                  // leaving the loop cancels the generation
            }
        } catch is CancellationError {
            return
        } catch let error as AppleFoundationChatError {
            continuation.yield(.error(Self.serveError(for: error)))
            return
        } catch {
            continuation.yield(.error(Self.serveError(for: .failed)))
            return
        }
        if Task.isCancelled { return }

        if !stopped {
            let tail = stop.finish()
            if !tail.isEmpty { reply += tail; continuation.yield(.textDelta(tail)) }
        }
        // Without OS token counts (before macOS 26.4) estimate at ~4 characters a token.
        // Implementer's call, pending owner confirmation — clients want integers, and none exists.
        let promptTokens = counted?.prompt ?? Self.estimate(instructions + prompt)
        let completionTokens = counted?.completion ?? Self.estimate(reply)
        continuation.yield(.usage(ServeUsage(promptTokens: promptTokens, completionTokens: completionTokens)))
        continuation.yield(.finish(!stopped && completionTokens >= request.effectiveMaxTokens ? .length : .stop))
    }

    /// With no earlier turns the prompt is the user's message verbatim. Otherwise the earlier turns
    /// come first as `User:` / `Assistant:` lines, then the message to answer — the framing that
    /// made the on-device model answer "What is my name?" correctly (the same turns in the
    /// instructions did not). Implementer's call, pending owner confirmation.
    static func prompt(history: [ServeMessage], last: String) -> String {
        guard !history.isEmpty else { return last }
        let lines = history.map { ($0.role == .assistant ? "Assistant: " : "User: ") + $0.text }.joined(separator: "\n")
        return "Conversation so far:\n\(lines)\n\nReply to the latest message from the user:\nUser: \(last)"
    }

    static func estimate(_ text: String) -> Int { text.isEmpty ? 0 : (text.count + 3) / 4 }

    /// context window → 400 with the limit; guardrail → 400; anything else → a generic 500.
    static func serveError(for error: AppleFoundationChatError) -> ServeError {
        switch error {
        case .contextWindowExceeded(let limit, let tokens):
            var message = "This conversation is too long for Apple Foundation Models"
            if let limit { message += ": its context window is \(limit) tokens" }
            if let tokens { message += " and the request used \(tokens)" }
            return ServeError(status: 400, message: message + ".", code: "context_length_exceeded")
        case .guardrailViolation:
            return ServeError(status: 400, message: "Apple Intelligence declined this request.", code: "content_filter")
        case .unsupportedLanguage:
            return ServeError(status: 400, message: "Apple Intelligence doesn't support this language.",
                              code: "unsupported_language")
        case .failed:
            return ServeError(status: 500, message: "The model failed to generate a reply.",
                              type: "server_error", code: "generation_failed")
        }
    }
}
