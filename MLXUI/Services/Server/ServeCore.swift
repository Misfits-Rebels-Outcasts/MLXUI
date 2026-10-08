import Foundation

// S1-3 — the neutral core (design §3). Every dialect decodes into one `ServeRequest` and encodes one
// stream of `ServeEvent`, so S3's Anthropic / Responses work is two translators, not two servers.
// Fields S2/S3 will fill (tool calls, reasoning) exist now and are simply unset in S1.

/// Why a generation ended. `toolCalls` is S2's; S1 only produces `stop` and `length`.
nonisolated enum ServeFinishReason: String, Sendable, Equatable {
    case stop
    case length
    case toolCalls = "tool_calls"
}

nonisolated struct ServeUsage: Sendable, Equatable {
    var promptTokens: Int
    var completionTokens: Int
    /// PC-3: prompt tokens served from the prompt cache. **Part of `promptTokens`, not extra** (it
    /// is the OpenAI `prompt_tokens_details.cached_tokens` meaning). 0 = a miss or the cache is off.
    var cachedTokens: Int = 0
    /// PC-3: the cache path failed and the request ran as if the cache were off. Never on the wire;
    /// it exists so a failing cache can't hide as "no hit" in the request log.
    var cacheFellBack: Bool = false
    var totalTokens: Int { promptTokens + completionTokens }
}

/// One incremental piece of a tool call, as it goes out on the wire (S2-2). `argumentsFragment`
/// carries the whole JSON string in one piece today — the parsers hand back a finished call — and
/// the field name keeps the door open for real fragmentation.
nonisolated struct ServeToolCallDelta: Sendable, Equatable {
    var index: Int
    var id: String?
    var name: String?
    var argumentsFragment: String?
}

/// One complete tool call (S2-2): history that came back from a harness, and the non-streaming
/// response's `message.tool_calls[]`.
nonisolated struct ServeToolCall: Sendable, Equatable {
    /// The client's `call_…` id when it sent one; `nil` only if it omitted it.
    var id: String?
    /// The function name. Never logged (rule 13).
    var name: String
    /// The arguments as a JSON **string**, which is what the OpenAI wire format carries.
    var argumentsJSON: String
}

/// One `tools[]` entry a client sent, kept **verbatim** (S2-2).
///
/// The raw JSON is what the chat template renders and what `ToolCallFormat`'s parsers use for
/// type-aware arguments, so nothing is reconstructed from a field list — a client key we don't
/// model still reaches the template. `name` is pulled out for `tool_choice`'s named form only.
nonisolated struct ServeToolSpec: Sendable, Equatable {
    var name: String
    var rawJSON: String

    /// The object, for `UserInput(tools:)` and `parse(content:tools:)`.
    var object: [String: any Sendable]? {
        guard let data = rawJSON.data(using: .utf8),
              let decoded = try? JSONSerialization.jsonObject(with: data) as? [String: any Sendable]
        else { return nil }
        return decoded
    }
}

/// What the client asked for in `tool_choice` (S2-2). Only `auto` / `none` are honoured exactly;
/// `required` and a named function are best effort (the journal records what actually happens).
nonisolated enum ServeToolChoice: Sendable, Equatable {
    case auto
    case none
    case required
    case function(String)
}

nonisolated enum ServeEvent: Sendable, Equatable {
    case textDelta(String)
    case reasoningDelta(String)                 // S2-3 — never produced yet
    case toolCallDelta(ServeToolCallDelta)      // S2-2
    case usage(ServeUsage)
    case finish(ServeFinishReason)
    case error(ServeError)
}

nonisolated struct ServeMessage: Sendable, Equatable {
    enum Role: String, Sendable, Equatable { case system, user, assistant, tool }
    var role: Role
    var text: String
    /// An assistant turn's `tool_calls` (S2-2). Empty for every other role.
    var toolCalls: [ServeToolCall] = []
    /// A `tool` turn's `tool_call_id` (S2-2).
    var toolCallID: String? = nil
    /// A `tool` turn's optional `name`.
    var toolName: String? = nil
}

nonisolated struct ServeRequest: Sendable, Equatable {
    static let defaultMaxTokens = 4096          // R16: a request with no limit gets 4096

    var model: String
    var messages: [ServeMessage]
    /// The client's `tools[]`, verbatim and in order (S2-2).
    var tools: [ServeToolSpec] = []
    var toolChoice: ServeToolChoice = .auto
    var temperature: Double?
    var topP: Double?
    var maxTokens: Int?
    var stop: [String] = []
    var stream: Bool = false
    var includeUsage: Bool = false
    var seed: Int?
    var reasoning: Bool?                        // S2-3 — nil = the client didn't ask
    /// An `reasoning_effort` string this server doesn't know, kept only so the request log can say
    /// what was seen (S2-3b). A parameter value is metadata; no prompt or completion content is
    /// stored anywhere near it (rule 13).
    var unknownReasoningEffort: String? = nil
    /// The client sent a non-empty `tools[]`. Whether that array is *honoured* depends on the
    /// model (`ToolCallFormatTable.acceptsTools`) and is decided by the route, which is the only
    /// place that knows both; `RequestLogEntry.toolsIgnored` carries the outcome.
    var toolsRequested: Bool = false

    var effectiveMaxTokens: Int { maxTokens ?? Self.defaultMaxTokens }

    /// The tools to put in the prompt, after `tool_choice` (S2-2):
    ///
    /// - `.none` renders none (the client asked for no tool use), so the caller also stops
    ///   scanning for calls;
    /// - `.function(name)` renders just that tool when the client sent it, else every tool
    ///   (best effort — the model has no way to be *forced*);
    /// - `.auto` / `.required` render every tool.
    var renderedTools: [ServeToolSpec] {
        switch toolChoice {
        case .none: return []
        case .function(let name):
            let named = tools.filter { $0.name == name }
            return named.isEmpty ? tools : named
        case .auto, .required: return tools
        }
    }
}


/// A model backend: turns one request into a stream of events. The queue and the transport live
/// outside it, so a fake backend can stand in for MLX in tests.
nonisolated protocol ServeBackend: Sendable {
    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error>

    /// The checkpoint's `model_type`, when the backend knows it. The route asks this to decide
    /// whether a client's `tools` are honoured (S2-2) — the same value the backend itself uses to
    /// pick the parser, so the two answers can't disagree.
    var modelType: String? { get }

    /// The context window the model declares (S2-6, R23). `nil` when nothing declares one, in which
    /// case the route pre-counts nothing.
    var contextLimit: Int? { get }

    /// The **real** prompt's token count, before any response byte (S2-6, R23). `nil` means "this
    /// backend can't count", which is the honest answer for a streamer whose SDK has no tokenizer —
    /// S1's in-stream overflow error stays the behaviour there. Never an estimate.
    ///
    /// **Throws a `ServeError` only when the checkpoint's template *refuses* the request** (S2-8):
    /// that is a `400` the route can send before the head, in a stream as much as outside one. Any
    /// other failure to count stays `nil` — S2-6's fail-open, unchanged.
    func countPromptTokens(_ request: ServeRequest) async throws -> Int?
}

extension ServeBackend {
    /// Backends with no checkpoint behind them answer `nil` for the family fact.
    var modelType: String? { nil }
    var contextLimit: Int? { nil }
    func countPromptTokens(_ request: ServeRequest) async throws -> Int? { nil }
}

// MARK: - Queue

/// Design §3 "Queueing": one generation in flight per model, a global cap (1 in S1 — two models on
/// one GPU is worse than waiting), and at most `perModelDepth` requests *waiting* per model; the
/// next one is refused so the route can answer `503` + `Retry-After: 2`.
///
/// Two-phase on purpose: `reserve` decides admit/refuse immediately (before any response byte is
/// written, so a refusal can still be a proper 503), `waitTurn` then suspends until a slot is free.
/// "Depth" counts waiters, not the one running — implementer's call, pending owner confirmation.
actor ServeQueue {
    struct Ticket: Sendable, Hashable {
        let id: UInt64
        let model: String
    }

    static let defaultGlobalCap = 1
    static let defaultPerModelDepth = 4

    private struct Waiter {
        let ticket: Ticket
        var continuation: CheckedContinuation<Void, Error>?
    }

    private let globalCap: Int
    private let perModelDepth: Int
    private var nextID: UInt64 = 0
    private var running: [UInt64: String] = [:]
    private var waiting: [Waiter] = []              // FIFO

    init(globalCap: Int = ServeQueue.defaultGlobalCap, perModelDepth: Int = ServeQueue.defaultPerModelDepth) {
        self.globalCap = max(1, globalCap)
        self.perModelDepth = max(0, perModelDepth)
    }

    /// A place in line, or `nil` when `model`'s line is full.
    func reserve(model: String) -> Ticket? {
        if !canRun(model), waiting.filter({ $0.ticket.model == model }).count >= perModelDepth {
            return nil
        }
        nextID += 1
        let ticket = Ticket(id: nextID, model: model)
        waiting.append(Waiter(ticket: ticket, continuation: nil))
        pump()
        return ticket
    }

    /// Suspends until `ticket` may run. Throws `CancellationError` (and frees the place) if the
    /// calling task is cancelled while waiting.
    func waitTurn(_ ticket: Ticket) async throws {
        if running[ticket.id] != nil { return }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if running[ticket.id] != nil { continuation.resume(); return }
                guard let index = waiting.firstIndex(where: { $0.ticket == ticket }) else {
                    continuation.resume(throwing: CancellationError()); return
                }
                waiting[index].continuation = continuation
            }
        } onCancel: {
            Task { await self.release(ticket) }
        }
    }

    /// Give the place back — running or still waiting. Idempotent.
    func release(_ ticket: Ticket) {
        running[ticket.id] = nil
        if let index = waiting.firstIndex(where: { $0.ticket == ticket }) {
            let waiter = waiting.remove(at: index)
            waiter.continuation?.resume(throwing: CancellationError())
        }
        pump()
    }

    var runningCount: Int { running.count }
    var waitingCount: Int { waiting.count }

    // MARK: - Private

    private func canRun(_ model: String) -> Bool {
        running.count < globalCap && !running.values.contains(model)
    }

    /// Admit, in arrival order, every waiter that may now run.
    private func pump() {
        var index = 0
        while index < waiting.count {
            let ticket = waiting[index].ticket
            if canRun(ticket.model) {
                let waiter = waiting.remove(at: index)
                running[ticket.id] = ticket.model
                waiter.continuation?.resume()
            } else {
                index += 1
            }
        }
    }
}

// MARK: - Request log

/// One line of the request log (S1-3): metadata only. **No prompt or completion text, ever**
/// (rule 13, readme T12) — there is deliberately no field that could hold it.
nonisolated struct RequestLogEntry: Sendable, Equatable {
    var time: Date
    var userAgent: String?
    var model: String?
    var promptTokens: Int?
    var completionTokens: Int?
    var durationSeconds: Double
    var status: Int
    /// The client sent a `tools[]` and this model can't use it (S1's "tools ignored" note).
    var toolsIgnored: Bool
    /// Thinking was actually on for this request (S2-3) — the flag, never the thinking itself.
    var reasoningOn: Bool = false
    /// How many tool calls the model produced (S2-2). **A count only** — rule 13: no tool names,
    /// no arguments, no results, ever.
    var toolCalls: Int = 0
    /// PC-3: prompt tokens served from the prompt cache. A count only (rule 9).
    var cachedTokens: Int = 0
    /// PC-3: the cache path failed and the request ran uncached (metadata only).
    var cacheFellBack: Bool = false
}

/// A bounded in-memory log the Connect sheet / Settings pane will read (S1-5).
nonisolated final class RequestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [RequestLogEntry] = []
    private let capacity: Int

    init(capacity: Int = 200) { self.capacity = capacity }

    func record(_ entry: RequestLogEntry) {
        lock.lock(); defer { lock.unlock() }
        items.append(entry)
        if items.count > capacity { items.removeFirst(items.count - capacity) }
    }

    /// Newest last.
    var entries: [RequestLogEntry] {
        lock.lock(); defer { lock.unlock() }
        return items
    }
}
