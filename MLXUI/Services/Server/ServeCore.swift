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
    var totalTokens: Int { promptTokens + completionTokens }
}

/// One incremental piece of a tool call (S2). Unused in S1.
nonisolated struct ServeToolCallDelta: Sendable, Equatable {
    var index: Int
    var id: String?
    var name: String?
    var argumentsFragment: String?
}

nonisolated enum ServeEvent: Sendable, Equatable {
    case textDelta(String)
    case reasoningDelta(String)                 // S2 — never produced in S1
    case toolCallDelta(ServeToolCallDelta)      // S2 — never produced in S1
    case usage(ServeUsage)
    case finish(ServeFinishReason)
    case error(ServeError)
}

nonisolated struct ServeMessage: Sendable, Equatable {
    enum Role: String, Sendable, Equatable { case system, user, assistant, tool }
    var role: Role
    var text: String
    var toolCalls: [ServeToolCallDelta] = []    // S2 — unset
    var toolCallID: String?                     // S2 — unset
}

nonisolated struct ServeToolSpec: Sendable, Equatable {   // S2 — unset
    var name: String
    var descriptionText: String?
    var parametersJSON: String?
}

nonisolated struct ServeRequest: Sendable, Equatable {
    static let defaultMaxTokens = 4096          // R16: a request with no limit gets 4096

    var model: String
    var messages: [ServeMessage]
    var tools: [ServeToolSpec] = []             // S2 — unset
    var toolChoice: String?                     // S2 — unset
    var temperature: Double?
    var topP: Double?
    var maxTokens: Int?
    var stop: [String] = []
    var stream: Bool = false
    var includeUsage: Bool = false
    var seed: Int?
    var reasoning: Bool?                        // S2 — unset
    /// The client sent `tools`/`tool_choice` and S1 ignored them (R9). Drives the request log's
    /// "tools ignored" note; never changes behaviour.
    var toolsIgnored: Bool = false

    var effectiveMaxTokens: Int { maxTokens ?? Self.defaultMaxTokens }
}

/// A model backend: turns one request into a stream of events. The queue and the transport live
/// outside it, so a fake backend can stand in for MLX in tests.
nonisolated protocol ServeBackend: Sendable {
    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error>
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
    var toolsIgnored: Bool
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
