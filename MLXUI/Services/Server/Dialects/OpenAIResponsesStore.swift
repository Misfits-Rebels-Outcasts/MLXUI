import Foundation

/// S3-3c — the `previous_response_id` store (owner rulings R34, R39, R40).
///
/// **In RAM only, never on disk, never logged** (rule 13: the conversation is the user's document). It is owned
/// by the Responses dialect and injected through `ServeEnvironment`. Key = the response id; value = the whole
/// conversation up to and including that response's output, as the neutral `[ServeMessage]` (earlier stored turns
/// + this response's input + its assistant turn: text and function calls; reasoning is dropped, as on input).
///
/// Limits (implementer's call, pending owner confirmation; journal 2026-450): **64 entries, 64 MB of text, 30
/// minutes**, least-recently-used first. A Codex-style client sends a growing conversation as one chain, so each
/// turn adds one entry holding the whole history: 64 entries is hours of turns; 64 MB is ~16 M tokens of text, far
/// more than any prompt the server admits (S3-5a), so it only matters for pathological inputs; 30 minutes is long
/// enough to leave a session idle (OpenAI keeps responses 30 days, but a local server's RAM is not a database).
/// An unknown or expired id is an error, never a silent fresh start (see `notFound`).
nonisolated final class ResponseStore: @unchecked Sendable {
    static let defaultMaxEntries = 64
    static let defaultMaxBytes = 64 * 1_048_576
    static let defaultTTL: TimeInterval = 30 * 60

    private struct Entry {
        let model: String
        let conversation: [ServeMessage]
        let bytes: Int
        let storedAt: Date
        var lastUsed: UInt64
    }

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    private var clock: UInt64 = 0
    private let maxEntries: Int
    private let maxBytes: Int
    private let ttl: TimeInterval
    private let now: @Sendable () -> Date

    init(maxEntries: Int = defaultMaxEntries, maxBytes: Int = defaultMaxBytes, ttl: TimeInterval = defaultTTL,
         now: @escaping @Sendable () -> Date = { Date() }) {
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        self.ttl = ttl
        self.now = now
    }

    /// Stores `conversation` under `id`. A conversation larger than the whole byte cap is not kept (it could only
    /// evict everything else); returns whether it was stored.
    @discardableResult
    func save(id: String, model: String, conversation: [ServeMessage]) -> Bool {
        lock.lock(); defer { lock.unlock() }
        dropExpired()
        let bytes = Self.bytes(of: conversation)
        guard bytes <= maxBytes else { return false }
        clock += 1
        entries[id] = Entry(model: model, conversation: conversation, bytes: bytes, storedAt: now(), lastUsed: clock)
        while entries.count > maxEntries || totalBytes > maxBytes {
            guard let victim = entries.min(by: { $0.value.lastUsed < $1.value.lastUsed })?.key else { break }
            entries[victim] = nil
        }
        return entries[id] != nil
    }

    /// The stored conversation for `id`, or `nil` when it is unknown or expired. Using an entry makes it the most
    /// recently used.
    func conversation(for id: String) -> [ServeMessage]? {
        lock.lock(); defer { lock.unlock() }
        dropExpired()
        guard var entry = entries[id] else { return nil }
        clock += 1
        entry.lastUsed = clock
        entries[id] = entry
        return entry.conversation
    }

    /// Serve turned off for a model, or the model uninstalled: its responses go.
    @discardableResult
    func evict(model: String) -> Int {
        lock.lock(); defer { lock.unlock() }
        let gone = entries.filter { $0.value.model == model }.keys
        gone.forEach { entries[$0] = nil }
        return gone.count
    }

    /// The server stopped.
    func removeAll() {
        lock.lock(); defer { lock.unlock() }
        entries.removeAll()
    }

    // MARK: Inspection (tests)

    var count: Int { lock.lock(); defer { lock.unlock() }; dropExpired(); return entries.count }
    var totalBytesHeld: Int { lock.lock(); defer { lock.unlock() }; dropExpired(); return totalBytes }

    // MARK: Private (lock held)

    private var totalBytes: Int { entries.values.reduce(0) { $0 + $1.bytes } }

    private func dropExpired() {
        let cutoff = now().addingTimeInterval(-ttl)
        for (id, entry) in entries where entry.storedAt <= cutoff { entries[id] = nil }
    }

    /// The text a conversation holds (UTF-8 bytes of every message, call id, name and arguments).
    static func bytes(of conversation: [ServeMessage]) -> Int {
        conversation.reduce(0) { sum, message in
            sum + message.text.utf8.count + (message.toolCallID?.utf8.count ?? 0) + (message.toolName?.utf8.count ?? 0)
                + message.toolCalls.reduce(0) { $0 + ($1.id?.utf8.count ?? 0) + $1.name.utf8.count + $1.argumentsJSON.utf8.count }
        }
    }

    /// The error for an id the store does not hold. OpenAI answers `400` with this wording (a community thread
    /// quotes `400 Bad Request` / "Previous response with id … not found"; the API docs are silent, journal
    /// 2026-450), so a Codex-style client sees what it would see against OpenAI.
    static func notFound(_ id: String) -> ServeError {
        ServeError(status: 400, message: "Previous response with id '\(id)' not found.", code: "previous_response_not_found")
    }
}
