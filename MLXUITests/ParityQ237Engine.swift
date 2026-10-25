import Foundation
@testable import MLXUI

// WW4-2-1 -- the engine seam of the SPEC-Q237 parity experiment. The harness (`ParityQ237Harness.swift`) never touches a
// model; it talks to a `ParityEngine`. WW4-2-1 plugs in `ScriptedParityEngine` (below); WW4-2-2 plugs in real engines behind
// the same protocol. Test target only -- nothing here ships (owner ruling 2026-10-25: the grammar one-shot lives in the tests).

/// One model reply plus what it cost. Token counts are the engine's own: a mock counts words (`approximateTokens`), a real
/// engine counts with the model's tokenizer.
nonisolated struct ParityReply: Sendable, Equatable {
    var text: String
    var promptTokens: Int
    var completionTokens: Int
    /// Model calls this reply cost. 1 for a mock; a real Q35 `gate` is 2 when the shipped `fireTag` retries.
    var modelCalls = 1
    /// Seconds of the reply's wall time that were setup, not generation (a grammar's vocabulary and compile). 0 for a mock.
    var setupSeconds = 0.0

    /// The mock's "tokenizer": whitespace-separated words, at least 1. Exposed so a test can compute the expected totals.
    static func approximateTokens(_ text: String) -> Int {
        max(1, text.split(whereSeparator: \.isWhitespace).count)
    }
}

/// The three model operations the two paths need. How a real engine slots in (WW4-2-2, implementer's call, pending owner
/// confirmation):
/// - `gate` and `line` are exactly the two closures `RealExecutor.llmExtractionCalls(stage:path:)` builds for the shipped
///   loop: `gate` = `fireTag(stage:prompt:tags: ["yes", "no"])` (returns the tag), `line` = `stage.run(.text(prompt))`
///   cut at the first newline, both from a stage built with `StageConfig(maxTokens: 32, temperature: 0)`. A `LLMEngine`-backed
///   adapter wraps those and adds token counts.
/// - `structured` is the grammar one-shot: build a grammar from `schema` with `MLXGuidedGeneration` (the way
///   `MLXChatBackend.prepareStructure` does), run ONE greedy generation (`temperature: 0`) on the same loaded model, and return
///   the reply text, which the harness decodes. Both real adapters load the model once through `ModelContainerPool`, from the
///   installed folder, and share it.
/// An engine instance serves one path of one case (the harness asks for a fresh one per path, see `ParityHarness.runCase`),
/// so a stateful mock can count calls without a reset.
nonisolated protocol ParityEngine: Sendable {
    /// The Q35 "is there another record?" decision: the reply text is exactly `yes` or `no`.
    func gate(prompt: String) async throws -> ParityReply
    /// The Q35 "what is the X value?" call: one bounded line (the adapter has already cut at the first newline).
    func line(prompt: String) async throws -> ParityReply
    /// The grammar one-shot: the whole table as one JSON reply, constrained by `schema` (a JSON Schema document whose root is an object).
    func structured(prompt: String, schema: String) async throws -> ParityReply
}

/// What one path cost: model calls, tokens, and wall time of the whole path (the model calls plus the shipped stage's own work).
nonisolated struct ParityAccounting: Sendable, Equatable {
    var modelCalls = 0
    var promptTokens = 0
    var completionTokens = 0
    var wallSeconds = 0.0
    /// Part of `wallSeconds` spent on setup (grammar vocabulary and compile), reported separately.
    var setupSeconds = 0.0

    var totalTokens: Int { promptTokens + completionTokens }
}

/// Counts every reply a path receives. An actor, because the shipped loop's closures are `@Sendable`.
actor ParityMeter {
    private var calls = 0
    private var prompt = 0
    private var completion = 0
    private var setup = 0.0

    func record(_ reply: ParityReply) {
        calls += reply.modelCalls
        setup += reply.setupSeconds
        prompt += reply.promptTokens
        completion += reply.completionTokens
    }

    func snapshot(wallSeconds: Double) -> ParityAccounting {
        ParityAccounting(modelCalls: calls, promptTokens: prompt, completionTokens: completion,
                         wallSeconds: wallSeconds, setupSeconds: setup)
    }
}

/// A mock that answers from ground-truth records, ignoring the prompt text. It models an engine that is right.
/// - The Q35 path: `gate` says `yes` for the first `records.count` calls then `no`; `line` walks the records cell by cell in
///   row-major order (the loop asks for every column of a record in order), `""` past the end.
/// - The one-shot path: `structured` returns `oneShotRecords` (default: the same records) as `{"records": [objects]}`.
/// A test makes the paths differ by passing different `oneShotRecords`.
actor ScriptedParityEngine: ParityEngine {
    let columns: [String]
    let records: [[String]]
    let oneShotRecords: [[String]]
    let delay: Duration?
    private var gateCalls = 0
    private var lineCalls = 0

    init(columns: [String], records: [[String]], oneShotRecords: [[String]]? = nil, delay: Duration? = nil) {
        self.columns = columns
        self.records = records
        self.oneShotRecords = oneShotRecords ?? records
        self.delay = delay
    }

    func gate(prompt: String) async throws -> ParityReply {
        try await pause()
        let answer = gateCalls < records.count ? "yes" : "no"
        gateCalls += 1
        return reply(answer, for: prompt)
    }

    func line(prompt: String) async throws -> ParityReply {
        try await pause()
        let record = lineCalls / max(columns.count, 1)
        let column = lineCalls % max(columns.count, 1)
        lineCalls += 1
        let value = record < records.count && column < records[record].count ? records[record][column] : ""
        return reply(value, for: prompt)
    }

    func structured(prompt: String, schema: String) async throws -> ParityReply {
        try await pause()
        let objects = oneShotRecords.map { Dictionary(zip(columns, $0), uniquingKeysWith: { _, last in last }) }
        let data = try JSONSerialization.data(withJSONObject: ["records": objects], options: [.sortedKeys])
        return reply(String(decoding: data, as: UTF8.self), for: prompt)
    }

    private func pause() async throws {
        if let delay { try await Task.sleep(for: delay) }
    }

    private func reply(_ text: String, for prompt: String) -> ParityReply {
        ParityReply(text: text, promptTokens: ParityReply.approximateTokens(prompt),
                    completionTokens: ParityReply.approximateTokens(text))
    }
}

/// An engine that always answers the one-shot with a fixed string, for the failure-path tests. The Q35 calls delegate to a
/// scripted engine so path A still runs.
actor FixedReplyParityEngine: ParityEngine {
    private let reply: String
    private let loop: ScriptedParityEngine

    init(structuredReply: String, columns: [String], records: [[String]]) {
        reply = structuredReply
        loop = ScriptedParityEngine(columns: columns, records: records)
    }

    func gate(prompt: String) async throws -> ParityReply { try await loop.gate(prompt: prompt) }
    func line(prompt: String) async throws -> ParityReply { try await loop.line(prompt: prompt) }

    func structured(prompt: String, schema: String) async throws -> ParityReply {
        ParityReply(text: reply, promptTokens: ParityReply.approximateTokens(prompt),
                    completionTokens: ParityReply.approximateTokens(reply))
    }
}
