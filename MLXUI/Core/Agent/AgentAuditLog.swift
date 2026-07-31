//
//  AgentAuditLog.swift
//  MLXUI — agentic chat tools, slice AG6b-2 (safety rails, part 2).
//
//  Session-scoped audit trail of every tool-call lifecycle event, fed from the same
//  `AgentToolEvent` stream that drives the transcript's tool cards. Unlike the transcript
//  (which folds started→finished into one card and is cleared when switching models), the
//  audit log is a flat, append-only record: one row per event, timestamped, capped.
//
//  In-memory only, by design: results can contain user file contents (`read_file`) or web
//  pages, so the log is never written to disk. It lives for the app session and can be
//  cleared by the user from the audit sheet.
//
//  Pure reducer, no SwiftUI, `nonisolated` — drive it directly from tests.
//

import Foundation
import MLXLMCommon

/// One audit-trail row: a single tool lifecycle event.
nonisolated struct AuditEntry: Identifiable, Sendable, Equatable {
    /// Which lifecycle event this row records (mirrors `AgentToolEvent` cases).
    enum Outcome: Sendable, Equatable {
        case started, finished, failed, denied, unknownTool, budgetExceeded
    }

    let id: UUID
    let date: Date
    let toolName: String
    let outcome: Outcome
    /// Compact context: the call's arguments for `.started`, the result / error message for
    /// terminal events. Truncated to `AgentAuditLog.maxDetailLength`.
    let detail: String
}

/// Append-only, capped log of tool activity for the current app session.
nonisolated struct AgentAuditLog: Sendable, Equatable {
    /// Oldest rows are dropped past this, so a long session can't grow without bound.
    static let maxEntries = 500
    /// Detail strings are clipped — `read_file` results alone can be 256 KB.
    static let maxDetailLength = 300

    private(set) var entries: [AuditEntry] = []

    var isEmpty: Bool { entries.isEmpty }

    /// Fold one lifecycle event into the log.
    mutating func record(_ event: AgentToolEvent, at date: Date = Date()) {
        switch event {
        case .started(let name, let arguments):
            append(date, name, .started, Self.summary(of: arguments))
        case .finished(let name, let result):
            append(date, name, .finished, result)
        case .failed(let name, let message):
            append(date, name, .failed, message)
        case .denied(let name):
            append(date, name, .denied, "")
        case .unknownTool(let name):
            append(date, name, .unknownTool, "")
        case .budgetExceeded(let name):
            append(date, name, .budgetExceeded, "")
        }
    }

    mutating func clear() {
        entries.removeAll()
    }

    private mutating func append(
        _ date: Date, _ name: String, _ outcome: AuditEntry.Outcome, _ detail: String
    ) {
        entries.append(AuditEntry(
            id: UUID(), date: date, toolName: name, outcome: outcome,
            detail: Self.truncated(detail)))
        if entries.count > Self.maxEntries {
            entries.removeFirst(entries.count - Self.maxEntries)
        }
    }

    /// One-line JSON of a call's arguments (best-effort; empty when there are none).
    /// Slashes stay unescaped so file paths and URLs read naturally in the log.
    static func summary(of arguments: [String: JSONValue]) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes, .sortedKeys]
        guard !arguments.isEmpty,
              let data = try? encoder.encode(arguments),
              let text = String(data: data, encoding: .utf8) else { return "" }
        return text
    }

    static func truncated(_ text: String) -> String {
        text.count <= maxDetailLength ? text : String(text.prefix(maxDetailLength)) + "…"
    }
}
