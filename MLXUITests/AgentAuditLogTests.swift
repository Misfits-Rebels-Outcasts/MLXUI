import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG6b-2 — the session audit trail. The log is a pure reducer over `AgentToolEvent`, so every
/// behavior (per-event rows, detail truncation, entry cap, clear) is tested synchronously; the
/// sheet UI is verified by build + smoke.
struct AgentAuditLogTests {

    @Test func recordsOneRowPerLifecycleEvent() {
        var log = AgentAuditLog()
        log.record(.started(name: "read_file", arguments: ["path": .string("/tmp/a.txt")]))
        log.record(.finished(name: "read_file", result: "hello"))
        log.record(.failed(name: "fetch_url", message: "timed out after 15s"))
        log.record(.denied(name: "write_file"))
        log.record(.unknownTool(name: "nope"))
        log.record(.budgetExceeded(name: "calculator"))

        #expect(log.entries.count == 6)
        #expect(log.entries[0].outcome == .started)
        #expect(log.entries[0].toolName == "read_file")
        #expect(log.entries[0].detail.contains("/tmp/a.txt"))
        #expect(log.entries[1].outcome == .finished)
        #expect(log.entries[1].detail == "hello")
        #expect(log.entries[2].outcome == .failed)
        #expect(log.entries[2].detail == "timed out after 15s")
        #expect(log.entries[3].outcome == .denied)
        #expect(log.entries[4].outcome == .unknownTool)
        #expect(log.entries[5].outcome == .budgetExceeded)
    }

    @Test func stampsTheProvidedDate() {
        var log = AgentAuditLog()
        let date = Date(timeIntervalSince1970: 1_000_000)
        log.record(.denied(name: "t"), at: date)
        #expect(log.entries[0].date == date)
    }

    @Test func truncatesLongDetails() {
        var log = AgentAuditLog()
        let long = String(repeating: "x", count: AgentAuditLog.maxDetailLength + 50)
        log.record(.finished(name: "read_file", result: long))
        // maxDetailLength characters plus the ellipsis marker.
        #expect(log.entries[0].detail.count == AgentAuditLog.maxDetailLength + 1)
        #expect(log.entries[0].detail.hasSuffix("…"))
    }

    @Test func shortDetailsAreKeptVerbatim() {
        let text = String(repeating: "y", count: AgentAuditLog.maxDetailLength)
        #expect(AgentAuditLog.truncated(text) == text)
    }

    @Test func capsAtMaxEntriesDroppingOldest() {
        var log = AgentAuditLog()
        for i in 0..<(AgentAuditLog.maxEntries + 3) {
            log.record(.denied(name: "tool-\(i)"))
        }
        #expect(log.entries.count == AgentAuditLog.maxEntries)
        // The three oldest rows fell off the front.
        #expect(log.entries.first?.toolName == "tool-3")
        #expect(log.entries.last?.toolName == "tool-\(AgentAuditLog.maxEntries + 2)")
    }

    @Test func clearEmptiesTheLog() {
        var log = AgentAuditLog()
        log.record(.denied(name: "t"))
        #expect(!log.isEmpty)
        log.clear()
        #expect(log.isEmpty)
    }

    @Test func argumentSummaryIsCompactJSONOrEmpty() {
        #expect(AgentAuditLog.summary(of: [:]) == "")
        let summary = AgentAuditLog.summary(of: ["query": .string("cats"), "top_k": .int(3)])
        #expect(summary.contains("\"query\""))
        #expect(summary.contains("cats"))
        #expect(summary.contains("3"))
    }
}
