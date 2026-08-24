import Testing
import Foundation
import CryptoKit
@testable import MLXUI

/// CFM-R12-8 — staged effects: a Stage Send/Post row queues one JSON entry into the flow's
/// visible outbox and stops. Nothing sends. The entry shape, content-digest id, and status
/// sentence match the Python's `outbox.py` byte for byte.
struct CatFlowOutboxTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-outbox-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    @Test func stageWritesThePythonShapedEntryByteForByte() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let fixed = Date(timeIntervalSince1970: 1_752_000_000)
        let row = Row(id: UUID(), task: "Stage Post", settings: "threads")
        let inputs = [Asset(items: [Item(kind: .text, value: "ship the minutes", path: nil, sourceText: nil)])]

        let result = try OutboxStore.stage(row: row, inputs: inputs, kind: "post",
                                           workspace: ws, flowID: "f", now: { fixed })

        // The deterministic id: sha256("Stage Post|threads|ship the minutes")[:10].
        let basis = "Stage Post|threads|ship the minutes"
        let digest = CryptoKit.SHA256.hash(data: Data(basis.utf8))
            .map { String(format: "%02x", $0) }.joined()
        #expect(result.id == String(digest.prefix(10)))
        #expect(result.status == "queued to threads (\(result.id))")
        #expect(result.summary == "to threads: ship the minutes")

        let file = OutboxStore.directory(workspace: ws, flowID: "f")
            .appendingPathComponent("\(result.id).json")
        let written = try String(contentsOf: file, encoding: .utf8)
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let stagedAt = iso.string(from: fixed)
        // The Python `json.dumps(entry, indent=2)` shape — field order and spacing.
        #expect(written == """
        {
          "id": "\(result.id)",
          "kind": "post",
          "destination": "threads",
          "text": "ship the minutes",
          "status": "pending",
          "staged_at": "\(stagedAt)"
        }
        """)
    }

    @Test func collisionSuffixesTheId() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let row = Row(id: UUID(), task: "Stage Send", settings: "outbox")
        let inputs = [Asset(items: [Item(kind: .text, value: "same content", path: nil, sourceText: nil)])]
        let a = try OutboxStore.stage(row: row, inputs: inputs, kind: "send", workspace: ws, flowID: "f")
        let b = try OutboxStore.stage(row: row, inputs: inputs, kind: "send", workspace: ws, flowID: "f")
        #expect(a.id != b.id)
        #expect(b.id.hasPrefix(a.id))
        #expect(b.id.contains("-2"))
    }

    @Test func entriesListsPendingInOrder() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let row = Row(id: UUID(), task: "Stage Send", settings: "support")
        _ = try OutboxStore.stage(row: row,
                                  inputs: [Asset(items: [Item(kind: .text, value: "reply", path: nil, sourceText: nil)])],
                                  kind: "send", workspace: ws, flowID: "f")
        let entries = OutboxStore.entries(workspace: ws, flowID: "f")
        #expect(entries.count == 1)
        #expect(entries[0].kind == "send")
        #expect(entries[0].destination == "support")
        #expect(entries[0].text == "reply")
        #expect(entries[0].stagedAt.isEmpty == false)
    }

    @Test func stagedFlowsAreRunnableNow() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "40-StagedPost")
        #expect(FlowRunner.canRun(doc) == .runnable)
    }

    /// The spec's "nothing in the app can send one — a test that greps for a transport is
    /// fine": the outbox module contains no network transport at all.
    @Test func outboxHasNoTransport() throws {
        let filePath = #filePath
        let url = URL(fileURLWithPath: filePath)
        var dir = url.deletingLastPathComponent()
        while dir.lastPathComponent != "MLXUITests" { dir = dir.deletingLastPathComponent() }
        let source = try String(contentsOf: dir.deletingLastPathComponent()
            .appendingPathComponent("MLXUI/FlowKit/Tools/OutboxStore.swift"), encoding: .utf8)
        #expect(!source.contains("URLSession"))
        #expect(!source.contains("http://"))
        #expect(!source.contains("https://"))
        #expect(!source.contains("NWConnection"))
    }
}
