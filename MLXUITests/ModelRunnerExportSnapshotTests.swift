import Foundation
import Testing
@testable import MLXUI

/// CX-2 — `ModelRunner.exportSnapshot(for:)`, the one place the export reads a conversation
/// from. Exports are always taken through it because the stored copy can be a turn behind the
/// live transcript. The header button and the save panel are verified by build and smoke
/// (CX-S1…CX-S5).
@MainActor
struct ModelRunnerExportSnapshotTests {

    private func makeRunner() -> (ModelRunner, URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("export-snapshot-\(UUID().uuidString)")
        let runner = ModelRunner()
        runner.prepare(for: makeEntry(id: "model-a", displayName: "Test Model"), in: directory)
        return (runner, directory)
    }

    @Test func liveItemsWinForTheActiveConversation() throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }

        runner.appendUserMessageForTesting("What is 37 x 481?")
        runner.appendAssistantChunkForTesting("It's 17797.")
        let activeID = try #require(runner.activeConversationID)

        // Nothing has been saved: the stored copy is still empty, the live transcript isn't.
        let snapshot = try #require(runner.exportSnapshot(for: activeID))
        #expect(snapshot.id == activeID)
        #expect(snapshot.items == runner.transcript.items)
        #expect(snapshot.items.count == 2)
    }

    @Test func aChunkArrivingAfterTheLastSaveIsStillExported() throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }

        runner.appendUserMessageForTesting("hello")
        let activeID = try #require(runner.activeConversationID)

        // A second conversation saves the first one and moves the working copy off it.
        runner.newConversation()
        runner.selectConversation(activeID)

        // A turn streams in without ending, so the saved copy is now behind.
        runner.appendAssistantChunkForTesting("partial…")
        let snapshot = try #require(runner.exportSnapshot(for: activeID))
        #expect(snapshot.items.count == 2)
        guard case .message(let last) = snapshot.items.last else {
            Issue.record("expected a message")
            return
        }
        #expect(last.role == .assistant)
        #expect(last.text == "partial…")
    }

    @Test func anotherConversationExportsItsStoredItems() throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }

        runner.appendUserMessageForTesting("first chat")
        let firstID = try #require(runner.activeConversationID)

        runner.newConversation()
        let secondID = try #require(runner.activeConversationID)
        #expect(secondID != firstID)

        // The second conversation is active; the first keeps its stored items.
        let snapshot = try #require(runner.exportSnapshot(for: firstID))
        #expect(snapshot.id == firstID)
        #expect(snapshot.items.count == 1)
        guard case .message(let first) = snapshot.items.first else {
            Issue.record("expected a message")
            return
        }
        #expect(first.role == .user)
        #expect(first.text == "first chat")
    }

    @Test func anEmptyConversationHasNothingToExport() throws {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }

        let emptyID = try #require(runner.activeConversationID)
        #expect(runner.exportSnapshot(for: emptyID) == nil)
    }

    @Test func anUnknownConversationHasNothingToExport() {
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(runner.exportSnapshot(for: UUID()) == nil)
    }

    @Test func exportingRequiresNoModelWeights() throws {
        // `prepare` doesn't load weights, and `exportSnapshot` reads only in-memory state, so an
        // export is possible on an installed-but-never-run model.
        let (runner, directory) = makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }

        runner.appendUserMessageForTesting("hi")
        let id = try #require(runner.activeConversationID)
        let snapshot = try #require(runner.exportSnapshot(for: id))

        // The pure exporter runs on it unchanged.
        let markdown = ConversationExporter.markdown(
            snapshot,
            context: ConversationExportContext(
                modelDisplayName: "Test Model",
                appVersion: "0.1 (2)",
                edition: "AI Browser",
                exportedAt: Date(timeIntervalSince1970: 1_791_025_320),
                timeZone: TimeZone(secondsFromGMT: 8 * 3600) ?? .gmt))
        #expect(markdown.contains("### You\n\nhi"))
    }
}
