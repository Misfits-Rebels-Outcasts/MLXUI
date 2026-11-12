import Foundation
import CoreGraphics
import SwiftUI
import AppKit
import Testing
@testable import MLXUI

/// MT4-5: "Ask in Chat" from the Run UIs. The result becomes a file, the Chat page starts a new conversation with it
/// waiting in the composer, and the ordinary send path registers it as asset `a1`. Helpers are prefixed `MT45`.
@MainActor
@Suite(.serialized)
struct AskInChatTests {

    private let model = makeEntry(id: "mlx-community--mt45", displayName: "Qwen3-8B")

    private func MT45Dir() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mt45-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func MT45Image() throws -> CGImage {
        let context = try #require(CGContext(data: nil, width: 4, height: 3, bitsPerComponent: 8, bytesPerRow: 0,
                                             space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 4, height: 3))
        return try #require(context.makeImage())
    }

    private final class OneShot: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { $0.yield("reply"); $0.finish() }
        }
    }

    private final class Hold: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
            continuation.yield("streaming")
            return stream
        }
    }

    // MARK: - the result as a file

    @Test func textBecomesATxtFileWithTheExactText() throws {
        let base = try MT45Dir(); defer { try? FileManager.default.removeItem(at: base) }
        let url = try AskInChatResult.text("Hello\nworld", name: "transcript").writeTemporaryFile(in: base)
        #expect(url.lastPathComponent == "transcript.txt")
        #expect(url.path.hasPrefix(AskInChatResult.temporaryRoot(in: base).path))
        #expect(try String(contentsOf: url, encoding: .utf8) == "Hello\nworld")
    }

    @Test func anImageBecomesAPNGAndAudioBecomesAWAV() throws {
        let base = try MT45Dir(); defer { try? FileManager.default.removeItem(at: base) }
        let png = try AskInChatResult.image(try MT45Image(), name: "flux image").writeTemporaryFile(in: base)
        #expect(png.lastPathComponent == "flux-image.png")
        #expect(Array(try Data(contentsOf: png).prefix(4)) == [0x89, 0x50, 0x4E, 0x47])
        let buffer = AudioBuffer(samples: (0..<2_400).map { sinf(Float($0) / 10) }, sampleRate: 24_000)
        let wav = try AskInChatResult.audio(buffer, name: "speech").writeTemporaryFile(in: base)
        #expect(wav.lastPathComponent == "speech.wav")
        #expect(Array(try Data(contentsOf: wav).prefix(4)) == Array("RIFF".utf8))
    }

    @Test func anEmptyResultIsRefusedAndNamesAreSanitised() throws {
        let base = try MT45Dir(); defer { try? FileManager.default.removeItem(at: base) }
        #expect(throws: AskInChatResult.WriteError.empty) { try AskInChatResult.text("  \n", name: "t").writeTemporaryFile(in: base) }
        #expect(throws: AskInChatResult.WriteError.empty) {
            try AskInChatResult.audio(AudioBuffer(samples: [], sampleRate: 24_000), name: "a").writeTemporaryFile(in: base)
        }
        #expect(AskInChatResult.stem("../../etc/passwd") == "etc-passwd" || !AskInChatResult.stem("../../etc/passwd").contains("/"))
        #expect(AskInChatResult.stem("///") == "result")
        #expect(AskInChatResult.stem("") == "result")
        #expect(AskInChatResult.stem(String(repeating: "x", count: 200)).count == 40)
    }

    // MARK: - the page

    @Test func theRequestStartsANewConversationWithTheFileWaitingAndKeepsTheOldOne() throws {
        let dir = try MT45Dir(), base = try MT45Dir()
        defer { [dir, base].forEach { try? FileManager.default.removeItem(at: $0) } }
        ChatGate.overrideForTesting = true; defer { ChatGate.overrideForTesting = nil }
        let old = Conversation(modelID: model.id, title: "old", updatedAt: Date(timeIntervalSince1970: 1000),
                               items: [.message(ChatMessage(role: .user, text: "old chat"))])
        ConversationStore.save(old, in: dir)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.acquire(.page, model: model)
        controller.runner.selectConversation(old.id)
        controller.setDraft(ChatDraft(text: "half typed"), for: .page)
        let file = try AskInChatResult.text("the transcript", name: "transcript").writeTemporaryFile(in: base)

        #expect(controller.askInChat(files: [file]))
        #expect(controller.holder == .page)
        #expect(controller.activeConversationID != old.id)                              // a NEW conversation
        #expect(controller.runner.transcript.items.isEmpty)
        #expect(controller.conversations.contains { $0.id == old.id })                  // the old one is still listed
        #expect(controller.draft(for: .page).files == [file])
        #expect(controller.draft(for: .page).text == "half typed")                      // nothing the user typed is lost

        // the page's own binding afterwards does not apply it a second time or reset the chat
        let active = controller.activeConversationID
        controller.acquire(.page, model: model)
        #expect(controller.activeConversationID == active)
        #expect(controller.draft(for: .page).files == [file])
    }

    @Test func aPageThatIsNotBoundYetTakesTheRequestWhenItBinds() throws {
        let dir = try MT45Dir(), base = try MT45Dir()
        defer { [dir, base].forEach { try? FileManager.default.removeItem(at: $0) } }
        ChatGate.overrideForTesting = true; defer { ChatGate.overrideForTesting = nil }
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.acquire(.bench, model: model)                                        // a sheet holds the runner, idle
        let file = try AskInChatResult.text("x", name: "x").writeTemporaryFile(in: base)
        #expect(controller.askInChat(files: [file]))
        #expect(controller.holder == .page)
        #expect(controller.draft(for: .page).files == [file])
    }

    @Test func aClosedGateARunningTurnAndNoFilesAreRefusedWithNothingChanged() async throws {
        let dir = try MT45Dir(), base = try MT45Dir(), modelDir = try MT45Dir()
        defer { [dir, base, modelDir].forEach { try? FileManager.default.removeItem(at: $0) } }
        let file = try AskInChatResult.text("x", name: "x").writeTemporaryFile(in: base)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        ChatGate.overrideForTesting = false
        #expect(!controller.canAskInChat)
        #expect(!controller.askInChat(files: [file]))
        #expect(controller.holder == nil)
        ChatGate.overrideForTesting = true; defer { ChatGate.overrideForTesting = nil }
        #expect(!controller.askInChat(files: []))

        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.streamingSessionForTesting = { Hold() }
        controller.acquire(.page, model: model)
        await controller.send("hello", from: .page)
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while !(controller.runner.isRunning && controller.runner.transcript.items.count == 2), ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        let active = controller.activeConversationID
        #expect(controller.runner.isRunning && !controller.canAskInChat)
        #expect(!controller.askInChat(files: [file]))
        #expect(controller.activeConversationID == active)                              // the turn was not stopped
        #expect(controller.runner.isRunning)
        #expect(controller.draft(for: .page).files.isEmpty)
        controller.stop(from: .page)
    }

    /// The real send path: the file the request left in the composer is registered as `a1` of the new conversation.
    @Test(arguments: ["text", "image", "audio"])
    func theFirstSendRegistersTheResultAsAssetA1(kind: String) async throws {
        let dir = try MT45Dir(), base = try MT45Dir(), modelDir = try MT45Dir()
        defer { [dir, base, modelDir].forEach { try? FileManager.default.removeItem(at: $0) } }
        ChatGate.overrideForTesting = true; defer { ChatGate.overrideForTesting = nil }
        let result: AskInChatResult = switch kind {
        case "image": .image(try MT45Image(), name: "picture")
        case "audio": .audio(AudioBuffer(samples: (0..<2_400).map { sinf(Float($0) / 10) }, sampleRate: 24_000), name: "speech")
        default: .text("Total due: 42.00", name: "extracted-text")
        }
        let file = try result.writeTemporaryFile(in: base)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.streamingSessionForTesting = { OneShot() }
        controller.runner.usableTokensForTesting = { _ in 32_000 }
        let routed = Routed()
        controller.runner.routerTurnForTesting = { routed.set($0) }
        controller.acquire(.page, model: model)
        #expect(controller.askInChat(files: [file]))

        let draft = controller.draft(for: .page)
        #expect(await controller.send("what is this?", attachments: draft.files, from: .page))
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while controller.runner.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(10)) }

        let conversation = try #require(controller.conversations.first { $0.id == controller.activeConversationID })
        let sent = conversation.items.compactMap { item -> [ChatAttachment]? in
            if case .message(let m) = item, m.role == .user { return m.attachments } else { return nil }
        }.first ?? []
        #expect(sent.count == 1)
        #expect(sent.first?.assetHandle == "a1")
        #expect(sent.first?.assetKind == kind)
        let assets = ChatAssetStore(conversationID: conversation.id, in: dir)
        let record = try #require(assets.record("a1"))
        #expect(record.kind == kind)
        #expect(FileManager.default.fileExists(atPath: assets.url(for: record).path))
        if kind == "text" { #expect(assets.text(for: "a1") == "Total due: 42.00") }
        #expect(routed.get?.prompt.contains("asset=\"a1\"") == true)                      // the model is told the handle
    }

    private final class Routed: @unchecked Sendable {
        private let lock = NSLock()
        private var value: RouterTurn?
        var get: RouterTurn? { lock.withLock { value } }
        func set(_ turn: RouterTurn?) { lock.withLock { value = turn } }
    }

    @Test func theAppStateEntrySelectsTheChatItemOrChangesNothing() throws {
        let base = try MT45Dir(); defer { try? FileManager.default.removeItem(at: base) }
        let app = AppState()
        ChatGate.overrideForTesting = false; defer { ChatGate.overrideForTesting = nil }
        #expect(!app.askInChat(.text("x", name: "x")))
        #expect(app.selectedSection == .home)
        ChatGate.overrideForTesting = true
        #expect(!app.askInChat(.text("   ", name: "x")))                                  // nothing to send
        #expect(app.selectedSection == .home)
        #expect(app.askInChat(.text("a transcript", name: "transcript")))
        #expect(app.selectedSection == .chat)
        #expect(app.chat.holder == .page)
        #expect(app.chat.draft(for: .page).files.map(\.lastPathComponent) == ["transcript.txt"])
    }

    /// A Run UI hosted with no `AppState` in the environment (the render tests, previews) must not trap: the button is absent.
    @Test func theButtonHostedWithoutAppStateDoesNotCrashAndShowsNothing() {
        ChatGate.overrideForTesting = true; defer { ChatGate.overrideForTesting = nil }
        let host = NSHostingView(rootView: AskInChatButton { .text("x", name: "x") })
        host.frame = NSRect(x: 0, y: 0, width: 200, height: 40)
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.width < 1)                                              // empty: no button was built
        // and with the state present it is a real button
        let withState = NSHostingView(rootView: AskInChatButton { .text("x", name: "x") }.environment(AppState()))
        withState.layoutSubtreeIfNeeded()
        #expect(withState.fittingSize.width > 20)
    }
}
