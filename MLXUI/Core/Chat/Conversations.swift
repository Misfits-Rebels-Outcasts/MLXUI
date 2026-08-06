//
//  Conversations.swift
//  MLXUI — multiple chat conversations.
//
//  A `Conversation` is a persisted chat transcript for one model: the renderable items
//  (text bubbles + tool cards) plus identity and ordering metadata. `ConversationStore`
//  keeps one JSON file per conversation under `Application Support/AI Browser/conversations/`
//  so chats survive relaunches in both editions (path via `ModelStore`).
//
//  The transcript is the UI's record; the model's memory is rebuilt per send by mapping the
//  items to `Chat.Message` history (`chatHistory`) and rehydrating `ChatSession` with it.
//  Tool cards are folded into their surrounding assistant turn as plain text — the parser
//  formats (XML/JSON tool markup) differ per family, so replaying results as text is the
//  format-neutral way to keep them in context.
//
//  Pure value types + directory-injected statics, `nonisolated` — testable synchronously.
//

import Foundation
import MLXLMCommon

/// One chat conversation with one model.
struct Conversation: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    /// Catalog id of the model this conversation belongs to.
    let modelID: String
    /// Shown in the conversation list; derived from the first prompt once one exists.
    var title: String
    let createdAt: Date
    /// Bumped on every completed turn; the list is sorted by this, newest first.
    var updatedAt: Date
    var items: [TranscriptItem]

    init(
        id: UUID = UUID(),
        modelID: String,
        title: String = Conversation.untitled,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        items: [TranscriptItem] = []
    ) {
        self.id = id
        self.modelID = modelID
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.items = items
    }

    static let untitled = "New Chat"

    /// A list title from the first prompt: first line, trimmed, capped.
    static func title(forFirstPrompt prompt: String) -> String {
        let firstLine = prompt
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .newlines)[0]
            .trimmingCharacters(in: .whitespaces)
        guard !firstLine.isEmpty else { return untitled }
        let maxLength = 48
        guard firstLine.count > maxLength else { return firstLine }
        return firstLine.prefix(maxLength) + "…"
    }

    /// The model-facing history for rehydrating `ChatSession`: user/assistant text turns,
    /// with tool cards folded in as assistant text (`[tool …]` lines) so earlier tool
    /// results stay in context without re-encoding any family-specific tool-call markup.
    static func chatHistory(from items: [TranscriptItem]) -> [Chat.Message] {
        var history: [Chat.Message] = []
        for item in items {
            switch item {
            case .message(let message):
                guard !message.text.isEmpty else { continue }
                switch message.role {
                case .user: history.append(.user(message.text))
                case .assistant:
                    // A tool card can open an assistant turn; subsequent text belongs to it.
                    if let last = history.last, last.role == .assistant {
                        history[history.count - 1].content += "\n" + message.text
                    } else {
                        history.append(.assistant(message.text))
                    }
                }
            case .tool(let activity):
                guard case .finished = activity.status, let result = activity.result,
                      !result.isEmpty
                else { continue }
                let line = "[tool \(activity.name) returned: \(result)]"
                if let last = history.last, last.role == .assistant {
                    history[history.count - 1].content += "\n" + line
                } else {
                    history.append(.assistant(line))
                }
            }
        }
        return history
    }
}

/// Disk persistence: one `{uuid}.json` per conversation.
nonisolated enum ConversationStore {
    /// `…/Application Support/AI Browser/conversations`.
    static func defaultDirectory() -> URL {
        ModelStore.shared.baseDirectory.appendingPathComponent(
            "conversations", isDirectory: true)
    }

    /// All persisted conversations for a model, newest `updatedAt` first. Unreadable files
    /// are skipped (a damaged conversation shouldn't take down the rest).
    static func conversations(
        forModelID modelID: String, in directory: URL = defaultDirectory()
    ) -> [Conversation] {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { url -> Conversation? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? decoder.decode(Conversation.self, from: data)
            }
            .filter { $0.modelID == modelID }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    static func save(_ conversation: Conversation, in directory: URL = defaultDirectory()) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(conversation) else { return }
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try? data.write(to: fileURL(for: conversation.id, in: directory), options: .atomic)
    }

    static func delete(id: UUID, in directory: URL = defaultDirectory()) {
        try? FileManager.default.removeItem(at: fileURL(for: id, in: directory))
    }

    static func fileURL(for id: UUID, in directory: URL) -> URL {
        directory.appendingPathComponent("\(id.uuidString).json")
    }
}
