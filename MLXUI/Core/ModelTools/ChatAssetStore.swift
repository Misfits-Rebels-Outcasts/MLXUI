//
//  ChatAssetStore.swift
//  MLXUI — Model Tools, MT1-3 (`RSI/DelegateModelToolsBacklog.md`, plan §3.7 step 5).
//
//  The assets one conversation has: attachments and model-tool outputs, each addressed by a short
//  handle (`a1`, `a2`, …) the router passes between tools. Files live in
//  `conversations/<uuid>/assets/aN.<ext>` with an `index.json` beside them, so a relaunch finds
//  them again and `ConversationStore.delete` (which removes the whole `conversations/<uuid>/`
//  folder) takes them with the chat. The store holds paths and facts only — never request or
//  response content beyond the files themselves.
//

import Foundation

/// How an asset came to be: the row that made it, the model's display name, the handles it read
/// and how long it took. This is the trace MT5 turns back into a workflow.
nonisolated struct ChatAssetProvenance: Codable, Sendable, Equatable {
    var rowText: String?
    var model: String?
    var inputs: [String]
    var milliseconds: Int?

    init(rowText: String? = nil, model: String? = nil, inputs: [String] = [], milliseconds: Int? = nil) {
        self.rowText = rowText
        self.model = model
        self.inputs = inputs
        self.milliseconds = milliseconds
    }
}

nonisolated struct ChatAssetRecord: Codable, Sendable, Equatable, Identifiable {
    /// `a1`, `a2`, … — unique within the conversation, never reused.
    let handle: String
    /// A `Kind` raw value (`text`, `audio`, `image`, …).
    let kind: String
    /// A human name: the attachment's file name, or what the tool made.
    let name: String
    /// The file's name inside the assets folder.
    let fileName: String
    let bytes: Int
    /// Text assets only.
    let words: Int?
    /// Audio and video assets only, seconds.
    let duration: Double?
    let provenance: ChatAssetProvenance
    let createdAt: Date

    var id: String { handle }
}

nonisolated final class ChatAssetStore: @unchecked Sendable {
    let conversationID: UUID
    /// `conversations/<uuid>/assets/`.
    let directory: URL
    private let lock = NSLock()
    private var records: [ChatAssetRecord] = []
    /// The next handle number. Persisted, so a handle is never reused even when the newest asset's file vanishes.
    private var nextNumber = 1

    /// - Parameter conversationsDirectory: the `conversations` folder (default: the live one).
    init(conversationID: UUID, in conversationsDirectory: URL = ConversationStore.defaultDirectory()) {
        self.conversationID = conversationID
        self.directory = ConversationStore.folderURL(for: conversationID, in: conversationsDirectory)
            .appendingPathComponent("assets", isDirectory: true)
        let loaded = Self.loadIndex(from: directory)
        self.records = loaded.records
        self.nextNumber = max(loaded.next, (loaded.records.compactMap { Int($0.handle.dropFirst()) }.max() ?? 0) + 1)
    }

    private var indexURL: URL { directory.appendingPathComponent("index.json") }

    /// `conversations/<uuid>/assets/.blobs/` — where an executor writes before the store renames
    /// the output into `aN.<ext>` (same volume, so the rename is free).
    var blobDirectory: URL { directory.appendingPathComponent(".blobs", isDirectory: true) }

    /// The assets whose file is on disk: what a tool can use.
    var all: [ChatAssetRecord] { allRecords.filter { !isMissing($0) } }

    /// MT4-7-FIX-4 (E): every asset the chat ever recorded, including one whose file was deleted or trashed. The chip, the inline
    /// player's "no longer on disk" line and the inspector's Assets list read this, so the provenance (model, time) survives.
    var allRecords: [ChatAssetRecord] { lock.withLock { records } }

    /// Whether the record's file is gone from disk.
    func isMissing(_ record: ChatAssetRecord) -> Bool {
        !FileManager.default.fileExists(atPath: url(for: record).path)
    }

    func record(_ handle: String) -> ChatAssetRecord? {
        let wanted = Self.normalized(handle)
        return lock.withLock { records.first { $0.handle == wanted } }
    }

    /// MT4-7-FIX-4 (B): a handle (`a1`, forgiving case and spaces), else an asset's exact (case-sensitive) name or stored file
    /// name when that matches exactly one asset in the chat. Zero or several matches: `nil`.
    func resolve(_ reference: String) -> ChatAssetRecord? {
        if let byHandle = record(reference) { return byHandle }
        let wanted = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !wanted.isEmpty else { return nil }
        let matches = all.filter { $0.name == wanted || $0.fileName == wanted }
        return matches.count == 1 ? matches[0] : nil
    }

    func url(for record: ChatAssetRecord) -> URL {
        directory.appendingPathComponent(record.fileName)
    }

    /// `"A3"`, `" a3 "`, `"a3"` → `"a3"`. The router is a language model; forgive case and spaces.
    static func normalized(_ handle: String) -> String {
        handle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Whether `text` looks like a handle (`a` + digits). Says nothing about whether it exists.
    static func looksLikeHandle(_ text: String) -> Bool {
        let t = normalized(text)
        guard t.count >= 2, t.hasPrefix("a") else { return false }
        return t.dropFirst().allSatisfy(\.isNumber)
    }

    // MARK: - Adding

    /// Register a file the caller already wrote (a tool output) by **moving** it in as `aN.<ext>`.
    @discardableResult
    func adopt(file: URL, kind: String, name: String, duration: Double? = nil,
               provenance: ChatAssetProvenance = .init()) throws -> ChatAssetRecord {
        try add(kind: kind, name: name, ext: file.pathExtension, duration: duration, provenance: provenance,
                words: nil) { destination in
            try FileManager.default.moveItem(at: file, to: destination)
        }
    }

    /// Register a file by **copying** it in (an attachment the user owns).
    @discardableResult
    func importFile(_ file: URL, kind: String, name: String? = nil, duration: Double? = nil,
                    provenance: ChatAssetProvenance = .init()) throws -> ChatAssetRecord {
        try add(kind: kind, name: name ?? file.lastPathComponent, ext: file.pathExtension, duration: duration,
                provenance: provenance, words: nil) { destination in
            try FileManager.default.copyItem(at: file, to: destination)
        }
    }

    /// Register text as `aN.txt`.
    @discardableResult
    func addText(_ text: String, name: String,
                 provenance: ChatAssetProvenance = .init()) throws -> ChatAssetRecord {
        try add(kind: "text", name: name, ext: "txt", duration: nil, provenance: provenance,
                words: Self.wordCount(text)) { destination in
            try Data(text.utf8).write(to: destination, options: .atomic)
        }
    }

    static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
    }

    private func add(kind: String, name: String, ext: String, duration: Double?,
                     provenance: ChatAssetProvenance, words: Int?,
                     write: (URL) throws -> Void) throws -> ChatAssetRecord {
        try lock.withLock {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let handle = "a\(nextNumber)"
            let cleanExt = ext.filter { $0.isLetter || $0.isNumber }.lowercased()
            let fileName = cleanExt.isEmpty ? handle : "\(handle).\(cleanExt)"
            let destination = directory.appendingPathComponent(fileName)
            try write(destination)
            let size = (try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            let record = ChatAssetRecord(handle: handle, kind: kind, name: name, fileName: fileName,
                                         bytes: size, words: words, duration: duration,
                                         provenance: provenance,
                                         createdAt: Date(timeIntervalSince1970: Date().timeIntervalSince1970.rounded(.down)))  // whole seconds: iso8601 round-trips exactly
            records.append(record)
            nextNumber += 1
            try saveIndexLocked()
            return record
        }
    }

    // MARK: - Reading

    /// The text of a text asset, or `nil` when the handle is unknown / not text / unreadable.
    func text(for handle: String) -> String? {
        guard let record = record(handle), record.kind == "text" else { return nil }
        return try? String(contentsOf: url(for: record), encoding: .utf8)
    }

    // MARK: - Persistence

    private func saveIndexLocked() throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(Index(next: nextNumber, assets: records)).write(to: indexURL, options: .atomic)
    }

    private struct Index: Codable { var next: Int; var assets: [ChatAssetRecord] }

    /// Reads `{next, assets}`; also the first shape of this file, a bare array of records.
    private static func loadIndex(from directory: URL) -> (records: [ChatAssetRecord], next: Int) {
        let url = directory.appendingPathComponent("index.json")
        guard let data = try? Data(contentsOf: url) else { return ([], 1) }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var decoded: [ChatAssetRecord] = []
        var next = 1
        if let index = try? decoder.decode(Index.self, from: data) {
            decoded = index.assets; next = index.next
        } else if let array = try? decoder.decode([ChatAssetRecord].self, from: data) {
            decoded = array
        }
        // MT4-7-FIX-4 (E): a record whose file was removed (by the user or a crash) is kept: it is just not usable (`all` leaves it
        // out, `isMissing` says so), so its chip keeps the model and time and the inspector lists it as missing.
        return (decoded, max(next, (decoded.compactMap { Int($0.handle.dropFirst()) }.max() ?? 0) + 1))
    }
}
