//
//  ModelToolChip.swift
//  MLXUI — Model Tools, MT4-4 (plan §3.9: "‹Model› · ‹Task› · 2.1 s" chips, inline players, the inspector's Files).
//
//  Pure derivations for the Chat transcript and inspector. Nothing here is stored: a chip is read from a tool card's
//  result (the one-line JSON `ModelToolRunner` returns, which names the output handle) joined to the conversation's
//  `ChatAssetRecord`s (provenance: model, row text, input handles, timing). Both are already persisted with the
//  conversation, so chips and the Files list survive a relaunch with no new file format.
//

import Foundation

/// One model-tool call as the transcript shows it.
struct ModelToolChip: Equatable, Identifiable {

    enum State: Equatable {
        case running
        case done
        /// The tool's own sentence (an `Error: …` result, or a failed card).
        case failed(String)
        /// The user chose not to download the model (the card's Cancel).
        case declined
    }

    /// The tool card's id, so the inspector can scroll to it.
    let id: UUID
    let toolName: String
    /// The row's task, `OCR`, `Speak`, `Transcribe`…
    let task: String
    let state: State
    /// The model's display name as the row ran it; nil while running or when the record is gone.
    let model: String?
    let milliseconds: Int?
    let outputHandle: String?
    let outputKind: String?
    let inputHandles: [String]
    let rowText: String?

    /// `PaddleOCR-VL · OCR · 2.1 s`. While running or after a refusal the parts that are not known yet are left out.
    var title: String {
        switch state {
        case .running:
            return "\(task) · running…"
        case .failed:
            return [model, task, "failed"].compactMap { $0 }.joined(separator: " · ")
        case .declined:
            return "\(task) · not downloaded"
        case .done:
            return [model, task, milliseconds.map(Self.duration(milliseconds:))].compactMap { $0 }.joined(separator: " · ")
        }
    }

    /// The popover's lines, label then value, only for what is known.
    var details: [(label: String, value: String)] {
        var lines: [(String, String)] = []
        if !inputHandles.isEmpty { lines.append(("Input", inputHandles.joined(separator: ", "))) }
        if let outputHandle { lines.append(("Output", outputHandle + (outputKind.map { " (\($0))" } ?? ""))) }
        if let rowText { lines.append(("Row", Self.displayRow(rowText))) }
        if let milliseconds { lines.append(("Time", Self.duration(milliseconds: milliseconds))) }
        return lines
    }

    // MARK: - Building

    /// The chip for a tool card, or `nil` when the card is not a model tool or has nothing a chip would add (a plain-text result
    /// from before Model Tools keeps its old card). `records` are the conversation's assets.
    static func make(toolName: String, id: UUID, status: ToolActivity.Status, result: String?,
                     records: [ChatAssetRecord]) -> ModelToolChip? {
        guard let spec = ModelToolSpec.spec(named: toolName), let task = spec.task else { return nil }
        func chip(_ state: State, record: ChatAssetRecord? = nil, handle: String? = nil) -> ModelToolChip {
            ModelToolChip(id: id, toolName: toolName, task: task, state: state,
                          model: record?.provenance.model, milliseconds: record?.provenance.milliseconds,
                          outputHandle: handle, outputKind: record?.kind,
                          inputHandles: record?.provenance.inputs ?? [], rowText: record?.provenance.rowText)
        }
        switch status {
        case .running:
            return chip(.running)
        case .failed(let message):
            return chip(.failed(message))
        case .denied:
            return chip(.declined)
        case .unknownTool, .budgetExceeded:
            return nil
        case .finished:
            let text = result?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if text.hasPrefix("Error:") { return chip(.failed(text)) }
            if text.hasPrefix(ModelToolHost.declineSentencePrefix) { return chip(.declined) }
            guard let handle = outputHandle(inResult: text) else { return nil }
            let record = records.first { $0.handle == handle }
            return chip(.done, record: record, handle: handle)
        }
    }

    /// The `"asset"` handle in a result's JSON, or nil when the result is not that JSON.
    static func outputHandle(inResult result: String) -> String? {
        guard result.hasPrefix("{"),
              let object = try? JSONSerialization.jsonObject(with: Data(result.utf8)) as? [String: Any],
              let handle = object["asset"] as? String, ChatAssetStore.looksLikeHandle(handle) else { return nil }
        return ChatAssetStore.normalized(handle)
    }

    // MARK: - Words

    /// `0.8 s`, `2.1 s`, `1 m 12 s`. One decimal under a minute (it is a measured time, not a promise).
    static func duration(milliseconds: Int) -> String {
        let ms = max(0, milliseconds)
        if ms < 60_000 { return String(format: "%.1f s", Double(ms) / 1000) }
        let seconds = Int((Double(ms) / 1000).rounded())
        return "\(seconds / 60) m \(seconds % 60) s"
    }

    /// The row text without its leading `1.` index (`OCR   PaddleOCR-VL`), runs of spaces folded.
    static func displayRow(_ rowText: String) -> String {
        var text = rowText.trimmingCharacters(in: .whitespacesAndNewlines)
        if let dot = text.firstIndex(of: "."), text[text.startIndex..<dot].allSatisfy(\.isNumber), dot != text.startIndex {
            text = String(text[text.index(after: dot)...]).trimmingCharacters(in: .whitespaces)
        }
        return text.split(separator: " ", omittingEmptySubsequences: true).joined(separator: " ")
    }
}

/// What the inspector shows about a conversation's assets: the Files list and the "This chat" summary.
struct ChatAssetListing: Equatable {

    struct Row: Equatable, Identifiable {
        let handle: String
        let name: String
        /// A `Kind` raw value.
        let kind: String
        /// `1.2 MB`.
        let size: String
        /// `Kokoro 82M · Speak` for a tool output, `attached` for the user's file.
        let origin: String
        /// The tool card that made it (for "Show in transcript"); nil for an attachment.
        let activityID: UUID?
        /// The file is gone from disk (the row stays, marked).
        var missing = false
        var id: String { handle }
    }

    let rows: [Row]
    /// The distinct tool models that ran in this chat, in first-use order.
    let modelsUsed: [String]

    static func make(records: [ChatAssetRecord], items: [TranscriptItem], missingHandles: Set<String> = []) -> ChatAssetListing {
        var producer: [String: UUID] = [:]
        for item in items {
            guard case .tool(let activity) = item, case .finished = activity.status,
                  let handle = ModelToolChip.outputHandle(inResult: activity.result?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "")
            else { continue }
            producer[handle] = activity.id
        }
        var models: [String] = []
        let rows = records.map { record -> Row in
            let made = record.provenance.model != nil
            if let model = record.provenance.model, !models.contains(model) { models.append(model) }
            let task = record.provenance.rowText.flatMap(Self.task(inRowText:))
            let origin = made ? [record.provenance.model, task].compactMap { $0 }.joined(separator: " · ") : "attached"
            return Row(handle: record.handle, name: record.name, kind: record.kind, size: byteString(record.bytes),
                       origin: origin, activityID: producer[record.handle],
                       missing: missingHandles.contains(record.handle))
        }
        return ChatAssetListing(rows: rows, modelsUsed: models)
    }

    /// The task word of a row (`1. OCR   Model; …` → `OCR`): the leading words of the row after its index, matched against the tool tasks.
    static func task(inRowText rowText: String) -> String? {
        let display = ModelToolChip.displayRow(rowText)
        for spec in ModelToolSpec.modelTools {
            if let task = spec.task, display.hasPrefix(task + " ") { return task }
        }
        return nil
    }

    static func byteString(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
