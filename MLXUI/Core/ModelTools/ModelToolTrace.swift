//
//  ModelToolTrace.swift
//  MLXUI
//
//  MT5-1: what a chat did with model tools, as plain data. A `ModelToolTrace` is the chat's transcript order (user turns,
//  assistant text, tool calls) joined with the chat's asset records (attachments and tool outputs, with their provenance).
//  It carries no UI types and no file handles, so `ModelToolFlowBuilder` can turn it into a `FlowDocument` as a pure
//  function, and the fixtures under `Fixtures/ModelTools/` can state it as JSON.
//

import Foundation

nonisolated struct ModelToolTrace: Codable, Sendable, Equatable {

    /// One asset of the chat: an attachment (`rowText == nil`) or a tool output (`rowText` is the one-row text the tool ran).
    struct Asset: Codable, Sendable, Equatable {
        var handle: String
        /// A `Kind` raw value: `text`, `audio`, `image`, …
        var kind: String
        /// The attachment's file name, or what the tool made.
        var name: String
        /// The file's name inside the chat's assets folder (the copy source).
        var fileName: String
        /// Text assets only: the text.
        var text: String?
        var rowText: String?
        /// The model's display name as the row names it.
        var model: String?
        /// The file is gone from the chat's assets folder.
        var missing: Bool?

        init(handle: String, kind: String, name: String, fileName: String, text: String? = nil,
             rowText: String? = nil, model: String? = nil, missing: Bool? = nil) {
            self.handle = handle
            self.kind = kind
            self.name = name
            self.fileName = fileName
            self.text = text
            self.rowText = rowText
            self.model = model
            self.missing = missing
        }
    }

    struct Event: Codable, Sendable, Equatable {
        enum Kind: String, Codable, Sendable { case user, assistant, tool }
        var kind: Kind
        /// A user or assistant message's text; for a tool call, its `text` argument.
        var text: String?
        /// An assistant message's model: `apple-intelligence`, or a catalog id.
        var modelID: String?
        /// A user message's attachment handles.
        var attachments: [String]?
        /// A tool call's name, its `asset` argument, the handle it produced, and whether it ran.
        var tool: String?
        var assetArgument: String?
        var output: String?
        /// `false` for a call that failed or was declined. `nil` reads as `true`.
        var ok: Bool?

        init(kind: Kind, text: String? = nil, modelID: String? = nil, attachments: [String]? = nil, tool: String? = nil,
             assetArgument: String? = nil, output: String? = nil, ok: Bool? = nil) {
            self.kind = kind
            self.text = text
            self.modelID = modelID
            self.attachments = attachments
            self.tool = tool
            self.assetArgument = assetArgument
            self.output = output
            self.ok = ok
        }
    }

    var events: [Event]
    var assets: [Asset]

    init(events: [Event], assets: [Asset]) {
        self.events = events
        self.assets = assets
    }
}

/// What the builder needs from the live catalog, as plain values.
nonisolated struct ModelToolFlowCatalog: Sendable, Equatable {
    /// A chat model's id (`ChatMessage.modelID`) → the display name a row would name it. `apple-intelligence` is not here.
    var routerDisplays: [String: String]
    /// A display name → the HuggingFace id its `models:` pin points at (for a model the bridge does not name).
    var hfModelIDs: [String: String]

    init(routerDisplays: [String: String] = [:], hfModelIDs: [String: String] = [:]) {
        self.routerDisplays = routerDisplays
        self.hfModelIDs = hfModelIDs
    }
}

/// One line of the "what this flow does not carry" banner.
nonisolated struct ModelToolFlowBannerEntry: Sendable, Equatable {
    enum Kind: String, Sendable {
        case leftOutTools, failedCalls, droppedReplies, history
        case unwritableModel, unresolvedModel, noRouter
        case missingAttachment, unconsumedAttachment
        case braceToken, dependsOnLeftOut, contextTrimmed, unsavedOutput, emptyText
    }
    var kind: Kind
    var count: Int
    /// Names the line is about (tools with their counts, model names, file names).
    var items: [String]

    var line: String {
        let list = items.joined(separator: ", ")
        switch kind {
        case .leftOutTools: return "Left out, a saved flow runs model steps only: \(list)."
        case .failedCalls: return "Left out: \(count) tool call\(count == 1 ? "" : "s") that did not run."
        case .droppedReplies: return "Left out: \(count) earlier repl\(count == 1 ? "y" : "ies") of the conversation."
        case .history: return "Left out: the conversation history (\(count) earlier message\(count == 1 ? "" : "s"))."
        case .unwritableModel: return "Left out: a step whose model name a .cat file cannot hold (\(list))."
        case .unresolvedModel: return "Left out: a step whose model is not in the catalog (\(list))."
        case .noRouter: return "Left out: \(count) answer\(count == 1 ? "" : "s") with no known chat model."
        case .missingAttachment: return "Left out: an attachment whose file is gone (\(list))."
        case .unconsumedAttachment: return "Left out: an attachment no step used (\(list))."
        case .braceToken: return "Left out: \(count) text\(count == 1 ? "" : "s") containing a placeholder such as {input} or {1}."
        case .dependsOnLeftOut: return "Left out: \(count) step\(count == 1 ? "" : "s") that needed something left out above."
        case .contextTrimmed: return "An answer read more than four earlier results; the flow keeps the latest four."
        case .unsavedOutput: return "Not saved: an output of a kind a flow cannot save (\(list))."
        case .emptyText: return "Left out: \(count) step\(count == 1 ? "" : "s") with no text."
        }
    }
}

nonisolated struct ModelToolFlow: Sendable {
    /// A file to copy into the new flow's folder.
    struct Copy: Sendable, Equatable {
        let handle: String
        /// The file's name in the chat's assets folder.
        let source: String
        /// Its name in the flow folder, as the `Read` row writes it.
        let destination: String
    }

    var document: FlowDocument
    /// `CatSerializer.serialize(document)`.
    var text: String
    var copies: [Copy]
    var banner: [ModelToolFlowBannerEntry]
    /// The text re-parsed to the same rows (task and model per row) and the same `models:` block.
    var roundTrips: Bool
    var isEmpty: Bool { document.rows.isEmpty }
}
