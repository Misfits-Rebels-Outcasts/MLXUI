//
//  ModelToolWorkflowHandoff.swift
//  MLXUI
//
//  MT5-2: "Save as workflow" without the views. Decides whether the button is on, and builds the unsaved flow the editor
//  opens on: a new flow folder holding the chat's attachments under the names the `Read` rows write, and an edit target
//  with no saved text (so it opens dirty) carrying the "left out" lines. Nothing is written to a `.cat` here; Save does that.
//

import Foundation

nonisolated enum ModelToolWorkflowHandoff {

    struct Availability: Equatable, Sendable {
        let enabled: Bool
        /// Why it is off, one short sentence; nil when on.
        let reason: String?
    }

    enum Failure: Error, Equatable, CustomStringConvertible {
        case notSavable
        case missingSource(String)
        case copyFailed(String)

        var description: String {
            switch self {
            case .notSavable: return "This chat can't be written as a workflow."
            case .missingSource(let name): return "Couldn't copy \(name) into the new workflow: the file is no longer in this chat."
            case .copyFailed(let name): return "Couldn't copy \(name) into the new workflow's folder."
            }
        }
    }

    static func availability(flow: ModelToolFlow, isRunning: Bool, workflowsAvailable: Bool) -> Availability {
        if !workflowsAvailable { return Availability(enabled: false, reason: "AI Workflows is turned off.") }
        if isRunning { return Availability(enabled: false, reason: "Wait for the reply to finish.") }
        if flow.isEmpty { return Availability(enabled: false, reason: "Nothing to save yet. Use a model tool or attach a text file first.") }
        if !flow.roundTrips { return Availability(enabled: false, reason: "This chat can't be written as a workflow.") }
        return Availability(enabled: true, reason: nil)
    }

    /// The flow's name: the chat's title, or a plain default while the chat is untitled.
    static func workflowName(chatTitle: String?) -> String {
        let trimmed = (chatTitle ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == "New Chat" { return "Chat workflow" }
        return String(trimmed.prefix(60))
    }

    /// `name`, or `name 2`, `name 3`… when a saved flow already has it (compared as the file name the editor would write).
    static func uniqueName(_ name: String, among taken: Set<String>) -> String {
        func isTaken(_ candidate: String) -> Bool { taken.contains(FlowEditorModel.sanitizedFileName(candidate).lowercased()) }
        guard isTaken(name) else { return name }
        var n = 2
        while isTaken("\(name) \(n)") { n += 1 }
        return "\(name) \(n)"
    }

    /// Make the flow folder, copy the attachments in, and return the target for the editor. On any failure the folder is
    /// removed again and the error is plain: no half-built flow opens.
    static func prepare(flow: ModelToolFlow, chatTitle: String?, assetsDirectory: URL,
                        workspace: FlowWorkspace) throws -> FlowEditTarget {
        guard !flow.isEmpty, flow.roundTrips else { throw Failure.notSavable }
        let fm = FileManager.default
        // Check every source before creating anything.
        for copy in flow.copies where !fm.fileExists(atPath: assetsDirectory.appendingPathComponent(copy.source).path) {
            throw Failure.missingSource(copy.destination)
        }
        // Read before the folder exists, so it can only list flows that were already saved.
        let existingFlowNames = Set(UserFlowStore.scan(workspace: workspace, bundledFlowIDs: []).map { $0.title.lowercased() })
        let flowID = UUID().uuidString
        let dir = workspace.directory(for: flowID)
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            for copy in flow.copies {
                try fm.copyItem(at: assetsDirectory.appendingPathComponent(copy.source),
                                to: dir.appendingPathComponent(copy.destination))
            }
        } catch {
            try? fm.removeItem(at: dir)
            let failed = flow.copies.first?.destination ?? "a file"
            throw Failure.copyFailed(failed)
        }
        let name = uniqueName(workflowName(chatTitle: chatTitle), among: existingFlowNames)
        var target = FlowEditTarget(flowID: flowID, name: name, document: flow.document, savedText: nil)
        target.leftOut = flow.banner.map(\.line)
        target.removeFolderIfUnsaved = true
        return target
    }

    /// The editor closed: remove the folder `prepare` made unless a flow was saved into it, or the editor is still on the
    /// navigation stack (it also disappears when something is pushed over it). Never deletes a folder holding a flow file.
    @discardableResult
    static func discardIfUnsaved(flowID: String, workspace: FlowWorkspace, saved: Bool, stillOpen: Bool) -> Bool {
        guard !saved, !stillOpen else { return false }
        let fm = FileManager.default
        let dir = workspace.directory(for: flowID)
        let files = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        if files.contains(where: { $0.pathExtension == "cat" || $0.pathExtension == "catpipeline" }) { return false }
        guard fm.fileExists(atPath: dir.path) else { return false }
        return (try? fm.removeItem(at: dir)) != nil
    }
}
