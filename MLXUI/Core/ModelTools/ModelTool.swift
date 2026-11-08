//
//  ModelTool.swift
//  MLXUI — Model Tools, MT1-3: one `AgentTool` per `ModelToolSpec` row.
//
//  The schema is generated from the spec (name, one-line description, parameters), plus the input
//  arguments (`asset` / `path` / `url`, or `text`) that say where the input asset comes from. No
//  model id appears in anything the router reads. `list_capabilities` and `read_asset` run no
//  model. The per-conversation context arrives by `@TaskLocal` (`ModelToolHost`, `ModelToolRuntime`),
//  so `AgentTool` keeps its shape.
//

import Foundation
import MLXLMCommon

nonisolated struct ModelTool: AgentTool {
    let spec: ModelToolSpec

    var name: String { spec.name }
    var toolDescription: String { spec.description }

    var parameters: [ToolParameter] {
        var out: [ToolParameter] = []
        for input in ModelToolRunner.inputArgumentNames(for: spec) {
            out.append(.optional(input, type: .string, description: Self.inputDescription(input)))
        }
        for param in spec.params {
            let type: ToolParameterType = { if case .integer = param.type { return .int } else { return .string } }()
            out.append(param.required
                ? .required(param.name, type: type, description: param.description)
                : .optional(param.name, type: type, description: param.description))
        }
        return out
    }

    static func inputDescription(_ name: String) -> String {
        switch name {
        case "asset": return "Handle of a file or text in this chat, e.g. a1. Give one input: asset, path, url or text."
        case "path": return "Path of a file in a folder attached to this chat."
        case "url": return "The absolute http(s) URL of the file."
        default: return "The text itself."
        }
    }

    /// Every tool, in table order.
    static var all: [ModelTool] { ModelToolSpec.all.map { ModelTool(spec: $0) } }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        switch spec.name {
        case "list_capabilities": return Self.listCapabilities(arguments: arguments)
        case "read_asset": return Self.readAsset(arguments: arguments)
        default:
            guard let runtime = ModelToolRuntime.current else {
                return ToolFailure("\(spec.name) isn't available here.").text
            }
            return await ModelToolRunner(spec: spec, runtime: runtime).run(arguments: arguments)
        }
    }

    // MARK: - list_capabilities

    /// One line per model tool with its status, then the chat's files. Plain facts, no model names (rule 3).
    static func listCapabilities(arguments: [String: JSONValue]) -> String {
        if let extra = arguments.keys.sorted().first {
            return ToolFailure("unknown argument \"\(extra)\" for list_capabilities. Valid arguments: none.").text
        }
        guard let runtime = ModelToolRuntime.current else { return ToolFailure("list_capabilities isn't available here.").text }
        var lines: [String] = []
        for spec in ModelToolSpec.modelTools {
            let status = CapabilityResolver.status(
                for: spec, catalog: runtime.catalog, installedModelIDs: runtime.installedModelIDs,
                claimableModelIDs: runtime.claimableModelIDs, totalRAMGB: runtime.totalRAMGB)
            let note: String
            switch status {
            case .ready: note = "ready"
            case .needsDownload(_, let gb): note = "needs a one-time \(String(format: gb < 10 ? "%.1f" : "%.0f", max(gb, 0.1))) GB download"
            case .cantRun(let reason): note = "can't run on this Mac. \(reason)"
            case .none: note = "not available in this build"
            }
            lines.append("- \(spec.name): \(spec.description) [\(note)]")
        }
        if let host = ModelToolHost.current, !host.assets.all.isEmpty {
            lines.append("Files in this chat:")
            for record in host.assets.all { lines.append("- \(record.handle) \(record.name) (\(record.kind))") }
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - read_asset

    static let defaultPageCharacters = 4000

    /// A page of a text asset: `{"asset","start","length","total","text"[,"next"]}`. `start`/`length` count characters.
    static func readAsset(arguments: [String: JSONValue]) -> String {
        guard let spec = ModelToolSpec.spec(named: "read_asset") else { return ToolFailure("read_asset isn't available.").text }
        let typed: [String: ModelToolRunner.Value]
        switch ModelToolRunner.validate(arguments, for: spec) {
        case .failure(let failure): return failure.text
        case .success(let values): typed = values
        }
        guard let host = ModelToolHost.current else { return ToolFailure("read_asset is only available in a chat.").text }
        guard case .string(let handle)? = typed["asset"] else { return ToolFailure("read_asset needs the \"asset\" argument.").text }
        guard let record = host.assets.record(handle) else {
            let known = host.assets.all.map { "\($0.handle) (\($0.kind))" }.joined(separator: ", ")
            return ToolFailure("no asset \"\(handle)\". " + (known.isEmpty ? "This chat has no assets yet." : "Known assets: \(known).")).text
        }
        guard record.kind == "text", let text = host.assets.text(for: record.handle) else {
            return ToolFailure("asset \(record.handle) is \(record.kind), not text. Pass it to a tool that takes \(record.kind).").text
        }
        var start = 0
        if case .int(let n)? = typed["start"] { start = n }
        var length = defaultPageCharacters
        if case .int(let n)? = typed["length"] { length = n }
        let characters = Array(text)
        guard start <= characters.count else {
            return ToolFailure("start \(start) is past the end of \(record.handle), which has \(characters.count) characters.").text
        }
        let end = min(characters.count, start + length)
        var parts = [
            "\"asset\":\(ModelToolRunner.json(record.handle))", "\"start\":\(start)", "\"length\":\(end - start)",
            "\"total\":\(characters.count)", "\"text\":\(ModelToolRunner.json(String(characters[start..<end])))",
        ]
        if end < characters.count { parts.append("\"next\":\(end)") }
        return "{" + parts.joined(separator: ",") + "}"
    }
}
