//
//  DemoTools.swift
//  MLXUI — agentic chat tools, slice AG1.
//
//  A DEBUG-only scaffold tool so the AG1 chat UI (tool-call cards + approval sheet + tool
//  toggles) has a real subject to render and so the end-to-end loop — including the
//  `.xmlFunction` tool-call parsing wired at load — can be smoke-tested with a live model.
//
//  It ships in DEBUG builds only; release builds advertise no tools until the real ones land
//  (AG2 web, AG3 compute, AG4 in-app MLX, AG5 files). Do not grow this file — add real tools
//  in their own slices.
//

import Foundation
import MLXLMCommon

#if DEBUG
/// Echoes its `text` argument back. Marked `requiresApproval` purely to exercise the AG1
/// approval sheet in a live smoke — a real echo would need no gate.
nonisolated struct EchoDemoTool: AgentTool {
    let name = "echo"
    let toolDescription = "Echo the provided text back to the user. A demo tool for testing."
    let parameters: [ToolParameter] = [
        .required("text", type: .string, description: "The text to echo back.")
    ]
    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        "echo: \(arguments.string("text") ?? "")"
    }
}
#endif
