//
//  ShellTool.swift
//  MLXUI-Direct target ONLY.
//
//  The `AgentTool` that exposes the shell to a local model. `requiresApproval`
//  is true, so `AgentSession.dispatch` gates every call on the UI before
//  `execute` runs (see Core/Agent/ToolRegistry.swift).
//
//  Design note: a single "run any command line" tool is the honest shape here —
//  a local 4-bit model is not going to reliably drive a hand-rolled catalogue of
//  narrow tools, and pretending otherwise just moves the risk somewhere less
//  visible. The safety story is therefore *approval + transparency*, not
//  cleverness in the schema: show the user the exact argv, let them decline.
//

import Foundation
import MLXLMCommon
import MLXUIKit

nonisolated struct ShellTool: AgentTool {
    let shell: any ShellCapability
    /// Commands may only run with a working directory inside one of these.
    let allowedRoots: [URL]
    let defaultTimeout: TimeInterval

    init(
        shell: any ShellCapability = ProcessShell(),
        allowedRoots: [URL] = PathPolicy.defaultRoots,
        defaultTimeout: TimeInterval = 60
    ) {
        self.shell = shell
        self.allowedRoots = allowedRoots
        self.defaultTimeout = defaultTimeout
    }

    let name = "run_shell"

    var toolDescription: String {
        """
        Run a shell command on the user's Mac and return its exit code, stdout and \
        stderr. The user must approve each command before it runs. Prefer a single \
        non-interactive command; commands that wait for input will time out. Output \
        is truncated at 64 KB.
        """
    }

    var parameters: [ToolParameter] {
        [
            .required(
                "command",
                type: .string,
                description: "The command line to run, e.g. 'git status --short'."
            ),
            .optional(
                "working_directory",
                type: .string,
                description: "Absolute path to run in. Must be inside the user's home directory."
            ),
            .optional(
                "timeout_seconds",
                type: .int,
                description: "Kill the command after this many seconds (default 60, max 600)."
            ),
        ]
    }

    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let command = arguments.string("command"), !command.isEmpty else {
            return "Error: 'command' is required."
        }

        var workingDirectory: URL?
        if let path = arguments.string("working_directory") {
            let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            guard PathPolicy.isAllowed(url, roots: allowedRoots) else {
                return "Error: \(url.path) is outside the directories this app may use."
            }
            workingDirectory = url
        }

        let timeout = min(max(TimeInterval(arguments.int("timeout_seconds") ?? Int(defaultTimeout)), 1), 600)

        do {
            let result = try await shell.run(
                .commandLine(command, workingDirectory: workingDirectory, timeout: timeout)
            )
            return result.transcript
        } catch let error as ShellError {
            // Returned, not thrown: the dispatcher would turn a throw into the
            // same string, but returning keeps the model's transcript readable
            // and lets it retry with a fix.
            return "Error: \(error.localizedDescription)"
        }
    }
}

/// Tools that ship only in the direct-download edition. The App Store target
/// never compiles this file, so its `@main` has nothing to pass and no way to
/// pass it — see Design/dual-distribution.md § "Phase 3".
nonisolated enum DirectTools {
    static func all(shell: any ShellCapability = ProcessShell()) -> [any AgentTool] {
        [
            ShellTool(shell: shell),
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
        ]
    }
}
