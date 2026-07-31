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
        non-interactive command; commands that wait for input will time out. Only \
        the first 80 lines of output are returned, so filter in the command itself \
        (head, tail, grep, wc -l) when a command could print a lot.
        """
    }

    var parameters: [ToolParameter] {
        [
            .required(
                "command",
                type: .string,
                description: "The command line to run, e.g. 'git status --short'."
            ),
            // Wording matters more than it looks: an earlier description that read
            // "Must be inside the user's home directory" made a 4-bit model treat
            // the parameter as mandatory, then stall trying to guess the user's
            // home path. Say "omit it" explicitly and give the default.
            .optional(
                "working_directory",
                type: .string,
                description: """
                    Optional — omit this unless the command must run somewhere \
                    specific. Defaults to the user's home directory. If given, an \
                    absolute path inside the home directory.
                    """
            ),
            .optional(
                "timeout_seconds",
                type: .int,
                description: """
                    Optional — omit this to use the 60 second default. Maximum 600. \
                    The command is killed if it runs longer.
                    """
            ),
        ]
    }

    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let command = arguments.string("command"), !command.isEmpty else {
            return "Error: 'command' is required."
        }

        // Default to the home directory rather than leaving it nil: an app launched
        // from Finder has cwd `/`, so a relative command like `ls` would silently
        // list the filesystem root and look like a bug in the tool.
        var workingDirectory: URL? = FileManager.default.homeDirectoryForCurrentUser
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

    /// Present in the tool list but unticked until the user opts in. Reading is
    /// enabled by default; running a command or overwriting a file is not — a
    /// model shouldn't be able to propose either on first launch. Per-call
    /// approval still applies once they're on.
    static let disabledByDefault: Set<String> = ["run_shell", "write_file"]
}
