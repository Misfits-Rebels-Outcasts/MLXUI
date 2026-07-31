//
//  ShellCapability.swift
//  MLXUI — shared code (ships in BOTH editions).
//
//  The seam between the two distributions. This file declares *what* a shell
//  capability is; it deliberately contains no implementation that can spawn a
//  process. `ProcessShell` (the real one) lives in the MLXUI-Direct app target
//  only — see Apps/Direct/ProcessShell.swift and Design/dual-distribution.md.
//
//  The split is enforced two ways, and the first is the one that matters:
//
//   1. Target membership. `Apps/Direct/` is a file-system synchronized group on
//      the MLXUI-Direct target only, so the App Store binary cannot contain the
//      subprocess code — verifiable with `nm -u | grep posix_spawn`, which is a
//      much better answer in review than "it's behind a flag".
//   2. `#if DIRECT_BUILD` around the three-line registration in
//      `ModelRunner.defaultTools()`, where the excluded branch names a type that
//      doesn't exist in the App Store target.
//
//  Hence a protocol here rather than a concrete no-op: shared code can hold a
//  `ShellCapability` and never know which edition it's in.
//

import Foundation

// MARK: - Invocation

/// A single child-process invocation. Options live on the struct (rather than as
/// `run` parameters) so new knobs don't break every conformer.
public struct ShellInvocation: Sendable {
    /// Absolute path, or a bare name resolved against `environment["PATH"]`.
    public var executable: String
    /// Passed to the child as-is — never re-parsed, never shell-expanded.
    public var arguments: [String]
    public var workingDirectory: URL?
    /// `nil` means "use the implementation's minimal default", not "inherit ours".
    public var environment: [String: String]?
    /// Written to the child's stdin, which is then closed. `nil` = /dev/null.
    public var standardInput: String?
    /// Wall-clock limit. On expiry the child gets SIGTERM, then SIGKILL.
    public var timeout: TimeInterval
    /// Cap per stream. Output past the cap is dropped and flagged, but the pipe
    /// keeps draining so the child never blocks on a full buffer.
    ///
    /// Deliberately small. This output goes straight into a local model's context
    /// window, so the limit that matters is tokens, not disk or memory: 64 KB of
    /// command output is ~30k tokens, which overruns an 8B model's context and, in
    /// testing, made it echo the output back a token at a time. 8 KB is generous
    /// for the commands anyone actually wants (`git status`, `ls`, `wc -l`), and
    /// anything larger should be filtered in the command itself.
    public var maxOutputBytes: Int

    public init(
        executable: String,
        arguments: [String] = [],
        workingDirectory: URL? = nil,
        environment: [String: String]? = nil,
        standardInput: String? = nil,
        timeout: TimeInterval = 60,
        maxOutputBytes: Int = 8 * 1024
    ) {
        self.executable = executable
        self.arguments = arguments
        self.workingDirectory = workingDirectory
        self.environment = environment
        self.standardInput = standardInput
        self.timeout = timeout
        self.maxOutputBytes = maxOutputBytes
    }

    /// Run a command *line* through a login shell — i.e. with globbing, pipes,
    /// redirection and substitution all live.
    ///
    /// This is the sharp edge of the API. Anything reaching it from a model's
    /// tool call must be user-approved (`AgentTool.requiresApproval`); prefer
    /// `init(executable:arguments:)` whenever the command is known ahead of time
    /// and only its arguments vary.
    public static func commandLine(
        _ line: String,
        shell: String = "/bin/zsh",
        workingDirectory: URL? = nil,
        timeout: TimeInterval = 60
    ) -> ShellInvocation {
        ShellInvocation(
            executable: shell,
            arguments: ["-lc", line],
            workingDirectory: workingDirectory,
            timeout: timeout
        )
    }
}

// MARK: - Result

/// Outcome of a child process that ran to completion (any exit code).
public struct ShellResult: Sendable {
    public let standardOutput: String
    public let standardError: String
    public let exitCode: Int32
    public let outputWasTruncated: Bool
    public let duration: TimeInterval

    public init(
        standardOutput: String,
        standardError: String,
        exitCode: Int32,
        outputWasTruncated: Bool = false,
        duration: TimeInterval = 0
    ) {
        self.standardOutput = standardOutput
        self.standardError = standardError
        self.exitCode = exitCode
        self.outputWasTruncated = outputWasTruncated
        self.duration = duration
    }

    public var succeeded: Bool { exitCode == 0 }

    /// Model-facing rendering: exit status first so a model can't miss a failure
    /// that printed nothing to stderr.
    ///
    /// Clipped a second time, by lines, on top of `maxOutputBytes`. The byte cap
    /// bounds memory; this bounds *context*, which is the scarcer resource — 8 KB
    /// of `yes` output is still 4,000 useless lines. Keeps the head and the tail,
    /// since the interesting part of command output is almost always at one end.
    public var transcript: String {
        var lines: [String] = ["exit code: \(exitCode)"]
        if !standardOutput.isEmpty { lines.append("stdout:\n\(Self.clip(standardOutput))") }
        if !standardError.isEmpty { lines.append("stderr:\n\(Self.clip(standardError))") }
        if standardOutput.isEmpty && standardError.isEmpty { lines.append("(no output)") }
        if outputWasTruncated { lines.append("(output truncated at the byte limit)") }
        return lines.joined(separator: "\n")
    }

    static let maxTranscriptLines = 80
    static let headLines = 60

    /// Keeps the first `headLines` and the last `maxTranscriptLines - headLines`
    /// lines, with a count of what was dropped in between.
    static func clip(_ text: String) -> String {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        guard lines.count > maxTranscriptLines else { return text }
        let tailCount = maxTranscriptLines - headLines
        let head = lines.prefix(headLines).joined(separator: "\n")
        let tail = lines.suffix(tailCount).joined(separator: "\n")
        let dropped = lines.count - maxTranscriptLines
        return "\(head)\n… \(dropped) more lines …\n\(tail)"
    }
}

// MARK: - Errors

public enum ShellError: LocalizedError, Sendable, Equatable {
    /// This edition has no shell. Thrown by `UnavailableShell`.
    case unavailable
    case executableNotFound(String)
    case notExecutable(String)
    /// Refused before launch by a policy check (path allowlist, denied command…).
    case blockedByPolicy(reason: String)
    case launchFailed(String)
    case timedOut(after: TimeInterval)

    public var errorDescription: String? {
        switch self {
        case .unavailable:
            return "Shell commands are not available in this version of AI Browser."
        case .executableNotFound(let name):
            return "Executable not found: \(name)"
        case .notExecutable(let path):
            return "Not an executable file: \(path)"
        case .blockedByPolicy(let reason):
            return "Blocked: \(reason)"
        case .launchFailed(let message):
            return "Could not start the process: \(message)"
        case .timedOut(let seconds):
            return "The command did not finish within \(Int(seconds))s and was terminated."
        }
    }
}

// MARK: - Capability

/// Runs child processes on the user's machine.
///
/// `Sendable` + `nonisolated` conformers, matching `AgentTool` / `PipelineStage`:
/// the app target defaults to main-actor isolation and process work must not sit
/// on the main actor.
public protocol ShellCapability: Sendable {
    /// `false` in the sandboxed edition. Check this before offering shell UI —
    /// don't rely on catching `.unavailable`.
    var isAvailable: Bool { get }

    func run(_ invocation: ShellInvocation) async throws -> ShellResult
}

/// The App Store edition's implementation, and a safe default for previews and
/// unit tests. Lets shared code depend on `ShellCapability` unconditionally
/// instead of littering call sites with optionals.
public nonisolated struct UnavailableShell: ShellCapability {
    public init() {}
    public var isAvailable: Bool { false }
    public func run(_ invocation: ShellInvocation) async throws -> ShellResult {
        throw ShellError.unavailable
    }
}
