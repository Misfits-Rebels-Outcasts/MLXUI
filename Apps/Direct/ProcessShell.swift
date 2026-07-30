//
//  ProcessShell.swift
//  MLXUI-Direct target ONLY — never a member of the App Store target.
//
//  Real `ShellCapability`: spawns child processes with `Process`. Requires the
//  app to be un-sandboxed (Config/Direct.xcconfig, MLXUI-Direct.entitlements);
//  under the sandbox `execve` of anything outside the bundle fails with EPERM,
//  so this type would be useless there even if it compiled.
//
//  Hardened Runtime note: no entitlement is needed to spawn a child. The child
//  gets its own runtime policy — see MLXUI-Direct.entitlements.
//
//  Everything happens on a private concurrent queue behind one continuation, so
//  the non-`Sendable` `Process` never crosses a concurrency domain; only the
//  `Sendable` `ShellResult` comes back out.
//

import Foundation

nonisolated struct ProcessShell: ShellCapability {

    /// Environment handed to children when the invocation doesn't specify one.
    /// Deliberately minimal rather than `ProcessInfo.processInfo.environment`:
    /// a debug build inherits Xcode's `DYLD_*`, `__XPC_*` and `SDKROOT` vars,
    /// which make child behaviour differ between Debug and Release in ways that
    /// are miserable to reproduce.
    static var minimalEnvironment: [String: String] {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return [
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": home,
            "LANG": "en_US.UTF-8",
            "TERM": "dumb",
        ]
    }

    private static let queue = DispatchQueue(
        label: "net.connectcode.mlxui.shell",
        qos: .userInitiated,
        attributes: .concurrent
    )

    var isAvailable: Bool { true }

    func run(_ invocation: ShellInvocation) async throws -> ShellResult {
        let executable = try Self.resolve(
            invocation.executable,
            path: (invocation.environment ?? Self.minimalEnvironment)["PATH"] ?? ""
        )
        return try await withCheckedThrowingContinuation { continuation in
            Self.queue.async {
                do {
                    continuation.resume(returning: try Self.execute(executable, invocation))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Executable resolution

    /// Resolves a bare name against `PATH` ourselves rather than shelling out to
    /// `/usr/bin/env`, which would happily run whatever the name expanded to.
    private static func resolve(_ executable: String, path: String) throws -> URL {
        let fm = FileManager.default
        if executable.hasPrefix("/") {
            guard fm.fileExists(atPath: executable) else {
                throw ShellError.executableNotFound(executable)
            }
            guard fm.isExecutableFile(atPath: executable) else {
                throw ShellError.notExecutable(executable)
            }
            return URL(fileURLWithPath: executable)
        }
        guard !executable.contains("/") else {
            throw ShellError.blockedByPolicy(
                reason: "relative executable paths are not allowed: \(executable)"
            )
        }
        for dir in path.split(separator: ":") {
            let candidate = "\(dir)/\(executable)"
            if fm.isExecutableFile(atPath: candidate) {
                return URL(fileURLWithPath: candidate)
            }
        }
        throw ShellError.executableNotFound(executable)
    }

    // MARK: - Execution (synchronous, off the main actor)

    /// Mutable state shared across the two drain queues and the timeout item.
    /// `group.wait()` / `cancel()` provide the ordering; the lock covers the rest.
    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var _out = Data()
        private var _err = Data()
        private var _truncated = false
        private var _timedOut = false

        func append(out: Data) { lock.withLock { _out.append(out) } }
        func append(err: Data) { lock.withLock { _err.append(err) } }
        func markTruncated() { lock.withLock { _truncated = true } }
        func markTimedOut() { lock.withLock { _timedOut = true } }
        var out: Data { lock.withLock { _out } }
        var err: Data { lock.withLock { _err } }
        var truncated: Bool { lock.withLock { _truncated } }
        var timedOut: Bool { lock.withLock { _timedOut } }
    }

    private static func execute(
        _ executable: URL,
        _ invocation: ShellInvocation
    ) throws -> ShellResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = invocation.arguments
        process.environment = invocation.environment ?? minimalEnvironment
        if let workingDirectory = invocation.workingDirectory {
            process.currentDirectoryURL = workingDirectory
        }

        let outPipe = Pipe()
        let errPipe = Pipe()
        let inPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        process.standardInput = invocation.standardInput == nil
            ? FileHandle.nullDevice
            : inPipe

        let box = Box()
        let started = Date()

        do {
            try process.run()
        } catch {
            throw ShellError.launchFailed(error.localizedDescription)
        }

        if let input = invocation.standardInput {
            let handle = inPipe.fileHandleForWriting
            handle.write(Data(input.utf8))
            try? handle.close()
        }

        // Drain both pipes concurrently. A child that fills the 64 KB pipe
        // buffer blocks on write; if we were sitting in waitUntilExit() at that
        // moment, neither side would ever move again.
        let group = DispatchGroup()
        for (pipe, isStdout) in [(outPipe, true), (errPipe, false)] {
            group.enter()
            queue.async {
                drain(pipe, cap: invocation.maxOutputBytes, into: box, isStdout: isStdout)
                group.leave()
            }
        }

        // SIGTERM at the deadline, SIGKILL two seconds later if it's ignored.
        let timeoutItem = DispatchWorkItem {
            guard process.isRunning else { return }
            box.markTimedOut()
            process.terminate()
            queue.asyncAfter(deadline: .now() + 2) {
                if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            }
        }
        queue.asyncAfter(deadline: .now() + invocation.timeout, execute: timeoutItem)

        process.waitUntilExit()
        timeoutItem.cancel()
        group.wait()

        if box.timedOut {
            throw ShellError.timedOut(after: invocation.timeout)
        }

        return ShellResult(
            standardOutput: Self.text(box.out),
            standardError: Self.text(box.err),
            exitCode: process.terminationStatus,
            outputWasTruncated: box.truncated,
            duration: Date().timeIntervalSince(started)
        )
    }

    /// Reads to EOF, keeping at most `cap` bytes. Past the cap it keeps reading
    /// and discarding — closing the handle early would hand the child a SIGPIPE
    /// and misreport a working command as a crash.
    private static func drain(_ pipe: Pipe, cap: Int, into box: Box, isStdout: Bool) {
        let handle = pipe.fileHandleForReading
        var kept = 0
        while true {
            let chunk = handle.availableData
            if chunk.isEmpty { break }
            if kept < cap {
                let slice = chunk.prefix(cap - kept)
                kept += slice.count
                if isStdout { box.append(out: Data(slice)) } else { box.append(err: Data(slice)) }
                if slice.count < chunk.count { box.markTruncated() }
            } else {
                box.markTruncated()
            }
        }
        try? handle.close()
    }

    /// Child output is whatever the child felt like emitting — never assume it's
    /// valid UTF-8 (`ffmpeg` progress bars, `git` under a non-UTF-8 locale…).
    /// The lossy decoder substitutes U+FFFD rather than returning nil.
    private static func text(_ data: Data) -> String {
        String(data: data, encoding: .utf8) ?? String(decoding: data, as: UTF8.self)
    }
}
