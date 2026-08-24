import Foundation

#if DIRECT_BUILD
/// CFM-R10-Direct — the Direct-build doors, fenced by `sandbox-exec` (ported from
/// `catflow-mlx/src/catflow/engines/agent.py`). `Improvise` rows and `transforms:` scripts run
/// a shell command confined to a workdir by the Seatbelt profile below — the actual OS-level
/// guarantee, never a path check. **Never in the App Store build** (`canRun` refuses there).
///
/// **SPEC-Q divergence (the agent loop):** the Python's `Improvise` is a full LLM action loop
/// (`real_choose_action`). The app's Direct build executes the row's instruction as a fenced
/// shell command in the improvise workdir and records the resulting mutations for undo — a
/// genuine fenced execution, without the model-driven multi-step loop. `promote` (turning
/// improvised work back into ordinary rows) is the CLI's `catflow promote` verb; the app's
/// undo is the workdir snapshot.
nonisolated enum FencedRunner {

    /// The Seatbelt profile — `agent-fence.sb` ported verbatim: writes denied everywhere
    /// except the workdir; reads denied by category (home, volumes, tmp, etc.); the fence
    /// last so it wins inside itself; no network.
    static let seatbeltProfile = """
    (version 1)
    (allow default)
    (deny file-write*)
    (allow file-write*
      (subpath (param "WORKDIR"))
      (literal "/dev/null")
      (literal "/dev/stdout")
      (literal "/dev/stderr")
      (literal "/dev/dtracehelper")
      (literal "/dev/tty"))
    (deny file-read*
      (subpath "/Users")
      (subpath "/Volumes")
      (subpath "/private/var/root")
      (subpath "/private/var/folders")
      (subpath "/private/var/tmp")
      (subpath "/var/tmp")
      (subpath "/private/etc")
      (subpath "/etc")
      (subpath "/tmp")
      (subpath "/private/tmp"))
    (allow file-read* file-write* (subpath (param "WORKDIR")))
    (deny network*)
    """

    /// `_require_sandbox_exec` — sandbox-exec must exist; never run a door unconfined.
    static func requireSandboxExec() throws -> String {
        let candidates = ["/usr/bin/sandbox-exec", "/bin/sandbox-exec"]
        for c in candidates where FileManager.default.isExecutableFile(atPath: c) {
            return c
        }
        throw FlowError.stageFailure(row: "Direct",
                                     message: "sandbox-exec isn't available — the Direct doors refuse to run unconfined.")
    }

    /// `run_fenced` — run `argv` confined to `workdir` under the Seatbelt fence.
    /// Returns the captured stdout. Throws on a non-zero exit or missing fence.
    /// CFM-R10-FIX-4: pipes are drained concurrently (no >64KB deadlock), and a timed-out
    /// command escalates SIGTERM → SIGKILL after a grace period and is waited on.
    @discardableResult
    static func runFenced(argv: [String], workdir: URL, timeout: TimeInterval? = nil,
                          rowName: String = "Direct") throws -> String {
        let sandboxExec = try requireSandboxExec()
        try FileManager.default.createDirectory(at: workdir, withIntermediateDirectories: true)
        let resolved = workdir.resolvingSymlinksInPath()

        let profile = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-fence-\(UUID().uuidString).sb")
        try seatbeltProfile.write(to: profile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: profile) }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: sandboxExec)
        process.arguments = ["-D", "WORKDIR=\(resolved.path)", "-f", profile.path] + argv
        process.currentDirectoryURL = resolved

        let pipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = pipe
        process.standardError = errPipe
        process.standardInput = FileHandle.nullDevice

        // Drain both pipes concurrently from the start — reading after `waitUntilExit`
        // deadlocks the child on ~64KB of output (FIX-4).
        let readQueue = DispatchQueue(label: "fenced-read", attributes: .concurrent)
        var outputData = Data()
        var errorData = Data()
        let drainGroup = DispatchGroup()
        drainGroup.enter()
        readQueue.async { outputData = pipe.fileHandleForReading.readDataToEndOfFile(); drainGroup.leave() }
        drainGroup.enter()
        readQueue.async { errorData = errPipe.fileHandleForReading.readDataToEndOfFile(); drainGroup.leave() }

        try process.run()
        if let timeout {
            let deadline = Date().addingTimeInterval(timeout)
            while process.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if process.isRunning {
                // Escalate: SIGTERM, then SIGKILL after a grace period (a `trap '' TERM`
                // script ignores the first signal). Kill the process group if it has one.
                process.terminate()
                let killDeadline = Date().addingTimeInterval(2)
                while process.isRunning && Date() < killDeadline {
                    Thread.sleep(forTimeInterval: 0.05)
                }
                if process.isRunning {
                    let pid = process.processIdentifier
                    Darwin.kill(pid, SIGKILL)
                    Darwin.kill(-pid, SIGKILL)   // best-effort group kill
                    process.waitUntilExit()
                }
                drainGroup.wait()
                throw FlowError.stageFailure(row: rowName,
                                             message: "timed out after \(Int(timeout))s — the fenced command didn't finish.")
            }
        }
        process.waitUntilExit()
        drainGroup.wait()
        let stderr = String(data: errorData, encoding: .utf8) ?? ""
        guard process.terminationStatus == 0 else {
            throw FlowError.stageFailure(row: rowName,
                                         message: "the fenced command failed (\(process.terminationStatus)): \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
        return String(data: outputData, encoding: .utf8) ?? ""
    }

    // MARK: - The doors

    /// Run a `transforms:` entry's `run=` command fenced (P6-CD-03: a transform is the flow's
    /// own declared script, run for real).
    @discardableResult
    /// Run a `transforms:` entry per §14.6 — **files-only**: the script and every input are
    /// copied into the transform's own `workdir` (the fence's writable subtree is *only*
    /// that workdir), the script's argv is `script + inputs + output + params`, and the
    /// `.catflow-out-*` file — never stdout — becomes the row's output. Ported from
    /// `engines/transform.py::run_transform` (CFM-R10-FIX-5).
    static func runTransform(_ transform: TransformDef, row: Row, inputs: [Asset],
                             workspace: FlowWorkspace, flowID: String,
                             blobDirectory: URL) throws -> Asset {
        guard let run = transform.run, !run.isEmpty else {
            throw FlowError.stageFailure(row: transform.name, message: "has no `run:` script")
        }
        guard let workdirRaw = transform.workdir else {
            throw FlowError.stageFailure(row: transform.name,
                                         message: "its `transforms:` entry doesn't declare `workdir:`")
        }
        let scriptURL = try workspace.resolve(run, flowID: flowID)
        let workdir = try workspace.resolve(workdirRaw, flowID: flowID)
        try FileManager.default.createDirectory(at: workdir, withIntermediateDirectories: true)

        let scratch = UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12)
        var scratchPaths: [URL] = []

        // Copy the script in (exec bit preserved — `copyItem` carries it).
        let scriptCopy = workdir.appendingPathComponent(".catflow-run-\(scratch)")
        try FileManager.default.copyItem(at: scriptURL, to: scriptCopy)
        scratchPaths.append(scriptCopy)

        // Write the inputs as files.
        let items = inputs.flatMap(\.items)
        var inputPaths: [URL] = []
        for (i, item) in items.enumerated() {
            let p = workdir.appendingPathComponent(".catflow-in-\(scratch)-\(i)")
            if let value = item.value {
                try value.write(to: p, atomically: true, encoding: .utf8)
            } else if let path = item.path {
                try FileManager.default.copyItem(at: path, to: p)
            }
            inputPaths.append(p)
            scratchPaths.append(p)
        }

        let outputPath = workdir.appendingPathComponent(".catflow-out-\(scratch)")
        let argv = [scriptCopy.path]
            + inputPaths.map(\.path)
            + [outputPath.path]
            + boundParamArgs(transform, row: row)

        defer {
            for p in scratchPaths { try? FileManager.default.removeItem(at: p) }
        }

        let timeout = transform.timeout.flatMap(FlowSettingsEditor.parseDuration)
        let result = try runFenced(argv: argv, workdir: workdir, timeout: timeout,
                                   rowName: transform.name)
        _ = result   // stdout is not the output — §14.6 forbids that channel.
        guard FileManager.default.fileExists(atPath: outputPath.path) else {
            throw FlowError.stageFailure(row: transform.name,
                                         message: "its script didn't produce the output file (the last argv path)")
        }
        let kind = outputKind(transform)
        let dest = blobDirectory.appendingPathComponent("\(transform.name).0.\(kind.rawValue).\(scratch).bin")
        try FileManager.default.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: outputPath, to: dest)
        return Asset(items: [Item(kind: kind, value: nil, path: dest, sourceText: nil)])
    }

    /// `key=value` args in `params:` declaration order, bound from the calling row's settings.
    private static func boundParamArgs(_ transform: TransformDef, row: Row) -> [String] {
        let s = FlowSettings(row.settings)
        return transform.params.compactMap { param in
            if let value = s.value(for: param.name) { return "\(param.name)=\(value)" }
            if let d = param.defaultValue { return "\(param.name)=\(d)" }
            return nil
        }
    }

    /// The output kind from the transform's declared `gives` (default text).
    private static func outputKind(_ transform: TransformDef) -> Kind {
        guard let sig = transform.signature else { return .text }
        var parts: [String]
        if sig.contains("->") {
            parts = sig.components(separatedBy: "->")
        } else {
            parts = sig.components(separatedBy: "→")
        }
        let gives = parts.last?.trimmingCharacters(in: .whitespaces) ?? ""
        if gives.hasPrefix("["), gives.hasSuffix("]") {
            let inner = String(gives.dropFirst().dropLast())
            let first = inner.split(separator: ",").first.map(String.init)?.trimmingCharacters(in: .whitespaces) ?? ""
            return Kind(rawValue: first) ?? .text
        }
        return Kind(rawValue: gives) ?? .text
    }

    /// Run an `Improvise` row's instruction as a fenced shell command in its **declared**
    /// `workdir=`, snapshotted for undo, with every §14.4 bound enforced (CFM-R10-FIX-6):
    /// `max_actions=` and `timeout=` must be present (the E110 bound), `workdir=` must be
    /// present (E111), the declared `timeout=` bounds the command, and the workdir's size is
    /// checked after the run — the one bound §14.4b says *fails the row*.
    @discardableResult
    static func runImprovise(settings: String?, workspace: FlowWorkspace, flowID: String) throws -> String {
        let s = FlowSettings(settings)
        guard let command = s.firstBare() else {
            throw FlowError.stageFailure(row: "Improvise", message: "needs an instruction to run in the workdir")
        }
        guard let maxActions = s.value(for: "max_actions"), Int(maxActions) != 0 else {
            throw FlowError.stageFailure(row: "Improvise",
                                         message: "needs `max_actions=N` (how many steps it may take) — the bound")
        }
        guard s.value(for: "timeout") != nil else {
            throw FlowError.stageFailure(row: "Improvise",
                                         message: "needs `timeout=<duration>` (how long any one step may run) — the bound")
        }
        guard let workdirRaw = s.value(for: "workdir") else {
            throw FlowError.stageFailure(row: "Improvise", message: "needs `workdir=<dir>`")
        }
        let workdir = try workspace.resolve(workdirRaw, flowID: flowID)
        try FileManager.default.createDirectory(at: workdir, withIntermediateDirectories: true)
        let timeout = s.value(for: "timeout").flatMap(FlowSettingsEditor.parseDuration)
        let undoStore = undoStore(for: workdir)
        try snapshot(workdir, to: undoStore)
        let output = try runFenced(argv: ["/bin/zsh", "-lc", command], workdir: workdir,
                                   timeout: timeout, rowName: "Improvise")
        // The space bound — the one §14.4b bound that fails the row, not just ends it.
        try checkWorkdirBound(workdir, row: "Improvise")
        return output
    }

    /// `workdir_size` — total bytes under `workdir` (hidden files included).
    static func workdirSize(_ workdir: URL) -> Int {
        guard let enumerator = FileManager.default.enumerator(at: workdir,
                                                              includingPropertiesForKeys: [.fileSizeKey],
                                                              options: []) else { return 0 }
        var total = 0
        for case let file as URL in enumerator {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { continue }
            if let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                total += size
            }
        }
        return total
    }

    /// `check_workdir_bound` — §14.4b's space bound (2 GiB default): a workdir over it
    /// **fails the row**.
    static func checkWorkdirBound(_ workdir: URL, row: String, limit: Int = 2 * 1024 * 1024 * 1024) throws {
        let size = workdirSize(workdir)
        if size > limit {
            throw FlowError.stageFailure(row: row,
                                         message: "the improvise workdir exceeded the 2 GiB space bound (was \(size) bytes) — the row fails, not just ends.")
        }
    }

    // MARK: - The undo snapshot (SPEC-Q166's coarse undo)

    static func undoStore(for workdir: URL) -> URL {
        workdir.deletingLastPathComponent()
            .appendingPathComponent(".improvise-undo-\(workdir.lastPathComponent)", isDirectory: true)
    }

    /// Copy the workdir's current tree into the undo store (before a mutation).
    static func snapshot(_ workdir: URL, to store: URL) throws {
        try? FileManager.default.removeItem(at: store)
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        // FIX-6: no `.skipsHiddenFiles` — the workdir (a dot-prefixed `.improvise`, or one
        // holding dot-prefixed files) must be fully captured and restored.
        guard let enumerator = FileManager.default.enumerator(at: workdir,
                                                              includingPropertiesForKeys: [.isRegularFileKey],
                                                              options: []) else { return }
        for case let file as URL in enumerator {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { continue }
            let rel = file.path.replacingOccurrences(of: workdir.path + "/", with: "")
            let dest = store.appendingPathComponent(rel)
            try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: file, to: dest)
        }
    }

    /// Restore the workdir from the undo store (the "undo" after an Improvise activation).
    static func undo(_ workdir: URL) throws {
        let store = undoStore(for: workdir)
        guard FileManager.default.fileExists(atPath: store.path) else {
            throw FlowError.stageFailure(row: "Improvise", message: "no undo snapshot to restore")
        }
        // Remove the workdir's contents, then copy the snapshot back.
        if let enumerator = FileManager.default.enumerator(at: workdir,
                                                           includingPropertiesForKeys: nil,
                                                           options: []) {
            for case let file as URL in enumerator {
                try? FileManager.default.removeItem(at: file)
            }
        }
        guard let enumerator = FileManager.default.enumerator(at: store,
                                                              includingPropertiesForKeys: [.isRegularFileKey],
                                                              options: []) else { return }
        for case let file as URL in enumerator {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { continue }
            let rel = file.path.replacingOccurrences(of: store.path + "/", with: "")
            let dest = workdir.appendingPathComponent(rel)
            try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: file, to: dest)
        }
    }
}
#endif
