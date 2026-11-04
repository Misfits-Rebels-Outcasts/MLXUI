import Foundation
import MLX
import Testing
@testable import MLXUI

// TEMPLATE — copied to MLXUITests/Smoke/ModelSmokeSupport.swift by task AM-T0 of RSI/prompts/add-model.md (once; every
// later model reuses it). Written 2026-10-28 without a compiler at hand: AM-T0 compiles it, fixes what doesn't build,
// and records each fix in its journal. Keep the line formats below unchanged — RSI/evals/model-smoke.sh parses them.

/// The /add-mlxui automated smoke harness (AM-T). Every `<Name>SmokeTests` suite is gated on `ModelSmoke.enabled(_:)`
/// and talks to the runner only through the lines below. Driven by `RSI/evals/model-smoke.sh run`, which sets
/// (as `TEST_RUNNER_…`, so the test process sees the names without the prefix):
///
/// | Variable | Meaning |
/// |---|---|
/// | `MLXUI_SMOKE_REPO` | the HF id under test. A suite runs only when this equals its own model; otherwise it is skipped. |
/// | `MLXUI_SMOKE_OUT` | the folder for artifacts the owner judges in `review.html` |
/// | `MLXUI_SMOKE_STORE` | a scratch `ModelStore` base under `/private/tmp` — **never** the user's Application Support |
/// | `MLXUI_SMOKE_WEIGHTS` | an existing checkpoint folder (skips the install) |
/// | `MLXUI_SMOKE_INSTALL` | `1` = install into the scratch store through `InstallManager` (that is also the install check) |
/// | `MLXUI_SMOKE_REPO_ROOT` | the checkout, for bundled samples |
///
/// Lines (the only output in these formats; one line per fact):
///
///     SMOKE|<row>|<check>|PASS|<detail>          (or FAIL / INFO — INFO is recorded, never counted)
///     FOOTPRINT|<phase>|<base MiB>|<peak MiB>|<after MiB>|<detail>
///     REVIEW|<row>|<file under MLXUI_SMOKE_OUT>|<what the owner judges>
///     INSTALL|<repo>|files=<n>|bytes=<b>|seconds=<s>|state=<state>
///
/// The host runs unsandboxed for these runs (`CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements
/// ENABLE_APP_SANDBOX=NO`, the `ServeStructuredRealModelTests` precedent), so it can read the scratch store, write the
/// artifacts, and run `say`. A suite that is skipped never counts as a pass.
nonisolated enum ModelSmoke {

    // MARK: - Environment

    static func value(_ key: String) -> String? {
        guard let v = ProcessInfo.processInfo.environment[key], !v.isEmpty else { return nil }
        return v
    }
    static var repo: String? { value("MLXUI_SMOKE_REPO") }
    static var out: URL? { value("MLXUI_SMOKE_OUT").map { URL(fileURLWithPath: $0, isDirectory: true) } }
    static var storeBase: URL? { value("MLXUI_SMOKE_STORE").map { URL(fileURLWithPath: $0, isDirectory: true) } }
    static var weights: URL? { value("MLXUI_SMOKE_WEIGHTS").map { URL(fileURLWithPath: $0, isDirectory: true) } }
    static var mayInstall: Bool { value("MLXUI_SMOKE_INSTALL") == "1" }

    /// The checkout (for `MLXUI/Resources/Samples/…` and fixtures).
    static var repoRoot: URL {
        if let root = value("MLXUI_SMOKE_REPO_ROOT") { return URL(fileURLWithPath: root, isDirectory: true) }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Smoke
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
    }
    static func sample(_ name: String) -> URL {
        repoRoot.appendingPathComponent("MLXUI/Resources/Samples/\(name)")
    }

    /// Gate for `@Suite(.enabled(if: ModelSmoke.enabled(Self.hfID)))`.
    static func enabled(_ hfID: String) -> Bool {
        repo == hfID && out != nil && (weights != nil || (mayInstall && storeBase != nil))
    }

    // MARK: - Lines

    static func clean(_ s: String) -> String {
        String(s.replacingOccurrences(of: "|", with: "/").replacingOccurrences(of: "\n", with: " ⏎ ").prefix(400))
    }

    /// One check: prints the SMOKE line and, on failure, records a test issue (so the xcodebuild run fails too).
    @discardableResult
    static func check(_ row: String, _ name: String, _ ok: Bool, _ detail: @autoclosure () -> String = "",
                      sourceLocation: SourceLocation = #_sourceLocation) -> Bool {
        let d = clean(detail())
        print("SMOKE|\(row)|\(name)|\(ok ? "PASS" : "FAIL")|\(d)")
        if !ok { Issue.record("smoke \(row) \(name): \(d)", sourceLocation: sourceLocation) }
        return ok
    }

    /// A fact worth reporting that is not a pass/fail check (speed, a figure for the owner).
    static func info(_ row: String, _ name: String, _ detail: String) {
        print("SMOKE|\(row)|\(name)|INFO|\(clean(detail))")
    }

    /// Save an artifact under `MLXUI_SMOKE_OUT`; with `judge`, it also goes on the owner's review page.
    @discardableResult
    static func artifact(_ row: String, _ name: String, data: Data, judge: String? = nil) throws -> URL {
        let dir = try #require(out, "MLXUI_SMOKE_OUT not set")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try data.write(to: url, options: .atomic)
        if let judge { print("REVIEW|\(row)|\(name)|\(clean(judge))") }
        return url
    }

    @discardableResult
    static func artifact(_ row: String, _ name: String, copying source: URL, judge: String? = nil) throws -> URL {
        try artifact(row, name, data: Data(contentsOf: source), judge: judge)
    }

    @discardableResult
    static func artifact(_ row: String, _ name: String, text: String, judge: String? = nil) throws -> URL {
        try artifact(row, name, data: Data(text.utf8), judge: judge)
    }

    // MARK: - Memory (phys_footprint of this process — what Activity Monitor's Memory column shows)

    static func physMiB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    /// BCC-0 (R17): MLX's own accounting, MiB. `active` = live arrays, `cache` = freed buffers MLX is holding for reuse,
    /// `peak` = the high-water mark of active+cache since `resetPeakMemory`. Together with `phys_footprint` this says
    /// whether a big footprint is live data or the buffer pool.
    static func mlxMiB() -> (active: Int, cache: Int, peak: Int) {
        let s = MLX.Memory.snapshot()
        return (s.activeMemory / 1_048_576, s.cacheMemory / 1_048_576, s.peakMemory / 1_048_576)
    }

    /// `mlx[active a0>a1 cache c0>c1 peak p MiB]`, appended to the detail field so the five leading
    /// `FOOTPRINT|` fields (what `model-smoke.sh` parses by position) are unchanged.
    static func mlxDetail(start: (active: Int, cache: Int, peak: Int)) -> String {
        let e = mlxMiB()
        return "mlx[active \(start.active)>\(e.active) cache \(start.cache)>\(e.cache) peak \(e.peak) MiB, cacheLimit \(MLX.Memory.cacheLimit / 1_048_576)]"
    }

    final class PeakSampler: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Double = 0
        func note(_ v: Double) { lock.lock(); value = max(value, v); lock.unlock() }
        var peak: Double { lock.lock(); defer { lock.unlock() }; return value }
    }

    /// Runs `body`, sampling every 20 ms, and prints one FOOTPRINT line for the phase. Phases are what the Run UI
    /// does (load, a typical run, the worst input the UI allows), never a synthetic shape the app can't produce.
    static func measure<T>(_ phase: String, detail: @autoclosure () -> String = "",
                           _ body: () async throws -> T) async throws -> T {
        MLX.Memory.clearCache()
        MLX.Memory.peakMemory = 0
        let mlxStart = mlxMiB()
        let base = physMiB()
        let sampler = PeakSampler()
        sampler.note(base)
        let task = Task.detached {
            while !Task.isCancelled {
                sampler.note(ModelSmoke.physMiB())
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        let t0 = ContinuousClock.now
        let value: T
        do {
            value = try await body()
        } catch {
            task.cancel()
            print("FOOTPRINT|\(clean(phase))|\(Int(base))|\(Int(sampler.peak))|\(Int(physMiB()))|threw: \(clean("\(error)")) \(mlxDetail(start: mlxStart))")
            throw error
        }
        task.cancel()
        sampler.note(physMiB())
        let elapsed = (ContinuousClock.now - t0).components
        let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
        print("FOOTPRINT|\(clean(phase))|\(Int(base))|\(Int(sampler.peak))|\(Int(physMiB()))|\(String(format: "%.1f s", seconds)) \(mlxDetail(start: mlxStart)) \(clean(detail()))")
        return value
    }

    // MARK: - Catalog + install

    static func catalogEntry(_ hfID: String) throws -> ModelEntry {
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        return try #require(catalog.domains.flatMap { $0.allModels }.first { $0.hfModelId == hfID },
                            "\(hfID) is not in browser.json")
    }

    /// The checkpoint folder: `MLXUI_SMOKE_WEIGHTS` if given, otherwise a real install through `InstallManager` into
    /// the scratch store — the same file selection, download, verify and `.installed` marker the Browse button uses.
    /// Prints the INSTALL line. Throws when the install doesn't finish (gated repo: needsAuth — rerun with --weights).
    @MainActor private static var resolved: [String: URL] = [:]

    @MainActor
    static func checkpoint(for hfID: String, timeoutSeconds: Double = 7200) async throws -> URL {
        if let weights { return weights }
        if let done = resolved[hfID] { return done }
        let base = try #require(storeBase, "MLXUI_SMOKE_STORE not set")
        let entry = try catalogEntry(hfID)
        let store = ModelStore(baseDirectory: base)
        let manager = InstallManager(store: store)
        let started = Date()
        manager.install(entry)
        func finished(_ s: InstallState?) -> Bool {
            switch s {
            case .installed?, .error?, .needsAuth?: return true
            default: return false
            }
        }
        while !finished(manager.modelStates[entry.id]), Date().timeIntervalSince(started) < timeoutSeconds {
            try await Task.sleep(for: .milliseconds(500))
        }
        let state = manager.modelStates[entry.id]
        let dir = store.directory(forHFModelID: hfID)
        let files = (FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)?.allObjects.count) ?? 0
        let bytes = ModelStore.directorySize(at: dir)
        print("INSTALL|\(hfID)|files=\(files)|bytes=\(bytes)|seconds=\(String(format: "%.1f", Date().timeIntervalSince(started)))|state=\(clean(String(describing: state)))")
        try #require(state == .installed, "install did not finish: \(String(describing: state))")
        resolved[hfID] = dir
        return dir
    }

    // MARK: - Inputs

    /// Speech of known text from macOS `say` (16 kHz mono 16-bit WAV), for ASR / voice / diarization oracles.
    /// `voice` nil = the system voice; a voice that isn't installed throws, and the caller records INFO, not FAIL.
    static func say(_ text: String, voice: String? = nil, name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxui-smoke-say", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name).appendingPathExtension("wav")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        p.arguments = (voice.map { ["-v", $0] } ?? []) + ["-o", url.path, "--file-format=WAVE", "--data-format=LEI16@16000", text]
        try p.run()
        p.waitUntilExit()
        guard p.terminationStatus == 0, FileManager.default.fileExists(atPath: url.path) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSLocalizedDescriptionKey: "say failed for voice \(voice ?? "default")"])
        }
        return url
    }

    // MARK: - Oracles (objective stand-ins for "look at the reply")

    static let thinkingTags = ["<think>", "</think>", "<|channel>", "<channel|>", "<|think|>", "◁think▷", "◁/think▷",
                               "<|start_thinking|>", "<|end_thinking|>"]

    static func thinkingTagsFound(in text: String) -> [String] { thinkingTags.filter { text.contains($0) } }

    static func words(_ s: String) -> [String] {
        s.lowercased().split { !$0.isLetter && !$0.isNumber && $0 != "'" }.map(String.init)
    }

    /// Degenerate output: some 6-word run repeats at least `times` times (the "garbled or repeated tokens" check).
    static func repeatedRun(in text: String, length: Int = 6, times: Int = 3) -> String? {
        let w = words(text)
        guard w.count >= length * times else { return nil }
        var counts: [String: Int] = [:]
        for i in 0...(w.count - length) {
            let key = w[i..<(i + length)].joined(separator: " ")
            counts[key, default: 0] += 1
            if counts[key, default: 0] >= times { return key }
        }
        return nil
    }

    /// Word error rate of `hypothesis` against `reference` (lower-cased words, punctuation dropped).
    static func wordErrorRate(reference: String, hypothesis: String) -> Double {
        let r = words(reference), h = words(hypothesis)
        guard !r.isEmpty else { return h.isEmpty ? 0 : 1 }
        var prev = Array(0...h.count)
        for i in 1...r.count {
            var cur = [i] + Array(repeating: 0, count: h.count)
            for j in stride(from: 1, through: h.count, by: 1) {
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] == h[j - 1] ? 0 : 1))
            }
            prev = cur
        }
        return Double(prev[h.count]) / Double(r.count)
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .nan }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    /// RMS and peak of mono float samples: "not silent" and "not clipped" for TTS / audio generation / cleanup.
    static func levels(_ samples: [Float]) -> (rms: Float, peak: Float) {
        guard !samples.isEmpty else { return (0, 0) }
        var sum: Float = 0, peak: Float = 0
        for s in samples { sum += s * s; peak = max(peak, abs(s)) }
        return ((sum / Float(samples.count)).squareRoot(), peak)
    }
}
