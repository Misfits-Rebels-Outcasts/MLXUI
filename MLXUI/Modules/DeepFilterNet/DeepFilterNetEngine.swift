import Foundation
import MLX
import MLXAudioSTS

/// MLX-native speech enhancement (noise removal) via mlx-audio-swift's `DeepFilterNetModel` (`MLXAudioSTS`). Takes 48 kHz
/// mono samples and returns cleaned 48 kHz mono samples of the same length. Loads the **installed** directory (the A2 pattern,
/// `fromLocal`, never `fromPretrained`, which would download behind the app's back), version `v3`.
///
/// **Sample rate (readme T6, owner ruling 2026-10-25).** The model is 48 kHz. The caller resamples to 48 kHz on the way in
/// (`AudioResampler`, to ``sampleRate``, not the 16 kHz ASR path) and the output stays 48 kHz whatever the input rate.
///
/// **Chunking (MP5-3a; implementer's call, pending owner confirmation).** The processing sits behind one function,
/// ``enhance(_:modelDirectory:chunkSeconds:overlapSeconds:maxDurationSeconds:progress:)``. `chunkSeconds == nil` (not the default,
/// ``defaultChunkSeconds`` is now 8) runs the whole clip in one `enhance` call. A number splits the clip into windows of that length that
/// overlap by `overlapSeconds` and are joined with a linear crossfade over the overlap (``chunkPlan(sampleCount:chunkSamples:overlapSamples:)``,
/// ``accumulate(into:window:values:overlap:isFirst:isLast:)``). The library's offline `enhance` carries no state between calls, so
/// each window starts cold; the overlap is what hides that warm-up. The defaults (8 s, 1 s) come from the 1, 10 and 60 minute
/// measurements (owner ruling 2026-10-25: measure before any limit or `ramGB` is set).
/// ``maxDurationSeconds`` is the refusal limit: 3,600 s, the longest clip measured (MP5-3c; numbers on ``defaultChunkSeconds``).
///
/// **Thread-bound streams (KICKOFF rule 17; memory note `mlx-swift-032-thread-bound-streams`).**
/// - `DeepFilterNetForward.swift:179` @ 580e952 is `coefs[0..., k, 0..., 0..., 0...]` where `k` is a Swift `Int` loop counter, so
///   it is **not** the array-index-after-another-index pattern (no `MLXArray` index follows another index). The scan test and a
///   read of `DeepFilterNetModel.enhance` / `DeepFilterNetForward` / `MossFormer2DSP` found no such subscript; the only array
///   index is `output.at[indices].add(...)` in `MossFormer2DSP.istft` (a scatter, which the 0.32.3 measurement lists as fine).
/// - `enhance` returns a **lazy** array (`MLX.clip(audioOut, ...)`). This engine builds it, evaluates and reads it
///   (`MLXErrorGuard.read`, rule 17a, which evaluates under `withError` first) in the same synchronous function with no `await`
///   between, so the array never leaves the thread that built it (17b). That is why `MLXAudioThread.run` is **not** used
///   (``usesAudioThread`` stays `false`): there is no suspension point to cross. The main-actor probe test (17c) is the evidence.
enum DeepFilterNetEngine {
    nonisolated static let sampleRate = 48_000
    /// The version subfolder of the installed repo that MLXUI runs (the repo also holds v1 and v2).
    nonisolated static let versionSubfolder = "v3"

    /// Window length of the default (chunked) run, in seconds. `nil` would be the whole clip in one pass, which is only for tests.
    ///
    /// **Measured 2026-10-25 by the lead (MP5-3b harness, `MLXUI` scheme, `phys_footprint` through the Run UI path
    /// `DeepFilterNetInput.read` -> stage -> `wavData`; journal 2026-597/598).** A whole-file pass grows linearly with length:
    /// 1 minute 2.996 GB, 10 minutes 14.201 GB (60 minutes would be about 80 GB; not run). Windows of 8 s with 1 s overlap:
    /// 1 minute 0.620 GB, 10 minutes 0.861 GB, 60 minutes 2.349 GB (MLX's own peak stays flat at 0.253 GiB; the growth is the host
    /// arrays, the decode phase alone is 1.76 GB at 60 minutes), 188 s wall for 60 minutes. Chunking costs nothing in quality: against the whole-file run the SNR gain differs by at most
    /// 0.07 dB at every setting tried (4/0.5, 4/1, 8/0.5, 8/1, 8/2, 16/1 s) and the seam difference is at most 0.002 RMS.
    /// 8 s / 1 s is the lead's ruling (implementer-level; the owner ruled the method, 2026-10-25).
    nonisolated static let defaultChunkSeconds: Double? = 8
    /// Overlap between windows when chunking, in seconds (see ``defaultChunkSeconds``).
    nonisolated static let defaultOverlapSeconds: Double = 1
    /// Longest recording accepted, in seconds: 3,600 (60 minutes), the longest clip measured (chunked 8 s / 1 s, 2.349 GB). The
    /// owner's rule (2026-10-25) is that a longer one is refused rather than run unmeasured. Lead's ruling, implementer-level.
    nonisolated static let maxDurationSeconds: Double? = 3_600
    /// Slack on the limit, in seconds: resampling a 3,600.000 s file can return a few samples more than 3,600 x 48,000, and that
    /// must not turn the longest measured clip into a refusal. Half a second over is refused. Implementer's call, pending owner confirmation.
    nonisolated static let limitToleranceSeconds: Double = 0.1

    /// Rule 17's one-line switch (see the type comment). `false`: the engine runs wherever it is awaited.
    nonisolated static let usesAudioThread = false

    enum EngineError: Error, LocalizedError, Equatable {
        case emptyAudio
        case noWeights(String)
        var errorDescription: String? {
            switch self {
            case .emptyAudio: return "The audio file has no samples."
            case .noWeights(let name):
                return "The DeepFilterNet model files (\(versionSubfolder)/config.json and model.safetensors) are missing from \(name). Re-install the model."
            }
        }
    }

    /// A recording longer than the limit. A `LocalizedError`, so `MLXErrorGuard.sentence(for:)` shows it as written.
    struct LimitError: Error, LocalizedError, Equatable {
        let seconds: Double
        let limit: Double
        var errorDescription: String? {
            "This recording is \(spoken(seconds)) long. Audio cleanup handles up to \(spoken(limit)) at a time; split the recording into shorter parts and run each one."
        }
    }

    /// A length as the person reads it: "59 minutes 43 seconds", "2 minutes", "45 seconds".
    nonisolated static func spoken(_ seconds: Double) -> String {
        let total = max(Int(seconds.rounded()), 0)
        let (m, s) = (total / 60, total % 60)
        func unit(_ n: Int, _ word: String) -> String { "\(n) \(word)\(n == 1 ? "" : "s")" }
        if m == 0 { return unit(s, "second") }
        return s == 0 ? unit(m, "minute") : "\(unit(m, "minute")) \(unit(s, "second"))"
    }

    /// Throws ``LimitError`` for a recording over `limit` (checked before any model is loaded). No limit when `limit` is nil.
    nonisolated static func requireWithinLimit(seconds: Double, limit: Double? = maxDurationSeconds) throws {
        if let limit, seconds > limit + limitToleranceSeconds { throw LimitError(seconds: seconds, limit: limit) }
    }

    // MARK: - Chunk plan and crossfade (pure)

    /// Windows over `sampleCount` samples. `chunkSamples == nil`, or a clip that fits in one chunk, is one window. Otherwise
    /// windows of `chunkSamples` start every `chunkSamples - overlapSamples`; the last ends at `sampleCount` and is longer than
    /// the overlap. The overlap is clamped to under half a chunk, so a window never fades in and out over the same samples.
    nonisolated static func chunkPlan(sampleCount: Int, chunkSamples: Int?, overlapSamples: Int) -> (windows: [Range<Int>], overlap: Int) {
        guard sampleCount > 0 else { return ([], 0) }
        guard let chunk = chunkSamples, chunk > 0, sampleCount > chunk else { return ([0..<sampleCount], 0) }
        let overlap = min(max(overlapSamples, 0), (chunk - 1) / 2)
        let step = chunk - overlap
        var windows: [Range<Int>] = []
        var start = 0
        while true {
            let end = min(start + chunk, sampleCount)
            windows.append(start..<end)
            if end == sampleCount { break }
            start += step
        }
        return (windows, overlap)
    }

    /// Adds one window's processed `values` into `out` with the crossfade weights: a linear fade-in over the first `overlap`
    /// samples (unless first) and the complementary fade-out over the last `overlap` (unless last). The two ramps sum to 1 across
    /// the shared region, so a pass-through reproduces the input.
    nonisolated static func accumulate(into out: inout [Float], window: Range<Int>, values: [Float], overlap: Int, isFirst: Bool, isLast: Bool) {
        let count = min(window.count, values.count)
        let ramp = Float(max(overlap, 1))
        for i in 0..<count {
            var weight: Float = 1
            if !isFirst && i < overlap { weight = (Float(i) + 0.5) / ramp }
            let fromEnd = window.count - 1 - i
            if !isLast && fromEnd < overlap { weight *= (Float(fromEnd) + 0.5) / ramp }
            out[window.lowerBound + i] += values[i] * weight
        }
    }

    // MARK: - Run

    /// Enhance 48 kHz mono `samples` with the installed model in `modelDirectory`. Loads the model per call (as the other
    /// engines). `progress` gets 0...1 after each window.
    nonisolated static func enhance(_ samples: [Float], modelDirectory: URL,
                                    chunkSeconds: Double? = defaultChunkSeconds,
                                    overlapSeconds: Double = defaultOverlapSeconds,
                                    maxDurationSeconds: Double? = maxDurationSeconds,
                                    progress: @Sendable (Double) -> Void = { _ in }) async throws -> [Float] {
        do {
            guard !samples.isEmpty else { throw EngineError.emptyAudio }
            try requireWithinLimit(seconds: Double(samples.count) / Double(sampleRate), limit: maxDurationSeconds)
            let model = try loadModel(modelDirectory)
            let plan = chunkPlan(sampleCount: samples.count,
                                 chunkSamples: chunkSeconds.map { Int($0 * Double(sampleRate)) },
                                 overlapSamples: Int(overlapSeconds * Double(sampleRate)))
            // One synchronous pass, no `await` between building an array and reading it (rule 17b).
            let cleaned = try run(samples, model: model, plan: plan, progress: progress)
            Memory.clearCache()
            return cleaned
        } catch let error as StageError {
            throw error
        } catch let error as LimitError {
            throw error
        } catch let error as EngineError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw StageError.engineFailure(stage: "DeepFilterNet", underlying: error)
        }
    }

    nonisolated private static func run(_ samples: [Float], model: DeepFilterNetModel,
                                        plan: (windows: [Range<Int>], overlap: Int),
                                        progress: @Sendable (Double) -> Void) throws -> [Float] {
        var out = [Float](repeating: 0, count: samples.count)
        for (index, window) in plan.windows.enumerated() {
            try Task.checkCancellation()
            // A whole-clip window reads `samples` directly: `Array(samples[window])` would be a second full copy of a long clip.
            let input = MLXArray(window.count == samples.count ? samples : Array(samples[window]))
            try MLXErrorGuard.evaluate(input)
            let enhanced = try model.enhance(input)
            var values = try MLXErrorGuard.read(enhanced, as: Float.self)
            // The library returns the input length; pad or trim defensively so a stray sample can't shift the timeline.
            if values.count < window.count { values += [Float](repeating: 0, count: window.count - values.count) }
            accumulate(into: &out, window: window, values: values, overlap: plan.overlap,
                       isFirst: index == 0, isLast: index == plan.windows.count - 1)
            // Return MLX's freed buffers after every window, or they pile up between windows (MP4-3d lesson).
            Memory.clearCache()
            progress(Double(index + 1) / Double(plan.windows.count))
        }
        return out
    }

    /// The model, loaded from a folder that must hold `v3/config.json` and weights. Checked here so a missing file is a
    /// readable sentence, not the library's path error.
    nonisolated private static func loadModel(_ modelDirectory: URL) throws -> DeepFilterNetModel {
        let version = modelDirectory.appendingPathComponent(versionSubfolder)
        let fm = FileManager.default
        guard fm.fileExists(atPath: version.appendingPathComponent("config.json").path),
              fm.fileExists(atPath: version.appendingPathComponent("model.safetensors").path) else {
            throw EngineError.noWeights(modelDirectory.lastPathComponent)
        }
        return try DeepFilterNetModel.fromLocal(modelDirectory, subfolder: versionSubfolder)
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout.
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
