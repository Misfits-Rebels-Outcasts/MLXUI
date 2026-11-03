import Foundation
import MLX
import MLXAudioCodecs
import MLXAudioSTS

/// MLX-native **sound isolation** through mlx-audio-swift's `SAMAudio` (`MLXAudioSTS`). One recording plus a short
/// description of a sound ("speech", "music", "singing voice") go in; two tracks come out — the named sound and
/// everything else. Loads the **installed** folder only (the A2 pattern): `SAMAudio(config:)` +
/// `loadConvertedWeights`, never `fromPretrained` / `fromDirectory`, so no code path here can download weights
/// (`SAMAudioModuleTests.noModuleFileCanDownloadWeights` reads the source to keep it that way).
///
/// **Trap 1 — the text encoder is not in the SAM-Audio repo.** `SAMAudio`'s `T5TextEncoder` resolves
/// `config.textEncoder.name`; when that string is not an existing path it calls `HubApi().snapshot(from: "t5-base")`,
/// a ~0.9 GB download behind the app's back, with no progress UI, on the first run. So ``load(modelDirectory:)``
/// decodes `config.json`, proves the installed `t5/` folder holds the encoder, **then** points `textEncoder.name` at
/// that folder's absolute path. A missing `t5/` is a readable error before any weights are touched (and before the
/// network is in play). Do not use `fromDirectory`: it decodes the unpatched config.
///
/// **Trap 3 — sample rate.** The model's DAC-VAE codec is 48 kHz, read from `config.json`
/// (``sampleRate(in:)``) and reported by `model.sampleRate`; it is neither the ASR path's 16 kHz nor assumed. Saved
/// WAVs use it, and the stage resamples the input to it (not through the ASR resampler).
///
/// **Chunking — the library's `separateLong` algorithm (windows, overlap, crossfade), run here with 30 s windows**
/// (owner ruling 2026-11-03: 30 s windows scored 0 bad windows in 30 against 13 in 90 at the library's 10 s default,
/// MC1-9b). Implementer's call for the rest, pending owner confirmation (implementer's call, pending owner
/// confirmation).** Up to ``wholeClipLimitSeconds`` (30 s) the clip goes through one `model.separate` pass; above it
/// the clip is separated in ``chunkSeconds`` (30 s) windows that overlap by ``overlapSeconds`` (3 s) and are joined
/// with the library's cosine crossfade (``cosineFade(count:)``, ``accumulate(into:window:values:overlap:isFirst:isLast:)``).
/// The library's own `separateLong` does exactly these windows but offers no hook between them, so it cannot honour
/// Stop or report progress; this loop is that algorithm with a `Task.checkCancellation()` **before every window**.
/// (`separateLong` also cannot stop mid-file — the Run screen's Stop requirement wins here.)
///
/// **Thread-bound MLX arrays (mlx-swift 0.32).** Every array the library returns is read through
/// `MLXErrorGuard.read`, which evaluates under `withError` first, so a failed `eval` throws instead of segfaulting;
/// the window array this file builds is evaluated (`MLXErrorGuard.evaluate`) before it crosses an `await`. The
/// main-actor probe test is the evidence.
nonisolated enum SAMAudioEngine {
    /// The catalog entry's repo.
    nonisolated static let modelHFID = "mlx-community/sam-audio-base"
    /// The T5-base text encoder's folder inside the installed model folder (the companion download, MC1-3).
    nonisolated static let textEncoderSubfolder = "t5"
    /// Used only when `config.json` is unreadable; the codec in this repo is 48 kHz.
    nonisolated static let fallbackSampleRate = 48_000

    /// Up to this long, a clip is separated in one pass; longer clips are windowed (see the type comment).
    nonisolated static let wholeClipLimitSeconds: Double = 30
    /// Window length above ``wholeClipLimitSeconds``: 30 s, the same as that limit, so a clip that fits one pass and a
    /// window of a longer clip are the same size (owner ruling 2026-11-03; the library's own default is 10 s).
    nonisolated static let chunkSeconds: Float = 30
    /// Overlap between windows; the library's `separateLong` default.
    nonisolated static let overlapSeconds: Float = 3
    /// Longest recording accepted, in seconds: 30 minutes (owner decision 5).
    nonisolated static let maxDurationSeconds: Double = 30 * 60
    /// Slack on the limit, in seconds: resampling a 30:00.000 file can return a few samples more than 30 x 60 x 48,000.
    nonisolated static let limitToleranceSeconds: Double = 0.1
    /// The codec's decoder runs in chunks of this many frames (the model card's recommendation), which keeps the
    /// decode — the peak of a long run — from holding one array for the whole clip.
    nonisolated static let odeDecodeChunkSize = 50

    /// The two tracks. Both are the recording's length at the model's sample rate.
    struct Result: Sendable, Equatable {
        var isolated: [Float]
        var other: [Float]
        var sampleRate: Int
    }

    enum EngineError: Error, LocalizedError, Equatable {
        case emptyAudio
        case missingDescription
        case noWeights(String)
        case noTextEncoder(String)

        var errorDescription: String? {
            switch self {
            case .emptyAudio:
                return "The audio file has no samples."
            case .missingDescription:
                return "Type what to isolate first — for example “speech” or “music”."
            case .noWeights(let name):
                return "The SAM-Audio model files (config.json and the .safetensors weights) are missing from \(name). Re-install the model."
            case .noTextEncoder(let name):
                return "The SAM-Audio model folder \(name) is missing its \(SAMAudioEngine.textEncoderSubfolder)/ folder, which holds the text encoder the model needs. Re-install the model."
            }
        }
    }

    /// A recording longer than the limit. A `LocalizedError`, so `MLXErrorGuard.sentence(for:)` shows it as written.
    struct LimitError: Error, LocalizedError, Equatable {
        let seconds: Double
        let limit: Double
        var errorDescription: String? {
            "This recording is \(spoken(seconds)) long. Sound isolation handles up to \(spoken(limit)) at a time; split the recording into shorter parts and run each one."
        }
    }

    /// A length as the person reads it: "29 minutes 43 seconds", "2 minutes", "45 seconds".
    nonisolated static func spoken(_ seconds: Double) -> String {
        let total = max(Int(seconds.rounded()), 0)
        let (m, s) = (total / 60, total % 60)
        func unit(_ n: Int, _ word: String) -> String { "\(n) \(word)\(n == 1 ? "" : "s")" }
        if m == 0 { return unit(s, "second") }
        return s == 0 ? unit(m, "minute") : "\(unit(m, "minute")) \(unit(s, "second"))"
    }

    /// A time **estimate** for the Run screen, rounded the way a person reads an estimate: "about 17 minutes" above
    /// 90 seconds, the exact spoken length below it. Never shown as a measurement.
    nonisolated static func spokenEstimate(_ seconds: Double) -> String {
        guard seconds >= 90 else { return spoken(seconds) }
        let minutes = max(1, Int((seconds / 60).rounded()))
        return "\(minutes) minute\(minutes == 1 ? "" : "s")"
    }

    /// Throws ``LimitError`` for a recording over `limit` (checked before any model is loaded). No limit when `limit` is nil.
    nonisolated static func requireWithinLimit(seconds: Double, limit: Double? = maxDurationSeconds) throws {
        if let limit, seconds > limit + limitToleranceSeconds { throw LimitError(seconds: seconds, limit: limit) }
    }

    // MARK: - The time estimate shown before a run (implementer's call, pending owner confirmation)

    /// The fixed cost of a run, in seconds: loading both models and the T5 encode, before the first window. Measured
    /// 2026-11-02 (MC1-6, the `SAMAudioRealLoad` harness): a cold first run took about a minute, including the one-time
    /// Metal kernel compilation; a warm process reloads in about 15 s.
    nonisolated static let estimatedLoadSeconds: Double = 60
    /// Seconds of processing per second of audio. Measured 2026-11-02 (MC1-6): the 65 s windowed clip ran at 0.65x to
    /// 0.68x realtime warm (92-95 s wall), and the first, cold 10 s run was slower. Set to 3 s per audio second, a
    /// conservative value; it varies with the Mac, which is why the Run screen says "about".
    nonisolated static let estimatedProcessingSecondsPerAudioSecond: Double = 3.0

    /// The sentence the Run screen shows before a run starts: "It will take about 90 minutes" for a 30-minute
    /// recording. Only a basis for a decision, never a promise about another Mac.
    nonisolated static func estimate(seconds: Double) -> Double {
        estimatedLoadSeconds + max(seconds, 0) * estimatedProcessingSecondsPerAudioSecond
    }

    // MARK: - Sample rate

    /// The model's sample rate, read from the installed `config.json` **without** loading any weights (the Run screen and
    /// the stage resample before a load, and the load must only happen once the run starts). 48 kHz for this repo.
    nonisolated static func sampleRate(in modelDirectory: URL) -> Int {
        guard let data = try? Data(contentsOf: modelDirectory.appendingPathComponent("config.json")),
              let config = try? JSONDecoder().decode(SAMAudioConfig.self, from: data) else {
            return fallbackSampleRate
        }
        return config.audioCodec.sampleRate
    }

    // MARK: - Window plan and crossfade (pure)

    /// Windows over `sampleCount` samples, the shape `separateLong` uses: a clip that fits in one window is one window;
    /// otherwise windows of `chunkSamples` start every `chunkSamples - overlapSamples`, and the last ends at the last
    /// sample. `overlap` is clamped to under half a chunk, so a window never fades in and out over the same samples.
    nonisolated static func chunkPlan(sampleCount: Int, chunkSamples: Int, overlapSamples: Int) -> (windows: [Range<Int>], overlap: Int) {
        guard sampleCount > 0 else { return ([], 0) }
        guard chunkSamples > 0, sampleCount > chunkSamples else { return ([0..<sampleCount], 0) }
        let overlap = min(max(overlapSamples, 0), (chunkSamples - 1) / 2)
        let step = chunkSamples - overlap
        var windows: [Range<Int>] = []
        var start = 0
        while true {
            let end = min(start + chunkSamples, sampleCount)
            windows.append(start..<end)
            if end == sampleCount { break }
            start += step
        }
        return (windows, overlap)
    }

    /// The library's crossfade weights for `count` samples: `0.5 * (1 - cos(pi * t))`, t from 0 to 1. The fade-out is
    /// its complement, so a pass-through reproduces the input exactly and two correlated windows never sum above 1.
    nonisolated static func cosineFade(count: Int) -> [Float] {
        guard count > 1 else { return count == 1 ? [0] : [] }
        let denominator = Float(count - 1)
        return (0..<count).map { i in
            0.5 * (1 - cos(Float.pi * Float(i) / denominator))
        }
    }

    /// Adds one window's `values` into `out` with the library's cosine crossfade: a fade-in over the first `overlap`
    /// samples (unless first) and the complementary fade-out over the last `overlap` (unless last). The two windows
    /// around a seam are indexed from the **start of the overlap region** on both sides, so their weights sum to 1 at
    /// every shared sample and a pass-through reproduces the input exactly.
    nonisolated static func accumulate(into out: inout [Float], window: Range<Int>, values: [Float],
                                       overlap: Int, isFirst: Bool, isLast: Bool) {
        let count = min(window.count, values.count)
        guard overlap > 0 else {
            for i in 0..<count { out[window.lowerBound + i] += values[i] }
            return
        }
        let fade = cosineFade(count: overlap)
        let tailStart = window.count - overlap
        for i in 0..<count {
            var weight: Float = 1
            if !isFirst && i < overlap { weight *= fade[i] }
            if !isLast && i >= tailStart { weight *= 1 - fade[i - tailStart] }
            out[window.lowerBound + i] += values[i] * weight
        }
    }

    /// Puts `values` into `out` starting at `start`, trimming or zero-padding so the two tracks keep the recording's
    /// timeline whatever the codec's length rounding is (a codec hop can return a few samples fewer).
    nonisolated static func place(_ values: [Float], into out: inout [Float], startingAt start: Int) {
        for i in 0..<min(values.count, max(out.count - start, 0)) {
            out[start + i] = values[i]
        }
    }

    // MARK: - Load

    /// A loaded model and the sample rate its codec reports.
    struct Loaded {
        let model: SAMAudio
        let sampleRate: Int
    }

    /// The installed folder's config, patched for Trap 1: `textEncoder.name` becomes the **absolute path** of the
    /// installed `t5/` folder. `T5TextEncoder.resolveModelFolder` returns that folder as-is when
    /// `FileManager.fileExists(atPath:)` is true, so the library never reaches `HubApi().snapshot`. Throws
    /// ``EngineError`` — reads nothing but the two folders — when the folder or its encoder is missing.
    nonisolated static func installedConfig(modelDirectory: URL) throws -> SAMAudioConfig {
        let fm = FileManager.default
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: configURL.path), let configData = try? Data(contentsOf: configURL) else {
            throw EngineError.noWeights(modelDirectory.lastPathComponent)
        }
        // The companion text encoder is proved present before the 5 GB of weights are read, so a bad install fails in
        // a second with a sentence, not after a long load.
        let textFolder = modelDirectory.appendingPathComponent(textEncoderSubfolder, isDirectory: true)
        guard fm.fileExists(atPath: textFolder.appendingPathComponent("config.json").path),
              fm.fileExists(atPath: textFolder.appendingPathComponent("model.safetensors").path) else {
            throw EngineError.noTextEncoder(modelDirectory.lastPathComponent)
        }
        var config: SAMAudioConfig
        do {
            config = try JSONDecoder().decode(SAMAudioConfig.self, from: configData)
        } catch {
            throw StageError.engineFailure(stage: "SAM-Audio", underlying: error)
        }
        config.textEncoder.name = textFolder.path
        return config
    }

    /// Loads the **installed** model folder: `config.json` + its top-level `.safetensors`, with `textEncoder.name`
    /// pointed at the installed `t5/` folder (Trap 1). Throws ``EngineError`` before touching the network, and never
    /// downloads.
    nonisolated static func load(modelDirectory: URL) throws -> Loaded {
        let fm = FileManager.default
        let config = try installedConfig(modelDirectory: modelDirectory)
        let files = (try? fm.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        let weightFiles = files.filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !weightFiles.isEmpty else { throw EngineError.noWeights(modelDirectory.lastPathComponent) }

        var weights: [String: MLXArray] = [:]
        for file in weightFiles {
            let arrays = try MLX.loadArrays(url: file)
            weights.merge(arrays) { _, new in new }
        }
        let model = SAMAudio(config: config)
        try model.loadConvertedWeights(weights)
        weights.removeAll()
        return Loaded(model: model, sampleRate: model.sampleRate)
    }

    // MARK: - Run

    /// The run seed used when the caller gives none. Fixed, so the same recording gives the same two tracks every time
    /// (MC1-9, R3). Each window's noise is `key(seed + windowIndex)`; nothing here ever seeds or reads the global
    /// `MLXRandom` state.
    nonisolated static let defaultSeed: UInt64 = 20_260_503

    /// The seed a run uses: the caller's, or ``defaultSeed`` when it passes none. The Run screen passes a fresh random
    /// one per Run (R12); flows and other callers get the default and so reproducible tracks.
    nonisolated static func effectiveSeed(_ seed: UInt64?) -> UInt64 { seed ?? defaultSeed }

    /// What one window's separation returns: the named sound and everything else, for that window's samples.
    struct WindowTracks: Sendable {
        var target: [Float]
        var residual: [Float]
    }

    /// The per-window separate call, as the window loop sees it: the window's samples, its index and its seeded noise.
    /// The engine passes the real `SAMAudio.separate`; unit tests pass a fake, so the loop (Stop between windows,
    /// progress, the crossfade, the noise) is tested without weights.
    typealias WindowSeparator = @Sendable (_ window: [Float], _ index: Int, _ noise: MLXArray) async throws -> WindowTracks

    /// The noise for window `index`: standard normal, shape `[1, frames, channels]` (the shape of the model's audio
    /// features for that window: `frames` = the window's codec frames, `channels` = 2 x the codec's codebook dimension),
    /// from the explicit key `seed + index`. Evaluated here, before it can cross an `await` (MLX thread-bound streams).
    nonisolated static func windowNoise(seed: UInt64, index: Int, frames: Int, channels: Int) throws -> MLXArray {
        let key = MLXRandom.key(seed &+ UInt64(index))
        let noise = MLXRandom.normal([1, frames, channels], key: key)
        try MLXErrorGuard.evaluate(noise)
        return noise
    }

    /// The window loop. `Task.checkCancellation()` runs **before every window** (so Stop ends a long run between
    /// windows), `progress` gets 0...1 **after** each, and the tracks are joined with the cosine crossfade and keep
    /// the recording's length (trimmed or zero-padded when the codec's rounding returns a few samples more or fewer).
    /// `hopLength` and `noiseChannels` size each window's noise: `ceil(window / hopLength)` frames.
    nonisolated static func runWindows(sampleCount: Int, plan: (windows: [Range<Int>], overlap: Int),
                                       samples: [Float], seed: UInt64, hopLength: Int, noiseChannels: Int,
                                       separateWindow: WindowSeparator,
                                       progress: @Sendable (Double) -> Void) async throws -> (isolated: [Float], other: [Float]) {
        var isolated = [Float](repeating: 0, count: sampleCount)
        var other = [Float](repeating: 0, count: sampleCount)
        for (index, window) in plan.windows.enumerated() {
            try Task.checkCancellation()
            let isOnlyWindow = plan.windows.count == 1
            let frames = (window.count + hopLength - 1) / max(hopLength, 1)
            let noise = try windowNoise(seed: seed, index: index, frames: frames, channels: noiseChannels)
            let tracks = try await separateWindow(Array(samples[window]), index, noise)
            if isOnlyWindow {
                place(tracks.target, into: &isolated, startingAt: 0)
                place(tracks.residual, into: &other, startingAt: 0)
            } else {
                accumulate(into: &isolated, window: window, values: tracks.target, overlap: plan.overlap,
                           isFirst: index == 0, isLast: index == plan.windows.count - 1)
                accumulate(into: &other, window: window, values: tracks.residual, overlap: plan.overlap,
                           isFirst: index == 0, isLast: index == plan.windows.count - 1)
            }
            Memory.clearCache()
            progress(Double(index + 1) / Double(plan.windows.count))
        }
        return (isolated, other)
    }

    /// Separates the 48 kHz mono `samples` (the stage resamples first) with the installed model in `modelDirectory`.
    /// `description` is what to isolate, in the model's own words ("speech", "music"). `progress` gets 0...1 per window.
    /// Cancellation is checked **before every window**, so Stop ends a long run between windows. `seed` makes the run
    /// reproducible (see ``defaultSeed``). `chunkSeconds` and `overlapSeconds` default to the shipped window shape;
    /// they are parameters so a measurement can try others without changing what ships.
    @concurrent nonisolated static func separate(_ samples: [Float], description: String, modelDirectory: URL,
                                                 maxDurationSeconds: Double? = maxDurationSeconds,
                                                 seed: UInt64 = SAMAudioEngine.defaultSeed,
                                                 chunkSeconds: Float = SAMAudioEngine.chunkSeconds,
                                                 overlapSeconds: Float = SAMAudioEngine.overlapSeconds,
                                                 progress: @Sendable @escaping (Double) -> Void = { _ in }) async throws -> Result {
        do {
            guard !samples.isEmpty else { throw EngineError.emptyAudio }
            let text = description.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { throw EngineError.missingDescription }
            let rate = sampleRate(in: modelDirectory)
            try requireWithinLimit(seconds: Double(samples.count) / Double(rate), limit: maxDurationSeconds)
            try Task.checkCancellation()

            let loaded = try load(modelDirectory: modelDirectory)
            defer { Memory.clearCache() }
            // The text encoder runs once for the whole clip: every window shares the features (as `separateLong` does).
            let encoded = try await loaded.model.textEncoder.encode([text])
            try MLXErrorGuard.evaluate(encoded.features, encoded.attentionMask)

            let windowed = Double(samples.count) / Double(rate) > wholeClipLimitSeconds
            let plan = windowed
                ? chunkPlan(sampleCount: samples.count,
                            chunkSamples: Int(Float(rate) * chunkSeconds),
                            overlapSamples: Int(Float(rate) * overlapSeconds))
                : (windows: [0..<samples.count], overlap: 0)

            let model = loaded.model
            let codec = model.config.audioCodec
            let tracks = try await runWindows(
                sampleCount: samples.count, plan: plan, samples: samples, seed: seed,
                hopLength: codec.hopLength, noiseChannels: 2 * codec.codebookDim,
                separateWindow: { window, _, noise in
                    let chunk = MLXArray(window).reshaped([1, 1, window.count])
                    try MLXErrorGuard.evaluate(chunk)
                    let result = try await model.separate(
                        audios: chunk,
                        descriptions: [text],
                        noise: noise,
                        odeDecodeChunkSize: odeDecodeChunkSize,
                        _textFeatures: encoded.features,
                        _textMask: encoded.attentionMask
                    )
                    guard let target = result.target.first, let residual = result.residual.first else {
                        throw MLXError.caught("the model returned no tracks")
                    }
                    return WindowTracks(target: try MLXErrorGuard.read(target, as: Float.self),
                                        residual: try MLXErrorGuard.read(residual, as: Float.self))
                },
                progress: progress)
            return Result(isolated: tracks.isolated, other: tracks.other, sampleRate: rate)
        } catch let error as StageError {
            throw error
        } catch let error as LimitError {
            throw error
        } catch let error as EngineError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw StageError.engineFailure(stage: "SAM-Audio", underlying: error)
        }
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout.
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
