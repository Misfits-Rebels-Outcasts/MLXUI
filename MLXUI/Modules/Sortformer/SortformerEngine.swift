import Foundation
import MLX
import MLXAudioVAD

/// MLX-native speaker diarization via mlx-audio-swift's `SortformerModel` (`MLXAudioVAD`). Takes the whole 16 kHz mono
/// buffer and returns who spoke when. Loads the **installed** directory (the A2 pattern, `fromModelDirectory`, never
/// `fromPretrained`, which would download behind the app's back).
///
/// **Two paths, chosen by clip length (MP4-3b; measured 2026-10-25 on the owner's Mac with MLX's own peak counter, through
/// this engine; weights are 0.22 GiB).**
///   - **Offline** (`generate`, one forward pass over the whole clip) up to ``offlineMaxSeconds`` (60 s). Its memory grows
///     roughly with the square of the length: peak 0.836 GiB at 52.7 s, 1.000 at 60 s, 1.67 at 150 s, 2.27 at 180 s, 2.80 at
///     316 s, 9.89 at 632 s. Fine for a short clip, fatal for a long one, so it is capped where the peak is 1 GiB.
///   - **Streaming** (`feed`, 5 s chunks, the speaker state carried between calls) above that, up to ``maxSeconds`` (3,583 s).
///     Its peak is flat: 0.59 GiB at 181 s, 0.68 at 316 s and 632 s, 0.89 at 1,896 s, 0.77 at 3,583 s (79 s to run). On the 52.7 s
///     parity clip it finds the same eight turns as the ground truth, within 0.12 s. `generateStream` was measured too and is
///     **not** used: it pre-encodes the whole recording first, so its peak grows with length (0.64, 2.13 and 3.76 GiB at 52,
///     316 and 632 s).
///   - Those are MLX's own counter, which leaves out Metal and driver memory. The process's `phys_footprint` through the Run UI's
///     whole path (MP4-3d, clean test process of about 100 MiB at start) peaks at: 1,441 MiB offline at 52.7 s, **1,673 MiB offline
///     at 60 s**, 1,219 MiB streaming at 61 s, 1,244 MiB at 588 s, 1,437 MiB at 3,583 s (that one includes the 229 MB of decoded
///     samples the UI holds). The offline cap is the maximum, so the catalog's `ramGB` (1.8 GB) and the `ModelLoadProfile` row
///     (6.95 x weights, added over the decoded baseline) come from it. Streaming frees MLX's buffer cache after every chunk
///     (about 0.25 GB less than without, and faster).
///   - Past ``maxSeconds`` the engine refuses with ``LimitError`` before touching the model: a clear sentence, not a crash
///     (the "no request may kill the app" rule). The 60 s offline cap and the 5 s chunk are owner-confirmed (2026-10-25).
///     The limit is the longest clip actually measured on the streaming path, 3,583 s, by the owner's ruling (2026-10-25):
///     a longer one is refused rather than run unmeasured.
///
/// **`feed` timing fix.** The library's `feed` shifts each chunk by `framesProcessed x 80 ms`, but a 5 s chunk of audio makes 63
/// output frames, not 62.5, so the timeline drifts later by half a frame per chunk (0.4 s after 52 s). This engine adds back
/// the difference between where the chunk really starts (its sample offset) and where the library puts it. The trailing
/// padded frame can overlap the next chunk by a few milliseconds; same-speaker pieces are joined by `SpeakerSegments.cleaned`.
///
/// **Thread-bound streams (KICKOFF rule 17; journals 2026-557/558).** `Sortformer.swift:1318-1319` @ 580e952
/// (`embs[0..., topIndices, 0...]`, `preds[0..., topIndices, 0...]`) is the array-index-after-another-index gather. It is in
/// `compressSpkcacheSimple`, the *streaming* speaker-cache compression, so the streaming branch does build it. It evaluates
/// both results straight away (`eval(compressedEmbs, compressedPreds)`), so no unevaluated thread-bound array is carried from
/// one `feed` call (one library `Task.detached` each) to the next; the owner-run main-actor probe, the streaming-branch
/// probe and the tiny-model streaming test are the evidence. Rule 17 in this file:
///   - the input chunk is evaluated up front with `MLXErrorGuard.evaluate`, so a failed evaluation throws before the library
///     touches it (17b);
///   - **no library array is read here**: only `DiarizationOutput.segments` (Swift values) is used, never `speakerProbs` or the
///     state's tensors. If a later task reads them it must go through `MLXErrorGuard.read(_:as:)` (17a);
///   - `MLXAudioThread.run` is **not** used (``usesAudioThread`` stays `false`): the real checkpoint ran from the main actor
///     inside `guardedRun` on both paths with no stream error and no fallback message (lead's runs, 2026-10-25). The
///     scoped `withError` handler is task-local and does not reach the library's detached tasks; an MLX error there lands in the
///     process-wide fallback, and the Run UI's `MLXErrorGuard.guardedRun` turns that into a thrown error.
///
/// **Library behaviour worth knowing** (read from the source, not measured): offline `generate` trims leading and trailing
/// silence before the pass and adds the offset back and peak-normalises the waveform; `feed` does neither (v2.1 features). Both
/// return segments per speaker slot (0 to 3) with `minDuration` 0 and `mergeGap` 0, the library defaults, which this engine
/// leaves alone.
enum SortformerEngine {
    nonisolated static let sampleRate = 16_000
    nonisolated static let defaultThreshold: Float = 0.5
    nonisolated static let thresholdRange: ClosedRange<Float> = 0.05...0.95

    /// Longest clip the one-pass offline path takes (1 minute: 1.000 GiB MLX peak, 1.63 GiB process footprint, measured). Longer clips stream.
    nonisolated static let offlineMaxSeconds: Double = 60
    /// Longest recording the engine accepts at all: 3,583 s (59 min 43 s), the longest clip measured on the streaming path
    /// (2026-10-25). Owner ruling, 2026-10-25: the limit is the longest measured length, not a round 60 minutes.
    nonisolated static let maxSeconds: Double = 3_583

    /// A length as the person reads it: "59 minutes 43 seconds", "2 minutes", "45 seconds".
    nonisolated static func spoken(_ seconds: Double) -> String {
        let total = max(Int(seconds.rounded()), 0)
        let (m, s) = (total / 60, total % 60)
        func unit(_ n: Int, _ word: String) -> String { "\(n) \(word)\(n == 1 ? "" : "s")" }
        if m == 0 { return unit(s, "second") }
        return s == 0 ? unit(m, "minute") : "\(unit(m, "minute")) \(unit(s, "second"))"
    }
    /// Streaming chunk length (the library's own default for `generateStream`).
    nonisolated static let streamChunkSeconds: Double = 5

    /// Rule 17's one-line switch (see the type comment). `false`: the engine runs wherever it is awaited.
    nonisolated static let usesAudioThread = false

    enum EngineError: Error, CustomStringConvertible {
        case emptyAudio
        case noWeights(URL)
        var description: String {
            switch self {
            case .emptyAudio: return "the audio file has no samples"
            case .noWeights(let url): return "no model weights (.safetensors) in \(url.lastPathComponent)"
            }
        }
    }

    /// A recording longer than ``maxSeconds``. A `LocalizedError`, so `MLXErrorGuard.sentence(for:)` shows it as written.
    struct LimitError: Error, LocalizedError, Equatable {
        let seconds: Double
        var errorDescription: String? {
            return "This recording is \(SortformerEngine.spoken(seconds)) long. Speaker identification handles up to "
                + "\(SortformerEngine.spoken(SortformerEngine.maxSeconds)) at a time; split the recording into shorter parts and run each one."
        }
    }

    /// Throws ``LimitError`` for a recording over ``maxSeconds`` (checked before any model is loaded).
    nonisolated static func requireWithinLimit(seconds: Double) throws {
        if seconds > maxSeconds { throw LimitError(seconds: seconds) }
    }

    /// Which path a clip of `seconds` takes.
    enum Path: Equatable { case offline, streaming }
    nonisolated static func path(forSeconds seconds: Double, offlineLimit: Double = offlineMaxSeconds) -> Path {
        seconds > offlineLimit ? .streaming : .offline
    }

    /// The streaming chunks: consecutive sample ranges of `chunkSamples`, with a short remainder (under half a chunk) folded into
    /// the chunk before it, so the last `feed` call never gets a sliver too short to compute features for. Empty for no samples.
    nonisolated static func streamingChunks(sampleCount: Int, chunkSamples: Int) -> [Range<Int>] {
        guard sampleCount > 0, chunkSamples > 0 else { return [] }
        var ranges: [Range<Int>] = []
        var start = 0
        while start < sampleCount {
            ranges.append(start..<min(start + chunkSamples, sampleCount))
            start += chunkSamples
        }
        if ranges.count > 1, let last = ranges.last, last.count < chunkSamples / 2 {
            ranges.removeLast()
            let previous = ranges.removeLast()
            ranges.append(previous.lowerBound..<last.upperBound)
        }
        return ranges
    }

    /// A threshold the model can use: NaN falls back to the default; anything else is clamped to ``thresholdRange``.
    nonisolated static func clampedThreshold(_ value: Float) -> Float {
        guard value.isFinite else { return defaultThreshold }
        return min(max(value, thresholdRange.lowerBound), thresholdRange.upperBound)
    }

    /// Diarize 16 kHz mono float samples using the installed model directory. Loads the model per call (matches the other
    /// engines). Returns the library's segments with the model's own slot numbers; `SpeakerSegments.cleaned` is the caller's
    /// clean-up (clamp, join, renumber by first appearance). `offlineLimit` exists so tests can send a short clip down the
    /// streaming path.
    nonisolated static func diarize(_ samples: [Float], threshold: Float = defaultThreshold, modelDirectory: URL,
                                    offlineLimit: Double = offlineMaxSeconds) async throws -> [SpeakerSegment] {
        do {
            let seconds = Double(samples.count) / Double(sampleRate)
            try requireWithinLimit(seconds: seconds)
            let chosen = path(forSeconds: seconds, offlineLimit: offlineLimit)
            let segments = try await perform {
                switch chosen {
                case .offline: try await generate(samples, threshold: threshold, modelDirectory: modelDirectory)
                case .streaming: try await generateStreaming(samples, threshold: threshold, modelDirectory: modelDirectory)
                }
            }
            // As ParakeetEngine / Qwen3ASREngine: return MLX's buffer cache once the result is read, after the model (a
            // local of the generate functions) is released, so its weights don't land in the cache on return.
            Memory.clearCache()
            return segments
        } catch let error as StageError {
            throw error
        } catch let error as LimitError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "Sortformer", underlying: error)
        }
    }

    nonisolated private static func perform(_ body: @escaping @Sendable () async throws -> [SpeakerSegment]) async throws -> [SpeakerSegment] {
        usesAudioThread ? try await MLXAudioThread.run(body) : try await body()
    }

    /// The model, loaded from a folder that must hold weights: the library's loader applies whatever weights it finds and does
    /// not fail on missing ones, so a folder with no safetensors would run a randomly initialised model and return noise.
    nonisolated private static func loadModel(_ modelDirectory: URL) throws -> SortformerModel {
        let files = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        guard files.contains(where: { $0.pathExtension == "safetensors" }) else { throw EngineError.noWeights(modelDirectory) }
        return try SortformerModel.fromModelDirectory(modelDirectory)
    }

    nonisolated private static func generate(_ samples: [Float], threshold: Float, modelDirectory: URL) async throws -> [SpeakerSegment] {
        guard !samples.isEmpty else { throw EngineError.emptyAudio }
        let model = try loadModel(modelDirectory)
        let audio = MLXArray(samples)
        try MLXErrorGuard.evaluate(audio)
        let output = try await model.generate(audio: audio, sampleRate: sampleRate, threshold: clampedThreshold(threshold))
        return output.segments.map {
            SpeakerSegment(speaker: $0.speaker, start: Double($0.start), end: Double($0.end))
        }
    }

    nonisolated private static func generateStreaming(_ samples: [Float], threshold: Float, modelDirectory: URL) async throws -> [SpeakerSegment] {
        guard !samples.isEmpty else { throw EngineError.emptyAudio }
        let model = try loadModel(modelDirectory)
        let proc = model.config.processorConfig
        let frameSeconds = Double(proc.hopLength * model.config.fcEncoderConfig.subsamplingFactor) / Double(proc.samplingRate)
        let chunks = streamingChunks(sampleCount: samples.count, chunkSamples: Int(streamChunkSeconds * Double(sampleRate)))
        var state = model.initStreamingState()
        var segments: [SpeakerSegment] = []
        for range in chunks {
            try Task.checkCancellation()
            let chunk = MLXArray(Array(samples[range]))
            try MLXErrorGuard.evaluate(chunk)
            let framesBefore = state.framesProcessed
            let (output, next) = try await model.feed(chunk: chunk, state: state, sampleRate: sampleRate,
                                                      threshold: clampedThreshold(threshold))
            state = next
            // Return MLX's freed buffers after every chunk, or they pile up between the 5 s steps (MP4-3d).
            Memory.clearCache()
            // Where the chunk really starts, minus where the library put it (see the type comment).
            let correction = Double(range.lowerBound) / Double(sampleRate) - Double(framesBefore) * frameSeconds
            segments += output.segments.map {
                SpeakerSegment(speaker: $0.speaker, start: Double($0.start) + correction, end: Double($0.end) + correction)
            }
        }
        return segments
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
