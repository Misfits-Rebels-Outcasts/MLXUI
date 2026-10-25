import Foundation
import MLX
import MLXAudioVAD

/// MLX-native speaker diarization via mlx-audio-swift's `SortformerModel` (`MLXAudioVAD`). Takes the whole 16 kHz mono
/// buffer and returns who spoke when. Loads the **installed** directory (the A2 pattern, `fromModelDirectory`, never
/// `fromPretrained`, which would download behind the app's back).
///
/// **Thread-bound streams (KICKOFF rule 17; journals 2026-557/558).** `Sortformer.swift:1318-1319` @ 580e952
/// (`embs[0..., topIndices, 0...]`, `preds[0..., topIndices, 0...]`) is the array-index-after-another-index gather. It is in
/// `compressSpkcacheSimple`, the *streaming* speaker-cache compression (`feed` / `generateStream`). This engine uses the
/// **offline** `generate(audio:threshold:)`, which never calls it: one forward pass, `predsToSegments` (a plain bulk
/// read of the evaluated predictions inside the library), plain `DiarizationSegment` values out. So:
///   - the input buffer is evaluated up front with `MLXErrorGuard.evaluate`, so a failed evaluation throws before the
///     library touches it (rule 17b);
///   - **no library array is read here**: the output's `speakerProbs` (an `MLXArray`) is deliberately never touched, only
///     the segments, which are Swift values. If a later task reads `speakerProbs` it must go through
///     `MLXErrorGuard.read(_:as:)`, never a bare `asArray` (rule 17a);
///   - `generate` runs its whole body in the library's own `Task.detached`, with no suspension point inside, so every array
///     it builds is built and evaluated on one thread. That is also why `MLXAudioThread.run` would not change anything for
///     this call: the work hops to a detached task whatever executor awaits it. The switch exists anyway, as the one
///     constant `usesAudioThread` (default `false`); flip it only if the owner-run main-actor probe
///     (`SortformerMainActorProbeTests`) reports a stream error. The task-local `withError` handler does not reach the
///     detached task either; an MLX error there lands in the process-wide fallback, and the Run UI's
///     `MLXErrorGuard.guardedRun` turns that into a thrown error.
///
/// **Long audio (UNMEASURED).** The offline path is a single pass over the whole clip: no chunking, no windowing. The
/// FastConformer's relative-position self-attention grows with the square of the frame count (one frame per 80 ms), so
/// memory grows faster than linearly with clip length. That is reasoning from the architecture, not a measurement; no peak
/// has been recorded for any clip yet (`SortformerRealLoadTests` and `SortformerLongClipMeasurement` record one on the
/// owner's machine). The library also has a chunked streaming API (`generateStream`) that bounds the working set but is the
/// path that contains the gather above; it is not used. No clip-length limit is imposed here until a measurement exists.
///
/// **Library behaviour worth knowing** (read from the source, not measured): it trims leading and trailing silence before
/// the pass and adds the offset back, peak-normalises the waveform, and returns segments per speaker slot (0 to 3) merged
/// across no gaps (`minDuration` 0, `mergeGap` 0, the library defaults, which this engine leaves alone).
enum SortformerEngine {
    nonisolated static let sampleRate = 16_000
    nonisolated static let defaultThreshold: Float = 0.5
    nonisolated static let thresholdRange: ClosedRange<Float> = 0.05...0.95

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

    /// A threshold the model can use: NaN falls back to the default; anything else is clamped to ``thresholdRange``.
    nonisolated static func clampedThreshold(_ value: Float) -> Float {
        guard value.isFinite else { return defaultThreshold }
        return min(max(value, thresholdRange.lowerBound), thresholdRange.upperBound)
    }

    /// Diarize 16 kHz mono float samples using the installed model directory. Loads the model per call (matches the other
    /// engines). Returns the library's segments as they come (sorted by start there); `SpeakerSegments.normalized`
    /// is the caller's clean-up.
    nonisolated static func diarize(_ samples: [Float], threshold: Float = defaultThreshold, modelDirectory: URL) async throws -> [SpeakerSegment] {
        do {
            let segments = try await perform { try await generate(samples, threshold: threshold, modelDirectory: modelDirectory) }
            // As ParakeetEngine / Qwen3ASREngine: return MLX's buffer cache once the result is read, after the model (a
            // local of `generate`) is released, so its weights don't land in the cache on return.
            Memory.clearCache()
            return segments
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "Sortformer", underlying: error)
        }
    }

    nonisolated private static func perform(_ body: @escaping @Sendable () async throws -> [SpeakerSegment]) async throws -> [SpeakerSegment] {
        usesAudioThread ? try await MLXAudioThread.run(body) : try await body()
    }

    nonisolated private static func generate(_ samples: [Float], threshold: Float, modelDirectory: URL) async throws -> [SpeakerSegment] {
        guard !samples.isEmpty else { throw EngineError.emptyAudio }
        // The library's loader applies whatever weights it finds and does not fail on missing ones: a folder with no
        // safetensors would run a randomly initialised model and return noise. Refuse it instead (found by the synthetic test).
        let files = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        guard files.contains(where: { $0.pathExtension == "safetensors" }) else { throw EngineError.noWeights(modelDirectory) }
        let model = try SortformerModel.fromModelDirectory(modelDirectory)
        let audio = MLXArray(samples)
        try MLXErrorGuard.evaluate(audio)
        let output = try await model.generate(audio: audio, sampleRate: sampleRate, threshold: clampedThreshold(threshold))
        return output.segments.map {
            SpeakerSegment(speaker: $0.speaker, start: Double($0.start), end: Double($0.end))
        }
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
