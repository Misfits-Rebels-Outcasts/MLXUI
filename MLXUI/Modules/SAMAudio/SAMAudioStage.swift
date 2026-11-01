import Foundation

/// `audio → audio` stage backed by `SAMAudioEngine`. Resamples any input to the model's own rate (read from
/// `config.json`, 48 kHz here — not the 16 kHz ASR path), refuses a recording past the engine's 30-minute limit
/// before loading anything, isolates the described sound, and returns the **isolated** track as the stage's `Media`.
///
/// **Two tracks, one `Media`.** A sound-isolation run returns the isolated sound *and* everything else, but `Media`
/// carries one audio buffer. The stage hands the second track out through the ``SAMAudioTrackBox`` the Run view
/// creates for the run (``init(modelID:description:output:)-swift.method``); a generic pipeline that only wants audio
/// still gets the isolated track. Tests use the injectable ``init(id:name:separate:)`` seam, so the stage's logic is
/// testable without weights.
nonisolated struct SAMAudioStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .audio }

    /// Samples at the model's rate → the two tracks; the second argument reports 0...1.
    private let separate: @Sendable ([Float], @Sendable @escaping (Double) -> Void) async throws -> SAMAudioEngine.Result
    /// Where the "everything else" track goes; created per run by the Run view.
    private let output: SAMAudioTrackBox?
    private let modelRate: Int

    /// The seam tests use: an injectable separator, no weights, no folder.
    init(id: String, name: String, sampleRate: Int = SAMAudioEngine.fallbackSampleRate,
         output: SAMAudioTrackBox? = nil,
         separate: @escaping @Sendable ([Float], @Sendable @escaping (Double) -> Void) async throws -> SAMAudioEngine.Result) {
        self.id = id
        self.name = name
        self.separate = separate
        self.output = output
        self.modelRate = sampleRate
    }

    /// The real engine on an explicit folder. The app's path (`init(modelID:)`) and the owner-run tests (a temp copy of
    /// the installed folder) both go through here, so they call `SAMAudioEngine.separate` identically.
    init(modelDirectory: URL, description: String, output: SAMAudioTrackBox? = nil) {
        let rate = SAMAudioEngine.sampleRate(in: modelDirectory)
        self.init(id: "samaudio.\(modelDirectory.lastPathComponent)", name: "SAM-Audio",
                  sampleRate: rate, output: output) { samples, progress in
            try await SAMAudioEngine.separate(samples, description: description, modelDirectory: modelDirectory,
                                              progress: progress)
        }
    }

    /// The real engine on the installed folder of `modelID`.
    init(modelID: String, description: String, output: SAMAudioTrackBox? = nil) {
        self.init(modelDirectory: SAMAudioEngine.installedModelDirectory(id: modelID), description: description,
                  output: output)
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .audio)
        guard case let .audio(buffer) = input else {
            throw StageError.kindMismatch(expected: .audio, got: input.kind)
        }
        progress(0.05)
        let mono = try AudioResampler.resample(buffer, to: modelRate)
        let duration = Double(mono.samples.count) / Double(modelRate)
        try SAMAudioEngine.requireWithinLimit(seconds: duration)
        let result = try await separate(mono.samples) { fraction in progress(0.1 + 0.85 * fraction) }
        output?.store(AudioBuffer(samples: result.other, sampleRate: result.sampleRate))
        progress(1.0)
        return .audio(AudioBuffer(samples: result.isolated, sampleRate: result.sampleRate))
    }
}

/// Carries the "everything else" track out of ``SAMAudioStage``: `Media` holds one audio buffer, and a sound-isolation
/// run returns two. The Run view creates one box per run, passes it to the stage and reads it after the run; the stage
/// is `Sendable` and may run off the main actor, so access is locked.
nonisolated final class SAMAudioTrackBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: AudioBuffer?

    init() {}

    func store(_ audio: AudioBuffer) {
        lock.lock()
        value = audio
        lock.unlock()
    }

    var other: AudioBuffer? {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func clear() {
        lock.lock()
        value = nil
        lock.unlock()
    }
}
