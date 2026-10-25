import Foundation

/// `audio → text` stage backed by `SortformerEngine`. Resamples the input to 16 kHz mono, diarizes, cleans the segments
/// (`SpeakerSegments.normalized`, clamped to the clip length) and returns them as CSV (`speaker,start,end`). There is no
/// speaker-segments `Media` case, and adding one would touch every switch on `Media`; the CSV is the stage's output
/// and the Run UI reads it back with `SpeakerSegments.parseCSV`. The diarizer is injectable so the stage's logic is
/// unit-testable without a model (the seam `ASRStage` uses).
nonisolated struct SortformerStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .text }

    /// 16 kHz mono float samples → the model's raw segments.
    private let diarize: @Sendable ([Float]) async throws -> [SpeakerSegment]

    init(id: String, name: String, diarize: @escaping @Sendable ([Float]) async throws -> [SpeakerSegment]) {
        self.id = id
        self.name = name
        self.diarize = diarize
    }

    /// The real engine on the installed folder of `modelID`, at `threshold` (default 0.5).
    init(modelID: String, threshold: Float = SortformerEngine.defaultThreshold) {
        let directory = SortformerEngine.installedModelDirectory(id: modelID)
        self.init(id: "sortformer.\(modelID)", name: "Sortformer") { samples in
            try await SortformerEngine.diarize(samples, threshold: threshold, modelDirectory: directory)
        }
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .audio)
        guard case let .audio(buffer) = input else {
            throw StageError.kindMismatch(expected: .audio, got: input.kind)
        }
        progress(0.1)
        let mono16k = try AudioResampler.resample(buffer, to: SortformerEngine.sampleRate)
        let duration = Double(mono16k.samples.count) / Double(SortformerEngine.sampleRate)
        let raw = try await diarize(mono16k.samples)
        let segments = SpeakerSegments.normalized(raw, duration: duration)
        progress(1.0)
        return .text(SpeakerSegments.csv(segments))
    }
}
