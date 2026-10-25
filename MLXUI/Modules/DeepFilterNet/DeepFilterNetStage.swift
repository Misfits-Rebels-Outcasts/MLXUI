import Foundation

/// `audio → audio` stage backed by `DeepFilterNetEngine`. Resamples any input to **48 kHz mono** (`AudioResampler`; not the
/// 16 kHz ASR path, readme T6), refuses a recording past the engine's limit before loading anything, enhances, and returns a
/// 48 kHz buffer whatever the input rate (owner ruling 2026-10-25). The enhancer is injectable so the stage's logic is
/// unit-testable without a model (the seam `SortformerStage` uses).
nonisolated struct DeepFilterNetStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .audio }

    /// 48 kHz mono samples → cleaned samples; the second argument reports 0...1.
    private let enhance: @Sendable ([Float], @Sendable @escaping (Double) -> Void) async throws -> [Float]

    init(id: String, name: String,
         enhance: @escaping @Sendable ([Float], @Sendable @escaping (Double) -> Void) async throws -> [Float]) {
        self.id = id
        self.name = name
        self.enhance = enhance
    }

    /// The real engine on the installed folder of `modelID`.
    init(modelID: String, chunkSeconds: Double? = DeepFilterNetEngine.defaultChunkSeconds) {
        let directory = DeepFilterNetEngine.installedModelDirectory(id: modelID)
        self.init(id: "deepfilternet.\(modelID)", name: "DeepFilterNet") { samples, progress in
            try await DeepFilterNetEngine.enhance(samples, modelDirectory: directory, chunkSeconds: chunkSeconds, progress: progress)
        }
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .audio)
        guard case let .audio(buffer) = input else {
            throw StageError.kindMismatch(expected: .audio, got: input.kind)
        }
        progress(0.05)
        let mono48k = try AudioResampler.resample(buffer, to: DeepFilterNetEngine.sampleRate)
        let duration = Double(mono48k.samples.count) / Double(DeepFilterNetEngine.sampleRate)
        try DeepFilterNetEngine.requireWithinLimit(seconds: duration)
        let cleaned = try await enhance(mono48k.samples) { fraction in progress(0.1 + 0.85 * fraction) }
        progress(1.0)
        return .audio(AudioBuffer(samples: cleaned, sampleRate: DeepFilterNetEngine.sampleRate))
    }
}
