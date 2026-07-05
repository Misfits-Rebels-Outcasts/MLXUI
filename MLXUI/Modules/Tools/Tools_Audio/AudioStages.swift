import Foundation
import AppKit

/// Source node: reads an audio file path (`.text`) into an `.audio` buffer. Modeling
/// "Read Audio" as text(path) → audio keeps it a normal first stage in the linear chain.
nonisolated struct ReadAudioStage: PipelineStage {
    let id = "tool.read_audio"
    let name = "Read Audio"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .audio }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        guard case let .text(path) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.1)
        let buffer = try AudioFileReader.read(URL(fileURLWithPath: path))
        progress(1.0)
        return .audio(buffer)
    }
}

/// Format adapter: resamples `.audio` to `targetRate` (e.g. 16 kHz for Whisper).
nonisolated struct ResampleStage: PipelineStage {
    let id = "tool.resample"
    let name: String
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .audio }

    private let targetRate: Int

    init(targetRate: Int) {
        self.targetRate = targetRate
        self.name = "Resample \(targetRate / 1000)kHz"
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .audio)
        guard case let .audio(buffer) = input else {
            throw StageError.kindMismatch(expected: .audio, got: input.kind)
        }
        progress(0.1)
        let resampled = try AudioResampler.resample(buffer, to: targetRate)
        progress(1.0)
        return .audio(resampled)
    }
}

/// Sink node: writes `.audio` to a WAV file (passes the audio through), optionally
/// opening it in the default player (QuickTime). `openOnSave` defaults to `false` so
/// tests don't launch an app; M11's run passes `true`.
nonisolated struct SaveWAVStage: PipelineStage {
    let id = "tool.save_wav"
    let name = "Save WAV"
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .audio }

    private let url: URL
    private let openOnSave: Bool

    init(url: URL, openOnSave: Bool = false) {
        self.url = url
        self.openOnSave = openOnSave
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .audio)
        guard case let .audio(buffer) = input else {
            throw StageError.kindMismatch(expected: .audio, got: input.kind)
        }
        try AudioWriter.writeWAV(buffer, to: url)
        progress(1.0)
        if openOnSave {
            let fileURL = url
            await MainActor.run { NSWorkspace.shared.open(fileURL) }
        }
        return input
    }
}
