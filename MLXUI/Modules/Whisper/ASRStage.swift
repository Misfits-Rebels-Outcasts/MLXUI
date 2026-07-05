import Foundation

/// `audio → text` stage backed by WhisperKit (via `WhisperEngine`). Resamples the
/// input to 16 kHz mono (what Whisper expects) before transcribing. The transcriber is
/// injectable so the stage's logic is unit-testable without downloading a model — the
/// same seam `LLMStage` uses. See `Design/pipeline-stage-sketch.md` (ASRStage).
nonisolated struct ASRStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .text }

    /// 16 kHz mono float samples → transcript.
    private let transcribe: @Sendable ([Float]) async throws -> String

    /// Designated init with an injectable transcriber (tests pass a mock).
    init(
        id: String,
        name: String,
        transcribe: @escaping @Sendable ([Float]) async throws -> String
    ) {
        self.id = id
        self.name = name
        self.transcribe = transcribe
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
        // Whisper wants 16 kHz mono; resample defensively (cheap no-op if already 16k).
        let mono16k = try AudioResampler.resample(buffer, to: 16_000)
        let text = try await transcribe(mono16k.samples)
        progress(1.0)
        return .text(text)
    }
}

extension ASRStage {
    /// Stage backed by the real WhisperKit engine. `model` is a WhisperKit model name
    /// (e.g. "base", "small"); WhisperKit downloads/manages it itself — independent of the
    /// app's installed-model registry.
    init(model: String = "base") {
        self.init(id: "whisper.\(model)", name: "Whisper ASR (\(model))") { samples in
            try await WhisperEngine.transcribe(samples, model: model)
        }
    }
}
