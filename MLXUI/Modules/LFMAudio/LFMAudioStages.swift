import Foundation

/// The two linear stages MP6-W will hand to workflow rows, built on the engine. (A voice turn is not a linear stage: it has
/// memory and streams, so the Run UI drives ``LFMAudioEngine/voiceTurn(userSamples:conversation:style:modelDirectory:factory:onEvent:)``
/// directly.) Both take an injectable backend factory so the stage logic is testable without weights, the seam `ASRStage` and
/// `TTSStage` already offer.
nonisolated enum LFMAudioStages {
    /// `audio -> text`. `ASRStage` resamples any input to the model's 16 kHz before the closure runs.
    static func asr(modelID: String) -> ASRStage {
        asr(modelDirectory: LFMAudioEngine.installedModelDirectory(id: modelID))
    }

    static func asr(modelDirectory: URL, factory: (@Sendable (URL) async throws -> any LFMAudioBackend)? = nil) -> ASRStage {
        ASRStage(id: "lfmaudio.asr.\(modelDirectory.lastPathComponent)", name: "LFM2.5-Audio (speech to text)") { samples in
            try await LFMAudioEngine.transcribe(samples, modelDirectory: modelDirectory, factory: factory)
        }
    }

    /// `text -> audio` at 24 kHz. `style` is free text added to the system line (the model card's example is "Use a UK male voice.");
    /// no voice list is invented.
    static func tts(modelID: String, style: String? = nil) -> TTSStage {
        tts(modelDirectory: LFMAudioEngine.installedModelDirectory(id: modelID), style: style)
    }

    static func tts(modelDirectory: URL, style: String? = nil, factory: (@Sendable (URL) async throws -> any LFMAudioBackend)? = nil) -> TTSStage {
        TTSStage(id: "lfmaudio.tts.\(modelDirectory.lastPathComponent)", name: "LFM2.5-Audio (text to speech)") { text in
            let samples = try await LFMAudioEngine.synthesize(text, style: style, modelDirectory: modelDirectory, factory: factory)
            return AudioBuffer(samples: samples, sampleRate: LFMAudioEngine.outputSampleRate)
        }
    }
}
