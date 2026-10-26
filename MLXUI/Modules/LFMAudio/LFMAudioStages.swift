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

/// One voice turn as a stage, so the Run window's `LeasingStage` can book the model's memory and ask before it pushes another
/// model out, exactly as for the other modes. `audio -> text`: the output is the reply text; the spoken reply and the text again
/// go through `onEvent` as they are produced. A turn is not a flow step (it has memory and sound out); nothing but the Run UI builds this.
nonisolated struct LFMVoiceTurnStage: PipelineStage {
    let id: String
    let name = "LFM2.5-Audio (voice chat)"
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .text }

    private let modelDirectory: URL
    private let conversation: LFMConversation
    private let style: String?
    private let onEvent: @Sendable (LFMVoiceEvent) -> Void
    private let factory: (@Sendable (URL) async throws -> any LFMAudioBackend)?

    init(modelDirectory: URL, conversation: LFMConversation, style: String? = nil,
         factory: (@Sendable (URL) async throws -> any LFMAudioBackend)? = nil,
         onEvent: @escaping @Sendable (LFMVoiceEvent) -> Void) {
        self.id = "lfmaudio.voice.\(modelDirectory.lastPathComponent)"
        self.modelDirectory = modelDirectory
        self.conversation = conversation
        self.style = style
        self.factory = factory
        self.onEvent = onEvent
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .audio)
        guard case let .audio(buffer) = input else { throw StageError.kindMismatch(expected: .audio, got: input.kind) }
        progress(0.1)
        let mono16k = try AudioResampler.resample(buffer, to: LFMAudioEngine.inputSampleRate)
        let reply = try await LFMAudioEngine.voiceTurn(userSamples: mono16k.samples, conversation: conversation, style: style,
                                                       modelDirectory: modelDirectory, factory: factory, onEvent: onEvent)
        progress(1.0)
        return .text(reply.text)
    }
}
