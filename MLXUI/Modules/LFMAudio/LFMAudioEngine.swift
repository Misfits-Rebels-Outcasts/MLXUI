import Foundation

/// LFM2.5-Audio 1.5B (6-bit) through mlx-audio-swift's `MLXAudioSTS/LFMAudio`: speech-to-text, text-to-speech and a voice turn
/// (speech in, text and speech out). Loads the **installed** directory (`LFM2AudioModel.fromModelDirectory`, never
/// `fromPretrained`, which would download behind the app's back; KICKOFF rule 6).
///
/// **Sample rates (settled from the pinned source @ 5c4c12b, MP6-3a).** Input is **16 kHz**: the pinned `config.json`'s
/// `preprocessor.sample_rate` is 16000, and `LFM2AudioProcessor.preprocessAudio(_:sampleRate:)` *ignores* its `sampleRate`
/// argument, so the caller must hand it 16 kHz samples (the model card's "24kHz input" is its Python example's variable, not the
/// model's). Output is **24 kHz**: `LFM2AudioConfig.sampleRate` defaults to 24000 and the config does not set it, and the
/// detokenizer's waveform is that rate. Tests: `LFMAudioCatalogTests`-style `LFMAudioEngineTests.inputRateIsSixteenKHz...`.
///
/// **No root tokenizer checkpoint needed (readme T7).** The loader reads every top-level `*.safetensors` whose name does not
/// contain `tokenizer`, and the detokenizer reads `audio_detokenizer/` by exact name, so the selector's skip of
/// `tokenizer-e351c8d8-checkpoint125.safetensors` costs nothing. The real-load tests assert the file is absent and run all three
/// modes.
///
/// **Budgets are per call, never per session (backlog MP6-3a).** The library default `maxNewTokens` is 512 for a whole call.
/// Here a call is one STT window (``sttWindowSeconds``), one TTS text chunk (``ttsChunkCharacters``) or one voice turn, each with its own
/// ``LFMSampling/maxNewTokens``.
///
/// **Threading (KICKOFF rule 17).** Load, generate, decode and every read run inside one `MLXAudioThread.run` region (the Kokoro
/// precedent), so no array built on one thread is evaluated on another and nothing runs on the main actor. Every library array
/// is read through `MLXErrorGuard` (17a). Why the pinned thread and not only `@concurrent`: `generateInterleaved` yields arrays
/// that are results of the model's own graph, and on macOS 15+ the pinned executor keeps the iteration's resumptions on the
/// thread that built them (17b); on macOS 14 `MLXAudioThread.run` runs the body inline, as for Kokoro.
///
/// **Cancellation.** Checked before the load and before every window / chunk / decode slice. The library's generation loop is
/// synchronous and has no cancellation point, so a cancel during one call takes effect when that call returns (at most
/// ``LFMSampling/maxNewTokens`` tokens); its result is then dropped. A cancelled run throws `CancellationError`, which the stage
/// layer does not treat as a failure.
nonisolated enum LFMAudioEngine {
    /// Model input rate, Hz (see the type comment).
    static let inputSampleRate = 16_000
    /// Model output rate, Hz.
    static let outputSampleRate = 24_000

    // MARK: Budgets and sampling (implementer's call, pending owner confirmation; unmeasured until MP6-3c)

    /// STT: one window of audio per call. The library has no documented limit; 30 s keeps each call's prefill small.
    static let sttWindowSeconds = 30.0
    /// STT tokens per window (the library default for a whole call, applied per window). 30 s of speech is far under it.
    static let sttMaxNewTokens = 512
    /// Greedy for transcription (a transcript should not vary run to run).
    static let sttSampling = LFMSampling(maxNewTokens: sttMaxNewTokens, temperature: 0, topK: 0, audioTemperature: 1, audioTopK: 4)

    /// TTS: characters per call; the text is split at paragraph and sentence boundaries (`KokoroEngine.textSegments` / `subChunks`).
    static let ttsChunkCharacters = 300
    /// TTS frames per chunk. A 300-character chunk is about 20 s of speech; the frame rate is 12.5 per second, so 1,024 frames is
    /// about 80 s, a ceiling that a runaway generation cannot pass.
    static let ttsMaxNewTokens = 1_024
    /// The README's audio sampling for the sequential (TTS) example.
    static let ttsSampling = LFMSampling(maxNewTokens: ttsMaxNewTokens, temperature: 0.9, topK: 50, audioTemperature: 0.8, audioTopK: 4)
    static let ttsInstruction = "Perform TTS."

    /// Voice turn: text and audio tokens together, per turn (the README example uses 2,048; about 80 s of speech at the cap).
    static let voiceMaxNewTokens = 1_024
    static let voiceSampling = LFMSampling(maxNewTokens: voiceMaxNewTokens, temperature: 0.9, topK: 50, audioTemperature: 0.8, audioTopK: 4)
    static let voiceInstruction = "Respond with interleaved text and speech audio."
    /// Longest single user recording a voice turn accepts. Provisional: unmeasured (the owner's rule is to measure before a limit
    /// is final, MP6-3c). It exists so a stuck push-to-talk can't prefill an hour of audio.
    static let maxUserTurnSeconds = 120.0
    /// Frames per delivered audio slice in a voice turn (12.5 frames per second, so 2 s). The upstream streaming example decodes
    /// every 10 frames; slice seams are unverified by ear until the MP6-3b smoke.
    static let voiceSliceFrames = 25
    static let sttInstruction = "Transcribe the audio."

    enum EngineError: Error, LocalizedError, Equatable {
        case emptyAudio
        case emptyText
        case noAudio
        case noWeights(String)
        case unexpectedRates(input: Int, output: Int)
        var errorDescription: String? {
            switch self {
            case .emptyAudio: return "The recording has no samples."
            case .emptyText: return "There is no text to speak."
            case .noAudio: return "The model returned no speech for that text."
            case .noWeights(let name):
                return "The LFM2.5-Audio model files are missing from \(name) (config.json, model.safetensors and audio_detokenizer/). Re-install the model."
            case .unexpectedRates(let input, let output):
                return "This LFM2.5-Audio checkpoint expects \(input) Hz in and \(output) Hz out, not the 16 kHz / 24 kHz this app supports."
            }
        }
    }

    /// A recording longer than ``maxUserTurnSeconds``. A `LocalizedError`, so `MLXErrorGuard.sentence(for:)` shows it as written.
    struct LimitError: Error, LocalizedError, Equatable {
        let seconds: Double
        let limit: Double
        var errorDescription: String? {
            "That recording is \(Int(seconds.rounded())) seconds long. A voice turn handles up to \(Int(limit)) seconds; send a shorter one."
        }
    }

    // MARK: - Pure helpers

    /// Consecutive windows of `windowSamples` over `sampleCount` samples (the last may be shorter). Empty for no samples.
    static func windows(sampleCount: Int, windowSamples: Int) -> [Range<Int>] {
        guard sampleCount > 0, windowSamples > 0 else { return [] }
        return stride(from: 0, to: sampleCount, by: windowSamples).map { $0 ..< min($0 + windowSamples, sampleCount) }
    }

    /// The text the model hears chunk by chunk: paragraph segments, each split at sentence boundaries.
    static func textChunks(_ text: String, maxCharacters: Int = ttsChunkCharacters) -> [String] {
        KokoroEngine.textSegments(text).flatMap { KokoroEngine.subChunks($0, maxLength: maxCharacters) }
    }

    static func joinTranscripts(_ pieces: [String]) -> String {
        pieces.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.joined(separator: " ")
    }

    static func requireWithinTurnLimit(sampleCount: Int, limit: Double = maxUserTurnSeconds) throws {
        let seconds = Double(sampleCount) / Double(inputSampleRate)
        if seconds > limit { throw LimitError(seconds: seconds, limit: limit) }
    }

    static func sttPrompt(_ window: [Float]) -> LFMPrompt {
        LFMPrompt(turns: [.user(.audio(window), .text(sttInstruction))])
    }

    static func ttsPrompt(_ text: String, style: String?) -> LFMPrompt {
        let trimmed = style?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let system = trimmed.isEmpty ? ttsInstruction : ttsInstruction + " " + trimmed
        return LFMPrompt(turns: [.system(system), .user(.text(text))], assistantStartsAudio: true)
    }

    /// The history, replayed as user recording then assistant text, and the new recording last.
    static func voicePrompt(userSamples: [Float], conversation: LFMConversation, style: String?) -> LFMPrompt {
        let trimmed = style?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let system = trimmed.isEmpty ? voiceInstruction : voiceInstruction + " " + trimmed
        var turns: [LFMPromptTurn] = [.system(system)]
        for turn in conversation.turns {
            turns.append(.user(.audio(turn.userSamples)))
            turns.append(.assistant(turn.assistantText))
        }
        turns.append(.user(.audio(userSamples)))
        return LFMPrompt(turns: turns)
    }

    // MARK: - The three modes, on a backend

    /// Speech to text: 16 kHz mono `samples`, in windows, each with its own token budget; the pieces are joined with a space.
    static func transcribe(_ samples: [Float], backend: any LFMAudioBackend, windowSeconds: Double = sttWindowSeconds,
                           sampling: LFMSampling = sttSampling,
                           progress: @Sendable (Double) -> Void = { _ in }) async throws -> String {
        guard !samples.isEmpty else { throw EngineError.emptyAudio }
        let plan = windows(sampleCount: samples.count, windowSamples: max(Int(windowSeconds * Double(inputSampleRate)), 1))
        var pieces: [String] = []
        for (index, window) in plan.enumerated() {
            try Task.checkCancellation()
            let result = try await backend.generate(sttPrompt(Array(samples[window])), mode: .interleaved, sampling: sampling)
            try Task.checkCancellation()
            pieces.append(result.text)
            progress(Double(index + 1) / Double(plan.count))
        }
        return joinTranscripts(pieces)
    }

    /// Text to speech: 24 kHz mono samples for `text`, chunk by chunk, each with its own token budget.
    static func synthesize(_ text: String, style: String? = nil, backend: any LFMAudioBackend,
                           sampling: LFMSampling = ttsSampling, maxCharacters: Int = ttsChunkCharacters,
                           progress: @Sendable (Double) -> Void = { _ in }) async throws -> [Float] {
        let chunks = textChunks(text, maxCharacters: maxCharacters)
        guard !chunks.isEmpty else { throw EngineError.emptyText }
        var samples: [Float] = []
        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            let result = try await backend.generate(ttsPrompt(chunk, style: style), mode: .sequential, sampling: sampling)
            guard !result.audioFrames.isEmpty else { throw EngineError.noAudio }
            try Task.checkCancellation()
            samples += try await backend.decode(frames: result.audioFrames)
            progress(Double(index + 1) / Double(chunks.count))
        }
        guard !samples.isEmpty else { throw EngineError.noAudio }
        return samples
    }

    /// One voice turn: the user's 16 kHz recording plus the session's memory in, the reply out. `onEvent` gets the text, then
    /// the audio in slices of ``voiceSliceFrames`` frames. The caller records the exchange with
    /// `LFMConversation.record(userSamples:assistantText:)` once the turn succeeded (a failed or cancelled turn leaves the memory
    /// untouched).
    static func voiceTurn(userSamples: [Float], conversation: LFMConversation, style: String? = nil, backend: any LFMAudioBackend,
                          sampling: LFMSampling = voiceSampling, sliceFrames: Int = voiceSliceFrames,
                          onEvent: @Sendable (LFMVoiceEvent) -> Void) async throws -> LFMVoiceReply {
        guard !userSamples.isEmpty else { throw EngineError.emptyAudio }
        try requireWithinTurnLimit(sampleCount: userSamples.count)
        try Task.checkCancellation()
        let result = try await backend.generate(voicePrompt(userSamples: userSamples, conversation: conversation, style: style),
                                                mode: .interleaved, sampling: sampling)
        try Task.checkCancellation()
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        onEvent(.text(text))
        var delivered = 0
        let size = max(sliceFrames, 1)
        for start in stride(from: 0, to: result.audioFrames.count, by: size) {
            try Task.checkCancellation()
            let slice = Array(result.audioFrames[start ..< min(start + size, result.audioFrames.count)])
            let samples = try await backend.decode(frames: slice)
            delivered += samples.count
            onEvent(.audio(samples))
        }
        return LFMVoiceReply(text: text, audioSampleCount: delivered)
    }

    // MARK: - Entry points on the installed model

    /// Runs `body` with the real backend for `modelDirectory`, or `factory`'s (tests), inside the audio thread.
    static func withBackend<T: Sendable>(_ modelDirectory: URL,
                                         factory: @escaping @Sendable (URL) async throws -> any LFMAudioBackend = { try await LFMAudioRealBackend.load($0) },
                                         _ body: @escaping @Sendable (any LFMAudioBackend) async throws -> T) async throws -> T {
        do {
            return try await MLXAudioThread.run {
                try Task.checkCancellation()
                let backend = try await factory(modelDirectory)
                return try await body(backend)
            }
        } catch let error as StageError {
            throw error
        } catch let error as EngineError {
            throw error
        } catch let error as LimitError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw StageError.engineFailure(stage: "LFM2.5-Audio", underlying: error)
        }
    }

    /// 16 kHz mono samples to text.
    static func transcribe(_ samples: [Float], modelDirectory: URL,
                           factory: (@Sendable (URL) async throws -> any LFMAudioBackend)? = nil,
                           progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> String {
        guard !samples.isEmpty else { throw EngineError.emptyAudio }
        return try await run(modelDirectory, factory) { backend in
            try await transcribe(samples, backend: backend, progress: progress)
        }
    }

    /// Text to 24 kHz mono samples.
    static func synthesize(_ text: String, style: String? = nil, modelDirectory: URL,
                           factory: (@Sendable (URL) async throws -> any LFMAudioBackend)? = nil,
                           progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> [Float] {
        guard !textChunks(text).isEmpty else { throw EngineError.emptyText }
        return try await run(modelDirectory, factory) { backend in
            try await synthesize(text, style: style, backend: backend, progress: progress)
        }
    }

    /// One voice turn on the installed model. Loads the model per call (as the other engines); a cached session model is a
    /// later optimisation (journal 2026-609).
    static func voiceTurn(userSamples: [Float], conversation: LFMConversation, style: String? = nil, modelDirectory: URL,
                          factory: (@Sendable (URL) async throws -> any LFMAudioBackend)? = nil,
                          onEvent: @escaping @Sendable (LFMVoiceEvent) -> Void) async throws -> LFMVoiceReply {
        guard !userSamples.isEmpty else { throw EngineError.emptyAudio }
        try requireWithinTurnLimit(sampleCount: userSamples.count)
        return try await run(modelDirectory, factory) { backend in
            try await voiceTurn(userSamples: userSamples, conversation: conversation, style: style, backend: backend, onEvent: onEvent)
        }
    }

    private static func run<T: Sendable>(_ directory: URL, _ factory: (@Sendable (URL) async throws -> any LFMAudioBackend)?,
                                         _ body: @escaping @Sendable (any LFMAudioBackend) async throws -> T) async throws -> T {
        if let factory { return try await withBackend(directory, factory: factory, body) }
        return try await withBackend(directory, body)
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout.
    static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
