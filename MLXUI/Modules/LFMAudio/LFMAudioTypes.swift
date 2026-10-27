import Foundation

// MP6-3a: the value types the LFM2.5-Audio engine speaks, kept free of MLX so the engine's logic (chunking, budgets, history,
// cancellation, joining) is testable with a mock `LFMAudioBackend` and no weights.

/// One turn of the prompt the backend turns into a `ChatState`. Parts keep their order: audio then text is "this recording, and this
/// instruction".
nonisolated struct LFMPromptTurn: Sendable, Equatable {
    enum Part: Sendable, Equatable {
        case text(String)
        /// 16 kHz mono samples (``LFMAudioEngine/inputSampleRate``).
        case audio([Float])
    }
    var role: String
    var parts: [Part]

    static func system(_ text: String) -> LFMPromptTurn { LFMPromptTurn(role: "system", parts: [.text(text)]) }
    static func user(_ parts: Part...) -> LFMPromptTurn { LFMPromptTurn(role: "user", parts: parts) }
    static func assistant(_ text: String) -> LFMPromptTurn { LFMPromptTurn(role: "assistant", parts: [.text(text)]) }
}

/// The whole prompt: closed turns, then an open assistant turn (optionally pre-started in audio, which is what TTS needs).
nonisolated struct LFMPrompt: Sendable, Equatable {
    var turns: [LFMPromptTurn]
    /// `true`: the open assistant turn begins with the audio-start token, so the model goes straight to speech (TTS).
    var assistantStartsAudio = false
}

nonisolated enum LFMGenerationMode: Sendable, Equatable {
    /// Alternating text and audio (the library's `generateInterleaved`): speech-to-text and voice conversation.
    case interleaved
    /// All text first, then all audio (`generateSequential`): text-to-speech.
    case sequential
}

/// Sampling and the **per-call** token budget. A call is one STT window, one TTS text chunk or one voice turn, never a session.
nonisolated struct LFMSampling: Sendable, Equatable {
    var maxNewTokens: Int
    var temperature: Float
    var topK: Int
    var audioTemperature: Float
    var audioTopK: Int
}

/// What one backend call produced. `audioFrames` are 8-codebook frames with the end-of-audio frame already removed.
nonisolated struct LFMGenerated: Sendable, Equatable {
    var text: String
    var audioFrames: [[Int32]]
    /// Tokens the library produced for this call (text tokens plus audio frames, the end-of-audio frame included), for the
    /// measurement harness and the "hit the cap" report; 0 from a backend that doesn't count. Not used by the engine's logic.
    var tokens = 0
}

/// What the engine needs from the model. The real one wraps `LFM2AudioModel` and `LFM2AudioDetokenizer`
/// (``LFMAudioRealBackend``); tests supply a mock. Created and used inside one ``LFMAudioEngine/withBackend`` region, so it is
/// never shared between threads.
nonisolated protocol LFMAudioBackend {
    func generate(_ prompt: LFMPrompt, mode: LFMGenerationMode, sampling: LFMSampling) async throws -> LFMGenerated
    /// Frames (each `codebooks` wide) to 24 kHz mono samples.
    func decode(frames: [[Int32]]) async throws -> [Float]
}

/// Conversation memory for one voice session: the user's recordings (16 kHz) and the assistant's reply *text*, oldest first.
///
/// **RAM only, never written to disk** (backlog MP6-3). Deliberately not `Codable`, so persisting it takes a new type and a
/// decision, not a stray `JSONEncoder`. The assistant's spoken audio is not replayed into the next prompt, only its text
/// (the library's `ChatState` has no getter for output codes); implementer's call, pending owner confirmation.
/// Capped at ``maxTurns`` so a long conversation can't grow without bound.
nonisolated struct LFMConversation: Sendable, Equatable {
    struct Turn: Sendable, Equatable {
        var userSamples: [Float]
        var assistantText: String
    }

    /// Implementer's call, pending owner confirmation (unmeasured): the six most recent exchanges.
    static let maxTurns = 6

    private(set) var turns: [Turn] = []

    init() {}

    mutating func record(userSamples: [Float], assistantText: String) {
        turns.append(Turn(userSamples: userSamples, assistantText: assistantText))
        if turns.count > Self.maxTurns { turns.removeFirst(turns.count - Self.maxTurns) }
    }

    mutating func reset() { turns.removeAll() }

    /// The most seconds of earlier user audio a voice turn replays to the model (owner ruling, Boo, 2026-10-27: bound voice-chat
    /// memory with a replay budget, oldest exchanges dropped first). The new recording is counted separately. The model's prefill
    /// memory grows faster than linearly with the audio in the prompt (journal 2026-615: 413 s of audio peaked at 7.7 GiB of MLX
    /// memory, 118 s at 3.5 GiB), so this is what bounds a long chat. The value is the implementer's call from those
    /// measurements, pending owner confirmation.
    static let maxReplaySeconds = 120.0

    /// The exchanges a turn replays: the newest ones whose user audio adds up to at most `maxSeconds`, oldest dropped first and
    /// each dropped whole (never trimmed). A short chat replays everything. ``turns`` itself is untouched, so nothing is lost
    /// from the transcript or the ``maxTurns`` count; only the replay shrinks.
    static func replayed(_ turns: [Turn], maxSeconds: Double = maxReplaySeconds, sampleRate: Int = LFMAudioEngine.inputSampleRate) -> [Turn] {
        let budget = Int(maxSeconds * Double(sampleRate))
        var used = 0, kept = 0
        for turn in turns.reversed() {
            if used + turn.userSamples.count > budget { break }
            used += turn.userSamples.count
            kept += 1
        }
        return Array(turns.suffix(kept))
    }

    /// The exchanges the next turn will replay.
    var replayTurns: [Turn] { Self.replayed(turns) }
}

/// What a voice turn reports as it goes. The model's loop is synchronous (the library builds its whole result inside
/// `AsyncThrowingStream.init`), so these arrive after generation finishes, not while it runs; audio is decoded and delivered in
/// slices so playback can start on the first.
nonisolated enum LFMVoiceEvent: Sendable, Equatable {
    /// The reply text, once.
    case text(String)
    /// A slice of the spoken reply: 24 kHz mono samples.
    case audio([Float])
    /// The reply stopped at the token budget (``LFMAudioEngine/voiceMaxNewTokens``), not because the model finished. Sent right
    /// after ``text(_:)``; the Run UI shows a plain "(reply cut short)" line under the reply, never an error.
    case cutShort
}

nonisolated struct LFMVoiceReply: Sendable, Equatable {
    var text: String
    /// Total 24 kHz samples delivered through ``LFMVoiceEvent/audio(_:)``.
    var audioSampleCount: Int
}
