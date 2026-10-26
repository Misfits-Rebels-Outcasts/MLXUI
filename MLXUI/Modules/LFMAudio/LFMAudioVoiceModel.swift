import Foundation

/// View-model for the Voice chat tab. **Nothing here touches the disk**: the transcript, the user's recordings and the spoken
/// replies live in these properties and in `conversation`, and are gone when the view closes (`shutdown()`) or on "New
/// conversation". A structural test scans this file and the capture/playback files for any file or defaults API.
///
/// The turn is one hold-to-talk press. The press ends, the recording goes to the model as a stage run (so the Run window's
/// `LeasingStage` books memory), and the reply arrives as events. **The text and the audio arrive after the model's generation
/// finishes, not while it runs** (the library builds its whole result inside one synchronous call; journal 2026-609), so the view
/// shows "Thinking..." with a Stop button meanwhile, then the text, then plays the audio.
///
/// Stop is immediate in the view and the sound; the model's loop cannot be interrupted, so it runs on until its token budget or
/// its end and its result is dropped (`turn` no longer matches). A cancelled turn is never an error.
@MainActor
@Observable
final class LFMVoiceRunModel {
    enum Phase: Equatable { case idle, recording, thinking, speaking }

    struct Message: Identifiable, Equatable {
        enum Role { case user, assistant }
        let id = UUID()
        let role: Role
        /// The assistant's reply text; for the user, the label ("Voice message 0:07"): the model does not transcribe the user in a turn.
        var text: String
        /// The spoken reply (24 kHz), for the Replay button. RAM only.
        var audio: [Float] = []
    }

    private(set) var phase: Phase = .idle
    private(set) var messages: [Message] = []
    private(set) var hint: String?
    var errorText: String?

    let recorder: LFMMicRecorder
    private let player: any LFMReplyPlaying
    private(set) var conversation = LFMConversation()
    private let makeStage: @MainActor (LFMConversation, @escaping @Sendable (LFMVoiceEvent) -> Void) -> any PipelineStage

    private var turn = 0
    private var task: Task<Void, Never>?
    private var pressHeld = false

    /// Shorter than this is a tap, not speech (0.3 s).
    static let minimumSamples = LFMAudioEngine.inputSampleRate * 3 / 10

    init(recorder: LFMMicRecorder? = nil, player: (any LFMReplyPlaying)? = nil,
         makeStage: @escaping @MainActor (LFMConversation, @escaping @Sendable (LFMVoiceEvent) -> Void) -> any PipelineStage) {
        let recorder = recorder ?? LFMMicRecorder()
        let player = player ?? LFMReplyPlayer()
        self.recorder = recorder
        self.player = player
        self.makeStage = makeStage
        player.onDrained = { [weak self] in
            guard let self, self.phase == .speaking else { return }
            self.phase = .idle
        }
        recorder.onAutoStop = { [weak self] samples in
            guard let self else { return }
            self.pressHeld = false
            self.send(samples)
        }
    }

    var permissionDenied: Bool { recorder.permissionDenied }
    var canHold: Bool { phase == .idle || phase == .speaking || phase == .recording }

    // MARK: Hold to talk

    /// The press began. Synchronous, so a release right behind it is seen (`pressHeld`); the permission prompt and the capture
    /// start run in a task.
    func pressBegan() {
        guard canHold, !pressHeld else { return }
        pressHeld = true
        hint = nil
        errorText = nil
        // A new hold ends the previous reply's sound first, so the microphone never hears the speakers.
        player.stop()
        if phase == .speaking { phase = .idle }
        Task { [weak self] in
            guard let self else { return }
            let result = await self.recorder.begin(stillHeld: { self.pressHeld })
            switch result {
            case .started:
                if self.pressHeld { self.phase = .recording } else { self.finishCapture(send: true) }   // released while starting
            case .pressEnded: break
            case .denied: break   // the view shows the plain message; not an error
            case .failed(let sentence): self.errorText = sentence
            }
        }
    }

    /// The press ended: send what was captured.
    func pressEnded() {
        pressHeld = false
        finishCapture(send: true)
    }

    /// The press was interrupted (the app lost focus, the view closed): drop the recording.
    func pressCancelled() {
        pressHeld = false
        finishCapture(send: false)
    }

    private func finishCapture(send shouldSend: Bool) {
        guard recorder.isRecording else { return }
        let samples = recorder.end() ?? []
        if phase == .recording { phase = .idle }
        if shouldSend { send(samples) }
    }

    // MARK: Turn

    func send(_ samples: [Float]) {
        if phase == .recording { phase = .idle }
        guard samples.count >= Self.minimumSamples else {
            hint = "That was too short. Hold the button while you speak, and let go when you are done."
            return
        }
        player.stop()
        turn += 1
        let thisTurn = turn
        let seconds = Int((Double(samples.count) / Double(LFMAudioEngine.inputSampleRate)).rounded())
        let userMessage = Message(role: .user, text: "Voice message \(seconds / 60):" + String(format: "%02d", seconds % 60))
        messages.append(userMessage)
        phase = .thinking

        let (stream, continuation) = AsyncStream.makeStream(of: LFMVoiceEvent.self)
        let stage = makeStage(conversation) { continuation.yield($0) }
        let input = Media.audio(AudioBuffer(samples: samples, sampleRate: LFMAudioEngine.inputSampleRate))
        let consumer = Task { [weak self] in
            for await event in stream {
                guard let self, self.turn == thisTurn else { return }
                self.handle(event)
            }
        }
        task = Task { [weak self] in
            do {
                let output = try await MLXErrorGuard.guardedRun { try await LFMRunOffMain.run(stage, input) }
                continuation.finish()
                await consumer.value
                guard let self, self.turn == thisTurn else { return }
                guard case let .text(text) = output else { throw StageError.kindMismatch(expected: .text, got: output.kind) }
                self.conversation.record(userSamples: samples, assistantText: text)
                self.player.finish()   // drained -> idle (immediately if the reply had no sound)
                if self.phase == .thinking { self.phase = .idle }   // no text event arrived; nothing to speak
            } catch {
                continuation.finish()
                consumer.cancel()
                guard let self, self.turn == thisTurn else { return }
                // A cancelled turn is never an error (and records nothing): drop the pending message.
                if error is CancellationError || Task.isCancelled {
                    self.discardPending(userMessage.id)
                } else {
                    self.errorText = MLXErrorGuard.sentence(for: error)
                    self.discardPending(userMessage.id)
                }
                self.player.stop()
                self.phase = .idle
            }
        }
    }

    private func handle(_ event: LFMVoiceEvent) {
        switch event {
        case .text(let text):
            messages.append(Message(role: .assistant, text: text))
            phase = .speaking
        case .audio(let samples):
            if phase == .thinking { phase = .speaking }
            if let index = messages.indices.last, messages[index].role == .assistant { messages[index].audio += samples }
            player.append(samples)
        }
    }

    private func discardPending(_ id: UUID) {
        messages.removeAll { $0.id == id }
    }

    /// Stop while the model is thinking or its reply is playing. Immediate in the view; the model's loop runs on to its budget
    /// and its result is dropped.
    func stop() {
        turn += 1
        task?.cancel()
        task = nil
        player.stop()
        if let last = messages.last, last.role == .user { messages.removeLast() }
        phase = .idle
    }

    func replay(_ message: Message) {
        guard phase == .idle, !message.audio.isEmpty else { return }
        player.stop()
        player.append(message.audio)
        player.finish()
    }

    /// Clears the memory and the transcript. Nothing was ever on disk.
    func newConversation() {
        stop()
        recorder.cancel()
        pressHeld = false
        conversation.reset()
        messages = []
        hint = nil
        errorText = nil
    }

    /// The view is closing: stop sound and capture, drop the memory.
    func shutdown() {
        newConversation()
    }
}

/// Runs a stage off the main actor. `SWIFT_APPROACHABLE_CONCURRENCY` makes a plain `nonisolated async` call run on its caller's
/// actor, and the run models' tasks are main-actor; `@concurrent` is a child hop (the MP5-3g rule), so the `MLXErrorGuard`
/// task-local and cancellation carry across.
nonisolated enum LFMRunOffMain {
    @concurrent static func run(_ stage: any PipelineStage, _ input: Media,
                                progress: @Sendable @escaping (Double) -> Void = { _ in }) async throws -> Media {
        try await stage.run(input, progress: progress)
    }
}
