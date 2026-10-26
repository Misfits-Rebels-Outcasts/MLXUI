import Foundation
import AppKit
import AVFoundation
import UniformTypeIdentifiers

/// View-model for the Speech to text tab: reads the chosen file (off the main actor), runs the ASR stage, shows the transcript.
/// The file input and the 16 kHz resample are the ones the other ASR views use (`AudioFileReader`, `ASRStage`). A cancelled
/// run is never an error. Copy and Save are the user's own actions.
@MainActor
@Observable
final class LFMTranscribeRunModel {
    var isRunning = false
    var transcript: String?
    var errorText: String?
    private var task: Task<Void, Never>?
    private var run = 0

    func start(audioURL: URL, stage: any PipelineStage) {
        cancel()
        run += 1
        let thisRun = run
        isRunning = true
        transcript = nil
        errorText = nil
        let accessed = audioURL.startAccessingSecurityScopedResource()
        task = Task { [weak self] in
            defer { if accessed { audioURL.stopAccessingSecurityScopedResource() } }
            do {
                let buffer = try await Task.detached { try AudioFileReader.read(audioURL) }.value
                let result = try await MLXErrorGuard.guardedRun { try await LFMRunOffMain.run(stage, .audio(buffer)) }
                guard let self, self.run == thisRun else { return }
                if case let .text(text) = result { self.transcript = text }
            } catch {
                guard let self, self.run == thisRun else { return }
                if !(error is CancellationError || Task.isCancelled) { self.errorText = MLXErrorGuard.sentence(for: error) }
            }
            self?.isRunning = false
        }
    }

    /// Immediate in the view; the model's call that is already running finishes and is dropped.
    func cancel() {
        run += 1
        task?.cancel()
        task = nil
        isRunning = false
    }

    func copy() {
        guard let transcript else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcript, forType: .string)
    }

    func save(suggestedName: String) {
        guard let transcript else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = suggestedName
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try transcript.write(to: url, atomically: true, encoding: .utf8) } catch {
            errorText = "Save failed: \(error.localizedDescription)"
        }
    }
}

/// View-model for the Text to speech tab: text in, 24 kHz audio out, played with `AVAudioPlayer` and saved as a WAV on request.
@MainActor
@Observable
final class LFMSpeakRunModel {
    var isRunning = false
    var didSynthesize = false
    var isPlaying = false
    var errorText: String?
    private var buffer: AudioBuffer?
    private var player: AVAudioPlayer?
    private var task: Task<Void, Never>?
    private var run = 0

    func synthesize(text: String, stage: any PipelineStage) {
        cancel()
        run += 1
        let thisRun = run
        isRunning = true
        didSynthesize = false
        errorText = nil
        stopPlayback()
        task = Task { [weak self] in
            do {
                let result = try await MLXErrorGuard.guardedRun { try await LFMRunOffMain.run(stage, .text(text)) }
                guard let self, self.run == thisRun else { return }
                guard case let .audio(audio) = result else { throw StageError.kindMismatch(expected: .audio, got: result.kind) }
                self.buffer = audio
                self.didSynthesize = true
                self.isRunning = false
                self.play()
            } catch {
                guard let self, self.run == thisRun else { return }
                if !(error is CancellationError || Task.isCancelled) { self.errorText = MLXErrorGuard.sentence(for: error) }
                self.isRunning = false
            }
        }
    }

    func cancel() {
        run += 1
        task?.cancel()
        task = nil
        isRunning = false
    }

    func play() {
        guard let buffer else { return }
        do {
            let player = try AVAudioPlayer(data: AudioWriter.wavData(buffer))
            self.player = player
            player.play()
            isPlaying = true
        } catch {
            errorText = "Playback failed: \(error.localizedDescription)"
        }
    }

    func stopPlayback() {
        player?.stop()
        player = nil
        isPlaying = false
    }

    func save() {
        guard let buffer else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = "speech.wav"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try AudioWriter.writeWAV(buffer, to: url) } catch { errorText = "Save failed: \(error.localizedDescription)" }
    }
}
