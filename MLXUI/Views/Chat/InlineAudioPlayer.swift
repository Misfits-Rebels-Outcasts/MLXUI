//
//  InlineAudioPlayer.swift
//  MLXUI — Model Tools, MT4-4 (an audio output plays inline in the transcript).
//
//  Play / pause, a scrub bar and the time, over `AVAudioPlayer` (the same engine the TTS run view plays through). Only
//  one inline player plays at a time. No Combine: position is polled by a task that lives while the audio plays.
//

import SwiftUI
import AVFoundation

@MainActor @Observable
final class InlineAudioPlayerModel {
    private(set) var isPlaying = false
    private(set) var position: TimeInterval = 0
    private(set) var duration: TimeInterval = 0
    private(set) var failure: String?

    @ObservationIgnored private var player: AVAudioPlayer?
    /// The player that is playing now; starting another one stops it.
    @ObservationIgnored private static weak var current: InlineAudioPlayerModel?

    let url: URL

    init(url: URL) {
        self.url = url
        load()
    }

    private func load() {
        guard FileManager.default.fileExists(atPath: url.path) else {
            failure = "This audio file is no longer on disk."
            return
        }
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.prepareToPlay()
            self.player = player
            duration = player.duration
        } catch {
            failure = "This audio can't be played: \(error.localizedDescription)"
        }
    }

    func toggle() {
        guard let player else { return }
        if isPlaying {
            player.pause()
            isPlaying = false
            return
        }
        if let other = Self.current, other !== self { other.pause() }
        Self.current = self
        // At the end, play from the start again.
        if player.currentTime >= player.duration - 0.05 { player.currentTime = 0 }
        isPlaying = player.play()
    }

    func pause() {
        player?.pause()
        isPlaying = false
    }

    func seek(to time: TimeInterval) {
        guard let player else { return }
        player.currentTime = min(max(0, time), player.duration)
        position = player.currentTime
    }

    /// Runs while `isPlaying`; ends the playing state when the audio reaches its end.
    func follow() async {
        while isPlaying, !Task.isCancelled {
            try? await Task.sleep(for: .milliseconds(150))
            guard let player else { return }
            position = player.currentTime
            if !player.isPlaying {
                isPlaying = false
                if player.currentTime >= player.duration - 0.2 { position = 0; player.currentTime = 0 }
            }
        }
    }

    /// `0:06`.
    static func clock(_ seconds: TimeInterval) -> String {
        let total = Int(max(0, seconds).rounded(.down))
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

struct InlineAudioPlayer: View {
    @State private var model: InlineAudioPlayerModel

    init(url: URL) {
        _model = State(initialValue: InlineAudioPlayerModel(url: url))
    }

    var body: some View {
        if let failure = model.failure {
            Label(failure, systemImage: "speaker.slash")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            HStack(spacing: 8) {
                Button {
                    model.toggle()
                } label: {
                    Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                        .frame(width: 16)
                }
                .buttonStyle(.borderless)
                .help(model.isPlaying ? "Pause" : "Play")
                Slider(
                    value: Binding(get: { model.position }, set: { model.seek(to: $0) }),
                    in: 0...max(model.duration, 0.1))
                    .controlSize(.small)
                Text("\(InlineAudioPlayerModel.clock(model.position)) / \(InlineAudioPlayerModel.clock(model.duration))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .task(id: model.isPlaying) { await model.follow() }
            .frame(maxWidth: 360)
        }
    }
}
