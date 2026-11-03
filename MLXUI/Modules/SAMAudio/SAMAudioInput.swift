import Foundation
import AVFoundation

/// Reading the person's chosen file for sound isolation, with errors a person can act on. `AudioFileReader` reports a
/// file it can't decode as `StageError.engineFailure("Read Audio", <CoreAudio code>)`, which says nothing; a non-audio
/// file renamed `.wav` must give a sentence, not a code.
nonisolated enum SAMAudioInput {
    struct ReadError: Error, LocalizedError, Equatable {
        enum Reason: Equatable { case unreadable(String), empty }
        let reason: Reason
        var errorDescription: String? {
            switch reason {
            case .unreadable(let name):
                return "“\(name)” couldn't be read as audio. Choose a sound file such as WAV, MP3, M4A, AIFF or FLAC; a file that only has an audio extension won't work."
            case .empty:
                return "The audio file has no sound in it."
            }
        }
    }

    /// The file's length in seconds, from its header (no decoding), or nil when it can't be opened as audio. Lets the
    /// Run screen refuse an over-long file before the decode, and show the time estimate before a run.
    static func durationSeconds(of url: URL) -> Double? {
        guard let file = try? AVAudioFile(forReading: url), file.processingFormat.sampleRate > 0 else { return nil }
        return Double(file.length) / file.processingFormat.sampleRate
    }

    /// Decodes `url` to mono float samples at the file's own rate. Throws ``ReadError``.
    static func read(_ url: URL) throws -> AudioBuffer {
        let buffer: AudioBuffer
        do {
            buffer = try AudioFileReader.read(url)
        } catch {
            throw ReadError(reason: .unreadable(url.lastPathComponent))
        }
        guard !buffer.samples.isEmpty, buffer.sampleRate > 0 else { throw ReadError(reason: .empty) }
        return buffer
    }
}
