import Foundation
import AVFoundation

/// Reads an audio file (any format AVFoundation supports) into a mono `AudioBuffer`,
/// downmixing multi-channel input by averaging. Source node for the audio pipeline.
nonisolated enum AudioFileReader {
    static func read(_ url: URL) throws -> AudioBuffer {
        do {
            let file = try AVAudioFile(forReading: url)
            let format = file.processingFormat        // Float32, deinterleaved
            let rate = Int(format.sampleRate)
            let frames = AVAudioFrameCount(file.length)
            guard frames > 0,
                  let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
                return AudioBuffer(samples: [], sampleRate: rate)
            }
            try file.read(into: pcm)
            guard let channelData = pcm.floatChannelData else {
                return AudioBuffer(samples: [], sampleRate: rate)
            }
            let count = Int(pcm.frameLength)
            let channels = Int(format.channelCount)
            var samples = [Float](repeating: 0, count: count)
            for frame in 0..<count {
                var sum: Float = 0
                for channel in 0..<channels { sum += channelData[channel][frame] }
                samples[frame] = sum / Float(max(1, channels))
            }
            return AudioBuffer(samples: samples, sampleRate: rate)
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "Read Audio", underlying: error)
        }
    }
}
