import Foundation
import AVFoundation

/// Sample-rate conversion via `AVAudioConverter` — the format adapter that turns a
/// 44.1 kHz recording into the 16 kHz mono Whisper expects (M8). See open-pipeline
/// nodes §3a.
nonisolated enum AudioResampler {
    enum ResampleError: Error { case setupFailed, bufferAllocationFailed }

    static func resample(_ input: AudioBuffer, to targetRate: Int) throws -> AudioBuffer {
        guard targetRate > 0 else { return input }
        // No work to do (same rate, or nothing to convert).
        guard input.sampleRate != targetRate, !input.samples.isEmpty else {
            return AudioBuffer(samples: input.samples, sampleRate: targetRate)
        }

        guard
            let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: Double(input.sampleRate),
                                         channels: 1, interleaved: false),
            let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                          sampleRate: Double(targetRate),
                                          channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else {
            throw StageError.engineFailure(stage: "Resample", underlying: ResampleError.setupFailed)
        }

        guard
            let inputBuffer = AVAudioPCMBuffer(pcmFormat: inFormat,
                                               frameCapacity: AVAudioFrameCount(input.samples.count)),
            let inChannel = inputBuffer.floatChannelData
        else {
            throw StageError.engineFailure(stage: "Resample", underlying: ResampleError.bufferAllocationFailed)
        }
        inputBuffer.frameLength = AVAudioFrameCount(input.samples.count)
        for i in input.samples.indices { inChannel[0][i] = input.samples[i] }

        let ratio = Double(targetRate) / Double(input.sampleRate)
        let outCapacity = AVAudioFrameCount(Double(input.samples.count) * ratio) + 1024
        guard
            let outputBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity),
            let outChannel = outputBuffer.floatChannelData
        else {
            throw StageError.engineFailure(stage: "Resample", underlying: ResampleError.bufferAllocationFailed)
        }

        var supplied = false
        var conversionError: NSError?
        converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if supplied {
                outStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return inputBuffer
        }
        if let conversionError {
            throw StageError.engineFailure(stage: "Resample", underlying: conversionError)
        }

        let produced = Int(outputBuffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: outChannel[0], count: produced))
        return AudioBuffer(samples: samples, sampleRate: targetRate)
    }
}
