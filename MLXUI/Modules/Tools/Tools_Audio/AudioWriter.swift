import Foundation

/// Writes mono `AudioBuffer` samples as a canonical 16-bit PCM WAV. Kokoro TTS
/// produces 24 kHz audio (M10); this is the sink that makes it playable (open in
/// QuickTime). Pure byte assembly so the header is unit-testable. See open-pipeline
/// nodes §3a (format adapters).
nonisolated enum AudioWriter {
    /// Build the full WAV byte stream (44-byte header + 16-bit little-endian samples).
    static func wavData(_ buffer: AudioBuffer) -> Data {
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let sampleRate = UInt32(max(0, buffer.sampleRate))
        let bytesPerSample = UInt32(bitsPerSample / 8)
        let byteRate = sampleRate * UInt32(channels) * bytesPerSample
        let blockAlign = channels * UInt16(bytesPerSample)
        let dataBytes = UInt32(buffer.samples.count) * bytesPerSample

        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) {
            var le = value.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36) + dataBytes)          // RIFF chunk size
        data.append(contentsOf: Array("WAVE".utf8))

        data.append(contentsOf: Array("fmt ".utf8))
        append(UInt32(16))                       // fmt chunk size (PCM)
        append(UInt16(1))                        // audio format = PCM
        append(channels)
        append(sampleRate)
        append(byteRate)
        append(blockAlign)
        append(bitsPerSample)

        data.append(contentsOf: Array("data".utf8))
        append(dataBytes)
        for sample in buffer.samples {
            let clamped = max(-1.0, min(1.0, sample))
            append(Int16(clamped * Float(Int16.max)))
        }
        return data
    }

    /// Write the WAV to `url`. Throws `StageError.engineFailure` on I/O error.
    static func writeWAV(_ buffer: AudioBuffer, to url: URL) throws {
        do {
            try wavData(buffer).write(to: url)
        } catch {
            throw StageError.engineFailure(stage: "Save WAV", underlying: error)
        }
    }
}
