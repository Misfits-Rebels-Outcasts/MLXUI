import Testing
import Foundation
@testable import MLXUI

/// Covers M5 Tools_Audio: WAV header correctness, file read→write round-trip, the
/// resampler, and the Read/Resample/Save stages. See RSI/evals/eval-plan.md G2.
struct ToolsAudioTests {

    // MARK: - AudioWriter header correctness

    @Test func wavHeaderHasCorrectMagicAndFormatFields() throws {
        let buffer = AudioBuffer(samples: [0, 0.5, -0.5, 1.0], sampleRate: 24_000)
        let data = AudioWriter.wavData(buffer)

        // 44-byte header + 2 bytes per sample.
        #expect(data.count == 44 + buffer.samples.count * 2)
        #expect(ascii(data, 0, 4) == "RIFF")
        #expect(ascii(data, 8, 4) == "WAVE")
        #expect(ascii(data, 12, 4) == "fmt ")
        #expect(ascii(data, 36, 4) == "data")

        #expect(u16(data, 20) == 1)                 // PCM
        #expect(u16(data, 22) == 1)                 // mono
        #expect(u32(data, 24) == 24_000)            // sample rate
        #expect(u16(data, 34) == 16)                // bits per sample
        #expect(u32(data, 40) == UInt32(buffer.samples.count * 2))   // data chunk size
        #expect(u32(data, 4) == UInt32(36 + buffer.samples.count * 2)) // RIFF size
    }

    @Test func wavClampsOutOfRangeSamples() throws {
        let data = AudioWriter.wavData(AudioBuffer(samples: [2.0, -2.0], sampleRate: 16_000))
        // +full scale and -full scale (Int16.max / -Int16.max), little-endian.
        #expect(i16(data, 44) == Int16.max)
        #expect(i16(data, 46) == -Int16.max)
    }

    // MARK: - read → write round-trip

    @Test func writeThenReadRoundTripsSamples() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rt-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let original = AudioBuffer(samples: [0, 0.25, -0.25, 0.5, -0.5], sampleRate: 16_000)
        try AudioWriter.writeWAV(original, to: url)
        let read = try AudioFileReader.read(url)

        #expect(read.sampleRate == 16_000)
        #expect(read.samples.count == original.samples.count)
        for (a, b) in zip(original.samples, read.samples) {
            #expect(abs(a - b) < 1e-3)   // 16-bit quantization tolerance
        }
    }

    // MARK: - AudioResampler

    @Test func resampleToSameRateIsIdentity() throws {
        let buffer = AudioBuffer(samples: [0.1, 0.2, 0.3], sampleRate: 16_000)
        let out = try AudioResampler.resample(buffer, to: 16_000)
        #expect(out.sampleRate == 16_000)
        #expect(out.samples == buffer.samples)
    }

    @Test func resampleDownChangesRateAndShrinksCount() throws {
        // 1 second of 32 kHz → ~16 kHz should roughly halve the frame count.
        let buffer = AudioBuffer(samples: [Float](repeating: 0, count: 32_000), sampleRate: 32_000)
        let out = try AudioResampler.resample(buffer, to: 16_000)
        #expect(out.sampleRate == 16_000)
        #expect(out.samples.count > 14_000 && out.samples.count < 18_000)
    }

    // MARK: - Stages

    @Test func readAudioStageReadsFileToAudio() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("read-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioWriter.writeWAV(AudioBuffer(samples: [0.1, -0.1, 0.2], sampleRate: 16_000), to: url)

        let out = try await ReadAudioStage().run(.text(url.path)) { _ in }
        guard case let .audio(buffer) = out else { Issue.record("expected audio"); return }
        #expect(buffer.sampleRate == 16_000)
        #expect(buffer.samples.count == 3)
    }

    @Test func resampleStageConvertsRate() async throws {
        let input = Media.audio(AudioBuffer(samples: [Float](repeating: 0, count: 8_000),
                                            sampleRate: 8_000))
        let out = try await ResampleStage(targetRate: 16_000).run(input) { _ in }
        guard case let .audio(buffer) = out else { Issue.record("expected audio"); return }
        #expect(buffer.sampleRate == 16_000)
    }

    @Test func saveWAVStageWritesFileAndPassesThrough() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("save-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let input = Media.audio(AudioBuffer(samples: [0.3, -0.3], sampleRate: 24_000))
        let out = try await SaveWAVStage(url: url).run(input) { _ in }   // openOnSave defaults false

        #expect(FileManager.default.fileExists(atPath: url.path))
        guard case let .audio(buffer) = out else { Issue.record("expected audio passthrough"); return }
        #expect(buffer.sampleRate == 24_000)
    }

    @Test func saveWAVStageRejectsNonAudio() async {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("x.wav")
        await #expect(throws: StageError.self) {
            _ = try await SaveWAVStage(url: url).run(.text("nope")) { _ in }
        }
    }

    // MARK: - byte helpers

    private func ascii(_ d: Data, _ offset: Int, _ length: Int) -> String {
        String(decoding: d[offset..<offset + length], as: UTF8.self)
    }
    private func u16(_ d: Data, _ o: Int) -> UInt16 {
        UInt16(d[o]) | (UInt16(d[o + 1]) << 8)
    }
    private func i16(_ d: Data, _ o: Int) -> Int16 { Int16(bitPattern: u16(d, o)) }
    private func u32(_ d: Data, _ o: Int) -> UInt32 {
        UInt32(d[o]) | (UInt32(d[o + 1]) << 8) | (UInt32(d[o + 2]) << 16) | (UInt32(d[o + 3]) << 24)
    }
}
