import Testing
import Foundation
import MLX
@testable import MLXUI

/// MG-ENG4 — engine test: drive the autoregressive run loop with synthetic weights + a stub
/// codec (returns silence), assert a non-empty `AudioBuffer` and that `AudioWriter` produces a
/// valid non-empty WAV. No real model weights — plumbing + WAV-output check.
///
/// Serialized like the other MLX suites (MLX's default stream isn't parallel-safe).
@Suite(.serialized)
struct MusicGenEngineTests {

    private static func tinyConfig() -> MusicGenDecoderConfig {
        var c = MusicGenDecoderConfig()
        c.hidden = 64
        c.layers = 2
        c.heads = 4
        c.headDim = 16
        c.ffn = 128
        c.codebooks = 4
        c.vocab = 64
        c.maxPosition = 2048
        c.bosTokenID = 64
        return c
    }

    /// Stub codec: ignores codes, returns a `[T×640]` silence waveform.
    private struct StubEncodec: MusicGenAudioDecoding {
        func decode(_ codes: MLXArray) -> MLXArray {
            MLXArray.zeros([codes.dim(0) * 640])
        }
    }

    @Test func generateProducesNonEmptyAudioAndWAV() async throws {
        let config = Self.tinyConfig()
        let encoder = MusicGenT5Encoder()
        let decoder = MusicGenDecoder(config: config)
        let stub = StubEncodec()

        let buffer = try await MusicGenEngine.generate(
            tokenIDs: [1, 2, 3],
            encoder: encoder, decoder: decoder, encodec: stub,
            maxSteps: 4, topK: 4, temperature: 1.0, guidanceScale: 3.0) { _ in }

        // EnCodec stub returns `[T'×640]` silence; the audio buffer must exist and be non-empty.
        #expect(buffer.sampleRate == 32_000)
        #expect(!buffer.samples.isEmpty)

        // WAV output: write and verify a non-empty file with a RIFF header.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("musicgen-engine-test-\(UUID().uuidString).wav")
        try AudioWriter.writeWAV(buffer, to: tmp)
        let data = try Data(contentsOf: tmp)
        #expect(!data.isEmpty)
        #expect(Array(data.prefix(4)) == Array("RIFF".utf8))
        try? FileManager.default.removeItem(at: tmp)
    }

    /// The delay-undo helper preserves the sequence length contract: maxSteps+1 rows in →
    /// (maxSteps+1)-codebooks rows out.
    @Test func undoDelayTrimsToExpectedLength() {
        let codebooks = 4
        let maxSteps = 8
        let rows = (0 ..< (maxSteps + 1)).map { i in
            MLXArray([Int32(i), Int32(i), Int32(i), Int32(i)])
        }
        let codes = MusicGenEngine.undoDelay(rows, codebooks: codebooks)
        #expect(codes.shape == [(maxSteps + 1) - codebooks, codebooks])
    }
}
