import Testing
import Foundation
import MLX
import MLXNN
@testable import MLXUI

/// MG-ENG2 — shape tests for the bespoke EnCodec decoder port. A synthetic `[T, 4]` int32 code
/// sequence decodes to a `[T × 640]` mono float32 waveform (32 kHz codec, hop 8·5·4·4 = 640).
///
/// Serialized like the other MLX suites (MLX's default stream isn't parallel-safe).
@Suite(.serialized)
struct MusicGenEncodecTests {

    /// The 32 kHz config decodes from the bundled config.json shape and yields 50 frames/s.
    @Test func encodecConfigIs32kFourCodebook() throws {
        let config = MusicGenEncodecConfig.encodec32k
        #expect(config.samplingRate == 32000)
        #expect(config.codebookSize == 2048)
        #expect(config.upsamplingRatios == [8, 5, 4, 4])
        #expect(config.targetBandwidths == [2.2])
        #expect(32000 / config.hopLength == 50)
    }

    /// A synthetic `[10, 4]` int32 code sequence decodes to `[10 × 640]` mono samples.
    /// Random codec weights — a plumbing/shape check, not numerical parity.
    @Test func decodeUpscalesBy640() {
        let config = MusicGenEncodecConfig.encodec32k
        let codec = MusicGenEncodec(config: config)
        let t = 10
        let codes = MLXArray.zeros([t, 4]).asType(.int32)   // [T, 4]
        let audio = codec.decode(codes)
        #expect(audio.shape == [t * 640])
    }

    /// The bundled-weights load path accepts the real checkpoint layout (72 tensors:
    /// `decoder.layers.*`, `quantizer.layers.N.codebook.embed`) — but only when weights exist.
    /// Guard the error path when `encodec/` is absent (fresh install before MG-DL1 install).
    @Test func missingEncodecThrows() {
        #expect(throws: StageError.self) {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            _ = try MusicGenEncodec(modelDirectory: dir)
        }
    }

    /// MG-ENG2 load-path regression (the "unhandledKeys" bug): the full checkpoint dict has
    /// `decoder.*`, `encoder.*`, AND `quantizer.*` keys. Each submodule must receive only its
    /// own subtree (prefix-stripped), else `.noUnusedKeys` reports every foreign key as
    /// unhandled and the install-time load throws. Feed synthetic flat keys through the same
    /// path the real `init(modelDirectory:)` uses and assert the submodule updates cleanly.
    @Test func loadPathScopesSubmoduleKeys() throws {
        let config = MusicGenEncodecConfig.encodec32k
        let quantizer = EncodecRVQ(config: config)
        let decoder = EncodecDecoderPort(config: config)

        // Synthetic flat keys mirroring the checkpoint's three top-level prefixes.
        var flat: [String: MLXArray] = [:]
        for (k, arr) in quantizer.parameters().flattened() {
            flat["quantizer." + k] = MLXRandom.normal(arr.shape, dtype: .float32)
        }
        for (k, arr) in decoder.parameters().flattened() {
            flat["decoder." + k] = MLXRandom.normal(arr.shape, dtype: .float32)
        }
        // A few "encoder.*" keys the app never loads — they must be ignored by both submodules.
        flat["encoder.layers.0.conv.weight"] = MLXArray.zeros([64, 7, 1])
        flat["encoder.layers.0.conv.bias"] = MLXArray.zeros([64])

        let qKeys = MusicGenEncodec.subtree(flat, prefix: "quantizer.")
        let dKeys = MusicGenEncodec.subtree(flat, prefix: "decoder.")
        // Each subtree only contains its own stripped keys (foreign prefixes excluded).
        #expect(qKeys.keys.allSatisfy { !$0.hasPrefix("decoder.") && !$0.hasPrefix("encoder.") })
        #expect(dKeys.keys.allSatisfy { !$0.hasPrefix("quantizer.") && !$0.hasPrefix("encoder.") })

        // The exact update the engine performs — `.noUnusedKeys` must not throw now.
        try quantizer.update(parameters: ModuleParameters.unflattened(qKeys), verify: .noUnusedKeys)
        try decoder.update(parameters: ModuleParameters.unflattened(dKeys), verify: .noUnusedKeys)
        eval(quantizer, decoder)
    }
}
