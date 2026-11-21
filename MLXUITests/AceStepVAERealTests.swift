import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3h: the VAE decoder from the REAL weights (3c `loadVAEDecoder`, f16 compute, chunk 256 / overlap 64) against the oracle's `stage_c.safetensors`.
/// Owner-run, skipped by default. Case `2s`: `vae.latents_2s_in` [1,64,50] -> `vae.audio_2s_out` [1,2,96000]. Case `full`: `vae.latents_in`
/// [1,64,250] -> `vae.audio_out_full` [1,2,480000]; 250 <= 256 frames, so this is ONE direct decode and does NOT exercise the tiling (the tiling is
/// covered by the tiny-tier test); a third case decodes the 250 latents repeated to 600 frames through the tiled path for shape/finite checks only.
/// Thresholds: cosine >= 0.999, max abs <= 0.02. Prints `ACESTEP-VAE prompt=<id> case=2s|full cos=<x> maxAbs=<x> rms=<x>` (rms of the difference).
@Suite(.serialized, .enabled(if: AceStepRealLoad.canRun && AceStepVAEReal.haveDumps, "needs MLXUI_TEST_ACESTEP=1, the model folder and a stage_c.safetensors dump"))
struct AceStepVAERealTests {

    @Test func decoderMatchesTheOracle() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let config = try AceStepVAEConfig(json: Data(contentsOf: dir.appendingPathComponent("vae/config.json")))
        let vae = try AceStepVAE(config: config, weights: AceStepWeights.loadVAEDecoder(directory: dir), dtype: .float16)
        var ran = 0
        for id in ["instrumental", "english_vocal"] where FileManager.default.fileExists(atPath: AceStepVAEReal.dump(id).path) {
            let index = try SafetensorsIndex(url: AceStepVAEReal.dump(id))
            func r(_ k: String) throws -> MLXArray {
                guard let v = try index.read([k])[k] else { throw AceStepWeightsError.missingTensor(k) }
                return v
            }
            for (name, inKey, outKey) in [("2s", "vae.latents_2s_in", "vae.audio_2s_out"), ("full", "vae.latents_in", "vae.audio_out_full")] {
                let out = try vae.decodeChannelsFirst(try r(inKey))
                let want = try r(outKey)
                #expect(out.shape == want.shape, "\(id) \(name) shape \(out.shape) vs \(want.shape)")
                let cos = ((out * want).sum() / (sqrt(out.square().sum()) * sqrt(want.square().sum()))).item(Float.self)
                let maxAbs = abs(out - want).max().item(Float.self)
                let rms = sqrt((out - want).square().mean()).item(Float.self)
                #expect(cos >= 0.999, "\(id) \(name) cosine \(cos)")
                #expect(maxAbs <= 0.02, "\(id) \(name) max abs \(maxAbs)")
                print(String(format: "ACESTEP-VAE prompt=%@ case=%@ cos=%.5f maxAbs=%.5f rms=%.6f", id, name, cos, maxAbs, rms))
                ran += 1
            }
            // Tiled path on the real weights: 250 frames repeated to 600 (shape and finite only).
            let z = try r("vae.latents_in")
            let long = concatenated([z, z, z[0..., 0..., 0 ..< 100]], axis: 2)
            let tiled = try vae.decodeChannelsFirst(long)
            #expect(tiled.shape == [1, 2, 600 * 1920], "\(id) tiled shape \(tiled.shape)")
            #expect(abs(tiled).max().item(Float.self).isFinite, "\(id) tiled finite")
            Memory.clearCache()
        }
        #expect(ran > 0)
    }
}

enum AceStepVAEReal {
    nonisolated static func dump(_ id: String) -> URL { AceStepOracle.root.appendingPathComponent(id).appendingPathComponent("stage_c.safetensors") }
    nonisolated static var haveDumps: Bool { ["instrumental", "english_vocal"].contains { FileManager.default.fileExists(atPath: dump($0).path) } }
}
