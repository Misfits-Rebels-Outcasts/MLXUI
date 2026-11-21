import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3g: the full 8-step sampler through the REAL DiT (bf16 weights, 3c `loadDecoder` + 3f) against the oracle. Owner-run, skipped by default.
/// Injects `sampler.noise` from `<prompt>/stage_b.safetensors`, takes `cond.prepare.encoder_hidden_states` / `cond.prepare.context_latents` from
/// the same prompt's `stage_a.safetensors`, runs DCW on and off, and compares with `sampler.latents_dcw_on` / `sampler.latents_dcw_off`:
/// cosine >= 0.99 and norm ratio within 3%. Prints `ACESTEP-SAMPLER prompt=<id> dcw=on|off cos=<x> normRatio=<x> maxAbs=<x>`.
@Suite(.serialized, .enabled(if: AceStepRealLoad.canRun && AceStepDiTReal.haveDumps, "needs MLXUI_TEST_ACESTEP=1, the model folder and stage_a/stage_b dumps"))
struct AceStepSamplerRealTests {

    @Test func fullRunMatchesTheOracle() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let config = try AceStepDiTConfig(json: Data(contentsOf: dir.appendingPathComponent("acestep-v15-turbo/config.json")))
        let net = try AceStepDiT.load(config: config, weights: AceStepWeights.loadDecoder(directory: dir))
        var ran = 0
        for id in ["instrumental", "english_vocal"] {
            let b = AceStepDiTReal.dump(id), a = AceStepOracle.dump(id)
            guard FileManager.default.fileExists(atPath: b.path), FileManager.default.fileExists(atPath: a.path) else { continue }
            let bIndex = try SafetensorsIndex(url: b), aIndex = try SafetensorsIndex(url: a)
            func read(_ index: SafetensorsIndex, _ k: String) throws -> MLXArray {
                guard let v = try index.read([k])[k] else { throw AceStepWeightsError.missingTensor(k) }
                return v
            }
            let noise = try read(bIndex, "sampler.noise")
            let enc = try read(aIndex, "cond.prepare.encoder_hidden_states"), ctx = try read(aIndex, "cond.prepare.context_latents")
            for dcwOn in [true, false] {
                var cfg = AceStepSamplerConfig()
                cfg.dcw = dcwOn ? AceStepDCW() : nil
                let out = try AceStepSampler.run(noise: noise, config: cfg, velocity: { x, t in
                    let ts = MLXArray([t])
                    return net(hidden: x, context: ctx, encoder: enc, timestep: ts, timestepR: ts)
                })
                let want = try read(bIndex, dcwOn ? "sampler.latents_dcw_on" : "sampler.latents_dcw_off")
                #expect(out.shape == want.shape, "\(id) shape")
                let cos = ((out * want).sum() / (sqrt(out.square().sum()) * sqrt(want.square().sum()))).item(Float.self)
                let ratio = (sqrt(out.square().sum()) / sqrt(want.square().sum())).item(Float.self)
                let maxAbs = abs(out - want).max().item(Float.self)
                #expect(cos >= 0.99, "\(id) dcw \(dcwOn) cosine \(cos)")
                #expect(abs(ratio - 1) <= 0.03, "\(id) dcw \(dcwOn) norm ratio \(ratio)")
                print(String(format: "ACESTEP-SAMPLER prompt=%@ dcw=%@ cos=%.5f normRatio=%.4f maxAbs=%.5f", id, dcwOn ? "on" : "off", cos, ratio, maxAbs))
                ran += 1
            }
            Memory.clearCache()
        }
        #expect(ran > 0)
    }
}
