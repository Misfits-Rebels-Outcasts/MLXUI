import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3f: the DiT built from the REAL weights (bf16, 3c `loadDecoder`) against the Python oracle's `stage_b.safetensors`. Owner-run, skipped
/// by default (`TEST_RUNNER_MLXUI_TEST_ACESTEP=1`, the model folder, and `<ORACLE>/<prompt>/stage_b.safetensors`). For steps 0 and 7 it feeds
/// the recorded `hidden_states` (+ `timestep`, `timestep_r`, step 0's `encoder_hidden_states` and `context_latents`, reused for step 7; the encoder
/// mask is all-ones for these prompts, as in upstream) and compares `out`: cosine >= 0.999, norm ratio within 2%.
/// Prints `ACESTEP-DIT prompt=<id> step=<n> cos=<x> normRatio=<x> maxAbs=<x>`.
@Suite(.serialized, .enabled(if: AceStepRealLoad.canRun && AceStepDiTReal.haveDumps, "needs MLXUI_TEST_ACESTEP=1, the model folder and a stage_b.safetensors dump"))
struct AceStepDiTRealTests {

    @Test func ditStepsMatchTheOracle() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let config = try AceStepDiTConfig(json: Data(contentsOf: dir.appendingPathComponent("acestep-v15-turbo/config.json")))
        let net = try AceStepDiT.load(config: config, weights: AceStepWeights.loadDecoder(directory: dir))
        var ran = 0
        for id in ["instrumental", "english_vocal"] where FileManager.default.fileExists(atPath: AceStepDiTReal.dump(id).path) {
            let index = try SafetensorsIndex(url: AceStepDiTReal.dump(id))
            func r(_ k: String) throws -> MLXArray { try index.read([k]).values.first ?? { throw AceStepWeightsError.missingTensor(k) }() }
            let enc = try r("dit.step0.encoder_hidden_states"), ctx = try r("dit.step0.context_latents")
            for step in [0, 7] {
                let out = net(hidden: try r("dit.step\(step).hidden_states"), context: ctx, encoder: enc,
                              timestep: try r("dit.step\(step).timestep"), timestepR: try r("dit.step\(step).timestep_r"))
                let want = try r("dit.step\(step).out")
                eval(out)
                #expect(out.shape == want.shape, "\(id) step \(step) shape")
                let cos = ((out * want).sum() / (sqrt(out.square().sum()) * sqrt(want.square().sum()))).item(Float.self)
                let ratio = (sqrt(out.square().sum()) / sqrt(want.square().sum())).item(Float.self)
                let maxAbs = abs(out - want).max().item(Float.self)
                #expect(cos >= 0.999, "\(id) step \(step) cosine \(cos)")
                #expect(abs(ratio - 1) <= 0.02, "\(id) step \(step) norm ratio \(ratio)")
                print(String(format: "ACESTEP-DIT prompt=%@ step=%d cos=%.5f normRatio=%.4f maxAbs=%.5f", id, step, cos, ratio, maxAbs))
                ran += 1
            }
        }
        #expect(ran > 0)
    }
}

enum AceStepDiTReal {
    nonisolated static func dump(_ id: String) -> URL { AceStepOracle.root.appendingPathComponent(id).appendingPathComponent("stage_b.safetensors") }
    nonisolated static var haveDumps: Bool { ["instrumental", "english_vocal"].contains { FileManager.default.fileExists(atPath: dump($0).path) } }
}
