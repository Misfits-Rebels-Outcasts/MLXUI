import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3e: the condition encoder built from the REAL weights (bf16, the 3c reader) against the Python oracle's stage-a dump. Owner-run,
/// skipped by default (`TEST_RUNNER_MLXUI_TEST_ACESTEP=1`, the model folder, and `<ORACLE>/<prompt>/stage_a.safetensors`; same env as 3d).
/// It feeds the oracle's own `text.hidden` / `text.lyric_embed` (isolating 3e from 3d) and its timbre input. Per prompt it checks:
/// the final `encoder_hidden_states` cosine >= 0.9995 (per-token mean) with norm ratio within 2%, the mask exactly, `context_latents`
/// cosine >= 0.9999 and its mask half exactly, and our own timbre input against `cond.timbre_encoder.in0`.
/// Prints `ACESTEP-COND prompt=<id> cosMean=<x> cosMin=<x> normRatio=<x> ctxCos=<x> timbreInMaxAbs=<x>`.
@Suite(.serialized, .enabled(if: AceStepOracle.canRun, "needs MLXUI_TEST_ACESTEP=1, the model folder and a stage_a.safetensors oracle dump"))
struct AceStepConditionRealTests {

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> (mean: Float, min: Float) {
        let x = a.asType(.float32), y = b.asType(.float32)
        let c = (x * y).sum(axis: -1) / (sqrt(x.square().sum(axis: -1)) * sqrt(y.square().sum(axis: -1)) + 1e-12)
        return (c.mean().item(Float.self), c.min().item(Float.self))
    }

    @Test func conditionEncoderMatchesTheOracle() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let config = try AceStepConditionConfig(json: Data(contentsOf: dir.appendingPathComponent("acestep-v15-turbo/config.json")))
        let weights = try AceStepWeights.loadConditionEncoder(directory: dir)
        let silence = try AceStepSilenceLatent.load().asType(.bfloat16)
        let enc = try AceStepConditionEncoder(config: config, weights: weights, silenceLatent: silence)
        var ran = 0
        for id in AceStepOracle.promptIDs where FileManager.default.fileExists(atPath: AceStepOracle.dump(id).path) {
            let index = try SafetensorsIndex(url: AceStepOracle.dump(id))
            let ref = try index.read(index.names)
            func r(_ k: String) throws -> MLXArray { try #require(ref[k], "\(id): \(k)") }
            let textHidden = try r("text.hidden").asType(.bfloat16)
            let lyricEmbeds = try r("text.lyric_embed").asType(.bfloat16)
            let inputs = AceStepTextInputs(
                textIds: [], textMask: MLXArray.ones([1, textHidden.dim(1)], dtype: .int32), lyricIds: [],
                lyricMask: MLXArray.ones([1, lyricEmbeds.dim(1)], dtype: .int32), textHidden: textHidden, lyricEmbeds: lyricEmbeds)
            let out = try enc.encode(inputs)

            let want = try r("cond.prepare.encoder_hidden_states")
            #expect(out.hidden.shape == want.shape, "\(id) hidden shape \(out.hidden.shape) vs \(want.shape)")
            let (cosMean, cosMin) = cosine(out.hidden, want)
            let normRatio = (sqrt(out.hidden.asType(.float32).square().sum()) / sqrt(want.asType(.float32).square().sum())).item(Float.self)
            #expect(cosMean >= 0.9995, "\(id) encoder_hidden_states cosine \(cosMean)")
            #expect(abs(normRatio - 1) <= 0.02, "\(id) norm ratio \(normRatio)")
            let wantMask = try r("cond.prepare.encoder_attention_mask").asType(.int32).asArray(Int32.self)
            #expect(out.mask.asType(.int32).asArray(Int32.self) == wantMask, "\(id) encoder mask")

            // context_latents: [1, T, 128]; the oracle's T is its own latent length.
            let wantCtx = try r("cond.prepare.context_latents")
            let ctx = try enc.contextLatents(duration: Double(wantCtx.dim(1)) / 25)
            #expect(ctx.shape == wantCtx.shape, "\(id) context shape \(ctx.shape) vs \(wantCtx.shape)")
            let ctxCos = cosine(ctx, wantCtx).min
            #expect(ctxCos >= 0.9999, "\(id) context_latents cosine \(ctxCos)")
            #expect(abs(ctx[0, 0..., 64 ..< 128].asType(.float32) - wantCtx[0, 0..., 64 ..< 128].asType(.float32)).max().item(Float.self) == 0,
                    "\(id) chunk-mask half exact")

            // Our own timbre input (silence[:750] as one clip) against the oracle's `cond.timbre_encoder.in0`.
            let in0 = try r("cond.timbre_encoder.in0")
            let mine = AceStepSilenceLatent.slice(silence, length: AceStepSilenceLatent.timbreFrames).expandedDimensions(axis: 0)
            let timbreInMax = abs(mine.asType(.float32) - in0.asType(.float32)).max().item(Float.self)
            #expect(in0.shape == mine.shape && timbreInMax <= 1e-2, "\(id) timbre input max abs \(timbreInMax)")

            print(String(format: "ACESTEP-COND prompt=%@ cosMean=%.5f cosMin=%.5f normRatio=%.4f ctxCos=%.5f timbreInMaxAbs=%.6f",
                         id, cosMean, cosMin, normRatio, ctxCos, timbreInMax))
            ran += 1
        }
        #expect(ran > 0)
    }
}
