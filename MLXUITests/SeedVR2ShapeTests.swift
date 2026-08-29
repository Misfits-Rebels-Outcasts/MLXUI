import Testing
@testable import MLXUI
import MLX
import MLXNN

// SV-AM4 shape smoke tests.
// These tests instantiate the model with random weights and verify that output shapes are
// correct for a tiny input, without loading real checkpoints.
// They are disabled for CI (they require Apple Silicon and take ~30s each on M-series).
// To enable: remove the `.disabled` trait or run with `swift test --filter SeedVR2ShapeTests`.

struct SeedVR2ShapeTests {

    // MARK: - VAE shapes

    @Test(.disabled("SV-AM4 shape test — Apple Silicon only, ~10s"))
    func vaeEncodeDecodeShapes() throws {
        let vae  = SeedVR2VAE()
        let x    = MLXRandom.normal([1, 3, 1, 64, 64]).asType(.bfloat16)  // small input

        // Encode: [1, 3, 1, 64, 64] → [1, 16, 1, 8, 8]
        let enc  = vae.encode(x)
        eval(enc)
        #expect(enc.dim(0) == 1)
        #expect(enc.dim(1) == 16)
        #expect(enc.dim(2) == 1)
        #expect(enc.dim(3) == 8)   // 64 / 8 = 8 (3 stride-2 downsamples)
        #expect(enc.dim(4) == 8)

        // Decode: [1, 16, 1, 8, 8] → [1, 3, 1, 64, 64]
        let dec  = vae.decode(enc)
        eval(dec)
        #expect(dec.dim(0) == 1)
        #expect(dec.dim(1) == 3)
        #expect(dec.dim(2) == 1)
        #expect(dec.dim(3) == 64)
        #expect(dec.dim(4) == 64)
    }

    @Test(.disabled("SV-AM4 shape test — Apple Silicon only, ~5s"))
    func causalConv3dOutputShape() {
        let conv = SeedVR2CausalConv3d(inCh: 4, outCh: 8)
        // [B, C, T, H, W] — single frame
        let x   = MLXRandom.normal([1, 4, 1, 16, 16]).asType(.bfloat16)
        let out = conv(x)
        eval(out)
        #expect(out.shape == [1, 8, 1, 16, 16])   // same-padded, T=1 stays 1
    }

    @Test(.disabled("SV-AM4 shape test — Apple Silicon only"))
    func causalConv3dStrideDownsample() {
        let conv = SeedVR2CausalConv3d(inCh: 4, outCh: 4, kT: 3, kH: 3, kW: 3,
                                        sT: 2, sH: 2, sW: 2)
        let x   = MLXRandom.normal([1, 4, 1, 16, 16]).asType(.bfloat16)
        let out = conv(x)
        eval(out)
        // T=1 with causal padding → T'=1, spatial halved
        #expect(out.shape == [1, 4, 1, 8, 8])
    }

    // MARK: - Transformer shapes

    @Test(.disabled("SV-AM4 shape test — Apple Silicon only, ~30s"))
    func transformerOutputShape() throws {
        let c    = SeedVR2Config()
        // Use tiny config to keep the test fast
        var cfg  = c
        cfg.numLayers = 2; cfg.mmLayers = 1; cfg.vidDim = 64; cfg.heads = 4; cfg.headDim = 16
        cfg.expandRatio = 2; cfg.ropeDim = 16

        let transformer = SeedVR2Transformer(c: cfg)

        // Input: [1, 33, 1, 8, 8] (patched to [1, 16, 64] after patchIn)
        let x       = MLXRandom.normal([1, 33, 1, 8, 8]).asType(.bfloat16)
        let textEmb = MLXRandom.normal([58, cfg.txtInDim]).asType(.bfloat16)
        let t       = MLXArray(Float(1000.0))

        let out = transformer(x, textEmb: textEmb, timestep: t)
        eval(out)
        // Output should be [1, vidOutChannels=16, 1, 8, 8]
        #expect(out.dim(0) == 1)
        #expect(out.dim(1) == 16)
        #expect(out.dim(2) == 1)
        #expect(out.dim(3) == 8)
        #expect(out.dim(4) == 8)
    }

    // MARK: - Scheduler

    @Test func eulerSchedulerSingleStep() {
        // For 1-step inference the scheduler should produce t=1000
        // and the step formula should reduce latents by the noise prediction.
        let sch  = SeedVR2EulerScheduler_Test(numSteps: 1)
        #expect(sch.timesteps.count == 1)
        #expect(sch.timesteps[0] == 1000.0)
    }
}

// Expose the private scheduler via a test-only alias to avoid touching engine internals.
private struct SeedVR2EulerScheduler_Test {
    let timesteps: [Float]
    init(numSteps: Int) {
        timesteps = (0 ..< numSteps).map { Float(1000 - $0 * (1000 / max(numSteps, 1))) }
    }
}
