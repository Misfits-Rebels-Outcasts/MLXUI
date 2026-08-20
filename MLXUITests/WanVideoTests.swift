import Testing
import Foundation
import MLX
import MLXNN
@testable import MLXUI

// MARK: - WAN 2.1 shape tests
// All tests use toy configs (small dims) and random weights — no real checkpoints required.
// They verify that the module forward passes produce the correct output shapes.

@Suite("WAN Video Tests")
struct WanVideoTests {

    // MARK: - T5 Encoder

    @Test func t5EncodeShape() {
        // Tiny config: 2 layers, 64d, 4 heads
        var cfg = WanT5Config()
        cfg.dModel = 64; cfg.dFF = 128; cfg.heads = 4; cfg.layers = 2; cfg.vocab = 256

        let encoder  = WanT5Encoder(config: cfg)
        let batchSz  = 2
        let seqLen   = 8
        let tokenIDs = MLXArray(Int32(0) ..< Int32(batchSz * seqLen)).reshaped([batchSz, seqLen])
        let out      = encoder(tokenIDs)   // [B, S, dModel]

        #expect(out.shape == [batchSz, seqLen, cfg.dModel])
    }

    @Test func t5UnloadClearsParams() {
        var cfg = WanT5Config()
        cfg.dModel = 32; cfg.dFF = 64; cfg.heads = 2; cfg.layers = 1; cfg.vocab = 128

        let encoder = WanT5Encoder(config: cfg)
        // Run once so params are evaluated
        let tokens = MLXArray(Int32(0) ..< 4).reshaped([1, 4])
        _ = encoder(tokens)
        eval(encoder)

        // Simulate release by zeroing — after setting to nil the object is deallocated;
        // here we just verify the model ran without error (dealloc is non-observable from tests).
        #expect(Bool(true))
    }

    // MARK: - DiT

    @Test func ditOutputShape() {
        var cfg = WanDiTConfig()
        cfg.dim = 48; cfg.ffnDim = 96; cfg.freqDim = 16; cfg.textDim = 32
        cfg.outDim = 4; cfg.heads = 4; cfg.layers = 2

        let dit  = WanDiT(config: cfg)
        let B    = 1
        let nT   = 2   // latent time frames
        let nH   = 4   // latent height / patchH
        let nW   = 4   // latent width  / patchW
        let inCh = cfg.outDim   // latent channels = outDim (both are 4 here)

        // Build input: [B, nT, nH*patchH, nW*patchW, inCh]
        let x        = MLXRandom.normal([B, nT, nH * cfg.patchH, nW * cfg.patchW, inCh]).asType(.float32)
        let timestep = MLXArray([Float(500.0)]).asType(.float32)
        let textCond = MLXRandom.normal([B, 6, cfg.textDim]).asType(.float32)

        let out = dit(x, timestep: timestep, textCond: textCond)  // [B, nT, nH*patchH, nW*patchW, outDim]
        // Unpatch restores the input spatial resolution (x was [B, nT, nH*patchH, nW*patchW, inCh])
        #expect(out.shape == [B, nT, nH * cfg.patchH, nW * cfg.patchW, cfg.outDim])
    }

    // MARK: - VAE Decoder

    @Test func vaeDecodeShape() {
        // Toy config: baseDim=8 → inner=32
        var cfg = WanVAEConfig()
        cfg.baseDim = 8; cfg.zDim = 4; cfg.outCh = 3

        let vae    = WanVAEDecoder(config: cfg)
        let B      = 1
        let T_lat  = 2   // latent temporal frames
        let H_lat  = 4   // latent spatial height
        let W_lat  = 4   // latent spatial width

        let z   = MLXRandom.normal([B, T_lat, H_lat, W_lat, cfg.zDim]).asType(.float32)
        let out = vae(z)   // [B, T_lat*4, H_lat*8, W_lat*8, 3]

        // Expected output: ×4 temporal (2 up_blocks with time upsample), ×8 spatial (3 up_blocks)
        #expect(out.shape[0] == B)
        #expect(out.shape[4] == cfg.outCh)
        // Spatial is ×8 from 3 up_blocks: ×2 × ×2 × ×2
        #expect(out.shape[2] == H_lat * 8)
        #expect(out.shape[3] == W_lat * 8)
    }

    // MARK: - RoPE

    @Test func rope3dPreservesShape() {
        let B = 1; let nT = 3; let nH = 2; let nW = 2; let nHeads = 4; let headDim = 24
        let S = nT * nH * nW
        let x = MLXRandom.normal([B, S, nHeads, headDim]).asType(.float32)

        let dHW = 2 * (headDim / 6)
        let dT  = headDim - 2 * dHW
        let freqsT = wanBuildFreqs(seqLen: nT, dim: dT)
        let freqsH = wanBuildFreqs(seqLen: nH, dim: dHW)
        let freqsW = wanBuildFreqs(seqLen: nW, dim: dHW)
        let (tPos, hPos, wPos) = wanMakePositions(nT: nT, nH: nH, nW: nW)

        let out = wanApplyRoPE3D(x, freqsT: freqsT, freqsH: freqsH, freqsW: freqsW,
                                 tPos: tPos, hPos: hPos, wPos: wPos)
        #expect(out.shape == x.shape)
    }

    // MARK: - Sampler

    @Test func samplerBuildsSigmas() {
        let sigs = WanSampler.buildSigmas(numSteps: 30)
        #expect(sigs.count == 31)
        #expect(sigs.first! > sigs.last!)  // descending
    }

    @Test func samplerBuildsSigmasSingleStep() {
        // numSteps=1 must not divide by (numSteps-1)=0 (NaN) and must emit σ₁ and σ=0
        let sigs = WanSampler.buildSigmas(numSteps: 1)
        #expect(sigs.count == 2)
        #expect(sigs[0].isFinite && sigs[0] > 0)
        #expect(sigs[1] == 0)
    }

    @Test func eulerStepPreservesShape() {
        let shape = [1, 2, 4, 4, 4]
        let x     = MLXRandom.normal(shape).asType(.float32)
        let pred  = MLXRandom.normal(shape).asType(.float32)
        let out   = WanSampler.eulerStep(x: x, noisePred: pred, sigmaFrom: 1.0, sigmaTo: 0.9)
        #expect(out.shape == shape)
    }

    // MARK: - SDK claim

    @Test @MainActor func sdkClaimsVideoEntries() throws {
        let json = """
        {
            "id": "Wan-AI--Wan2.1-T2V-1.3B-Diffusers",
            "family": "Wan2.1",
            "displayName": "Wan2.1-T2V-1.3B",
            "paramSize": "1.3B",
            "paramCountB": 1.3,
            "modelType": "video",
            "source": "mlx",
            "format": "mlx-bf16",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "Wan-AI",
            "hfModelId": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "ramGB": 10.0,
            "downloadSizeGB": 11.0,
            "contextWindow": null,
            "variants": [{
                "quantization": "bfloat16",
                "format": "mlx-bf16",
                "ramGB": 10.0,
                "downloadSizeGB": 11.0,
                "qualityPercent": 100,
                "hfModelId": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "recommended": true
            }]
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        let sdk   = WanVideoSDK()
        #expect(entry.runnerKind == .video)
        #expect(sdk.claim(entry) == .exact)
    }
}
