import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3g: the sampler and DCW. Tiny tier replays upstream's recorded 8-step run (`mc5_dit_tiny.safetensors`: injected noise, DCW on and off)
/// through the 3f tiny DiT. No real weights, no RNG parity.
struct AceStepSamplerTests {

    static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { AceStepDiTTests.maxAbs(a, b) }

    private func velocityFn(_ f: AceStepDiTTests.Fixture, _ net: AceStepDiT) throws -> (MLXArray, Float) throws -> MLXArray {
        let ctx = try #require(f.tensors["sampler.ctx"]), enc = try #require(f.tensors["sampler.enc"])
        return { x, t in
            let ts = MLXArray([t])
            return net(hidden: x, context: ctx, encoder: enc, timestep: ts, timestepR: ts)
        }
    }

    // MARK: Replay

    @Test(arguments: [true, false])
    func replayMatchesUpstreamFinalLatents(dcwOn: Bool) throws {
        let f = try AceStepDiTTests.load()
        let net = try AceStepDiT.load(config: f.config, weights: f.weights)
        let noise = try #require(f.tensors["sampler.noise"])
        var cfg = AceStepSamplerConfig()
        cfg.dcw = dcwOn ? AceStepDCW() : nil
        let out = try AceStepSampler.run(noise: noise, config: cfg, velocity: try velocityFn(f, net))
        let want = try #require(f.tensors[dcwOn ? "sampler.latents_dcw_on" : "sampler.latents_dcw_off"])
        let d = Self.maxAbs(out, want)
        print("ACESTEP-SAMPLER-TINY dcw=\(dcwOn ? "on" : "off") maxAbs=\(d)")
        #expect(out.shape == want.shape)
        #expect(d <= 1e-5, "final latents max abs \(d)")
    }

    /// The recorded per-step inputs (x_t) and times, with DCW on: each step's input is what upstream fed its decoder.
    @Test func everyStepSeesUpstreamsInputAndTime() throws {
        let f = try AceStepDiTTests.load()
        let net = try AceStepDiT.load(config: f.config, weights: f.weights)
        let inner = try velocityFn(f, net)
        var step = 0
        var worst: Float = 0
        _ = try AceStepSampler.run(noise: try #require(f.tensors["sampler.noise"]), config: AceStepSamplerConfig(), velocity: { x, t in
            let wantX = try #require(f.tensors["sampler.step\(step).xt"]), wantT = try #require(f.tensors["sampler.step\(step).t"])
            worst = max(worst, Self.maxAbs(x, wantX))
            #expect(abs(wantT.item(Float.self) - t) < 1e-6, "step \(step) t")
            step += 1
            return try inner(x, t)
        })
        #expect(step == 8)
        #expect(worst <= 1e-5, "worst per-step input max abs \(worst)")
    }

    // MARK: Schedule, cancellation, progress

    @Test func scheduleIsLinspaceAndShiftOneIsIdentity() {
        #expect(AceStepSampler.schedule(shift: 1.0, steps: 8) == [1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125])
        let s3 = AceStepSampler.schedule(shift: 3.0, steps: 4)
        #expect(zip(s3, [1.0, 0.75, 0.5, 0.25].map { 3 * $0 / (1 + 2 * $0) }).allSatisfy { abs($0 - $1) < 1e-12 })
    }

    @Test func lastStepGoesStraightToX0AndReportsProgress() throws {
        // v == x gives x0 = x - x*t on the last step; with 1 step (t = 1) the result is exactly 0.
        let x = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 2, 2)
        var cfg = AceStepSamplerConfig(); cfg.steps = 1; cfg.dcw = nil
        var seen: [(Int, Int)] = []
        let out = try AceStepSampler.run(noise: x, config: cfg, velocity: { x, _ in x }, progress: { seen.append(($0, $1)) })
        #expect(Self.maxAbs(out, MLXArray.zeros([1, 2, 2])) == 0)
        #expect(seen.count == 1 && seen[0] == (1, 1))
    }

    @Test func cancellationBetweenStepsThrows() async {
        let task = Task { () throws -> Int in
            var calls = 0
            _ = try AceStepSampler.run(noise: MLXArray.zeros([1, 2, 2]), config: AceStepSamplerConfig(), velocity: { x, _ in
                calls += 1
                withUnsafeCurrentTask { $0?.cancel() }
                return x
            })
            return calls
        }
        do { _ = try await task.value; Issue.record("expected CancellationError") } catch is CancellationError {} catch { Issue.record("\(error)") }
    }

    // MARK: Haar and DCW

    @Test(arguments: [8, 7])
    func haarRoundTrips(length: Int) {
        let x = MLXRandom.normal([2, length, 3], key: MLXRandom.key(7))
        let (l, h) = AceStepHaar.forward(x)
        #expect(l.dim(1) == (length + 1) / 2)
        #expect(Self.maxAbs(AceStepHaar.inverse(l, h, length: length), x) < 1e-6)
    }

    @Test func haarOfAHandCheckedPair() throws {
        let x = MLXArray([Float(3), 1, 2, 6]).reshaped(1, 4, 1)
        let (l, h) = AceStepHaar.forward(x)
        let r = Float(2).squareRoot()
        #expect(abs(try MLXErrorGuard.read(l, as: Float.self)[0] - 4 / r) < 1e-6)
        #expect(abs(try MLXErrorGuard.read(h, as: Float.self)[0] - 2 / r) < 1e-6)
        #expect(abs(try MLXErrorGuard.read(l, as: Float.self)[1] - 8 / r) < 1e-6)
        #expect(abs(try MLXErrorGuard.read(h, as: Float.self)[1] - -4 / r) < 1e-6)
    }

    /// x = [3, 1], y = [1, 1] (one Haar pair, one channel), double mode, t = 0.5, scalers 0.4 / 0.2:
    /// xL = 4/r, yL = 2/r, xH = 2/r, yH = 0; low s = 0.5 x 0.4 = 0.2, high s = 0.5 x 0.2 = 0.1.
    /// xL' = 4/r + 0.2 x 2/r = 4.4/r, xH' = 2/r + 0.1 x 2/r = 2.2/r; inverse: ((xL'+xH')/r, (xL'-xH')/r) = (3.3, 1.1).
    @Test func doubleModeHandComputed() throws {
        let dcw = AceStepDCW(mode: .double, scaler: 0.4, highScaler: 0.2)
        let out = dcw.apply(MLXArray([Float(3), 1]).reshaped(1, 2, 1), denoised: MLXArray([Float(1), 1]).reshaped(1, 2, 1), t: 0.5)
        let v = try MLXErrorGuard.read(out, as: Float.self)
        #expect(abs(v[0] - 3.3) < 1e-5 && abs(v[1] - 1.1) < 1e-5, "\(v)")
    }

    @Test func dcwChangesTheResultAndIsIdentityWhenInactive() throws {
        let x = MLXRandom.normal([1, 10, 4], key: MLXRandom.key(1)), y = MLXRandom.normal([1, 10, 4], key: MLXRandom.key(2))
        #expect(Self.maxAbs(AceStepDCW().apply(x, denoised: y, t: 0.5), x) > 1e-3)
        #expect(!AceStepDCW(mode: .double, scaler: 0, highScaler: 0).isActive)
        #expect(AceStepDCW(mode: .double, scaler: 0, highScaler: 0.02).isActive)
        #expect(!AceStepDCW(mode: .low, scaler: 0, highScaler: 0.02).isActive)
        // t = 0: low scaler vanishes, high scaler is at full strength (complementary schedule); t = 1: the reverse.
        #expect(Self.maxAbs(AceStepDCW(mode: .low, scaler: 0.05).apply(x, denoised: y, t: 0), x) == 0)
        #expect(Self.maxAbs(AceStepDCW(mode: .high, scaler: 0.05).apply(x, denoised: y, t: 1), x) == 0)
        // pix mode is the raw scaler regardless of t.
        let pix = AceStepDCW(mode: .pix, scaler: 0.1).apply(x, denoised: y, t: 0.3)
        #expect(Self.maxAbs(pix, x + 0.1 * (x - y)) < 1e-6)
    }

    @Test func oddLengthLatentKeepsItsLength() {
        let x = MLXRandom.normal([1, 9, 4], key: MLXRandom.key(3)), y = MLXRandom.normal([1, 9, 4], key: MLXRandom.key(4))
        #expect(AceStepDCW().apply(x, denoised: y, t: 0.5).shape == [1, 9, 4])
    }
}
