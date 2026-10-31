import Darwin
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXUI

// MP7-3f, rule 13b: the Klein DiT on the REAL installed weights, opt-in and measurement-only (no accuracy claim: there is no diffusers golden here; see
// Flux2KleinRealDiTGoldenTests for that). Needs the installed `transformer/` (header check: Klein3eHeader) AND TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_BITS set to
// "4", "8" or "16" (16 = bf16, nothing quantized) so ONE variant runs per process and its peak is not polluted by an earlier one. Unsandboxed host:
//   TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_BITS=4 xcodebuild test -scheme MLXUI -destination 'platform=macOS' -parallel-testing-enabled NO \
//     CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements ENABLE_APP_SANDBOX=NO -only-testing:MLXUITests/Flux2KleinTransformerRealLoadTests
// Nothing is downloaded and the installed folder is only read.

enum KleinDiTLoad {
    nonisolated static let bits: Int?? = {
        guard let raw = ProcessInfo.processInfo.environment["MLXUI_TEST_KLEIN_DIT_BITS"], let n = Int(raw) else { return nil }
        return n == 16 ? .some(nil) : .some(n)
    }()
    nonisolated static let enabled = Klein3eHeader.present && bits != nil
    nonisolated static let skipReason: Comment = "needs the installed FLUX.2 Klein transformer/ (or TEST_RUNNER_MLXUI_TEST_KLEIN_DIR), TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_BITS=4|8|16, and an unsandboxed test host"
}

@Suite(.enabled(if: KleinDiTLoad.enabled, KleinDiTLoad.skipReason), .serialized)
struct Flux2KleinTransformerRealLoadTests {
    @Test func streamedLoadAndAForwardAt128x128() throws {
        let dir = try #require(Klein3eHeader.directory)
        let bits = try #require(KleinDiTLoad.bits)
        let label = bits.map { "int\($0)" } ?? "bf16"
        let sampler = KleinFootprint()
        Memory.clearCache()
        Memory.peakMemory = 0
        let before = KleinFootprint.gb(KleinFootprint.now())
        sampler.start()
        let t0 = Date()
        var model: Flux2KleinTransformer? = try {
            func stage(_ s: String) {
                if s.hasPrefix("shard") || s == "double block 0 ready" || s == "double block 4 ready" || s == "single block 0 ready" || s == "single block 19 ready" || s == "front end ready" {
                    print(String(format: "STAGE %@ %@: mlx active %.2f GB, mlx peak %.2f GB, footprint %.2f GB", label, s, KleinFootprint.gb(Memory.activeMemory),
                                 KleinFootprint.gb(Memory.peakMemory), KleinFootprint.gb(KleinFootprint.now())))
                }
            }
            if let bits { return try Flux2KleinTransformer.load(transformerDirectory: dir, bits: bits, stage: stage) }
            return try Flux2KleinTransformer.loadUnquantizedForTesting(transformerDirectory: dir, stage: stage)
        }()
        let loadSeconds = Date().timeIntervalSince(t0)
        let loadMLXPeak = KleinFootprint.gb(Memory.peakMemory)
        let loadFootprintPeak = KleinFootprint.gb(sampler.stop())
        do {
        let m = try #require(model)
        let weightBytes = m.parameterBytes()
        let blocksBytes = m.parameterBytes { $0.hasPrefix("transformer_blocks.") || $0.hasPrefix("single_transformer_blocks.") }
        let activeAfterLoad = KleinFootprint.gb(Memory.activeMemory)
        print(String(format: "SMOKE-NUMBERS %@ LOAD: %.1f s, mlx peak %.2f GB, footprint peak %.2f GB (before %.2f), weights %.3f GB (blocks %.3f GB, rest %.3f GB), mlx active %.2f GB",
                     label, loadSeconds, loadMLXPeak, loadFootprintPeak, before, KleinFootprint.gb(weightBytes), KleinFootprint.gb(blocksBytes),
                     KleinFootprint.gb(weightBytes - blocksBytes), activeAfterLoad))

        // Forward at 128 x 128: 8 x 8 = 64 image tokens (128 channels) + 16 text rows. Random input, bf16: a timing/memory run, not an accuracy claim.
        MLXRandom.seed(1234)
        let hidden = MLXRandom.normal([1, 64, 128]).asType(.bfloat16), encoder = MLXRandom.normal([1, 16, 7680]).asType(.bfloat16)
        let hh = MLXArray((0 ..< 64).map { Float($0 / 8) }), ww = MLXArray((0 ..< 64).map { Float($0 % 8) }), z = MLXArray.zeros([64]), zt = MLXArray.zeros([16])
        let imgIDs = stacked([z, hh, ww, z], axis: -1), txtIDs = stacked([zt, zt, zt, MLXArray((0 ..< 16).map { Float($0) })], axis: -1)
        let sigma = MLXArray([0.5] as [Float])
        let sampler2 = KleinFootprint()
        Memory.peakMemory = 0
        sampler2.start()
        var times: [Double] = []
        var out = MLXArray.zeros([1])
        for _ in 0 ..< 2 {
            let t = Date()
            out = try m(hidden: hidden, encoder: encoder, sigma: sigma, imgIDs: imgIDs, txtIDs: txtIDs)
            times.append(Date().timeIntervalSince(t))
        }
        let finite = try MLXErrorGuard.read(isFinite(out.asType(.float32)).all().reshaped(1), as: Bool.self)[0]
        let norm = try MLXErrorGuard.read(sqrt((out.asType(.float32) * out.asType(.float32)).sum()).reshaped(1), as: Float.self)[0]
        print(String(format: "SMOKE-NUMBERS %@ FORWARD 128x128: first %.2f s, second %.2f s, mlx peak %.2f GB, footprint peak %.2f GB, output shape %@, finite %@, L2 %.3f",
                     label, times[0], times[1], KleinFootprint.gb(Memory.peakMemory), KleinFootprint.gb(sampler2.stop()), "\(out.shape)", "\(finite)", norm))
        #expect(out.shape == [1, 64, 128] && finite)
        }   // `m` and `out` are released here, before `model = nil`

        model = nil
        Memory.clearCache()
        print(String(format: "SMOKE-NUMBERS %@ AFTER FREE: mlx active %.3f GB, footprint %.2f GB", label, KleinFootprint.gb(Memory.activeMemory), KleinFootprint.gb(KleinFootprint.now())))
        Thread.sleep(forTimeInterval: 2)
        print(String(format: "SMOKE-NUMBERS %@ 2 s after free: footprint %.2f GB", label, KleinFootprint.gb(KleinFootprint.now())))
        // The owner's stop line for the streamed load is about 8 GB; bf16 is a reference measurement and carries no limit.
        if bits != nil { #expect(loadMLXPeak < 8.0 && loadFootprintPeak < 8.0, "STOP LINE: streamed \(label) load peaked at mlx \(loadMLXPeak) GB, footprint \(loadFootprintPeak) GB") }
    }
}
