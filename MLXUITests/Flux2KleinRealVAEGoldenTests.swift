import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXUI

// MP7-3g, TASK B: this port's VAE decoder on the REAL checkpoint against diffusers 0.37.1 on CPU, from a file the OWNER makes with
// RSI/notes/claude-code/modelpack/klein-ref/real_vae_goldens.py (not committed, not run by the implementer). TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_GOLDEN_DIR names the
// folder holding real_vae.safetensors; the installed Klein model (Direct store, or TEST_RUNNER_MLXUI_TEST_KLEIN_DIR) supplies the weights. Without both the suite SKIPS with the
// reason. Unsandboxed host. The test feeds the same normalised packed latents to this port's bf16 decode (weights as shipped) and to its fp32 decode (weights upcast), and PRINTS
// PSNR / cosine / max-abs per case against diffusers' float32 image (REALVAE lines), plus against diffusers' bf16 image for context.
//
// The floors below are PROVISIONAL (implementer's call, pending owner confirmation). The follow-up cycle replaces them with measured values plus a margin after the owner's first run.

enum KleinVAEGolden {
    nonisolated static let file: URL? = {
        guard let dir = ProcessInfo.processInfo.environment["MLXUI_TEST_KLEIN_VAE_GOLDEN_DIR"] else { return nil }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("real_vae.safetensors")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }()
    nonisolated static let enabled = file != nil && Klein3gReal.present
    nonisolated static let skipReason: Comment = "needs TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_GOLDEN_DIR (folder with real_vae.safetensors from real_vae_goldens.py), the installed Klein vae/, and an unsandboxed test host"

    /// PROVISIONAL floors against diffusers float32 (implementer's call, pending owner confirmation; to be replaced by measured values plus a margin).
    static let provisionalPSNR = 40.0
    static let provisionalCosine = 0.99

    static let cases = ["128", "256"]

    static func requiredKeys() -> [String] {
        cases.flatMap { ["in.packed_\($0)", "out.fp32_\($0)", "out.bf16_\($0)", "stats.fp32_\($0)", "stats.bf16_\($0)"] }
    }

    struct Row { var label: String, size: String, reference: String, errors: Klein3gFix.Errors }

    /// Decodes every case with `vae`, compares with the golden's fp32 and bf16 images, prints a REALVAE line each, returns the rows. Throws a named error for a missing key.
    static func compare(vae: Flux2KleinVAE, label: String, golden: [String: MLXArray]) throws -> [Row] {
        for key in requiredKeys() where golden[key] == nil { throw Flux2KleinVAEError.badInput("the golden file has no \(key)") }
        var rows: [Row] = []
        for c in cases {
            guard let packed = golden["in.packed_\(c)"] else { continue }
            let image = try vae.decode(packedLatents: packed.asType(vae.computeDType))
            for ref in ["fp32", "bf16"] {
                guard let want = golden["out.\(ref)_\(c)"] else { continue }
                let e = try Klein3gFix.compare(image, want)
                print(String(format: "REALVAE %@ %@x%@ vs diffusers %@: PSNR %.2f dB, cosine %.8f, max-abs %.5f", label, c, c, ref, e.psnr, e.cosine, e.maxAbs))
                rows.append(Row(label: label, size: c, reference: ref, errors: e))
            }
        }
        return rows
    }
}

@Suite(.enabled(if: KleinVAEGolden.enabled, KleinVAEGolden.skipReason), .serialized)
struct Flux2KleinRealVAEGoldenTests {
    @Test(arguments: [("bf16", nil as DType?), ("fp32", DType.float32)])
    func theRealDecoderMatchesDiffusersOnFixedLatents(variant: (String, DType?)) throws {
        let dir = try #require(Klein3gReal.directory)
        let golden = try loadArrays(url: try #require(KleinVAEGolden.file))
        let vae = try Flux2KleinVAE.load(vaeDirectory: dir, computeDType: variant.1)
        let rows = try KleinVAEGolden.compare(vae: vae, label: variant.0, golden: golden)
        for r in rows where r.reference == "fp32" {          // floors against the card's reference precision only; bf16 rows are printed context
            #expect(r.errors.psnr >= KleinVAEGolden.provisionalPSNR, Comment(rawValue: "PROVISIONAL \(r.label) \(r.size) psnr \(r.errors.psnr)"))
            #expect(r.errors.cosine >= KleinVAEGolden.provisionalCosine, Comment(rawValue: "PROVISIONAL \(r.label) \(r.size) cosine \(r.errors.cosine)"))
        }
        #expect(rows.count == 4)
    }
}

/// Always runs: the golden file's key names and the comparison plumbing, on a tiny model with a synthesized golden in the script's format. It cannot prove the Python
/// side's names match (eyeball `requiredKeys` against real_vae_goldens.py on the first run), but it proves the Swift side reads what the format says and flags a missing key.
@Suite(.serialized) struct Flux2KleinRealVAEGoldenHarnessTests {
    @Test func theHarnessReadsTheScriptsKeyFormatAndFlagsAMissingKey() throws {
        let shape = Flux2KleinVAEShape(blockOutChannels: [8, 8, 16, 16], layersPerBlock: 1, groups: 4, latentChannels: 4)
        let (source, tensors) = Klein3gFix.randomCheckpoint(shape: shape)
        let vae = try Flux2KleinVAE.load(shape: shape, tensors: tensors, computeDType: .float32)
        // the harness hard-codes cases "128" / "256" (grids 8 / 16 -> 128 / 256 px at the real 16 px per cell); the tiny VAE has the same 16x factor
        var golden: [String: MLXArray] = [:]
        MLXRandom.seed(1)
        for (c, g) in [("128", 8), ("256", 16)] {
            let packed = MLXRandom.normal([1, 16, g / 4, g / 4])        // keep the tiny decode small: the factor is what matters, not the size
            golden["in.packed_\(c)"] = packed
            let img = try source.decode(packedLatents: packed)
            golden["out.fp32_\(c)"] = img
            golden["out.bf16_\(c)"] = img + 0.01
            golden["stats.fp32_\(c)"] = MLXArray([1, 1, 1] as [Float]); golden["stats.bf16_\(c)"] = MLXArray([1, 1, 1] as [Float])
        }
        let rows = try KleinVAEGolden.compare(vae: vae, label: "tiny", golden: golden)
        #expect(rows.count == 4)
        let exact = try #require(rows.first { $0.reference == "fp32" && $0.size == "128" })
        #expect(exact.errors.maxAbs == 0 && exact.errors.psnr == .infinity)
        let off = try #require(rows.first { $0.reference == "bf16" && $0.size == "128" })
        #expect(abs(off.errors.maxAbs - 0.01) < 1e-5 && off.errors.psnr.isFinite)
        var broken = golden
        broken.removeValue(forKey: "out.fp32_256")
        #expect(throws: Flux2KleinVAEError.self) { _ = try KleinVAEGolden.compare(vae: vae, label: "tiny", golden: broken) }
        // a perturbed weight lowers the agreement
        var bent = tensors
        bent["decoder.conv_out.bias"] = (bent["decoder.conv_out.bias"] ?? MLXArray(0)) + 0.5
        let vae2 = try Flux2KleinVAE.load(shape: shape, tensors: bent, computeDType: .float32)
        let worse = try KleinVAEGolden.compare(vae: vae2, label: "bent", golden: golden)
        #expect((worse.first { $0.reference == "fp32" && $0.size == "128" }?.errors.maxAbs ?? 0) > 0.4)
    }
}
