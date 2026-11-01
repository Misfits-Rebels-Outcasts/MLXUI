import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXUI

// MP7-3f, TASK B: this port's DiT on the REAL checkpoint against diffusers 0.37.1 bf16 on CPU, from a file the OWNER makes with
// RSI/notes/claude-code/modelpack/klein-ref/real_dit_goldens.py (not committed, not run by the implementer). TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_GOLDEN_DIR names the
// folder holding real_dit.safetensors; the installed Klein model (Direct store, or TEST_RUNNER_MLXUI_TEST_KLEIN_DIR) supplies the weights. Without both the suite
// SKIPS with the reason; it never passes without them. Unsandboxed host (see the script's header for the command).
//
// What it prints (REALDIT lines): cosine and relative L2 against diffusers at the taps (double 0 / last, single 0 / last, final prediction), and for EVERY block the
// L2 norm, mean |x| and max |x| of this port's output next to diffusers' and their ratio -- a bf16 intermediate that is silently upcast, or an outlier channel that is
// lost, shows up as a norm that drifts away from the diffusers list. Limits are the measured worst values plus a margin (MP7-3f-FIX-1; see `Limits`).

enum KleinDiTGolden {
    nonisolated static let file: URL? = {
        guard let dir = ProcessInfo.processInfo.environment["MLXUI_TEST_KLEIN_DIT_GOLDEN_DIR"] else { return nil }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("real_dit.safetensors")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }()
    nonisolated static let enabled = file != nil && Klein3eHeader.present
    nonisolated static let skipReason: Comment = "needs TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_GOLDEN_DIR (folder with real_dit.safetensors from real_dit_goldens.py), the installed Klein transformer/, and an unsandboxed test host"

    /// Limits per variant, set (MP7-3f-FIX-1) AFTER the owner's first real run, with a margin under the measured worst values. They pin measured behaviour of this port
    /// against diffusers bf16 on CPU; they are not quality claims. The margins are the implementer's call, pending owner confirmation.
    /// Measured worst over the six taps + pred / over every block's L2 ratio (ours/diffusers):
    ///   bf16: cosine 0.999264 (pred), relL2 0.03838, L2 ratio 0.9956...1.0038     -> cosine >= 0.998, relL2 <= 0.06, L2 ratio within 2%
    ///   int8: cosine 0.997260 (pred), relL2 0.07399, L2 ratio 0.9943...1.0069     -> cosine >= 0.995, relL2 <= 0.10, L2 ratio within 2%
    ///   int4: cosine 0.943265 (pred), relL2 0.33473, L2 ratio 0.9536...1.0365     -> cosine >= 0.93,  relL2 <= 0.40, L2 ratio within 8%
    struct Limits { let cosine: Double, relL2: Double, normBand: Double }
    static let bf16Limits = Limits(cosine: 0.998, relL2: 0.06, normBand: 0.02)
    static let int8Limits = Limits(cosine: 0.995, relL2: 0.10, normBand: 0.02)
    static let int4Limits = Limits(cosine: 0.93, relL2: 0.40, normBand: 0.08)

    static func check(_ label: String, _ r: Result, _ l: Limits) {
        for t in r.taps {
            #expect(t.cosine >= l.cosine, Comment(rawValue: "\(label) \(t.name) cosine \(t.cosine) < \(l.cosine)"))
            #expect(t.relL2 <= l.relL2, Comment(rawValue: "\(label) \(t.name) relL2 \(t.relL2) > \(l.relL2)"))
        }
        for group in groups {
            for (i, (a, b)) in zip(r.stats[group] ?? [], r.theirs[group] ?? []).enumerated() {
                #expect(abs(a.l2 / b.l2 - 1) <= l.normBand, Comment(rawValue: "\(label) \(group)[\(i)] L2 ratio \(a.l2 / b.l2)"))
            }
        }
    }

    /// Per-block statistics of one output: L2 norm, mean |x|, max |x| (float32).
    struct BlockStats { var l2: Double, meanAbs: Double, maxAbs: Double }

    struct Result {
        var taps: [(name: String, cosine: Double, relL2: Double, maxAbs: Double)] = []
        var stats: [String: [BlockStats]] = [:]            // group -> per block
        var theirs: [String: [BlockStats]] = [:]
    }

    static let groups = ["double_txt", "double_img", "single", "pred"]

    static func statsOf(_ a: MLXArray) throws -> BlockStats {
        let x = a.asType(.float32)
        let r = try MLXErrorGuard.read(stacked([sqrt((x * x).sum()), abs(x).mean(), abs(x).max()]), as: Float.self)
        return BlockStats(l2: Double(r[0]), meanAbs: Double(r[1]), maxAbs: Double(r[2]))
    }

    /// (group, index) of a probe stage name.
    static func group(of stage: String) -> (String, Int)? {
        if stage == "out" { return ("pred", 0) }
        let p = stage.split(separator: "_")
        if p.count == 3, p[0] == "double", let i = Int(p[1]) { return (p[2] == "txt" ? "double_txt" : "double_img", i) }
        if p.count == 2, p[0] == "single", let i = Int(p[1]) { return ("single", i) }
        return nil
    }

    static func maxAbsDiff(_ x: [Float], _ y: [Float]) -> Double { zip(x, y).reduce(0.0) { max($0, Double(abs($1.0 - $1.1))) } }

    static func floats(_ a: MLXArray) throws -> [Float] { try MLXErrorGuard.read(a.asType(.float32).reshaped(-1), as: Float.self) }

    /// The golden keys the comparison needs for a model with these block counts (the script writes them for 5 + 20).
    static func requiredKeys(doubleBlocks: Int, singleBlocks: Int) -> [String] {
        let d = doubleBlocks - 1, s = singleBlocks - 1
        var keys = ["in.hidden", "in.encoder", "in.sigma", "in.img_ids", "in.txt_ids", "out.pred",
                    "out.double_0_txt", "out.double_0_img", "out.double_\(d)_txt", "out.double_\(d)_img", "out.single_0", "out.single_\(s)"]
        for g in groups { keys += ["stats.\(g).l2", "stats.\(g).meanabs", "stats.\(g).maxabs"] }
        return keys
    }

    /// Runs `model` on the golden's inputs (cast to `dtype`) with the probe, and compares with diffusers' outputs. Throws a named error when a golden key is missing.
    static func compare(model: Flux2KleinTransformer, golden: [String: MLXArray], dtype: DType) throws -> Result {
        let d = model.doubleBlocks.count - 1, s = model.singleBlocks.count - 1
        for key in requiredKeys(doubleBlocks: model.doubleBlocks.count, singleBlocks: model.singleBlocks.count) where golden[key] == nil {
            throw Flux2KleinTransformerError.badInput("the golden file has no \(key)")
        }
        func g(_ k: String) throws -> MLXArray {
            guard let a = golden[k] else { throw Flux2KleinTransformerError.badInput("the golden file has no \(k)") }
            return a
        }
        let wanted: [(String, String)] = [("double_0_txt", "out.double_0_txt"), ("double_0_img", "out.double_0_img"), ("double_\(d)_txt", "out.double_\(d)_txt"),
                                          ("double_\(d)_img", "out.double_\(d)_img"), ("single_0", "out.single_0"), ("single_\(s)", "out.single_\(s)")]
        var result = Result()
        var failure: Error?
        var perGroup: [String: [Int: BlockStats]] = [:]
        let pred = try model(hidden: try g("in.hidden").asType(dtype), encoder: try g("in.encoder").asType(dtype), sigma: try g("in.sigma"),
                             imgIDs: try g("in.img_ids"), txtIDs: try g("in.txt_ids")) { stage, value in
            do {
                eval(value)
                if let (group, index) = group(of: stage) { perGroup[group, default: [:]][index] = try statsOf(value) }
                if let key = wanted.first(where: { $0.0 == stage })?.1 {
                    let want = try g(key)
                    let x = try floats(value), y = try floats(want)
                    result.taps.append((stage, KleinCosine.cosine(x, y), Flux2KleinEncoderQuantizationTests.relL2(x, y), maxAbsDiff(x, y)))
                }
            } catch { failure = error }
        }
        if let failure { throw failure }
        let want = try g("out.pred")
        let x = try floats(pred), y = try floats(want)
        result.taps.append(("pred", KleinCosine.cosine(x, y), Flux2KleinEncoderQuantizationTests.relL2(x, y), maxAbsDiff(x, y)))
        for group in groups {
            let l2 = try floats(g("stats.\(group).l2")), mean = try floats(g("stats.\(group).meanabs")), mx = try floats(g("stats.\(group).maxabs"))
            result.theirs[group] = (0 ..< l2.count).map { BlockStats(l2: Double(l2[$0]), meanAbs: Double(mean[$0]), maxAbs: Double(mx[$0])) }
            result.stats[group] = (0 ..< l2.count).map { perGroup[group]?[$0] ?? BlockStats(l2: .nan, meanAbs: .nan, maxAbs: .nan) }
        }
        return result
    }

    static func report(_ label: String, _ r: Result) {
        for t in r.taps { print(String(format: "REALDIT %@ tap %@: cosine %.6f relL2 %.5f max-abs %.4f", label, t.name, t.cosine, t.relL2, t.maxAbs)) }
        for group in groups {
            guard let ours = r.stats[group], let theirs = r.theirs[group] else { continue }
            func list(_ f: (BlockStats) -> Double) -> String { ours.indices.map { String(format: "%.3f", f(ours[$0])) }.joined(separator: " ") }
            func their(_ f: (BlockStats) -> Double) -> String { theirs.indices.map { String(format: "%.3f", f(theirs[$0])) }.joined(separator: " ") }
            func ratio(_ f: (BlockStats) -> Double) -> String { ours.indices.map { String(format: "%.4f", f(ours[$0]) / max(f(theirs[$0]), 1e-30)) }.joined(separator: " ") }
            print("REALDIT \(label) \(group) L2 port:    [\(list { $0.l2 })]")
            print("REALDIT \(label) \(group) L2 diffusers: [\(their { $0.l2 })]")
            print("REALDIT \(label) \(group) L2 ratio:   [\(ratio { $0.l2 })]")
            print("REALDIT \(label) \(group) mean|x| ratio: [\(ratio { $0.meanAbs })]   max|x| ratio: [\(ratio { $0.maxAbs })]")
        }
    }

    /// Writes a golden in the script's format from a model's own outputs (the harness test only: it proves the key names and the comparison, not parity).
    static func synthesizeGolden(model: Flux2KleinTransformer, inputs: (hidden: MLXArray, encoder: MLXArray, sigma: MLXArray, imgIDs: MLXArray, txtIDs: MLXArray),
                                 dtype: DType) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = ["in.hidden": inputs.hidden, "in.encoder": inputs.encoder, "in.sigma": inputs.sigma, "in.img_ids": inputs.imgIDs, "in.txt_ids": inputs.txtIDs]
        var perGroup: [String: [Int: BlockStats]] = [:]
        let d = model.doubleBlocks.count - 1, s = model.singleBlocks.count - 1
        let keep: Set<String> = ["double_0_txt", "double_0_img", "double_\(d)_txt", "double_\(d)_img", "single_0", "single_\(s)"]
        let pred = try model(hidden: inputs.hidden.asType(dtype), encoder: inputs.encoder.asType(dtype), sigma: inputs.sigma, imgIDs: inputs.imgIDs, txtIDs: inputs.txtIDs) { stage, value in
            eval(value)
            if let (group, index) = group(of: stage), let st = try? statsOf(value) { perGroup[group, default: [:]][index] = st }
            if keep.contains(stage) { out["out." + stage] = value.asType(.float32) }
        }
        out["out.pred"] = pred.asType(.float32)
        for group in groups {
            let all = (perGroup[group] ?? [:]).sorted { $0.key < $1.key }.map(\.value)
            out["stats.\(group).l2"] = MLXArray(all.map { Float($0.l2) })
            out["stats.\(group).meanabs"] = MLXArray(all.map { Float($0.meanAbs) })
            out["stats.\(group).maxabs"] = MLXArray(all.map { Float($0.maxAbs) })
        }
        return out
    }
}

/// Always runs: the comparison harness itself, on a tiny random model and a golden synthesized from that same model (so cosine must be 1 and every norm ratio 1).
/// This pins the key names, the tap selection and the printing, NOT parity (that needs the owner's real golden).
@Suite(.serialized) struct Flux2KleinRealDiTGoldenHarnessTests {
    @Test func theHarnessReadsTheScriptsKeysAndReportsPerfectAgreementWithItself() throws {
        let (model, _) = Klein3fTiny.source()
        let i = Klein3fTiny.inputs()
        let golden = try KleinDiTGolden.synthesizeGolden(model: model, inputs: i, dtype: .float32)
        for key in KleinDiTGolden.requiredKeys(doubleBlocks: 2, singleBlocks: 2) { #expect(golden[key] != nil, Comment(rawValue: key)) }
        let r = try KleinDiTGolden.compare(model: model, golden: golden, dtype: .float32)
        KleinDiTGolden.report("harness", r)
        #expect(r.taps.map(\.name) == ["double_0_txt", "double_0_img", "double_1_txt", "double_1_img", "single_0", "single_1", "pred"])
        #expect(r.taps.allSatisfy { $0.cosine > 0.9999999 && $0.relL2 < 1e-6 })
        #expect(r.stats["double_txt"]?.count == 2 && r.stats["single"]?.count == 2 && r.stats["pred"]?.count == 1)
        for group in KleinDiTGolden.groups {
            for (a, b) in zip(r.stats[group] ?? [], r.theirs[group] ?? []) { #expect(abs(a.l2 / b.l2 - 1) < 1e-5 && abs(a.maxAbs / b.maxAbs - 1) < 1e-5) }
        }
    }

    @Test func aMissingKeyIsNamed() throws {
        let (model, _) = Klein3fTiny.source()
        var golden = try KleinDiTGolden.synthesizeGolden(model: model, inputs: Klein3fTiny.inputs(), dtype: .float32)
        golden.removeValue(forKey: "out.single_1")
        #expect(throws: Flux2KleinTransformerError.badInput("the golden file has no out.single_1")) { _ = try KleinDiTGolden.compare(model: model, golden: golden, dtype: .float32) }
    }

    @Test func aWrongPortIsSeenAsAWorseCosine() throws {
        // Sanity that the harness can fail: perturb one block's weights and the taps after it move.
        let (model, _) = Klein3fTiny.source()
        let golden = try KleinDiTGolden.synthesizeGolden(model: model, inputs: Klein3fTiny.inputs(), dtype: .float32)
        model.singleBlocks[0].update(parameters: ModuleParameters.unflattened(["attn.to_out.weight": model.singleBlocks[0].attn.toOut.weight * 1.5]))
        eval(model)
        let r = try KleinDiTGolden.compare(model: model, golden: golden, dtype: .float32)
        #expect((r.taps.first { $0.name == "single_0" }?.cosine ?? 1) < 0.9999)
        #expect((r.taps.first { $0.name == "double_0_txt" }?.cosine ?? 0) > 0.9999999)     // before the change: untouched
    }
}

/// Owner-run, opt-in. Skips with the reason when the golden or the install is absent.
@Suite(.enabled(if: KleinDiTGolden.enabled, KleinDiTGolden.skipReason), .serialized)
struct Flux2KleinRealDiTGoldenTests {
    static func golden() throws -> [String: MLXArray] { try loadArrays(url: try #require(KleinDiTGolden.file)) }

    @Test func bf16PortVersusDiffusers() throws {
        let dir = try #require(Klein3eHeader.directory)
        let golden = try Self.golden()
        let r: KleinDiTGolden.Result
        do {
            let model = try Flux2KleinTransformer.loadUnquantizedForTesting(transformerDirectory: dir)
            r = try KleinDiTGolden.compare(model: model, golden: golden, dtype: .bfloat16)
        }
        Memory.clearCache()
        KleinDiTGolden.report("bf16", r)
        KleinDiTGolden.check("bf16", r, KleinDiTGolden.bf16Limits)
    }

    @Test func int8PortVersusDiffusers() throws { try quantized(bits: 8, keepFrontEndBF16: false, label: "int8", limits: KleinDiTGolden.int8Limits) }

    @Test func int4PortVersusDiffusers() throws { try quantized(bits: 4, keepFrontEndBF16: false, label: "int4", limits: KleinDiTGolden.int4Limits) }

    /// Print-only extra (implementer's call, pending owner confirmation): int4 blocks, but context_embedder, the three modulation Linears and norm_out.linear left in bf16.
    /// Its limits are the int4 ones, so it can only fail if it is WORSE than the shipping int4 policy's measured bound.
    @Test func int4KeepingTheFrontEndBF16VersusDiffusers() throws {
        try quantized(bits: 4, keepFrontEndBF16: true, label: "int4-frontbf16", limits: KleinDiTGolden.int4Limits)
    }

    private func quantized(bits: Int, keepFrontEndBF16: Bool, label: String, limits: KleinDiTGolden.Limits) throws {
        let dir = try #require(Klein3eHeader.directory)
        let golden = try Self.golden()
        let r: KleinDiTGolden.Result
        do {
            let model = try Flux2KleinTransformer.loadForTesting(transformerDirectory: dir, bits: bits, keepFrontEndBF16: keepFrontEndBF16)
            print(String(format: "REALDIT %@ weight bytes %.3f GB (blocks %.3f GB)", label, Double(model.parameterBytes()) / 1e9,
                         Double(model.parameterBytes { $0.hasPrefix("transformer_blocks.") || $0.hasPrefix("single_transformer_blocks.") }) / 1e9))
            r = try KleinDiTGolden.compare(model: model, golden: golden, dtype: .bfloat16)
        }
        Memory.clearCache()
        KleinDiTGolden.report(label, r)
        KleinDiTGolden.check(label, r, limits)
    }
}
