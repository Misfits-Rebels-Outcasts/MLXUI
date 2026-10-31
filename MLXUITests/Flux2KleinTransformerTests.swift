import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXUI

// MP7-3f: the Klein single-stream block, the whole MMDiT forward and the streamed loader.
// Parity: diffusers 0.37.1 `Flux2Transformer2DModel` on g3f_dit_stack.safetensors (tiny, fp32: 3 double + 3 single blocks; `mid.*` after EVERY block, `out.pred`).
// Loader: tiny random-weight shards written/held by the test; no real weights in this file's always-run tests.

final class Klein3fProbe {
    var seen: [(String, MLXArray)] = []
    func callAsFunction(_ name: String, _ value: MLXArray) { seen.append((name, value)) }
}

enum Klein3f {
    static var fixtures: URL { Klein3d.fixtures }
    static let g3f: [String: MLXArray] = (try? MLX.loadArrays(url: fixtures.appendingPathComponent("g3f_dit_stack.safetensors"))) ?? [:]

    /// TF of dit_stack_goldens.py: heads 2 x head_dim 8 = inner 16, in 8, joint 12, 16 timestep channels, axes (2, 2, 2, 2), theta 2000; 3 double + 3 single.
    static let shape = Flux2KleinTransformerShape(
        embedder: Flux2KleinEmbedderShape(inChannels: 8, jointAttentionDim: 12, inner: 16, timestepChannels: 16, eps: 1e-6, axesDims: [2, 2, 2, 2], ropeTheta: 2000),
        heads: 2, headDim: 8, mlpRatio: 3.0, doubleBlocks: 3, singleBlocks: 3)

    static func weights() -> [String: MLXArray] {
        var w: [String: MLXArray] = [:]
        for (k, v) in g3f where k.hasPrefix("w.") { w[String(k.dropFirst(2))] = v }
        return w
    }

    static func model() throws -> Flux2KleinTransformer {
        let m = Flux2KleinTransformer(shape: shape)
        try m.update(parameters: ModuleParameters.unflattened(weights()), verify: .all)
        eval(m)
        return m
    }

    /// The golden holds two runs of the same weights and inputs: the primary one at sigma 0.005 and an `alt.` one at the realistic sigma 0.7 (see dit_stack_goldens.py:
    /// sigma 0.7 puts sinusoid angles up to 700 rad in float32, where torch and MLX differ by ~1e-5 and the stack passes about half of that to the output).
    enum Run: String, CaseIterable {
        case primary = "", alt = "alt."
        var timestep: String { self == .primary ? "in.timestep" : "alt.timestep" }
        var mid: String { self == .primary ? "mid." : "alt.mid." }
        var pred: String { self == .primary ? "out.pred" : "alt.pred" }
    }

    static func run(_ m: Flux2KleinTransformer, _ run: Run = .primary, probe: Klein3fProbe? = nil) throws -> MLXArray {
        try m(hidden: try #require(g3f["in.hidden_states"]), encoder: try #require(g3f["in.encoder_hidden_states"]), sigma: try #require(g3f[run.timestep]),
              imgIDs: try #require(g3f["in.img_ids"]), txtIDs: try #require(g3f["in.txt_ids"]), probe: probe.map { p in { p($0, $1) } })
    }

    struct Err { let maxAbs: Float; let oneMinusCos: Double }

    static func measure(_ name: String, _ mine: MLXArray, _ key: String) throws -> Err {
        let want = try #require(g3f[key], Comment(rawValue: key))
        #expect(mine.shape == want.shape, Comment(rawValue: "\(name) shape \(mine.shape) vs \(want.shape)"))
        let d = try MLXErrorGuard.read(abs(mine.asType(.float32) - want).max().reshaped(1), as: Float.self)[0]
        let c = try KleinCosine.of(mine, want)
        print("DITPARITY \(name): max-abs \(d) 1-cos \(1 - c)")
        return Err(maxAbs: d, oneMinusCos: 1 - c)
    }

    // Floors set AFTER the first measurement (2026-10-31, see MP7-3f-report.md); the owner confirmed the targets max-abs 1e-6 / cosine 0.999999 for tiny fp32 block parity.
    static let maxAbsFloor: Float = 1e-6
    static let cosineFloor = 0.999999
    // Alt run (sigma 0.7), measured 2026-10-31: isolated worst 4.05e-6, stacked worst 7.51e-6 (1-cos <= 3.0e-12). Floor 2e-5 (~2.7x): implementer's call, pending owner confirmation.
    static let altMaxAbsFloor: Float = 2e-5

    /// The block outputs in forward order, with the stage name the probe uses.
    static let stages: [String] = (0 ..< 3).flatMap { ["double_\($0)_txt", "double_\($0)_img"] } + (0 ..< 3).map { "single_\($0)" }
}

@Suite(.serialized) struct Flux2KleinTransformerParityTests {
    @Test func theGoldenStacksAtLeastTwoDoubleAndTwoSingleBlocks() {
        let w = Klein3f.weights()
        let doubles = Set(w.keys.compactMap { $0.hasPrefix("transformer_blocks.") ? $0.split(separator: ".")[1] : nil })
        let singles = Set(w.keys.compactMap { $0.hasPrefix("single_transformer_blocks.") ? $0.split(separator: ".")[1] : nil })
        #expect(doubles.count == 3 && singles.count == 3)
    }

    @Test func singleBlockKeysAreTheFourDiffusersKeys() {
        let b = Flux2KleinSingleBlock(dim: 16, heads: 2, headDim: 8, mlpRatio: 3.0, eps: 1e-6)
        let keys = Set(b.parameters().flattened().map { $0.0 })
        #expect(keys == ["attn.to_qkv_mlp_proj.weight", "attn.to_out.weight", "attn.norm_q.weight", "attn.norm_k.weight"])
        #expect(keys == Flux2KleinSingleBlock.expectedTensorNames)
        let shapes = Dictionary(uniqueKeysWithValues: b.parameters().flattened().map { ($0.0, $0.1.shape) })
        #expect(shapes["attn.to_qkv_mlp_proj.weight"] == [3 * 16 + 2 * 48, 16])        // [27648, 3072] at real size
        #expect(shapes["attn.to_out.weight"] == [16, 16 + 48])                          // [3072, 12288]
        #expect(shapes["attn.norm_q.weight"] == [8] && shapes["attn.norm_k.weight"] == [8])
    }

    @Test func theTransformerKeysAreExactlyTheGoldensKeys() {
        let mine = Set(Flux2KleinTransformer(shape: Klein3f.shape).parameters().flattened().map { $0.0 })
        #expect(mine == Set(Klein3f.weights().keys))
        #expect(mine.count == 9 + 3 * 16 + 3 * 4)
    }

    /// Floors per run. Primary (sigma 0.005): the owner-confirmed block-parity targets. Alt (sigma 0.7): see `Klein3f.Run`; the max-abs floor there is the implementer's call,
    /// pending owner confirmation, set after the measurement (about 2.7x the worst measured stacked value).
    static func floors(_ run: Klein3f.Run) -> (maxAbs: Float, cosine: Double) {
        run == .primary ? (Klein3f.maxAbsFloor, Klein3f.cosineFloor) : (Klein3f.altMaxAbsFloor, Klein3f.cosineFloor)
    }

    /// Each block alone, fed the golden's own previous-stage output: isolates a block's error from the accumulated one.
    @Test(arguments: Klein3f.Run.allCases) func everyBlockMatchesDiffusersFedTheGoldensPreviousStage(run: Klein3f.Run) throws {
        let m = try Klein3f.model()
        let g = Klein3f.g3f
        let cond = m.conditioningForTesting(sigma: try #require(g[run.timestep]))
        let (cos, sin) = Flux2KleinRoPE.jointCosSin(textIDs: try #require(g["in.txt_ids"]), imageIDs: try #require(g["in.img_ids"]), axesDims: [2, 2, 2, 2], theta: 2000)
        var img = m.xEmbedder(try #require(g["in.hidden_states"]))
        var txt = m.contextEmbedder(try #require(g["in.encoder_hidden_states"]))
        var worst = Klein3f.Err(maxAbs: 0, oneMinusCos: 0)
        func note(_ e: Klein3f.Err) { worst = Klein3f.Err(maxAbs: max(worst.maxAbs, e.maxAbs), oneMinusCos: max(worst.oneMinusCos, e.oneMinusCos)) }
        for (i, block) in m.doubleBlocks.enumerated() {
            let out = block(img: img, txt: txt, imgMod: cond.doubleImg, txtMod: cond.doubleTxt, cos: cos, sin: sin)
            eval(out.txt, out.img)
            note(try Klein3f.measure("\(run) isolated double_\(i)_txt", out.txt, "\(run.mid)double_\(i)_txt"))
            note(try Klein3f.measure("\(run) isolated double_\(i)_img", out.img, "\(run.mid)double_\(i)_img"))
            // feed the GOLDEN's output forward, not ours
            txt = try #require(g["\(run.mid)double_\(i)_txt"]); img = try #require(g["\(run.mid)double_\(i)_img"])
        }
        var joint = concatenated([txt, img], axis: 1)
        for (i, block) in m.singleBlocks.enumerated() {
            let out = block(joint, mod: cond.single[0], cos: cos, sin: sin)
            eval(out)
            note(try Klein3f.measure("\(run) isolated single_\(i)", out, "\(run.mid)single_\(i)"))
            joint = try #require(g["\(run.mid)single_\(i)"])
        }
        print("DITPARITY \(run) isolated blocks worst: max-abs \(worst.maxAbs) 1-cos \(worst.oneMinusCos)")
        let f = Self.floors(run)
        #expect(worst.maxAbs <= f.maxAbs)
        #expect(1 - worst.oneMinusCos >= f.cosine)
    }

    /// The whole forward, chained: every block output and the final prediction against diffusers. This is the stacking check (residual order, the shared modulation reused
    /// by every block, the [text, image] hand-off between the double and the single stage).
    @Test(arguments: Klein3f.Run.allCases) func theWholeStackMatchesDiffusersAtEveryBlockAndAtTheOutput(run: Klein3f.Run) throws {
        let m = try Klein3f.model()
        let probe = Klein3fProbe()
        let pred = try Klein3f.run(m, run, probe: probe)
        eval(pred)
        #expect(probe.seen.map(\.0) == Klein3f.stages + ["out"])
        var errs: [Klein3f.Err] = []
        for name in Klein3f.stages {
            let mine = try #require(probe.seen.first { $0.0 == name }).1
            eval(mine)
            errs.append(try Klein3f.measure("\(run) stack \(name)", mine, run.mid + name))
        }
        errs.append(try Klein3f.measure("\(run) stack pred", pred, run.pred))
        let f = Self.floors(run)
        for e in errs {
            #expect(e.maxAbs <= f.maxAbs)
            #expect(1 - e.oneMinusCos >= f.cosine)
        }
        print("DITPARITY \(run) stack worst max-abs \(errs.map(\.maxAbs).max() ?? 0), worst 1-cos \(errs.map(\.oneMinusCos).max() ?? 0)")
    }

    @Test func theSingleStageRunsTextFirst() throws {
        // A [image, text] hand-off with the same (text-first) rotary table is a different computation, by a wide margin.
        let m = try Klein3f.model()
        let g = Klein3f.g3f
        let cond = m.conditioningForTesting(sigma: try #require(g["in.timestep"]))
        let (cos, sin) = Flux2KleinRoPE.jointCosSin(textIDs: try #require(g["in.txt_ids"]), imageIDs: try #require(g["in.img_ids"]), axesDims: [2, 2, 2, 2], theta: 2000)
        let txt = try #require(g["mid.double_2_txt"]), img = try #require(g["mid.double_2_img"])
        let right = m.singleBlocks[0](concatenated([txt, img], axis: 1), mod: cond.single[0], cos: cos, sin: sin)
        let wrong = m.singleBlocks[0](concatenated([img, txt], axis: 1), mod: cond.single[0], cos: cos, sin: sin)
        // compare the same stream rows: right's image rows (4...) vs wrong's image rows (0..<6)
        let d = try MLXErrorGuard.read(abs(right[0..., 4..., 0...] - wrong[0..., 0 ..< 6, 0...]).max().reshaped(1), as: Float.self)[0]
        print("DITPARITY wrong hand-off order moves the image rows by \(d)")
        #expect(d > 1e-3)
    }

    @Test func zeroGatesMakeEveryBlockTheIdentity() throws {
        let m = try Klein3f.model()
        let g = Klein3f.g3f
        let cond = m.conditioningForTesting(sigma: try #require(g["in.timestep"]))
        let (cos, sin) = Flux2KleinRoPE.jointCosSin(textIDs: try #require(g["in.txt_ids"]), imageIDs: try #require(g["in.img_ids"]), axesDims: [2, 2, 2, 2], theta: 2000)
        let x = try #require(g["mid.double_2_txt"]), y = try #require(g["mid.double_2_img"])
        let joint = concatenated([x, y], axis: 1)
        let s = cond.single[0]
        let zero = Flux2KleinModulationSet(shift: s.shift, scale: s.scale, gate: MLXArray.zeros(like: s.gate))
        let out = m.singleBlocks[0](joint, mod: zero, cos: cos, sin: sin)
        #expect(try MLXErrorGuard.read(abs(out - joint).max().reshaped(1), as: Float.self)[0] == 0)
    }

    @Test func theTextRowsAreDroppedAndTheyStillMatterThroughAttention() throws {
        let m = try Klein3f.model()
        let base = try Klein3f.run(m)
        #expect(base.shape == [1, 6, 8])
        let g = Klein3f.g3f
        let enc = try #require(g["in.encoder_hidden_states"])
        let ramp = MLXArray((0 ..< 12).map { Float($0) / 4 }).reshaped(1, 1, 12)
        let bumped = concatenated([enc[0 ..< 1, 0 ..< 1] + ramp, enc[0 ..< 1, 1...]], axis: 1)
        let out = try m(hidden: try #require(g["in.hidden_states"]), encoder: bumped, sigma: try #require(g["in.timestep"]),
                        imgIDs: try #require(g["in.img_ids"]), txtIDs: try #require(g["in.txt_ids"]))
        #expect(try MLXErrorGuard.read(abs(out - base).max().reshaped(1), as: Float.self)[0] > 1e-4)
    }

    @Test func badInputsAreRefusedWithAReason() throws {
        let m = try Klein3f.model()
        let g = Klein3f.g3f
        #expect(throws: Flux2KleinTransformerError.self) {
            _ = try m(hidden: MLXArray.zeros([1, 6, 9]), encoder: try #require(g["in.encoder_hidden_states"]), sigma: try #require(g["in.timestep"]),
                      imgIDs: try #require(g["in.img_ids"]), txtIDs: try #require(g["in.txt_ids"]))
        }
        #expect(throws: Flux2KleinTransformerError.self) {            // 5 image ids for 6 tokens
            _ = try m(hidden: try #require(g["in.hidden_states"]), encoder: try #require(g["in.encoder_hidden_states"]), sigma: try #require(g["in.timestep"]),
                      imgIDs: MLXArray.zeros([1, 5, 4]), txtIDs: try #require(g["in.txt_ids"]))
        }
    }

    @Test func aCancelledForwardThrowsCancellation() async throws {
        let m = try Klein3f.model()
        let task = Task { () throws -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try Klein3f.run(m)
            return true
        }
        if case let .failure(error) = await task.result { #expect(error is CancellationError) } else { Issue.record("a cancelled forward must stop") }
    }
}

extension Flux2KleinTransformer {
    /// The same conditioning products `Flux2KleinEmbedders.conditioning(sigma:dtype:)` builds, against this model's own modules (tests use it to run single blocks).
    func conditioningForTesting(sigma: MLXArray, dtype: DType = .float32) -> Flux2KleinEmbedders.Conditioning {
        let temb = timeGuidanceEmbed(sigma: sigma, dtype: dtype)
        return Flux2KleinEmbedders.Conditioning(temb: temb, doubleImg: doubleModulationImg.chunks(temb), doubleTxt: doubleModulationTxt.chunks(temb),
                                                single: singleModulation.chunks(temb))
    }
}

// MARK: - Fixture integrity (new file; the existing integrity test is untouched)

@Suite struct Flux2KleinDiTStackFixtureTests {
    static let sha256 = "d85451a5f2b5c53f3c27932719c4b7446f5710c4a5f0dbebed8bca2b6e68cc1a"
    static let scriptSha256 = "561066efb4001a36120abe9b1208dd4c24fc29bb8ae9f532051082ff99952e6a"

    @Test func theDiTStackGoldenIsByteIdenticalToWhatTheScriptWrote() throws {
        for (name, expected) in [("g3f_dit_stack.safetensors", Self.sha256), ("dit_stack_goldens.py", Self.scriptSha256)] {
            let data = try Data(contentsOf: Klein3f.fixtures.appendingPathComponent(name))
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            #expect(digest == expected, Comment(rawValue: name))
        }
    }

    @Test func theFixtureFolderStaysUnderTheOwnersBudget() throws {
        let files = try FileManager.default.contentsOfDirectory(at: Klein3f.fixtures, includingPropertiesForKeys: [.fileSizeKey])
        let total = try files.reduce(0) { $0 + (try $1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) }
        print("DITFIXTURE folder total \(total) bytes in \(files.count) files")
        #expect(total < 2_000_000, "owner ruling 2026-10-31: the fixture folder stays under 2 MB")
    }
}

// MARK: - Streamed loader (tiny random-weight shards)

enum Klein3fTiny {
    /// Widths that are multiples of 64, so MLX can really quantize: inner 2 x 32 = 64, mlp hidden 192, in/joint/timestep 64.
    static let shape = Flux2KleinTransformerShape(
        embedder: Flux2KleinEmbedderShape(inChannels: 64, jointAttentionDim: 64, inner: 64, timestepChannels: 64, eps: 1e-6, axesDims: [8, 8, 8, 8], ropeTheta: 2000),
        heads: 2, headDim: 32, mlpRatio: 3.0, doubleBlocks: 2, singleBlocks: 2)

    /// Random weights with real magnitude (matrices N(0, 1/fan_in), norms 1 + 0.1 N(0, 1)); the SOURCE model holds them.
    static func source() -> (model: Flux2KleinTransformer, weights: [String: MLXArray]) {
        MLXRandom.seed(20261031)
        let m = Flux2KleinTransformer(shape: shape)
        var w: [String: MLXArray] = [:]
        for (k, p) in m.parameters().flattened() {
            w[k] = p.ndim == 1 ? 1 + 0.1 * MLXRandom.normal(p.shape) : MLXRandom.normal(p.shape) / Float(p.dim(-1)).squareRoot()
        }
        try? m.update(parameters: ModuleParameters.unflattened(w), verify: .all)
        eval(m)
        return (m, w)
    }

    static func inputs() -> (hidden: MLXArray, encoder: MLXArray, sigma: MLXArray, imgIDs: MLXArray, txtIDs: MLXArray) {
        MLXRandom.seed(7)
        let z = MLXArray.zeros([4]), n = MLXArray(Array(0 ..< 4).map { Float($0) })
        let txt = stacked([z, z, z, n], axis: -1)
        let hh = MLXArray([0, 0, 0, 1, 1, 1] as [Float]), ww = MLXArray([0, 1, 2, 0, 1, 2] as [Float]), z6 = MLXArray.zeros([6])
        let img = stacked([z6, hh, ww, z6], axis: -1)
        return (MLXRandom.normal([1, 6, 64]), MLXRandom.normal([1, 4, 64]), MLXArray([0.5] as [Float]), img, txt)
    }

    static func forward(_ m: Flux2KleinTransformer) throws -> MLXArray {
        let i = inputs()
        return try m(hidden: i.hidden, encoder: i.encoder, sigma: i.sigma, imgIDs: i.imgIDs, txtIDs: i.txtIDs)
    }

    static func maxAbs(_ a: MLXArray, _ b: MLXArray) throws -> Float {
        try MLXErrorGuard.read(abs(a.asType(.float32) - b.asType(.float32)).max().reshaped(1), as: Float.self)[0]
    }

    static func load(_ shards: [Flux2KleinTransformer.Shard], bits: Int?) throws -> Flux2KleinTransformer {
        try Flux2KleinTransformer.load(shape: shape, shards: shards, bits: bits, groupSize: 64)
    }
}

@Suite(.serialized) struct Flux2KleinTransformerLoaderTests {
    @Test func anUnquantizedStreamedLoadReproducesTheSourceModelExactly() throws {
        let (source, weights) = Klein3fTiny.source()
        let reference = try Klein3fTiny.forward(source)
        let loaded = try Klein3fTiny.load([{ weights }], bits: nil)
        #expect(try Klein3fTiny.maxAbs(try Klein3fTiny.forward(loaded), reference) == 0)
    }

    @Test func aBlockThatStraddlesTwoShardsWaitsForItsSecondHalf() throws {
        let (source, weights) = Klein3fTiny.source()
        let reference = try Klein3fTiny.forward(source)
        let late: Set<String> = ["single_transformer_blocks.1.attn.to_out.weight", "transformer_blocks.0.ff.linear_out.weight", "proj_out.weight"]
        var first = weights, second: [String: MLXArray] = [:]
        for k in late { second[k] = first.removeValue(forKey: k) }
        var events: [String] = []
        let loaded = try Flux2KleinTransformer.load(shape: Klein3fTiny.shape, shards: [{ first }, { second }], bits: nil) { events.append($0) }
        #expect(try Klein3fTiny.maxAbs(try Klein3fTiny.forward(loaded), reference) == 0)
        // the straddling blocks became ready only after the second shard opened
        let secondOpened = try #require(events.firstIndex { $0.hasPrefix("shard 1") })
        #expect(try #require(events.firstIndex(of: "double block 0 ready")) > secondOpened)
        #expect(try #require(events.firstIndex(of: "single block 1 ready")) > secondOpened)
        #expect(try #require(events.firstIndex(of: "double block 1 ready")) < secondOpened)
        #expect(events.last == "front end ready")
    }

    @Test func aDirectoryOfShardsLoadsWithoutAnIndexFile() throws {
        let (source, weights) = Klein3fTiny.source()
        let reference = try Klein3fTiny.forward(source)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("klein3f-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let firstHalf = Set(weights.keys.sorted().prefix(weights.count / 2))
        try MLX.save(arrays: weights.filter { firstHalf.contains($0.key) }, url: dir.appendingPathComponent("diffusion_pytorch_model-00001-of-00002.safetensors"))
        try MLX.save(arrays: weights.filter { !firstHalf.contains($0.key) }, url: dir.appendingPathComponent("diffusion_pytorch_model-00002-of-00002.safetensors"))
        let loaded = try Klein3fTiny.load(try Flux2KleinTransformer.weightShards(in: dir), bits: nil)
        #expect(try Klein3fTiny.maxAbs(try Klein3fTiny.forward(loaded), reference) == 0)
        // an empty directory is a named error, not a crash
        let empty = dir.appendingPathComponent("empty")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        #expect(throws: Flux2KleinTransformerError.noWeightFiles(empty.path)) { _ = try Flux2KleinTransformer.weightShards(in: empty) }
    }

    @Test func missingTensorsAreNamed() throws {
        let (_, weights) = Klein3fTiny.source()
        var short = weights
        short.removeValue(forKey: "x_embedder.weight")
        short.removeValue(forKey: "single_transformer_blocks.1.attn.to_out.weight")
        #expect(throws: Flux2KleinTransformerError.missingWeight("x_embedder.weight, single_transformer_blocks.1.attn.to_out.weight")) {
            _ = try Klein3fTiny.load([{ short }], bits: nil)
        }
        // a block that never arrives lists all of its tensors
        let noBlock = weights.filter { !$0.key.hasPrefix("transformer_blocks.1.") }
        do {
            _ = try Klein3fTiny.load([{ noBlock }], bits: nil)
            Issue.record("a model without a double block must be refused")
        } catch let Flux2KleinTransformerError.missingWeight(names) {
            let list = names.components(separatedBy: ", ")
            #expect(list.count == 16 && list.allSatisfy { $0.hasPrefix("transformer_blocks.1.") })
        }
        let noSingle = weights.filter { !$0.key.hasPrefix("single_transformer_blocks.0.") }
        do {
            _ = try Klein3fTiny.load([{ noSingle }], bits: nil)
            Issue.record("a model without a single block must be refused")
        } catch let Flux2KleinTransformerError.missingWeight(names) {
            #expect(names.components(separatedBy: ", ").count == 4)
        }
    }

    @Test func extraRenamedOutOfRangeAndDuplicatedTensorsAreRefusedByName() throws {
        let (_, weights) = Klein3fTiny.source()
        var extra = weights
        extra["transformer_blocks.0.attn.to_q.bias"] = MLXArray.zeros([64])
        #expect(throws: Flux2KleinTransformerError.unexpectedWeight("transformer_blocks.0.attn.to_q.bias")) { _ = try Klein3fTiny.load([{ extra }], bits: nil) }
        var renamed = weights
        renamed["single_transformer_blocks.0.attn.to_out_proj.weight"] = renamed.removeValue(forKey: "single_transformer_blocks.0.attn.to_out.weight")
        #expect(throws: Flux2KleinTransformerError.unexpectedWeight("single_transformer_blocks.0.attn.to_out_proj.weight")) { _ = try Klein3fTiny.load([{ renamed }], bits: nil) }
        var outOfRange = weights
        outOfRange["single_transformer_blocks.2.attn.norm_q.weight"] = MLXArray.ones([32])
        #expect(throws: Flux2KleinTransformerError.unexpectedWeight("single_transformer_blocks.2.attn.norm_q.weight")) { _ = try Klein3fTiny.load([{ outOfRange }], bits: nil) }
        var unknown = weights
        unknown["final_layer.weight"] = MLXArray.ones([4])
        #expect(throws: Flux2KleinTransformerError.unexpectedWeight("final_layer.weight")) { _ = try Klein3fTiny.load([{ unknown }], bits: nil) }
        // the same tensor in two shards, before and after its block is built
        #expect(throws: Flux2KleinTransformerError.unexpectedWeight("x_embedder.weight (duplicate)")) {
            _ = try Klein3fTiny.load([{ weights }, { weights.filter { $0.key == "x_embedder.weight" } }], bits: nil)
        }
        #expect(throws: Flux2KleinTransformerError.unexpectedWeight("transformer_blocks.0.attn.to_q.weight")) {
            _ = try Klein3fTiny.load([{ weights }, { weights.filter { $0.key == "transformer_blocks.0.attn.to_q.weight" } }], bits: nil)
        }
    }

    @Test func aWrongShapeIsRefused() throws {
        let (_, weights) = Klein3fTiny.source()
        var bad = weights
        bad["single_transformer_blocks.0.attn.to_out.weight"] = MLXArray.zeros([64, 100])
        #expect(throws: (any Error).self) { _ = try Klein3fTiny.load([{ bad }], bits: nil) }
    }

    @Test func unsupportedBitWidthsAreRefusedBeforeReadingAnything() throws {
        for bits in [1, 2, 7, 9, 16] {
            #expect(throws: Flux2KleinTransformerError.unsupportedBits(bits)) { _ = try Klein3fTiny.load([{ [:] }], bits: bits) }
        }
        #expect(Flux2KleinTransformerError.unsupportedBits(7).errorDescription?.contains("3, 4, 5, 6 or 8") == true)
    }

    @Test func aCancelledLoadThrowsCancellationNotAnMLXError() async {
        let (_, weights) = Klein3fTiny.source()
        let task = Task { () throws -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try Klein3fTiny.load([{ weights }], bits: nil)
            return true
        }
        if case let .failure(error) = await task.result { #expect(error is CancellationError) } else { Issue.record("a cancelled load must stop") }
    }

    @Test func cancellingBetweenBlocksStopsTheLoad() async {
        let (_, weights) = Klein3fTiny.source()
        // cancel from the stage callback, after the first block is built: the NEXT check (per block) must stop it
        let task = Task { () throws -> Bool in
            _ = try Flux2KleinTransformer.load(shape: Klein3fTiny.shape, shards: [{ weights }], bits: nil) { stage in
                if stage == "double block 0 ready" { withUnsafeCurrentTask { $0?.cancel() } }
            }
            return true
        }
        if case let .failure(error) = await task.result { #expect(error is CancellationError) } else { Issue.record("cancellation between blocks must stop the load") }
    }
}

// MARK: - Quantization

@Suite(.serialized) struct Flux2KleinTransformerQuantizationTests {
    static func loaded(bits: Int) throws -> (source: MLXArray, quantized: Flux2KleinTransformer, reference: Flux2KleinTransformer) {
        let (source, weights) = Klein3fTiny.source()
        let reference = try Klein3fTiny.load([{ weights }], bits: nil)
        let q = try Klein3fTiny.load([{ weights }], bits: bits)
        return (try Klein3fTiny.forward(source), q, reference)
    }

    @Test(arguments: [3, 4, 5, 6, 8]) func everyBlockLinearAndTheNamedFrontLinearsBecomeQuantized(bits: Int) throws {
        let (_, q, _) = try Self.loaded(bits: bits)
        for (path, leaf) in q.leafModules().flattened() where leaf is Linear {
            let isBlock = path.hasPrefix("transformer_blocks.") || path.hasPrefix("single_transformer_blocks.")
            let want = isBlock || Flux2KleinTransformer.quantizesFront(path)
            #expect((leaf is QuantizedLinear) == want, Comment(rawValue: "\(path) bits \(bits): quantized \(leaf is QuantizedLinear), wanted \(want)"))
            if let ql = leaf as? QuantizedLinear { #expect(ql.bits == bits && ql.groupSize == 64, Comment(rawValue: path)) }
        }
        // the policy: these stay in the checkpoint dtype
        #expect(!(q.xEmbedder is QuantizedLinear) && !(q.projOut is QuantizedLinear))
        #expect(!(q.timeGuidanceEmbed.timestepEmbedder.linear1 is QuantizedLinear) && !(q.timeGuidanceEmbed.timestepEmbedder.linear2 is QuantizedLinear))
        #expect(q.contextEmbedder is QuantizedLinear && q.normOut.linear is QuantizedLinear)
        #expect(q.doubleModulationImg.linear is QuantizedLinear && q.doubleModulationTxt.linear is QuantizedLinear && q.singleModulation.linear is QuantizedLinear)
    }

    @Test func quantizedForwardTracksTheUnquantizedModel() throws {
        for bits in [8, 4] {
            let (_, q, ref) = try Self.loaded(bits: bits)
            let a = try Klein3fTiny.forward(q), b = try Klein3fTiny.forward(ref)
            let c = try KleinCosine.of(a, b)
            let rel = try Klein3fTiny.maxAbs(a, b)
            print("DITQUANT tiny random int\(bits): cosine \(c) max-abs \(rel)")
            #expect(c > (bits == 8 ? Klein3fQuantFloors.int8 : Klein3fQuantFloors.int4), "int\(bits) cosine \(c)")
        }
    }

    @Test func aLayerThatCannotBeQuantizedThrowsInsteadOfFallingBack() throws {
        // The fp32 golden shape has inner 16: not a multiple of the group size 64.
        let w = Klein3f.weights()
        do {
            _ = try Flux2KleinTransformer.load(shape: Klein3f.shape, shards: [{ w }], bits: 4, groupSize: 64)
            Issue.record("a non-quantizable layer must throw")
        } catch let Flux2KleinTransformerError.quantizationFailed(what) {
            #expect(what.contains("multiple of the group size 64"), Comment(rawValue: what))
        }
    }

    @Test func quantizingPerBlockKeepsEveryBlockEvaluatedAndSmaller() throws {
        let (_, weights) = Klein3fTiny.source()
        let bf16 = try Klein3fTiny.load([{ weights.mapValues { $0.asType(.bfloat16) } }], bits: nil)
        let int4 = try Klein3fTiny.load([{ weights.mapValues { $0.asType(.bfloat16) } }], bits: 4)
        let big = bf16.parameterBytes(), small = int4.parameterBytes()
        print("DITQUANT tiny parameter bytes bf16 \(big) int4 \(small)")
        #expect(small < big / 2)
    }
}

/// Floors for quantized-vs-unquantized on the tiny random shape (cosine). Set AFTER measuring (MP7-3f-report.md); they pin measured behaviour, not image quality.
enum Klein3fQuantFloors {
    static let int8 = 0.9999       // measured 0.999997
    static let int4 = 0.998         // measured 0.999349
}

// MARK: - Real header check (opt-in: names + shapes only, no tensor data)

@Suite(.serialized) struct Flux2KleinTransformerRealHeaderTests {
    @Test(.enabled(if: Klein3eHeader.present, Klein3eHeader.skipReason))
    func everyTensorUnderTheRealHeaderIsAccountedForByTheFullModelAndViceVersa() throws {
        let dir = try #require(Klein3eHeader.directory)
        let header = try Klein3eHeader.read(dir.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let config = try Flux2KleinTransformerConfig.decode(from: Data(contentsOf: dir.appendingPathComponent("config.json")))
        let before = Klein3fRSS.now()
        let model = Flux2KleinTransformer(config)                           // lazy: parameters are never evaluated
        print("REALKEYS full model build maxrss delta: \(Klein3fRSS.now() - before) bytes")
        let mine = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        print("REALKEYS header \(header.count) tensors, model \(mine.count) parameters")
        #expect(header.count == 169 && mine.count == 169)
        #expect(Set(header.keys) == Set(mine.keys), "names (header minus model: \(Set(header.keys).subtracting(mine.keys).sorted().prefix(5)), model minus header: \(Set(mine.keys).subtracting(header.keys).sorted().prefix(5)))")
        var mismatches: [String] = []
        for (k, t) in header {
            if mine[k] != t.shape { mismatches.append("\(k): model \(String(describing: mine[k])) header \(t.shape)") }
            if t.dtype != "BF16" { mismatches.append("\(k): dtype \(t.dtype)") }
        }
        #expect(mismatches.isEmpty, Comment(rawValue: mismatches.prefix(5).joined(separator: "; ")))
        // the single blocks 0...19 specifically
        for n in 0 ..< config.numSingleLayers {
            let p = "single_transformer_blocks.\(n).attn."
            #expect(header[p + "to_qkv_mlp_proj.weight"]?.shape == [27648, 3072], Comment(rawValue: "single \(n) to_qkv_mlp_proj"))
            #expect(header[p + "to_out.weight"]?.shape == [3072, 12288], Comment(rawValue: "single \(n) to_out"))
            #expect(header[p + "norm_q.weight"]?.shape == [128] && header[p + "norm_k.weight"]?.shape == [128], Comment(rawValue: "single \(n) norms"))
            #expect(header.keys.filter { $0.hasPrefix("single_transformer_blocks.\(n).") }.count == 4)
        }
        // total bytes of the header, for the memory table: bf16 = 2 bytes per element
        let elements = header.values.reduce(0) { $0 + $1.shape.reduce(1, *) }
        print("REALKEYS header elements \(elements) = \(Double(elements) * 2 / 1e9) GB bf16")
    }
}

enum Klein3fRSS {
    static func now() -> Int {
        var u = rusage()
        getrusage(RUSAGE_SELF, &u)
        return Int(u.ru_maxrss)
    }
}
