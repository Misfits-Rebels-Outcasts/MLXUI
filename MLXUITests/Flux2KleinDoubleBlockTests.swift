import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXUI

// MP7-3e: the Klein double-stream block against diffusers 0.37.1 on the tiny fp32 golden g3a_transformer.safetensors (weights `w.transformer_blocks.0.*`,
// inputs `in.*`, block output `mid.double_0_txt` / `mid.double_0_img`). The tiny config has ONE double layer, so "block 0" and "block N-1" are the same block.
// The block inputs are built by chaining the 3d pieces (embedders, conditioning, joint rope), exactly as the transformer's forward does.

enum Klein3e {
    static var g3a: [String: MLXArray] { Klein3d.g3a }
    static let axes = [4, 4, 4, 4]

    static func block() throws -> Flux2KleinDoubleBlock {
        var w: [String: MLXArray] = [:]
        for (k, v) in g3a where k.hasPrefix("w.transformer_blocks.0.") { w[String(k.dropFirst("w.transformer_blocks.0.".count))] = v }
        let b = Flux2KleinDoubleBlock(dim: 64, heads: 4, headDim: 16, mlpRatio: 3.0, eps: 1e-6)
        try b.update(parameters: ModuleParameters.unflattened(w), verify: .all)
        eval(b)
        return b
    }

    struct Inputs { let img: MLXArray; let txt: MLXArray; let cond: Flux2KleinEmbedders.Conditioning; let cos: MLXArray; let sin: MLXArray }

    static func inputs() throws -> Inputs {
        let e = try Klein3d.embedders()
        let img = e.xEmbedder(try #require(g3a["in.hidden_states"]))
        let txt = e.contextEmbedder(try #require(g3a["in.encoder_hidden_states"]))
        let cond = e.conditioning(sigma: try #require(g3a["in.timestep"]))
        let (c, s) = Flux2KleinRoPE.jointCosSin(textIDs: try #require(g3a["in.txt_ids"]), imageIDs: try #require(g3a["in.img_ids"]), axesDims: axes, theta: 2000)
        eval(img, txt, c, s)
        return Inputs(img: img, txt: txt, cond: cond, cos: c, sin: s)
    }

    static func run(_ b: Flux2KleinDoubleBlock, _ i: Inputs, cos: MLXArray? = nil, sin: MLXArray? = nil) -> (txt: MLXArray, img: MLXArray) {
        b(img: i.img, txt: i.txt, imgMod: i.cond.doubleImg, txtMod: i.cond.doubleTxt, cos: cos ?? i.cos, sin: sin ?? i.sin)
    }

    struct Err { let maxAbs: Float; let oneMinusCos: Double }

    static func measure(_ name: String, _ mine: MLXArray, _ key: String) throws -> Err {
        let want = try #require(g3a[key], Comment(rawValue: key))
        #expect(mine.shape == want.shape, Comment(rawValue: "\(name) shape \(mine.shape) vs \(want.shape)"))
        let d = try MLXErrorGuard.read(abs(mine.asType(.float32) - want).max().reshaped(1), as: Float.self)[0]
        let c = try KleinCosine.of(mine, want)
        print("DBLPARITY \(name): max-abs \(d) 1-cos \(1 - c)")
        return Err(maxAbs: d, oneMinusCos: 1 - c)
    }

    /// The 18 diffusers keys of one double block (4 RMSNorm weights + 8 attention Linears + 4 feed-forward Linears; the four LayerNorms have none).
    static let expectedKeys: Set<String> = [
        "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight",
        "attn.to_out.0.weight", "attn.to_add_out.weight",
        "attn.norm_q.weight", "attn.norm_k.weight", "attn.norm_added_q.weight", "attn.norm_added_k.weight",
        "ff.linear_in.weight", "ff.linear_out.weight", "ff_context.linear_in.weight", "ff_context.linear_out.weight",
    ]
}

@Suite(.serialized) struct Flux2KleinDoubleBlockTests {
    @Test func parameterKeysAreTheDiffusersKeys() {
        let b = Flux2KleinDoubleBlock(dim: 64, heads: 4, headDim: 16, mlpRatio: 3.0, eps: 1e-6)
        let keys = Set(b.parameters().flattened().map { $0.0 })
        #expect(keys == Klein3e.expectedKeys)
        #expect(keys.count == 16)
    }

    @Test func g3aCarriesExactlyTheseKeysForBlockZero() {
        let prefix = "w.transformer_blocks.0."
        let keys = Set(Klein3e.g3a.keys.filter { $0.hasPrefix(prefix) }.map { String($0.dropFirst(prefix.count)) })
        #expect(keys == Klein3e.expectedKeys)
    }

    @Test func blockMatchesDiffusers() throws {
        let i = try Klein3e.inputs()
        let out = Klein3e.run(try Klein3e.block(), i)
        eval(out.txt, out.img)
        let t = try Klein3e.measure("double_0_txt", out.txt, "mid.double_0_txt")
        let m = try Klein3e.measure("double_0_img", out.img, "mid.double_0_img")
        // Measured 2026-10-31: txt max-abs 1.1e-7 (1-cos 1.2e-12), img max-abs 1.2e-7 (1-cos 2.1e-15). Floor 1e-6 (about 8x), cosine >= 0.999999; the spike's 1e-4 is looser.
        for e in [t, m] {
            #expect(e.maxAbs <= Klein3e.maxAbsFloor)
            #expect(1 - e.oneMinusCos >= 0.999999)
        }
    }

    @Test func outputOrderIsTextThenImage() throws {
        let i = try Klein3e.inputs()
        let out = Klein3e.run(try Klein3e.block(), i)
        #expect(out.txt.shape == [1, 8, 64] && out.img.shape == [1, 16, 64])
    }

    @Test func theJointSequenceIsTextThenImage() throws {
        // Right order matches diffusers; a rotary table built image-first (a [image, text] sequence) must NOT, by a wide margin.
        let b = try Klein3e.block()
        let i = try Klein3e.inputs()
        // The wrong table: image ids first, text ids second (an [image, text] sequence), while the block still treats the first 8 rows as text.
        let (c2, s2) = Flux2KleinRoPE.jointCosSin(textIDs: try #require(Klein3e.g3a["in.img_ids"]), imageIDs: try #require(Klein3e.g3a["in.txt_ids"]),
                                                 axesDims: Klein3e.axes, theta: 2000)
        let wrong = Klein3e.run(b, i, cos: c2, sin: s2)
        let right = Klein3e.run(b, i)
        let dTxt = try MLXErrorGuard.read(abs(wrong.txt - right.txt).max().reshaped(1), as: Float.self)[0]
        let dImg = try MLXErrorGuard.read(abs(wrong.img - right.img).max().reshaped(1), as: Float.self)[0]
        print("DBLPARITY wrong-order table moves txt by \(dTxt), img by \(dImg)")
        #expect(dTxt > 1e-3 && dImg > 1e-3)      // measured 1.1e-2 / 9.5e-3, about 10^4 x the parity error
    }

    @Test func perturbingTheFirstTextRowMovesTheImageRowsThroughAttention() throws {
        // Joint attention: a change in one text token reaches every image token; a change in one image token reaches every text token.
        let b = try Klein3e.block()
        let i = try Klein3e.inputs()
        let base = Klein3e.run(b, i)
        // not a constant shift: LayerNorm removes those
        let ramp = MLXArray((0 ..< 64).map { Float($0) / 8 }).reshaped(1, 1, 64)
        let bumpedTxt = concatenated([i.txt[0 ..< 1, 0 ..< 1] + ramp, i.txt[0 ..< 1, 1...]], axis: 1)
        let t = Klein3e.run(b, Klein3e.Inputs(img: i.img, txt: bumpedTxt, cond: i.cond, cos: i.cos, sin: i.sin))
        let bumpedImg = concatenated([i.img[0 ..< 1, 0 ..< 1] + ramp, i.img[0 ..< 1, 1...]], axis: 1)
        let m = Klein3e.run(b, Klein3e.Inputs(img: bumpedImg, txt: i.txt, cond: i.cond, cos: i.cos, sin: i.sin))
        // the LAST image row / LAST text row change although only row 0 of the other stream was touched
        let lastImg = try MLXErrorGuard.read(abs(t.img[0 ..< 1, 15...] - base.img[0 ..< 1, 15...]).max().reshaped(1), as: Float.self)[0]
        let lastTxt = try MLXErrorGuard.read(abs(m.txt[0 ..< 1, 7...] - base.txt[0 ..< 1, 7...]).max().reshaped(1), as: Float.self)[0]
        #expect(lastImg > 1e-6 && lastTxt > 1e-6)
    }

    @Test func gatesAreOnTheResiduals() throws {
        // Zero gates: both streams pass through unchanged.
        let b = try Klein3e.block()
        let i = try Klein3e.inputs()
        func zeroed(_ s: [Flux2KleinModulationSet]) -> [Flux2KleinModulationSet] {
            s.map { Flux2KleinModulationSet(shift: $0.shift, scale: $0.scale, gate: MLXArray.zeros(like: $0.gate)) }
        }
        let out = b(img: i.img, txt: i.txt, imgMod: zeroed(i.cond.doubleImg), txtMod: zeroed(i.cond.doubleTxt), cos: i.cos, sin: i.sin)
        let a = try MLXErrorGuard.read(abs(out.img - i.img).max().reshaped(1), as: Float.self)[0]
        let c = try MLXErrorGuard.read(abs(out.txt - i.txt).max().reshaped(1), as: Float.self)[0]
        #expect(a == 0 && c == 0)
    }
}

extension Klein3e {
    // Floor set after the first measurement (2026-10-31): worst 1.2e-7, floor 1e-6. Implementer's call, pending owner confirmation (the spike target was 1e-4).
    static let maxAbsFloor: Float = 1e-6
}

// MARK: - Real checkpoint header check (opt-in, reads ONLY the safetensors header: no tensor data, nothing downloaded)

enum Klein3eHeader {
    struct Tensor: Equatable { let dtype: String; let shape: [Int] }

    /// Same location rule as `KleinRealInstall` (TEST_RUNNER_MLXUI_TEST_KLEIN_DIR, else the Direct store); the transformer folder must have its weights and config.
    static let directory: URL? = {
        let url = ProcessInfo.processInfo.environment["MLXUI_TEST_KLEIN_DIR"].map { URL(fileURLWithPath: $0) }
            ?? ModelStore.shared.directory(forHFModelID: "mlx-community/FLUX.2-klein-4B-bf16")
        let t = url.appendingPathComponent("transformer")
        let fm = FileManager.default
        return fm.fileExists(atPath: t.appendingPathComponent("diffusion_pytorch_model.safetensors").path) && fm.fileExists(atPath: t.appendingPathComponent("config.json").path) ? t : nil
    }()
    static var present: Bool { directory != nil }
    static let skipReason: Comment = "needs the installed FLUX.2 Klein (mlx-community--FLUX.2-klein-4B-bf16) or TEST_RUNNER_MLXUI_TEST_KLEIN_DIR, and an unsandboxed test host (reads the safetensors header only)"

    /// First 8 bytes (little-endian u64 N) then N bytes of JSON. Nothing else is read.
    static func read(_ url: URL) throws -> [String: Tensor] {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        guard let lenData = try h.read(upToCount: 8), lenData.count == 8 else { throw CocoaError(.fileReadCorruptFile) }
        let n = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard n < 16_000_000, let json = try h.read(upToCount: Int(n)), json.count == Int(n) else { throw CocoaError(.fileReadCorruptFile) }
        guard let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any] else { throw CocoaError(.fileReadCorruptFile) }
        var out: [String: Tensor] = [:]
        for (k, v) in obj where k != "__metadata__" {
            guard let d = v as? [String: Any], let dtype = d["dtype"] as? String, let shape = d["shape"] as? [Int] else { continue }
            out[k] = Tensor(dtype: dtype, shape: shape)
        }
        return out
    }
}

@Suite(.serialized) struct Flux2KleinDoubleBlockRealKeyTests {
    static func config() throws -> Flux2KleinTransformerConfig {
        let dir = try #require(Klein3eHeader.directory)
        return try Flux2KleinTransformerConfig.decode(from: Data(contentsOf: dir.appendingPathComponent("config.json")))
    }

    static func shapes(_ m: Module, prefix: String = "") -> [String: [Int]] {
        Dictionary(uniqueKeysWithValues: m.parameters().flattened().map { (prefix + $0.0, $0.1.shape) })
    }

    @Test(.enabled(if: Klein3eHeader.present, Klein3eHeader.skipReason))
    func everyDoubleBlockMatchesTheRealHeaderByNameAndShape() throws {
        let header = try Klein3eHeader.read(try #require(Klein3eHeader.directory).appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let c = try Self.config()
        #expect(c.numLayers == 5)
        let before = rusage_maxrss()
        let block = Flux2KleinDoubleBlock(c)                 // lazy: parameters are never evaluated
        print("REALKEYS block build maxrss delta: \(rusage_maxrss() - before) bytes")
        let mine = Self.shapes(block)
        #expect(mine.count == 16)
        for n in 0 ..< c.numLayers {
            let prefix = "transformer_blocks.\(n)."
            let theirs = Dictionary(uniqueKeysWithValues: header.filter { $0.key.hasPrefix(prefix) }.map { (String($0.key.dropFirst(prefix.count)), $0.value) })
            #expect(Set(theirs.keys) == Set(mine.keys), Comment(rawValue: "block \(n) names"))
            for (k, t) in theirs {
                #expect(mine[k] == t.shape, Comment(rawValue: "block \(n) \(k): module \(String(describing: mine[k])) header \(t.shape)"))
                #expect(t.dtype == "BF16", Comment(rawValue: "block \(n) \(k) dtype \(t.dtype)"))
            }
        }
    }

    @Test(.enabled(if: Klein3eHeader.present, Klein3eHeader.skipReason))
    func theEmbeddersKeysMatchTheRealHeaderByNameAndShape() throws {
        let header = try Klein3eHeader.read(try #require(Klein3eHeader.directory).appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let c = try Self.config()
        let mine = Self.shapes(Flux2KleinEmbedders(Flux2KleinEmbedderShape(c)))
        let outside = header.filter { !$0.key.hasPrefix("transformer_blocks.") && !$0.key.hasPrefix("single_transformer_blocks.") }
        #expect(Set(outside.keys) == Set(mine.keys), Comment(rawValue: "non-block names"))
        for (k, t) in outside { #expect(mine[k] == t.shape, Comment(rawValue: "\(k): module \(String(describing: mine[k])) header \(t.shape)")) }
        // Everything under single_transformer_blocks. is 3f's; just pin that it is the fused 4-key layout the spike describes.
        let singles = Set(header.keys.filter { $0.hasPrefix("single_transformer_blocks.0.") }.map { String($0.dropFirst("single_transformer_blocks.0.".count)) })
        #expect(singles == ["attn.norm_k.weight", "attn.norm_q.weight", "attn.to_out.weight", "attn.to_qkv_mlp_proj.weight"])
        #expect(header.count == 5 * 16 + 20 * 4 + 9)
    }
}

private func rusage_maxrss() -> Int {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    return Int(u.ru_maxrss)
}
