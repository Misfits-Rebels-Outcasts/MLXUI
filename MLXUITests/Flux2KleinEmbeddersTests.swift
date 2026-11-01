import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXUI

// MP7-3d: the Klein MMDiT front end against diffusers 0.37.1 on the tiny fp32 golden (g3d_embedders.safetensors, made by embedders_goldens.py with
// g3a_transformer.safetensors' own weights, never edited). Each piece prints its measured error (EMBPARITY lines) before the floor below it was set.

enum Klein3d {
    static var fixtures: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Flux2Klein") }
    static let golden: [String: MLXArray] = (try? MLX.loadArrays(url: fixtures.appendingPathComponent("g3d_embedders.safetensors"))) ?? [:]
    static let g3a: [String: MLXArray] = (try? MLX.loadArrays(url: fixtures.appendingPathComponent("g3a_transformer.safetensors"))) ?? [:]

    /// TINY_TF of tiny_goldens.py: inner 4 x 16 = 64, in 16, joint 48, 32 timestep channels, axes (4, 4, 4, 4), theta 2000.
    static let shape = Flux2KleinEmbedderShape(inChannels: 16, jointAttentionDim: 48, inner: 64, timestepChannels: 32, eps: 1e-6,
                                               axesDims: [4, 4, 4, 4], ropeTheta: 2000)

    /// The embedder module with g3a's own weights loaded (every key must be used and present).
    static func embedders() throws -> Flux2KleinEmbedders {
        let keys = ["x_embedder.", "context_embedder.", "time_guidance_embed.", "double_stream_modulation_img.", "double_stream_modulation_txt.",
                    "single_stream_modulation.", "norm_out.", "proj_out."]
        var w: [String: MLXArray] = [:]
        for (k, v) in g3a where k.hasPrefix("w.") {
            let name = String(k.dropFirst(2))
            if keys.contains(where: { name.hasPrefix($0) }) { w[name] = v }
        }
        let m = Flux2KleinEmbedders(shape)
        try m.update(parameters: ModuleParameters.unflattened(w), verify: .all)
        eval(m)
        return m
    }

    struct Err { let maxAbs: Float; let oneMinusCos: Double }

    static func measure(_ name: String, _ mine: MLXArray, _ key: String) throws -> Err {
        let want = try #require(golden[key], Comment(rawValue: key))
        #expect(mine.shape == want.shape, Comment(rawValue: "\(name) shape \(mine.shape) vs \(want.shape)"))
        let d = try MLXErrorGuard.read(abs(mine.asType(.float32) - want).max().reshaped(1), as: Float.self)[0]
        let c = try KleinCosine.of(mine, want)
        print("EMBPARITY \(name): max-abs \(d) 1-cos \(1 - c)")
        return Err(maxAbs: d, oneMinusCos: 1 - c)
    }

    // Floors: the spike's target (max-abs <= 1e-5). First measurement 2026-10-31: every piece below it (worst: norm_out 9.5e-7), except the raw
    // sinusoid and the real-size rope table, which carry their own measured floors in their tests. EMBPARITY lines are in MP7-3d-report.md.
    static let maxAbsFloor: Float = 1e-5
}

@Suite(.serialized) struct Flux2KleinEmbeddersTests {
    static let sigma = MLXArray([Float(0.7)])

    @Test func goldenIsPresent() {
        #expect(Klein3d.golden["out.temb"] != nil && Klein3d.g3a["w.x_embedder.weight"] != nil)
    }

    @Test func tembMatchesDiffusers() throws {
        let m = try Klein3d.embedders()
        let proj = Flux2KleinTimeEmbed.projection(sigma: Self.sigma, channels: 32)
        let p = try Klein3d.measure("time_proj", proj, "out.time_proj")
        let e = try Klein3d.measure("temb", m.timeGuidanceEmbed(sigma: Self.sigma), "out.temb")
        // Measured 2026-10-31: temb max-abs 1.4e-6 (1-cos 7e-12), within the 1e-5 target. The raw sinusoid is looser (max-abs 1.5e-5, 1-cos 9e-12) because its
        // arguments reach 700 rad (sigma 0.7 x 1000) and float32 rounds those at about 4e-5; torch and MLX round them differently. The MLP's small weights shrink
        // that to 1.4e-6 in temb, so the sinusoid's own floor is 5e-5 (about 3x the measurement) and temb keeps the 1e-5 target.
        #expect(p.maxAbs <= 5e-5)
        #expect(e.maxAbs <= Klein3d.maxAbsFloor)
    }

    @Test func embedderOutputsMatchDiffusers() throws {
        let m = try Klein3d.embedders()
        let a = try Klein3d.measure("x_embedder", m.xEmbedder(try #require(Klein3d.g3a["in.hidden_states"])), "out.x_embedder")
        let b = try Klein3d.measure("context_embedder", m.contextEmbedder(try #require(Klein3d.g3a["in.encoder_hidden_states"])), "out.context_embedder")
        #expect(a.maxAbs <= Klein3d.maxAbsFloor && b.maxAbs <= Klein3d.maxAbsFloor)
    }

    @Test func modulationOutputsAndChunksMatchDiffusers() throws {
        let m = try Klein3d.embedders()
        let temb = m.timeGuidanceEmbed(sigma: Self.sigma)
        let cond = m.conditioning(sigma: Self.sigma)
        #expect(cond.doubleImg.count == 2 && cond.doubleTxt.count == 2 && cond.single.count == 1)
        let groups: [(String, Flux2KleinModulation, [Flux2KleinModulationSet])] = [
            ("img", m.doubleModulationImg, cond.doubleImg), ("txt", m.doubleModulationTxt, cond.doubleTxt), ("single", m.singleModulation, cond.single),
        ]
        for (name, module, sets) in groups {
            let raw = try Klein3d.measure("mod_\(name)", module(temb), "out.mod_\(name)")
            #expect(raw.maxAbs <= Klein3d.maxAbsFloor)
            for (i, set) in sets.enumerated() {
                for (part, array) in [("shift", set.shift), ("scale", set.scale), ("gate", set.gate)] {
                    #expect(array.shape == [1, 1, 64])
                    let e = try Klein3d.measure("mod_\(name)_\(i)_\(part)", array.reshaped(1, -1), "out.mod_\(name)_\(i)_\(part)")
                    #expect(e.maxAbs <= Klein3d.maxAbsFloor, Comment(rawValue: "mod_\(name)_\(i)_\(part)"))
                }
            }
        }
    }

    @Test func normOutAndProjOutMatchDiffusers() throws {
        let m = try Klein3d.embedders()
        let temb = m.timeGuidanceEmbed(sigma: Self.sigma)
        let x = try #require(Klein3d.golden["in.norm_hidden"])
        let n = try Klein3d.measure("norm_out", m.normOut(x, temb: temb), "out.norm_out")
        let p = try Klein3d.measure("proj_out", m.finalLayer(x, temb: temb), "out.proj_out")
        #expect(n.maxAbs <= Klein3d.maxAbsFloor && p.maxAbs <= Klein3d.maxAbsFloor)
    }

    @Test func theSigmaIsMultipliedBy1000Inside() throws {
        // sigma 0.7 and a pre-multiplied 700 must NOT be interchangeable: passing 700 would be 700000 inside.
        let a = Flux2KleinTimeEmbed.projection(sigma: MLXArray([Float(0.7)]), channels: 32)
        let b = Flux2KleinTimeEmbed.projection(sigma: MLXArray([Float(700)]), channels: 32)
        let d = try MLXErrorGuard.read(abs(a - b).max().reshaped(1), as: Float.self)[0]
        #expect(d > 0.1)
        // the first half is cos, the second half sin (flip_sin_to_cos): at t = 0 that is ones then zeros.
        let z = try MLXErrorGuard.read(Flux2KleinTimeEmbed.projection(sigma: MLXArray([Float(0)]), channels: 32), as: Float.self)
        #expect(Array(z[0 ..< 16]) == [Float](repeating: 1, count: 16) && Array(z[16 ..< 32]) == [Float](repeating: 0, count: 16))
    }

    @Test func parameterKeysAreTheDiffusersKeys() {
        let m = Flux2KleinEmbedders(Klein3d.shape)
        let keys = Set(m.parameters().flattened().map { $0.0 })
        #expect(keys == [
            "x_embedder.weight", "context_embedder.weight",
            "time_guidance_embed.timestep_embedder.linear_1.weight", "time_guidance_embed.timestep_embedder.linear_2.weight",
            "double_stream_modulation_img.linear.weight", "double_stream_modulation_txt.linear.weight", "single_stream_modulation.linear.weight",
            "norm_out.linear.weight", "proj_out.weight",
        ])
    }
}

@Suite(.serialized) struct Flux2KleinRoPETests {
    static let axes = [4, 4, 4, 4]

    static func ids(_ key: String) throws -> MLXArray { try #require(Klein3d.golden[key]) }

    @Test func fluxLayersEmbedNDMatchesDiffusersCosSin() throws {
        // The reuse-vs-port decision: FluxLayers.embedND (unedited) against diffusers Flux2PosEmbed.
        for name in ["img", "txt", "img_alt"] {
            let (c, s) = Flux2KleinRoPE.cosSin(ids: try Self.ids("rope.\(name)_ids"), axesDims: Self.axes, theta: 2000)
            let ec = try Klein3d.measure("rope_\(name)_cos", c, "rope.\(name)_cos")
            let es = try Klein3d.measure("rope_\(name)_sin", s, "rope.\(name)_sin")
            #expect(ec.maxAbs <= Klein3d.maxAbsFloor && es.maxAbs <= Klein3d.maxAbsFloor, Comment(rawValue: name))
        }
    }

    @Test func rotaryApplicationMatchesDiffusers() throws {
        let (c, s) = Flux2KleinRoPE.cosSin(ids: try Self.ids("rope.img_ids"), axesDims: Self.axes, theta: 2000)
        let q = try #require(Klein3d.golden["in.q"]), k = try #require(Klein3d.golden["in.k"])
        let eq = try Klein3d.measure("q_rot", Flux2KleinRoPE.applyRotary(q, cos: c, sin: s), "out.q_rot")
        let ek = try Klein3d.measure("k_rot", Flux2KleinRoPE.applyRotary(k, cos: c, sin: s), "out.k_rot")
        #expect(eq.maxAbs <= Klein3d.maxAbsFloor && ek.maxAbs <= Klein3d.maxAbsFloor)
    }

    @Test func twoIDLayoutsOfEqualLengthDoNotAlias() throws {
        // 4x4 and 2x8 image grids are both 16 rows. A cache keyed on length alone would hand the second call the first call's tables.
        let a = try Self.ids("rope.img_ids"), b = try Self.ids("rope.img_alt_ids")
        #expect(a.shape == b.shape && a.dim(0) == 16)
        let (ca, sa) = Flux2KleinRoPE.cosSin(ids: a, axesDims: Self.axes, theta: 2000)
        let (cb, sb) = Flux2KleinRoPE.cosSin(ids: b, axesDims: Self.axes, theta: 2000)
        let (ca2, _) = Flux2KleinRoPE.cosSin(ids: a, axesDims: Self.axes, theta: 2000)      // asking for the first layout again, after the second
        let apart = try MLXErrorGuard.read(maximum(abs(ca - cb).max(), abs(sa - sb).max()).reshaped(1), as: Float.self)[0]
        let same = try MLXErrorGuard.read(abs(ca - ca2).max().reshaped(1), as: Float.self)[0]
        print("EMBPARITY alias: layouts differ by \(apart), repeat call differs by \(same)")
        #expect(apart > 0.1)
        #expect(same == 0)
        let e = try Klein3d.measure("rope_img_alt_cos_after_img", cb, "rope.img_alt_cos")
        #expect(e.maxAbs <= Klein3d.maxAbsFloor)
        // and the rotated queries differ the way diffusers' do
        let q = try #require(Klein3d.golden["in.q"])
        let r = try Klein3d.measure("q_rot_alt", Flux2KleinRoPE.applyRotary(q, cos: cb, sin: sb), "out.q_rot_alt")
        #expect(r.maxAbs <= Klein3d.maxAbsFloor)
    }

    @Test func jointCosSinPutsTextBeforeImage() throws {
        let t = try Self.ids("rope.txt_ids"), i = try Self.ids("rope.img_ids")
        let (c, _) = Flux2KleinRoPE.jointCosSin(textIDs: t, imageIDs: i, axesDims: Self.axes, theta: 2000)
        let (ct, _) = Flux2KleinRoPE.cosSin(ids: t, axesDims: Self.axes, theta: 2000)
        #expect(c.dim(0) == t.dim(0) + i.dim(0) && c.dim(1) == 16)
        let head = try MLXErrorGuard.read(abs(c[0 ..< t.dim(0)] - ct).max().reshaped(1), as: Float.self)[0]
        #expect(head == 0)
    }

    /// Real Klein sizes (axes 32 x 4, theta 2000, positions up to 511 on the L axis and 63 on H/W) against a Double-precision CPU reference.
    /// diffusers builds the table in float64; MLX's GPU path is float32. This measures what that costs at the positions the real model uses.
    @Test func realSizedTablesAgainstADoublePrecisionReference() throws {
        var rows: [[Float]] = []
        for l in stride(from: 0, to: 512, by: 7) { rows.append([0, 0, 0, Float(l)]) }
        for h in stride(from: 0, to: 64, by: 5) { for w in stride(from: 0, to: 64, by: 9) { rows.append([0, Float(h), Float(w), 0]) } }
        let ids = MLXArray(rows.flatMap { $0 }, [rows.count, 4])
        let (c, s) = Flux2KleinRoPE.cosSin(ids: ids, axesDims: [32, 32, 32, 32], theta: 2000)
        let mc = try MLXErrorGuard.read(c, as: Float.self), ms = try MLXErrorGuard.read(s, as: Float.self)
        var worst: Double = 0
        for (r, row) in rows.enumerated() {
            for axis in 0 ..< 4 {
                for j in 0 ..< 16 {
                    let omega = 1.0 / pow(2000.0, Double(2 * j) / 32.0)
                    let angle = Double(row[axis]) * omega
                    for rep in 0 ..< 2 {
                        let col = axis * 32 + 2 * j + rep
                        worst = max(worst, abs(Double(mc[r * 128 + col]) - cos(angle)), abs(Double(ms[r * 128 + col]) - sin(angle)))
                    }
                }
            }
        }
        print("EMBPARITY real-size rope vs Double reference: max-abs \(worst) over \(rows.count) rows")
        // Measured 2.4e-5 (float32 angle rounding at position ~500, not a formula difference); floor 1e-4, about 4x. Implementer's call, pending owner confirmation.
        #expect(worst <= 1e-4)
    }
}

@Suite struct Flux2KleinEmbeddersFixtureTests {
    static let sha256 = "1b7695bf722fc266c5a14d87c92e79f91b168fe331097766e30d5da893d8fc33"
    static let scriptSha256 = "e47115ad06ab8cdb9b9aa6fa6a413ce369a110895016eb7c633904929a12d784"

    @Test func theEmbeddersGoldenIsByteIdenticalToWhatTheScriptWrote() throws {
        for (name, expected) in [("g3d_embedders.safetensors", Self.sha256), ("embedders_goldens.py", Self.scriptSha256)] {
            let data = try Data(contentsOf: Klein3d.fixtures.appendingPathComponent(name))
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            #expect(digest == expected, Comment(rawValue: name))
        }
    }
}
