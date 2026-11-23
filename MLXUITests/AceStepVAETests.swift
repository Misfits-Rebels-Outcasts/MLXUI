import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3h: the VAE decoder against upstream's tiny-config output (`Fixtures/MC5/mc5_vae_tiny.safetensors`), the tiling plan, cancellation and
/// strict names. Weights go through the real 3c `convertVAE`. No real weights.
struct AceStepVAETests {

    static let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures/MC5/mc5_vae_tiny.safetensors")

    struct Fixture {
        let config: AceStepVAEConfig
        let weights: [String: MLXArray]      // converted, names under `decoder.`
        let tensors: [String: MLXArray]
    }

    /// The fixture holds F64 tensors (the checksum), which the 3c `SafetensorsIndex` rejects by design, so this file has its own tiny reader:
    /// F32 / I32 tensors become arrays, F64 ones are returned as raw doubles.
    static func loadRaw() throws -> (config: Data, tensors: [String: MLXArray], doubles: [String: [Double]]) {
        let data = try Data(contentsOf: fixtureURL)
        let headerLength = Int(data.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        let header = try #require(try JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< 8 + headerLength)) as? [String: Any])
        let base = 8 + headerLength
        let meta = try #require(header["__metadata__"] as? [String: String])
        var tensors: [String: MLXArray] = [:], doubles: [String: [Double]] = [:]
        for (name, value) in header where name != "__metadata__" {
            let entry = try #require(value as? [String: Any])
            let shape = try #require(entry["shape"] as? [Int]), dtype = try #require(entry["dtype"] as? String)
            let range = try #require(entry["data_offsets"] as? [Int])
            let bytes = data.subdata(in: base + range[0] ..< base + range[1])
            switch dtype {
            case "F32": tensors[name] = MLXArray(bytes, shape, dtype: .float32)
            case "I32": tensors[name] = MLXArray(bytes, shape, dtype: .int32)
            case "F64": doubles[name] = bytes.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
            default: Issue.record("unexpected dtype \(dtype) for \(name)")
            }
        }
        return (Data((meta["config"] ?? "").utf8), tensors, doubles)
    }

    static func load() throws -> Fixture {
        let (json, all, _) = try loadRaw()
        var raw: [String: MLXArray] = [:]
        for (k, v) in all where k.hasPrefix("w.") { raw[String(k.dropFirst(2))] = v }
        return Fixture(config: try AceStepVAEConfig(json: json), weights: try AceStepWeights.convertVAE(raw), tensors: all)
    }

    static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { AceStepDiTTests.maxAbs(a, b) }

    @Test func configParsesAndHopIs1920() throws {
        let f = try Self.load()
        #expect(f.config.hop == 1920 && f.config.decoderChannels == 4 && f.config.inputChannels == 64 && f.config.audioChannels == 2)
        #expect(AceStepVAEConfig().hop == 1920)
    }

    @Test func shortDecodeMatchesTorchAndUpstreamMLXInF32() throws {
        let f = try Self.load()
        let vae = try AceStepVAE(config: f.config, weights: f.weights, dtype: .float32)
        let out = vae.decode(try #require(f.tensors["short.in.z"]))
        let torch = try #require(f.tensors["short.out.torch"]), mlx = try #require(f.tensors["short.out.f32"])
        #expect(out.shape == [1, 16 * 1920, 2] && out.shape == torch.shape)
        let dT = Self.maxAbs(out, torch), dM = Self.maxAbs(out, mlx)
        print("ACESTEP-VAE-TINY f32 maxAbs vs torch=\(dT) vs upstream-mlx=\(dM)")
        #expect(dT <= 1e-5, "vs torch \(dT)")
        #expect(dM <= 1e-5, "vs upstream MLX \(dM)")
    }

    @Test func shortDecodeInF16StaysWithinTheStatedToleranceOfUpstreamsF16() throws {
        let f = try Self.load()
        let vae = try AceStepVAE(config: f.config, weights: f.weights, dtype: .float16)
        let out = vae.decode(try #require(f.tensors["short.in.z"]))
        #expect(out.dtype == .float16)
        let want = try #require(f.tensors["short.out.f16"])
        let d = Self.maxAbs(out, want)
        let peak = abs(want).max().item(Float.self)
        print("ACESTEP-VAE-TINY f16 maxAbs vs upstream f16=\(d) (peak \(peak))")
        // Stated tolerance: 1e-4 absolute (measured 1.5e-5 on outputs peaking at 0.021; f16 has ~1e-3 relative precision).
        #expect(d <= 1e-4, "f16 max abs \(d)")
    }

    @Test func channelsFirstIsTheDumpsLayoutAndF32() throws {
        let f = try Self.load()
        let vae = try AceStepVAE(config: f.config, weights: f.weights, dtype: .float32)
        let z = try #require(f.tensors["short.in.z"])
        let cf = try vae.decodeChannelsFirst(z.transposed(0, 2, 1))
        #expect(cf.shape == [1, 2, 16 * 1920] && cf.dtype == .float32)
        #expect(Self.maxAbs(cf.transposed(0, 2, 1), vae.decode(z)) == 0)
    }

    // MARK: Tiling

    @Test func planFollowsUpstreamsWindows() {
        #expect(AceStepVAE.plan(frames: 250, chunk: 256).isEmpty, "250 <= 256: one direct decode, no tiling")
        let p = AceStepVAE.plan(frames: 200, chunk: 192)             // stride 64
        #expect(p.count == 4)
        #expect(p[0] == (0, 64, 0, 128) && p[1] == (64, 128, 0, 192) && p[2] == (128, 192, 64, 200) && p[3] == (192, 200, 128, 200))
        // chunk below 192 is raised to 192; a real 6000-frame (240 s) run at chunk 256 has stride 128.
        #expect(AceStepVAE.plan(frames: 200, chunk: 10).count == 4)
        let big = AceStepVAE.plan(frames: 6000, chunk: 256)
        #expect(big.count == 47 && big.last?.coreEnd == 6000)
        #expect(zip(big, big.dropFirst()).allSatisfy { $0.coreEnd == $1.coreStart })
    }

    @Test func tiledDecodeMatchesUpstreamWindowsAndTotals() throws {
        let f = try Self.load()
        let vae = try AceStepVAE(config: f.config, weights: f.weights, dtype: .float32)
        var progress: [Int] = []
        let out = try vae.decodeTiled(try #require(f.tensors["tiled.in.z"]), chunk: 192, progress: { done, total in
            #expect(total == 4); progress.append(done)
        })
        let total = try MLXErrorGuard.read(try #require(f.tensors["tiled.out.total_samples"]), as: Int32.self)[0]
        #expect(out.shape == [1, Int(total), 2] && total == 200 * 1920)
        #expect(progress == [1, 2, 3, 4])
        var i = 0
        var worst: Float = 0
        while let want = f.tensors["tiled.out.win\(i)"] {
            let start = Int(try MLXErrorGuard.read(try #require(f.tensors["tiled.out.win\(i).start"]), as: Int32.self)[0])
            worst = max(worst, Self.maxAbs(out[0, start ..< start + want.dim(0), 0...], want))
            i += 1
        }
        #expect(i == 5)
        print("ACESTEP-VAE-TINY tiled windows=\(i) worst maxAbs=\(worst)")
        #expect(worst <= 1e-5, "window max abs \(worst)")
        // Upstream's checksum (two F64 values: sum, sum of abs).
        let check = try #require(try Self.loadRaw().doubles["tiled.out.checksum"])
        let wantSum = check[0], wantAbs = check[1]
        let values = try MLXErrorGuard.read(out.asType(.float32).reshaped(-1), as: Float.self)
        let gotSum = values.reduce(0.0) { $0 + Double($1) }, gotAbs = values.reduce(0.0) { $0 + Double(abs($1)) }
        print("ACESTEP-VAE-TINY checksum abs got=\(gotAbs) want=\(wantAbs) sum got=\(gotSum) want=\(wantSum)")
        #expect(abs(gotAbs - wantAbs) <= 1e-4 * wantAbs, "abs checksum")
        #expect(abs(gotSum - wantSum) <= 1e-4 * wantAbs, "sum checksum")
    }

    @Test func cancellationBetweenChunksThrows() async throws {
        let f = try Self.load()
        let vae = try AceStepVAE(config: f.config, weights: f.weights, dtype: .float32)
        let z = try #require(f.tensors["tiled.in.z"])
        let task = Task { () throws -> Int in
            var seen = 0
            _ = try vae.decodeTiled(z, chunk: 192, progress: { done, _ in
                seen = done
                withUnsafeCurrentTask { $0?.cancel() }
            })
            return seen
        }
        do { _ = try await task.value; Issue.record("expected CancellationError") } catch is CancellationError {} catch { Issue.record("\(error)") }
    }

    @Test func strictNamesRefuseMissingAndExtra() throws {
        let f = try Self.load()
        #expect(Set(AceStepVAE.expectedParameterNames(f.config)) == Set(f.weights.keys))
        var missing = f.weights
        missing["decoder.block.2.res_unit2.conv1.bias"] = nil
        #expect(throws: AceStepWeightsError.self) { _ = try AceStepVAE(config: f.config, weights: missing, dtype: .float32) }
        var extra = f.weights
        extra["decoder.block.9.snake1.alpha"] = MLXArray.zeros([1])
        #expect(throws: AceStepWeightsError.self) { _ = try AceStepVAE(config: f.config, weights: extra, dtype: .float32) }
    }
}
