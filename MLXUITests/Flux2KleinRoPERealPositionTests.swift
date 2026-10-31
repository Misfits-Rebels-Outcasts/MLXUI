import Foundation
import MLX
import Testing
@testable import MLXUI

// MP7-3d-FIX-1: Flux2KleinRoPE at the positions the real model uses (every text position 0...511, a full 64 x 64 image grid), real axes [32, 32, 32, 32],
// theta 2000, against a Double-precision CPU reference computed here. Owner ruling (Boo, 2026-10-31): the float32 gap is accepted, no change; this test
// pins it. The floor 1e-4 is owner-set.
@Suite struct Flux2KleinRoPERealPositionTests {
    static let axes = [32, 32, 32, 32]

    /// Max-abs of cos and sin against the Double reference: inv_freq = 1 / theta^(arange(0, dim, 2) / dim), angle = pos * inv_freq, repeat-interleave 2, axes concatenated.
    static func maxError(rows: [[Float]]) throws -> Double {
        let ids = MLXArray(rows.flatMap { $0 }, [rows.count, 4])
        let (c, s) = Flux2KleinRoPE.cosSin(ids: ids, axesDims: axes, theta: 2000)
        let mc = try MLXErrorGuard.read(c, as: Float.self), ms = try MLXErrorGuard.read(s, as: Float.self)
        let width = axes.reduce(0, +)
        var worst = 0.0
        for (r, row) in rows.enumerated() {
            var col = 0
            for (axis, dim) in axes.enumerated() {
                for j in 0 ..< dim / 2 {
                    let angle = Double(row[axis]) * (1.0 / pow(2000.0, Double(2 * j) / Double(dim)))
                    for _ in 0 ..< 2 {
                        worst = max(worst, abs(Double(mc[r * width + col]) - cos(angle)), abs(Double(ms[r * width + col]) - sin(angle)))
                        col += 1
                    }
                }
            }
        }
        return worst
    }

    @Test func textPositionsUpTo511MatchADoubleReference() throws {
        let rows = (0 ..< 512).map { [Float(0), 0, 0, Float($0)] }
        let worst = try Self.maxError(rows: rows)
        print("ROPEREAL text ids L=0..511 (512 rows): max-abs \(worst)")
        #expect(worst <= 1e-4)
    }

    @Test func aFull64x64ImageGridMatchesADoubleReference() throws {
        var rows: [[Float]] = []
        for h in 0 ..< 64 { for w in 0 ..< 64 { rows.append([0, Float(h), Float(w), 0]) } }
        let worst = try Self.maxError(rows: rows)
        print("ROPEREAL image ids 64x64 grid (4096 rows): max-abs \(worst)")
        #expect(worst <= 1e-4)
    }
}
