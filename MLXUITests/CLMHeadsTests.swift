import Testing
import Foundation
import MLX
@testable import MLXUI

/// CL-4a golden test — `CLMHeadPair` against `Fixtures/CLM/heads.json` (provenance in
/// `Fixtures/CLM/PROVENANCE.md`). Loads the real, shipped heads weights from
/// `Fixtures/CLM/heads_weights/` (gitignored, not committed — downloaded from `RealityCat/CLM-v0.1-8B-MLX-8bit`'s public
/// `heads/` folder, sha256 `293646f3dca1900ac4038a1afcdee769ce64687c1636b7b311774b0663539523`
/// — matches §0 of `RSI/DelegateCLMBacklog.md` exactly), runs the same 3 fixed input vectors
/// through it, and checks cosine similarity against Python's projections — cosine ≥ 0.99999
/// per CL-4a's hard gate. Skips cleanly when the weights aren't present (a fresh clone) —
/// the two `curl` lines in `PROVENANCE.md` fetch them.
@Suite(.serialized, .enabled(if: CLMHeadsTests.headsWeightsPresent))
struct CLMHeadsTests {
    static let headsWeightsPresent: Bool = {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let weights = repoRoot.appendingPathComponent(
            "Fixtures/CLM/heads_weights/CLM_v0.1-8B.safetensors")
        return FileManager.default.fileExists(atPath: weights.path)
    }()

    private struct GoldenVector: Decodable {
        var input: [Double]
        var state_projection: [Double]
        var action_projection: [Double]
    }

    private struct Golden: Decodable {
        var scale: Double
        var logit_scale: Double
        var vectors: [GoldenVector]
    }

    private func loadGolden() throws -> Golden {
        let filePath = #filePath                      // .../MLXUITests/CLMHeadsTests.swift
        let repoRoot = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("Fixtures/CLM/heads.json")
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

    private func headsDirectory() -> URL {
        let filePath = #filePath
        let repoRoot = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return repoRoot.appendingPathComponent("Fixtures/CLM/heads_weights")
    }

    private func cosine(_ a: [Double], _ b: [Double]) -> Double {
        let dot = zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
        let normA = sqrt(a.reduce(0.0) { $0 + $1 * $1 })
        let normB = sqrt(b.reduce(0.0) { $0 + $1 * $1 })
        return dot / (normA * normB)
    }

    @Test func scaleAndLogitScaleMatchGolden() throws {
        let golden = try loadGolden()
        let heads = try CLMHeadPair(headsDirectory: headsDirectory())
        #expect(abs(Double(heads.scale) - golden.scale) < 1e-6)
        #expect(abs(Double(heads.logitScale) - golden.logit_scale) < 1e-4)
    }

    @Test func projectionsMatchGoldenToFivedNines() throws {
        let golden = try loadGolden()
        let heads = try CLMHeadPair(headsDirectory: headsDirectory())

        for (index, vector) in golden.vectors.enumerated() {
            let input = MLXArray(vector.input.map { Float($0) })
            let stateProjection = heads.project(input, which: .state).asArray(Float.self).map(Double.init)
            let actionProjection = heads.project(input, which: .action).asArray(Float.self).map(Double.init)

            #expect(stateProjection.count == 512, "vector \(index): state projection wrong size")
            #expect(actionProjection.count == 512, "vector \(index): action projection wrong size")

            let stateCos = cosine(stateProjection, vector.state_projection)
            let actionCos = cosine(actionProjection, vector.action_projection)
            #expect(stateCos >= 0.99999, "vector \(index): state cosine \(stateCos) below 0.99999")
            #expect(actionCos >= 0.99999, "vector \(index): action cosine \(actionCos) below 0.99999")
        }
    }

    @Test func projectionsAreL2Normalized() throws {
        let golden = try loadGolden()
        let heads = try CLMHeadPair(headsDirectory: headsDirectory())
        let input = MLXArray(golden.vectors[0].input.map { Float($0) })
        let stateProjection = heads.project(input, which: .state).asArray(Float.self).map(Double.init)
        let norm = sqrt(stateProjection.reduce(0.0) { $0 + $1 * $1 })
        #expect(abs(norm - 1.0) < 1e-4)
    }
}
