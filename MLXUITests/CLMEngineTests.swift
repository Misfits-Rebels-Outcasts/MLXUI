import Testing
import Foundation
@testable import MLXUI

/// CL-4b golden test — `CLMEngine.answer` against `Fixtures/CLM/answers.json` (provenance in
/// `Fixtures/CLM/PROVENANCE.md`): 24 real questions from the owner's MLXCLM parity corpus,
/// including 4 of the 8 `parity.json` flips. Needs the real encoder + heads weights at
/// `Fixtures/CLM/{encoder_weights,heads_weights}/` — `encoder_weights/` is gitignored (~8GB,
/// see `CLMEncoderTests.swift`'s header comment for how to re-download it).
///
/// CL-4b's hard gate, per the owner's 2026-09-28 ruling: Swift agrees with Python MLX on
/// 24/24 argmax (unchanged); per-question probability difference ≤ 0.03, AND at least 18/24
/// questions ≤ 0.01. Against Python MLX itself (this checkout's own `clm_mlx.engine`, run in
/// a venv pinned to the exact core mlx-swift 0.31.4 vendors — `mlx==0.31.1`/`mlx-lm==0.31.2`,
/// `Fixtures/CLM/PROVENANCE.md`), not the upstream vLLM server `parity.json` compares against.
/// The 0.01 ceiling widened to 0.03 because the 5 measured misses (max 0.0278, all still
/// unanimous on argmax) are below the upstream server's own run-to-run noise in `parity.json`
/// (median 0.022, p95 0.060) — this port isn't chasing a tighter number than the reference
/// server achieves against itself.
///
/// Parses `answers.json` with `CLMJSONParser`, not `JSONDecoder`/`JSONSerialization` — several
/// real corpus states are JSON objects with float fields (real invoices), and losing the
/// literal number token there was the root cause of 6 of the 7 original disagreements this
/// gate found (`RSI/DelegateCLMBacklog.md` CL-4a, "Numbers: corrected 2026-09-28").
@Suite(.serialized)
struct CLMEngineTests {
    private struct GoldenAnswer {
        var requestID: String
        var questionID: String
        var type: String
        var instructionsJSON: CLMJSON?
        var criteriaJSON: CLMJSON?
        var stateJSON: CLMJSON
        var isKnownFlip: Bool
        var argmaxLabel: String
        var argmaxProbability: Double

        init(json: CLMJSON) throws {
            guard case .object(let fields) = json else {
                throw CLMJSONError.unexpectedCharacter(" ", at: 0)
            }
            func field(_ key: String) -> CLMJSON? { fields.first { $0.key == key }?.value }

            guard case .string(let rid)? = field("request_id") else { throw CLMJSONError.unexpectedEnd }
            guard case .string(let qid)? = field("question_id") else { throw CLMJSONError.unexpectedEnd }
            guard case .string(let t)? = field("type") else { throw CLMJSONError.unexpectedEnd }
            guard case .bool(let flip)? = field("is_known_flip") else { throw CLMJSONError.unexpectedEnd }
            guard let state = field("state") else { throw CLMJSONError.unexpectedEnd }

            requestID = rid
            questionID = qid
            type = t
            isKnownFlip = flip
            stateJSON = state
            instructionsJSON = field("instructions")
            criteriaJSON = field("criteria")

            // `python_mlx_answer`'s numbers are only ever compared as `Double` (never
            // re-rendered as text), so losing the literal token here is fine.
            guard case .object(let answerFields)? = field("python_mlx_answer") else {
                throw CLMJSONError.unexpectedEnd
            }
            func answerField(_ key: String) -> CLMJSON? { answerFields.first { $0.key == key }?.value }

            var probabilities: [String: Double] = [:]
            if case .object(let probFields)? = answerField("probabilities") {
                for pf in probFields {
                    if case .number(let token) = pf.value, let d = Double(token) {
                        probabilities[pf.key] = d
                    }
                }
            } else if case .number(let token)? = answerField("noul"), let noul = Double(token) {
                probabilities = ["false": 1 - noul, "true": noul]
            }
            let best = probabilities.max { $0.value < $1.value }
            argmaxLabel = best?.key ?? ""
            argmaxProbability = best?.value ?? 0
        }
    }

    private static let repoRoot: URL = {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath).deletingLastPathComponent().deletingLastPathComponent()
    }()

    private func loadGolden() throws -> [GoldenAnswer] {
        let url = Self.repoRoot.appendingPathComponent("Fixtures/CLM/answers.json")
        let text = try String(contentsOf: url, encoding: .utf8)
        guard case .array(let items) = try CLMJSONParser.parse(text) else {
            throw CLMJSONError.unexpectedEnd
        }
        return try items.map(GoldenAnswer.init(json:))
    }

    private func requireModelWeights() throws {
        let encoder = Self.repoRoot.appendingPathComponent(
            "Fixtures/CLM/encoder_weights/model-00001-of-00002.safetensors")
        let heads = Self.repoRoot.appendingPathComponent(
            "Fixtures/CLM/heads_weights/CLM_v0.1-8B.safetensors")
        guard FileManager.default.fileExists(atPath: encoder.path),
              FileManager.default.fileExists(atPath: heads.path)
        else {
            throw CLMHeadsError.invalidConfig(
                "Fixtures/CLM/{encoder_weights,heads_weights}/ not present locally — "
                    + "re-download from the HF repo to run this test.")
        }
    }

    /// A model directory shaped like an installed CLM checkpoint (`encoder/` + `heads/`
    /// underneath it), built from the two separately-fetched fixture directories so
    /// `CLMEngine.load` can be pointed at it unchanged. Symlinks the individual **files**
    /// into real subdirectories, rather than symlinking `encoder`/`heads` themselves —
    /// `MLXLMCommon.loadWeights`'s `FileManager.enumerator(at:)` walks the real directory
    /// tree looking for `.safetensors` files, and a symlinked directory *root* isn't
    /// reliably descended into the same way real subdirectories holding symlinked files are.
    private func modelDirectory() throws -> URL {
        try requireModelWeights()
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("CLMEngineTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        try linkContents(
            of: Self.repoRoot.appendingPathComponent("Fixtures/CLM/encoder_weights"),
            into: tmp.appendingPathComponent("encoder"))
        try linkContents(
            of: Self.repoRoot.appendingPathComponent("Fixtures/CLM/heads_weights"),
            into: tmp.appendingPathComponent("heads"))
        return tmp
    }

    private func linkContents(of sourceDir: URL, into destDir: URL) throws {
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
        for name in try FileManager.default.contentsOfDirectory(atPath: sourceDir.path) {
            try FileManager.default.createSymbolicLink(
                at: destDir.appendingPathComponent(name),
                withDestinationURL: sourceDir.appendingPathComponent(name))
        }
    }

    @Test func swiftAgreesWithPythonMLXOn24Of24Questions() async throws {
        let golden = try loadGolden()
        #expect(golden.count == 24)
        let engine = try await CLMEngine.load(modelDirectory: modelDirectory())

        var disagreements: [String] = []
        var within001 = 0
        for item in golden {
            guard let type = LayaQuestionType(rawValue: item.type) else {
                Issue.record("unknown question type \(item.type)")
                continue
            }
            let results = try await engine.answer(
                state: item.stateJSON,
                questions: [(
                    id: item.questionID, type: type,
                    instructions: item.instructionsJSON, criteria: item.criteriaJSON
                )])
            guard let answer = results.first?.answer else {
                disagreements.append("\(item.requestID)/\(item.questionID): no answer")
                continue
            }

            let swiftBest = zip(answer.optionLabels, answer.probabilities).max { $0.1 < $1.1 }
            let swiftLabel = swiftBest?.0 ?? ""
            let swiftProb = swiftBest?.1 ?? 0

            if swiftLabel != item.argmaxLabel {
                let flip = item.isKnownFlip ? " (known flip)" : ""
                disagreements.append(
                    "\(item.requestID)/\(item.questionID)\(flip): argmax \(swiftLabel) vs \(item.argmaxLabel)")
            }
            let diff = abs(swiftProb - item.argmaxProbability)
            if diff <= 0.01 { within001 += 1 }
            #expect(
                diff <= 0.03,
                "\(item.requestID)/\(item.questionID): prob \(swiftProb) vs \(item.argmaxProbability)")
        }
        #expect(disagreements.isEmpty, "argmax disagreements: \(disagreements.joined(separator: "; "))")
        #expect(within001 >= 18, "only \(within001)/24 questions within 0.01 probability difference")
    }
}
