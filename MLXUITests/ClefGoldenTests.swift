import Darwin
import Foundation
import MLX
import Testing

@testable import MLXUI

/// CF-5a's hard gate (rule 8): the goldens `RSI/notes/clef/make-goldens.py` produced with the
/// **reference** `clef_mlx.py` — `Fixtures/Clef/prompt.json`, `Fixtures/Clef/head.json` and
/// `Fixtures/Clef/head_inputs.safetensors` (provenance in `Fixtures/Clef/PROVENANCE.md`).
///
/// The suite needs two things and skips cleanly without either:
/// - the fixtures (until the owner runs the script, `Fixtures/Clef/` does not exist);
/// - the **installed model** — the prompt golden re-tokenizes with its `tokenizer.json` and the
///   head golden needs its 244 MB `joint_head.safetensors`. That is why the weights are never a
///   fixture: the gitignore rule is about committed weights, and these tests skip instead.
private let clefFixtureDirectory: URL? = {
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // MLXUITests
        .deletingLastPathComponent()  // repo root
    let directory = repoRoot.appendingPathComponent("Fixtures/Clef")
    let files = ["prompt.json", "head.json", "head_inputs.safetensors", "PROVENANCE.md"]
    guard files.allSatisfy({ FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) })
    else { return nil }
    return directory
}()

/// The test host is the **sandboxed** `MLXUI` app (`MLXUI-Direct` has no test action), so
/// `ModelStore`'s `applicationSupportDirectory` resolves inside the container
/// (`~/Library/Containers/…`), where the owner's install is not. The install lives in the real home
/// (`CLAUDE.md`, "Install & storage"), and `getpwuid` is the one call that still reports it from
/// inside a sandbox.
private func realUserHome() -> URL? {
    guard let entry = getpwuid(getuid()), let path = entry.pointee.pw_dir else { return nil }
    return URL(fileURLWithPath: String(cString: path), isDirectory: true)
}

private let clefModelDirectory: URL? = {
    let identifier = "mlx-community--clef-flash-4bit"
    var candidates = [ModelStore.shared.directory(forHFModelID: "mlx-community/clef-flash-4bit")]
    if let home = realUserHome() {
        candidates.append(
            home.appendingPathComponent("Library/Application Support/AI Browser/models/\(identifier)"))
    }
    let needed = ["joint_head.safetensors", "joint_head_config.json", "tokenizer.json"]
    for candidate in candidates
    where needed.allSatisfy({ FileManager.default.fileExists(atPath: candidate.appendingPathComponent($0).path) }) {
        return candidate
    }
    return nil
}()

private let clefGoldensReady = clefFixtureDirectory != nil && clefModelDirectory != nil

/// The reference's head computes and **returns** bfloat16 logits, so a logit is quantized to ~1e-3 at
/// the magnitudes here (one ulp is 2⁻⁷ of the value). Measured on `Fixtures/Clef/head.json`
/// (journal `2026-496` §7): `prior` and `cosine` agree with the reference to the bit, and the
/// residual MLP differs by **one ulp** on 2 of 3 `single-score` options — a fused-vs-unfused
/// `addmm` accumulation — which reaches the logit as ≈4 ulps through the gate. The cycle's 1e-4
/// absolute number is unreachable for any implementation that is not bit-exact in bf16, so the gate
/// is stated in the units bf16 actually has: ≈4 ulps on a logit, 1e-3 on a probability, and an
/// **identical argmax** — the thing a decision is made from.
/// owner 2026-10-22, mlx core 0.32.2 drift, journal 552: logit tolerance is a fixed absolute 1e-2 (worst observed
/// 7.8e-3); probability tolerance 2e-3 (was 2e-3 / 1e-3). Small logits inherit error from larger intermediates,
/// so a per-value ulp bound does not hold. Tight Swift-side drift detection is the snapshot test's job.
private let clefLogitTolerance = 1e-2
private let clefProbabilityTolerance = 2e-3

/// Reading the generator's JSON through the port's own parser keeps the fixture's number kinds
/// (`int` vs `float`) and object order intact — which is exactly what the port must reproduce.
private enum ClefGolden {
    static func load(_ url: URL) throws -> ClefJSON {
        try ClefJSONParser.parse(try String(contentsOf: url, encoding: .utf8))
    }
}

@Suite(.serialized, .enabled(if: clefGoldensReady))
struct ClefGoldenTests {
    private func promptGolden() throws -> ClefJSON {
        try ClefGolden.load(try #require(clefFixtureDirectory).appendingPathComponent("prompt.json"))
    }

    private func headGolden() throws -> ClefJSON {
        try ClefGolden.load(try #require(clefFixtureDirectory).appendingPathComponent("head.json"))
    }

    private func modelDirectory() throws -> URL {
        try #require(clefModelDirectory)
    }

    private func decodedRecord(from golden: ClefJSON) throws -> ClefEncodedRecord {
        let questions = try #require(golden["questions"]?.arrayValue)
        let encoded = try questions.map { question -> ClefEncodedQuestion in
            let optionIDs = (question["option_ids"]?.arrayValue ?? []).compactMap { $0.stringValue }
            return ClefEncodedQuestion(
                questionID: try #require(question["question_id"]?.stringValue),
                type: try #require(question["type"]?.intValue),
                questionSpan: try #require(question["question_span"]?.spanValue),
                optionSpans: try (question["option_spans"]?.arrayValue ?? []).map {
                    try #require($0.spanValue)
                },
                optionIDs: optionIDs)
        }
        let ids = try #require(golden["input_ids"]?.intArray)
        return ClefEncodedRecord(
            inputIDs: ids, questions: encoded,
            stateTokensKept: golden["state_tokens_kept"]?.intValue ?? ids.count,
            stateTokensTotal: golden["state_tokens_total"]?.intValue ?? ids.count)
    }

    /// The index of the largest value, first-wins (Python's `max`/`argmax` convention).
    private static func argmax(_ values: [Double]) -> Int? {
        guard let first = values.first else { return nil }
        var best = 0
        for index in values.indices where values[index] > values[best] { best = index }
        _ = first
        return best
    }

    /// Option indices best-first (ties by index), the order a top-k read is made from.
    private static func ranking(_ values: [Double]) -> [Int] {
        values.indices.sorted { values[$0] != values[$1] ? values[$0] > values[$1] : $0 < $1 }
    }

    /// The reference takes `lexical_lookup` as a parameter (`:370`), and calls it once per option
    /// span in question-then-option order. The golden stores those rows, so a cursor over them
    /// reproduces the seam exactly (the head mean-pools the rows it is handed).
    private func lexicalCursor(
        for testCase: ClefJSON, inputs: [String: MLXArray]
    ) throws -> (makeLookup: () -> (MLXArray) -> MLXArray, count: Int) {
        var rows: [MLXArray] = []
        for key in try #require(testCase["lexical_keys"]?.arrayValue).compactMap(\.stringValue) {
            let matrix = try #require(inputs[key])
            rows += (0 ..< matrix.dim(0)).map { matrix[$0] }
        }
        return (
            {
                var cursor = 0
                return { _ in
                    defer { cursor += 1 }
                    return rows[cursor].reshaped(1, -1)
                }
            },
            rows.count
        )
    }

    // MARK: - prompt.json

    /// The reference's `encode_record` output, token for token: the per-fragment tokenization,
    /// every span, and the app-side truncation numbers. Swift must match **exactly** (CF-5a).
    @Test func promptGoldensMatchTheReferenceTokenForToken() async throws {
        let golden = try promptGolden()
        let modelDirectory = try modelDirectory()
        let tokenizer = ClefMLXTokenizer(upstream: try await HFTokenizerLoader().load(from: modelDirectory))
        let defaultMaxLength = golden["max_length"]?.intValue ?? ClefPrompt.defaultMaxLength
        let cases = try #require(golden["cases"]?.arrayValue)
        #expect(cases.count >= 6)

        for testCase in cases {
            let name = testCase["name"]?.stringValue ?? "?"
            let request = try ClefRequest(json: try #require(testCase["request"]))
            let record = try ClefPrompt.encodeRecord(
                tokenizer: tokenizer, record: request,
                maxLength: testCase["max_length"]?.intValue ?? defaultMaxLength)

            #expect(record.inputIDs == testCase["input_ids"]?.intArray, "\(name): token ids")
            #expect(record.stateTokensTotal == testCase["state_tokens_total"]?.intValue, "\(name): state total")
            #expect(record.stateTokensKept == testCase["state_tokens_kept"]?.intValue, "\(name): state kept")
            #expect(record.stateTruncated == (testCase["state_truncated"]?.boolValue ?? false), "\(name): truncated flag")
            #expect(record.questions.count == request.questions.count, "\(name): question count")

            let expectedQuestions = try #require(testCase["questions"]?.arrayValue)
            for (index, expected) in expectedQuestions.enumerated() {
                let actual = record.questions[index]
                #expect(actual.questionID == expected["question_id"]?.stringValue, "\(name): question id")
                #expect(actual.type == expected["type"]?.intValue, "\(name): question type")
                #expect(actual.questionSpan == expected["question_span"]?.spanValue, "\(name): question span")
                #expect(actual.optionSpans == (expected["option_spans"]?.arrayValue ?? []).compactMap(\.spanValue),
                        "\(name): option spans")
                #expect(actual.optionIDs == expected["option_ids"]?.arrayValue?.compactMap(\.stringValue),
                        "\(name): option ids")
            }
        }
    }

    /// The same requests' **answers**: `ClefAnswer` fed the reference's unrounded probabilities must
    /// reproduce the reference's own `systemone_answer` body — including the tie case, where the
    /// model's two identical descriptions land closest together.
    @Test func promptGoldenObservedAnswersMatchClefAnswer() throws {
        let cases = try #require(try promptGolden()["cases"]?.arrayValue)
        for testCase in cases {
            let name = testCase["name"]?.stringValue ?? "?"
            let request = try ClefRequest(json: try #require(testCase["request"]))
            let observed = try #require(testCase["observed"]?.dictionaryValue)
            let raw = try #require(testCase["probabilities_raw"]?.dictionaryValue)

            for question in request.questions {
                let probabilities = try #require(raw[question.id]?.dictionaryValue).compactMapValues(\.doubleValue)
                let answer = try ClefAnswer.answer(
                    question: question, probabilities: probabilities,
                    stateTruncated: testCase["state_truncated"]?.boolValue ?? false)
                let wire = try #require(observed[question.id]?.dictionaryValue, "\(name): no observed answer")

                switch question.type {
                case .noul:
                    #expect(answer.noulProbability == wire["noul"]?.doubleValue, "\(name): noul")

                case .choice:
                    #expect(answer.choiceLabel == wire["choice"]?.stringValue, "\(name): choice")
                    #expect(
                        abs(answer.confidence - (wire["confidence"]?.doubleValue ?? -1)) < 1e-9,
                        "\(name): choice confidence")
                    let expected = try #require(wire["probabilities"]?.dictionaryValue)
                    for (index, id) in answer.optionLabels.enumerated() {
                        #expect(
                            abs(answer.probabilities[index] - (expected[id]?.doubleValue ?? -1)) < 1e-9,
                            "\(name): P(\(id))")
                    }

                case .score:
                    #expect(
                        abs((answer.scoreValue ?? -1) - (wire["score"]?.doubleValue ?? -2)) < 1e-9,
                        "\(name): score")
                    #expect(
                        abs(answer.confidence - (wire["confidence"]?.doubleValue ?? -1)) < 1e-9,
                        "\(name): score confidence")
                    let legend = try #require(wire["legend"]?.dictionaryValue)
                    for (index, level) in answer.optionLabels.enumerated() {
                        #expect(answer.scoreLevels[index] == legend[level]?.stringValue, "\(name): legend \(level)")
                    }
                }
            }
        }
    }

    // MARK: - head.json

    /// `JointSchemaHead` on the reference's three seeded hidden-state cases: every option logit
    /// within bf16 resolution (see `clefLogitTolerance`) and the **same argmax** per question — the
    /// first case and the mixed case in fact match the reference bit for bit.
    @Test func headGoldensMatchTheReferenceWithinBfloat16Resolution() throws {
        let golden = try headGolden()
        let modelDirectory = try modelDirectory()
        let inputs = try MLX.loadArrays(url: try #require(clefFixtureDirectory).appendingPathComponent("head_inputs.safetensors"))
        let head = try ClefHead.load(
            configURL: modelDirectory.appendingPathComponent("joint_head_config.json"),
            weightsURL: modelDirectory.appendingPathComponent("joint_head.safetensors"))

        let cases = try #require(golden["cases"]?.arrayValue)
        #expect(cases.count == 3)
        for testCase in cases {
            let name = testCase["name"]?.stringValue ?? "?"
            let record = try decodedRecord(from: testCase)
            let hiddenKey = try #require(testCase["hidden_key"]?.stringValue)
            let hidden = try #require(inputs[hiddenKey])

            let lexical = try lexicalCursor(for: testCase, inputs: inputs)
            #expect(lexical.count > 0, "\(name): the golden carries the lexical rows")
            let logits = head(hidden: hidden, record: record, lexicalLookup: lexical.makeLookup())

            let expectedQuestions = try #require(testCase["questions"]?.arrayValue)
            #expect(logits.count == expectedQuestions.count)
            for (questionIndex, expected) in expectedQuestions.enumerated() {
                let expectedLogits = try #require(expected["logits"]?.arrayValue).compactMap(\.doubleValue)
                let actual = logits[questionIndex].asArray(Float.self).map(Double.init)
                #expect(actual.count == expectedLogits.count, "\(name): logit count")
                for (offset, pair) in zip(actual, expectedLogits).enumerated() {
                    #expect(
                        abs(pair.0 - pair.1) <= clefLogitTolerance,
                        "\(name): logit \(offset) \(pair.0) vs \(pair.1)")
                }
                #expect(Self.argmax(actual) == Self.argmax(expectedLogits), "\(name): argmax")
                // owner 2026-10-22, mlx core 0.32.2 drift, journal 552: the whole option order, not only the argmax.
                #expect(Self.ranking(actual) == Self.ranking(expectedLogits), "\(name): top-k order")

                // …and the same probabilities the reference's own softmax produced.
                let expectedProbabilities = try #require(expected["probabilities"]?.arrayValue).compactMap(\.doubleValue)
                let probabilities = softmax(logits[questionIndex]).asArray(Float.self).map(Double.init)
                for (offset, pair) in zip(probabilities, expectedProbabilities).enumerated() {
                    #expect(
                        abs(pair.0 - pair.1) <= clefProbabilityTolerance,
                        "\(name): P \(offset) \(pair.0) vs \(pair.1)")
                }
            }
        }
    }

    /// WW2-3-1 (owner 2026-10-22, journal 552 §3): this branch's own head logits and probabilities for the same
    /// three cases (`Fixtures/Clef/swift-snapshot-mlx-swift-0.32.3/head.json`); probability <= 1e-4, no ranking change.
    @Test func headMatchesTheBranchSnapshot() throws {
        let golden = try headGolden()
        let modelDirectory = try modelDirectory()
        let inputs = try MLX.loadArrays(url: try #require(clefFixtureDirectory).appendingPathComponent("head_inputs.safetensors"))
        let head = try ClefHead.load(
            configURL: modelDirectory.appendingPathComponent("joint_head_config.json"),
            weightsURL: modelDirectory.appendingPathComponent("joint_head.safetensors"))

        var recorded: [String: Any] = [:]
        for testCase in try #require(golden["cases"]?.arrayValue) {
            let name = testCase["name"]?.stringValue ?? "?"
            let record = try decodedRecord(from: testCase)
            let hiddenKey = try #require(testCase["hidden_key"]?.stringValue)
            let hidden = try #require(inputs[hiddenKey])
            let lexical = try lexicalCursor(for: testCase, inputs: inputs)
            let logits = head(hidden: hidden, record: record, lexicalLookup: lexical.makeLookup())
            recorded[name] = logits.map { vector in
                [
                    "logits": vector.asArray(Float.self).map(Double.init),
                    "probabilities": softmax(vector).asArray(Float.self).map(Double.init),
                ]
            }
        }
        let url = SwiftSnapshot.url("Clef", "head.json")
        if SwiftSnapshot.isRecording { try SwiftSnapshot.write(recorded, to: url); return }

        let stored = try #require(try SwiftSnapshot.read(url) as? [String: [[String: [Double]]]], "no snapshot at \(url.path)")
        #expect(Set(stored.keys) == Set(recorded.keys))
        for (name, was) in stored {
            let now = try #require(recorded[name] as? [[String: [Double]]])
            #expect(now.count == was.count, "\(name): question count")
            for (index, pair) in zip(now, was).enumerated() {
                let a = pair.0["probabilities"] ?? [], b = pair.1["probabilities"] ?? []
                #expect(a.count == b.count, "\(name)/\(index): option count")
                for (x, y) in zip(a, b) {
                    #expect(abs(x - y) <= SwiftSnapshot.probabilityTolerance, "\(name)/\(index): P \(x) vs \(y)")
                }
                #expect(Self.ranking(pair.0["logits"] ?? []) == Self.ranking(pair.1["logits"] ?? []), "\(name)/\(index): ranking")
            }
        }
    }

    /// The CF-5a dtype check (readme T6): bfloat16 has float32's exponent range, so a large-magnitude
    /// hidden state must not push a LayerNorm or the scorer to inf/nan the way Laya's fp16 port did
    /// (journal `2026-334`).
    @Test func headStaysFiniteOnLargeMagnitudeInputs() throws {
        let golden = try headGolden()
        let modelDirectory = try modelDirectory()
        let inputs = try MLX.loadArrays(url: try #require(clefFixtureDirectory).appendingPathComponent("head_inputs.safetensors"))
        let head = try ClefHead.load(
            configURL: modelDirectory.appendingPathComponent("joint_head_config.json"),
            weightsURL: modelDirectory.appendingPathComponent("joint_head.safetensors"))

        let cases = try #require(golden["cases"]?.arrayValue)
        let testCase = try #require(cases.first)
        let record = try decodedRecord(from: testCase)
        let hiddenKey = try #require(testCase["hidden_key"]?.stringValue)
        let base = try #require(inputs[hiddenKey])
        let lexical = try lexicalCursor(for: testCase, inputs: inputs)
        for scale in [Float(1), Float(1e4), Float(1e8)] {
            let hidden = (base * scale).asType(.bfloat16)
            let logits = head(hidden: hidden, record: record, lexicalLookup: lexical.makeLookup())
            for vector in logits {
                for value in vector.asArray(Float.self) {
                    #expect(value.isFinite, "scale \(scale): \(value)")
                }
            }
        }
    }

    /// The strict load is strict: a config whose width disagrees with the checkpoint must throw
    /// rather than score with half a head (the reference's `strict=True`, `:577`).
    @Test func headLoadRefusesAMismatchedConfig() throws {
        let modelDirectory = try modelDirectory()
        let weights = modelDirectory.appendingPathComponent("joint_head.safetensors")
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("clef-head-config-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: temporary) }

        let wrong = ClefHeadConfig(
            hiddenSize: 4096, width: 512, routingLayers: 2, layers: 4, heads: 16, feedforward: 4096)
        let data = try JSONEncoder().encode(wrong)
        try data.write(to: temporary)
        #expect(throws: ClefHeadError.self) {
            _ = try ClefHead.load(configURL: temporary, weightsURL: weights)
        }
    }
}
