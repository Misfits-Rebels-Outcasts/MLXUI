import Darwin
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXUI

/// **CF-5b-FIX-3 — isolate the long-state defect to the backbone or our head.**
///
/// The review's point: "Swift" is `mlx-swift-lm`'s backbone *plus* CF-5a's port, and `2026-500` compared
/// end to end. Two cross-feeds on the two worst records (`4k noul payable`, `8k noul hold`):
///
/// - **A** — the reference's hidden states (`Fixtures/Clef/xfeed/reference.safetensors`, dumped bf16 and
///   gitignored) through **this** `ClefHead` and this `ClefLexicalRows`, compared with the reference answer.
/// - **B** — this backbone's hidden states, written to a safetensors the **reference** `JointSchemaHead`
///   then consumes (`RSI/notes/clef/cross-feed.py --head-on-swift`).
///
/// A matches and B doesn't → the backbone. A doesn't → our head. Both fail → both.
///
/// The sandboxed test host can only write inside its container, so the dumps go to its temporary
/// directory and the path is printed for the Python side.
private let clefCrossFeedDirectory: URL? = {
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let directory = repoRoot.appendingPathComponent("Fixtures/Clef/xfeed")
    guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("records.json").path),
          FileManager.default.fileExists(atPath: directory.appendingPathComponent("reference.safetensors").path)
    else { return nil }
    return directory
}()

private let clefCrossFeedModelDirectory: URL? = {
    let identifier = "mlx-community--clef-flash-4bit"
    var candidates = [ModelStore.shared.directory(forHFModelID: "mlx-community/clef-flash-4bit")]
    if let entry = getpwuid(getuid()), let path = entry.pointee.pw_dir {
        candidates.append(
            URL(fileURLWithPath: String(cString: path), isDirectory: true)
                .appendingPathComponent("Library/Application Support/AI Browser/models/\(identifier)"))
    }
    let needed = ["config.json", "tokenizer.json", "joint_head.safetensors"]
    for candidate in candidates
    where needed.allSatisfy({ FileManager.default.fileExists(atPath: candidate.appendingPathComponent($0).path) }) {
        return candidate
    }
    return nil
}()

private let clefCrossFeedReady = clefCrossFeedDirectory != nil && clefCrossFeedModelDirectory != nil

private func crossFeedCosine(_ a: [Float], _ b: [Float]) -> Double {
    var dot = 0.0
    var left = 0.0
    var right = 0.0
    for index in 0 ..< min(a.count, b.count) {
        dot += Double(a[index]) * Double(b[index])
        left += Double(a[index]) * Double(a[index])
        right += Double(b[index]) * Double(b[index])
    }
    return dot / max((left * right).squareRoot(), 1e-12)
}

@Suite(.serialized, .enabled(if: clefCrossFeedReady))
struct ClefCrossFeedTests {
    private func golden() throws -> ClefJSON {
        try ClefJSONParser.parse(try String(
            contentsOf: try #require(clefCrossFeedDirectory).appendingPathComponent("records.json"),
            encoding: .utf8))
    }

    private func modelDirectory() throws -> URL { try #require(clefCrossFeedModelDirectory) }

    private func isCrossFed(_ record: ClefJSON) -> Bool { record["index"]?.intValue != nil
        && (record["band"]?.stringValue == "4k" && record["question"]?["id"]?.stringValue == "payable"
            || record["band"]?.stringValue == "8k" && record["question"]?["id"]?.stringValue == "hold") }

    /// (b) All twelve long records — not just the five `tokens.json` prompts — encode to the same ids and
    /// spans as the reference. Rules the prompt builder out.
    @Test func spansAndIdsMatchForAllLongRecords() async throws {
        let tokenizer = ClefMLXTokenizer(
            upstream: try await HFTokenizerLoader().load(from: try modelDirectory()))
        let records = try #require(try golden()["records"]?.arrayValue)
        #expect(records.count == 12)
        for record in records {
            let request = try ClefRequest(json: try #require(record["request"]))
            let encoded = try ClefPrompt.encodeRecord(
                tokenizer: tokenizer, record: request, maxLength: ClefPrompt.defaultMaxLength)
            let name = "\(record["band"]?.stringValue ?? "?") \(record["question"]?["id"]?.stringValue ?? "?")"
            #expect(encoded.inputIDs == record["input_ids"]?.intArray, "\(name): ids")
            let span = try #require(record["question_span"]?.intArray)
            #expect(encoded.questions[0].questionSpan == span[0] ..< span[1], "\(name): question span")
            let optionSpans = try #require(record["option_spans"]?.arrayValue).compactMap(\.intArray)
            #expect(encoded.questions[0].optionSpans.map { $0.lowerBound ..< $0.upperBound }
                == optionSpans.map { $0[0] ..< $0[1] }, "\(name): option spans")
            #expect(encoded.questions[0].optionIDs == record["option_ids"]?.arrayValue?.compactMap(\.stringValue),
                    "\(name): option ids")
        }
    }

    /// (a) both directions, and (c) the backbone's per-position and per-span cosines.
    @Test func crossFeedBothDirections() async throws {
        let directory = try modelDirectory()
        let golden = try golden()
        let tokenizer = ClefMLXTokenizer(upstream: try await HFTokenizerLoader().load(from: directory))
        let head = try ClefHead.load(
            configURL: directory.appendingPathComponent("joint_head_config.json"),
            weightsURL: directory.appendingPathComponent("joint_head.safetensors"))
        let backbone = try await ClefBackbone.load(
            modelDirectory: directory, hiddenSize: head.config.hiddenSize)
        let dumpDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("clef-cross-feed", isDirectory: true)
        try? FileManager.default.createDirectory(at: dumpDirectory, withIntermediateDirectories: true)
        print("XF dump directory \(dumpDirectory.path)")

        for record in try #require(golden["records"]?.arrayValue) where isCrossFed(record) {
            let index = try #require(record["index"]?.intValue)
            let name = "\(record["band"]?.stringValue ?? "?") \(record["question"]?["id"]?.stringValue ?? "?")"
            let request = try ClefRequest(json: try #require(record["request"]))
            let encoded = try ClefPrompt.encodeRecord(
                tokenizer: tokenizer, record: request, maxLength: ClefPrompt.defaultMaxLength)
            let optionIDs = encoded.questions[0].optionIDs
            // `noul` reports one number; the other types a probability per option.
            let observed = try #require(record["observed"])
            let expected: [String: Double]
            if let probabilities = observed["probabilities"]?.dictionaryValue {
                expected = probabilities.compactMapValues(\.doubleValue)
            } else {
                let noul = try #require(observed["noul"]?.doubleValue)
                expected = ["true": noul, "false": 1 - noul]
            }

            let measured = try await backbone.withHidden(ids: encoded.inputIDs) { hidden, rows in
                // (a) A: the reference's states through this head and this lexical lookup.
                let reference = try MLX.loadArrays(
                    url: try #require(clefCrossFeedDirectory).appendingPathComponent("reference.safetensors"))[
                        "case\(index).hidden"]!.asType(.bfloat16)
                let logits = head(hidden: reference, record: encoded) { ids in rows(ids) }
                let probabilities = softmax(logits[0].asType(.float32)).asArray(Float.self)

                // (c): per-position full-vector cosine, and the per-span means.
                let count = min(hidden.dim(0), reference.dim(0))
                let width = hidden.dim(1)
                let swiftFlat = hidden[0 ..< count, .ellipsis].asType(.float32).asArray(Float.self)
                let referenceFlat = reference[0 ..< count, .ellipsis].asType(.float32).asArray(Float.self)
                var cosines: [Double] = []
                cosines.reserveCapacity(count)
                for position in 0 ..< count {
                    let lower = position * width
                    cosines.append(crossFeedCosine(
                        Array(swiftFlat[lower ..< lower + width]),
                        Array(referenceFlat[lower ..< lower + width])))
                }
                var spanCosines: [Double] = []
                for span in encoded.questions[0].optionSpans + [encoded.questions[0].questionSpan] {
                    let swiftMean = hidden[span, .ellipsis].asType(.float32).mean(axis: 0).asArray(Float.self)
                    let referenceMean = reference[span, .ellipsis].asType(.float32).mean(axis: 0)
                        .asArray(Float.self)
                    spanCosines.append(crossFeedCosine(swiftMean, referenceMean))
                }
                try save(arrays: ["hidden": hidden],
                         url: dumpDirectory.appendingPathComponent("swift_case\(index).safetensors"))
                let sorted = cosines.sorted()
                return (probabilities, sorted.first ?? 0, sorted[sorted.count / 2],
                        cosines.firstIndex(of: sorted.first ?? 0) ?? -1, spanCosines)
            }

            let predicted = Dictionary(uniqueKeysWithValues: zip(optionIDs, measured.0.map(Double.init)))
            let worst = expected.map { abs((predicted[$0.key] ?? 0) - $0.value) }.max() ?? 1
            let reference = expected.map { "\($0.key)=\(String(format: "%.3f", $0.value))" }
                .sorted().joined(separator: " ")
            print("XF A \(name): Δp=\(String(format: "%.4f", worst)) (reference \(reference))")
            let spans = measured.4.map { String(format: "%.4f", $0) }.joined(separator: " ")
            print("XF C \(name): position cosine min=\(String(format: "%.6f", measured.1)) "
                + "median=\(String(format: "%.6f", measured.2)) argmin=\(measured.3)/\(encoded.inputIDs.count) "
                + "| span means [\(spans)]")
        }
    }
}
