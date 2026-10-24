import Darwin
import Foundation
import MLX
import MLXGuidedGeneration
import MLXLMCommon
import Testing

@testable import MLXUI

/// WW4-1-1's measurement row (design note: "every cost number is a thing WW4-1 must measure"). Prints `GRAMMAR-MEASURE …` lines
/// for the journal; the assertions are only sanity — the numbers are machine-dependent and belong in prose. Uses the installed
/// MLX chat model with the largest vocabulary (Qwen3-8B, the brief's choice, isn't installed here). Skips on a machine
/// without it. Reads the tokenizer only; no weights are loaded.
private let grammarMeasurementDirectory: URL? = {
    let identifier = "mlx-community--gemma-4-e4b-it-4bit"
    var candidates = [ModelStore.shared.directory(forHFModelID: "mlx-community/gemma-4-e4b-it-4bit")]
    if let entry = getpwuid(getuid()), let path = entry.pointee.pw_dir {
        candidates.append(URL(fileURLWithPath: String(cString: path), isDirectory: true)
            .appendingPathComponent("Library/Application Support/AI Browser/models/\(identifier)"))
    }
    return candidates.first { dir in
        ["tokenizer.json", "generation_config.json"].allSatisfy { FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path) }
    }
}()

@Suite(.serialized, .enabled(if: grammarMeasurementDirectory != nil))
struct ServeGrammarMeasurementTests {
    private static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    private static func residentMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count) }
        }
        return status == KERN_SUCCESS ? Double(info.resident_size) / 1_048_576 : -1
    }

    private static func ms(_ body: () throws -> Void) rethrows -> Double {
        let start = ContinuousClock.now
        try body()
        let d = ContinuousClock.now - start
        return (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000
    }

    private static func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }

    private static let small = #"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}"#
    private static let large = """
    {"type":"object","properties":{
      "id":{"type":"string"},"title":{"type":"string"},"status":{"enum":["open","closed","blocked","review"]},
      "priority":{"type":"integer"},"labels":{"type":"array","items":{"type":"string"}},
      "assignee":{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"}},"required":["name"]},
      "estimate":{"type":"number"},"done":{"type":"boolean"},
      "children":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"title":{"type":"string"},"done":{"type":"boolean"}},"required":["id","title"]}},
      "notes":{"anyOf":[{"type":"string"},{"type":"null"}]}},
     "required":["id","title","status","priority"]}
    """

    @Test func measure() async throws {
        let directory = try #require(grammarMeasurementDirectory)
        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        let generation = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("generation_config.json"))) as? [String: Any])
        let eosIDs = Set((generation["eos_token_id"] as? [Int]) ?? [])
        let stop = eosIDs.union(tokenizer.eosTokenId.map { [$0] } ?? [])
        let eos = try #require(ServeGrammarEOS.pick(stopIDs: stop, tokenizerEOS: tokenizer.eosTokenId, tokenString: { tokenizer.convertIdToToken($0) }))
        print("GRAMMAR-MEASURE model=\(directory.lastPathComponent) stopIDs=\(stop.sorted()) tokenizerEOS=\(String(describing: tokenizer.eosTokenId)) pick=\(eos) token=\(tokenizer.convertIdToToken(eos) ?? "nil")")
        #expect(tokenizer.convertIdToToken(eos) == "<turn|>")

        // --- resident cost, measured FIRST (before any build has warmed the allocator) over three retained vocabularies ---
        malloc_zone_pressure_relief(nil, 0)
        let footprint0 = Self.footprintMB(), resident0 = Self.residentMB()
        var retained: [ServeGrammarVocabulary] = []
        var totals: [Double] = []
        for index in 0 ..< 3 {
            totals.append(try Self.ms {
                retained.append(try ServeGrammarVocabulary(directory: directory.appendingPathComponent("copy\(index)"), tokenizer: tokenizer, eosTokenID: eos))
            })
            print("GRAMMAR-MEASURE after vocabulary #\(index + 1): footprint=\(String(format: "%.1f", Self.footprintMB())) MB resident=\(String(format: "%.1f", Self.residentMB())) MB")
        }
        let footprint1 = Self.footprintMB(), resident1 = Self.residentMB()
        let held = try #require(retained.first)
        print("GRAMMAR-MEASURE vocabulary_total_ms=\(totals.map { Int($0) }) whitespaceTokens=\(held.whitespaceTokenIDs.count) footprint_before=\(String(format: "%.1f", footprint0)) MB; per-vocabulary footprint_delta_MB=\(String(format: "%.1f", (footprint1 - footprint0) / 3)) resident_delta_MB=\(String(format: "%.1f", (resident1 - resident0) / 3))")
        let vocabSize = held.vocabularyCount

        // --- vocabulary build, split into its parts (three runs; the first is the cold one) ---
        var extractMs: [Double] = [], whitespaceMs: [Double] = [], tablesMs: [Double] = []
        for _ in 0 ..< 3 {
            var extracted: TokenizerVocabExtractor.GrammarVocab?
            extractMs.append(Self.ms { extracted = TokenizerVocabExtractor.extractForGrammar(from: tokenizer) })
            whitespaceMs.append(Self.ms { _ = WhitespaceTokenBias.compute(tokenizer: tokenizer) })
            let vocab = try #require(extracted)
            tablesMs.append(try Self.ms { _ = try GrammarTokenizer(vocab: vocab.vocab, vocabType: vocab.vocabType, eosTokenId: Int32(eos)) })
        }
        print("GRAMMAR-MEASURE vocab=\(vocabSize) extract_ms=\(extractMs.map { Int($0) }) whitespace_ms=\(whitespaceMs.map { Int($0) }) grammarTokenizer_ms=\(tablesMs.map { Int($0) })")

        // --- compile: cold vs warm (lease from idle after a rollback) ---
        let cache = ServeGrammarCache()
        for (name, format) in [("small", ServeGrammarFormat.jsonSchema(Self.small)), ("large", .jsonSchema(Self.large)), ("json_object", .jsonObject)] {
            let before = Self.footprintMB()
            var cold: [Double] = [], warm: [Double] = []
            for run in 0 ..< 3 {
                let fresh = ServeGrammarCache()                                  // a fresh cache: every run is a real compile
                var lease: ServeGrammarLease?
                cold.append(try await {
                    let start = ContinuousClock.now
                    lease = try await fresh.lease(vocabulary: held, format: format)
                    let d = ContinuousClock.now - start
                    return (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000
                }())
                if run == 0, let lease { await cache.release(lease, committedTokens: 0) }
            }
            for _ in 0 ..< 20 {
                let start = ContinuousClock.now
                let lease = try await cache.lease(vocabulary: held, format: format)
                let d = ContinuousClock.now - start
                warm.append((Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000)
                await cache.release(lease, committedTokens: 0)
            }
            let after = Self.footprintMB()
            print("GRAMMAR-MEASURE compile schema=\(name) cold_ms=\(cold.map { Int($0) }) warm_median_ms=\(String(format: "%.3f", Self.median(warm))) footprint_delta_MB=\(String(format: "%.1f", after - before))")
        }

        // --- per-token: the mask, its unpacking on the device, and the whole process() ---
        let lease = try await cache.lease(vocabulary: held, format: .jsonSchema(Self.large))
        let processor = ServeGrammarProcessor(constraint: lease.constraint, vocabulary: held, format: .jsonSchema(Self.large))
        let width = vocabSize + 256                                              // a padded embedding matrix, as in the real models
        var maskMs: [Double] = [], unpackMs: [Double] = [], processMs: [Double] = []
        let text = Array(#"{"id":"A-1","title":"Ship the grammar processor","status":"open","priority":3,"labels":["a","b"]}"#)
        var committed = 0
        for character in text {
            var words: MaskResult?
            maskMs.append(try Self.ms { words = try lease.constraint.computeMask() })
            let mask = try #require(words)
            unpackMs.append(try Self.ms { _ = try MLXErrorGuard.read(ServeGrammarProcessor.additiveMask(words: mask.mask, vocabularyCount: vocabSize, width: width), as: Float.self) })
            let logits = MLXArray.zeros([1, width])
            try MLXErrorGuard.evaluate(logits)
            processMs.append(try Self.ms { _ = try MLXErrorGuard.read(processor.process(logits: logits), as: Float.self) })
            let ids = tokenizer.encode(text: String(character), addSpecialTokens: false)
            guard ids.count == 1 else { continue }                                 // the large vocabulary has single-character tokens for these
            processor.didSample(token: MLXArray([Int32(ids[0])]))
            committed += 1
        }
        print("GRAMMAR-MEASURE per_token computeMask_median_ms=\(String(format: "%.3f", Self.median(maskMs))) unpack_eval_median_ms=\(String(format: "%.3f", Self.median(unpackMs))) process_median_ms=\(String(format: "%.3f", Self.median(processMs))) steps=\(text.count) committed=\(committed) failure=\(processor.failure ?? "none")")
        #expect(processor.failure == nil)
        await cache.release(lease, committedTokens: processor.committedTokens)
    }
}

/// Why the first measurement's 10-property schema took 58 s to compile on a 262 k vocabulary when a 2-property one took 0.48 s:
/// one cold compile per feature, each alone, so the journal can say which constructs are expensive. Run on demand
/// (`MLXUI_MEASURE_COMPILE=1` via `TEST_RUNNER_MLXUI_MEASURE_COMPILE=1`) — it takes minutes.
@Suite(.serialized, .enabled(if: grammarMeasurementDirectory != nil && ProcessInfo.processInfo.environment["MLXUI_MEASURE_COMPILE"] != nil))
struct ServeGrammarCompileCostTests {
    private static let features: [(String, String)] = [
        ("plain 10 props", #"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"},"c":{"type":"integer"},"d":{"type":"number"},"e":{"type":"boolean"},"f":{"type":"string"},"g":{"type":"string"},"h":{"type":"integer"},"i":{"type":"string"},"j":{"type":"string"}},"required":["a","c"]}"#),
        ("enum", #"{"type":"object","properties":{"s":{"enum":["open","closed","blocked","review"]}},"required":["s"]}"#),
        ("array of strings", #"{"type":"object","properties":{"l":{"type":"array","items":{"type":"string"}}},"required":["l"]}"#),
        ("array of objects", #"{"type":"object","properties":{"c":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"done":{"type":"boolean"}},"required":["id"]}}},"required":["c"]}"#),
        ("anyOf string|null", #"{"type":"object","properties":{"n":{"anyOf":[{"type":"string"},{"type":"null"}]}},"required":["n"]}"#),
        ("integer min/max", #"{"type":"object","properties":{"p":{"type":"integer","minimum":1,"maximum":5}},"required":["p"]}"#),
        ("string maxLength 50", #"{"type":"object","properties":{"t":{"type":"string","maxLength":50}},"required":["t"]}"#),
        ("string maxLength 200", #"{"type":"object","properties":{"t":{"type":"string","maxLength":200}},"required":["t"]}"#),
        ("array maxItems 8", #"{"type":"object","properties":{"l":{"type":"array","items":{"type":"string"},"maxItems":8}},"required":["l"]}"#),
        ("string pattern", #"{"type":"object","properties":{"z":{"type":"string","pattern":"^[A-Z]{2}-[0-9]{4}$"}},"required":["z"]}"#),
    ]

    @Test func compileCostByFeature() async throws {
        let directory = try #require(grammarMeasurementDirectory)
        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        let eos = 106
        let vocabulary = try ServeGrammarVocabulary(directory: directory, tokenizer: tokenizer, eosTokenID: eos)
        for (name, schema) in Self.features {
            let cache = ServeGrammarCache()
            let start = ContinuousClock.now
            _ = try await cache.lease(vocabulary: vocabulary, format: .jsonSchema(schema))
            let d = ContinuousClock.now - start
            print("GRAMMAR-MEASURE compile-cost feature=\"\(name)\" cold_ms=\(Int((Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000))")
        }
    }
}
