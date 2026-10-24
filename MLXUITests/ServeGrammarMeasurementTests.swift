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

/// WW4-1-2 / ruling AR — the cold-compile cost of every bounded construct, as a ladder on Gemma 4's 262 k vocabulary, so
/// `ServeSchemaCheck.Bounds` is encoded from numbers and not from a guess. Opt-in (`TEST_RUNNER_MLXUI_MEASURE_COMPILE=1`); a
/// ladder stops at the first point over 2 s (that point is recorded). Prints `BOUND construct=… n=… ms=…`.
@Suite(.serialized, .enabled(if: grammarMeasurementDirectory != nil && ProcessInfo.processInfo.environment["MLXUI_MEASURE_COMPILE"] != nil))
struct ServeSchemaBoundMeasurementTests {
    private struct Ladder {
        let construct: String
        let points: [Int]
        let schema: (Int) -> String
    }

    private static func wrap(_ property: String) -> String {
        #"{"type":"object","properties":{"t":\#(property)},"required":["t"]}"#
    }

    private static let ladders: [Ladder] = [
        Ladder(construct: "string maxLength", points: [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 50]) { wrap(#"{"type":"string","maxLength":\#($0)}"#) },
        Ladder(construct: "string minLength", points: [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 50]) { wrap(#"{"type":"string","minLength":\#($0)}"#) },
        Ladder(construct: "string minLength+maxLength(2n)", points: [1, 2, 4, 8, 16, 32]) { wrap(#"{"type":"string","minLength":\#($0),"maxLength":\#($0 * 2)}"#) },
        Ladder(construct: "array maxItems (strings)", points: [1, 4, 16, 64, 256, 1024, 4096]) { wrap(#"{"type":"array","items":{"type":"string"},"maxItems":\#($0)}"#) },
        Ladder(construct: "array minItems (strings)", points: [1, 4, 16, 64, 256, 1024, 4096]) { wrap(#"{"type":"array","items":{"type":"string"},"minItems":\#($0)}"#) },
        Ladder(construct: "array maxItems (integers)", points: [1, 4, 16, 64, 256, 1024]) { wrap(#"{"type":"array","items":{"type":"integer"},"maxItems":\#($0)}"#) },
        Ladder(construct: "array maxItems (objects)", points: [1, 4, 16, 64, 256]) {
            wrap(#"{"type":"array","items":{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}},"required":["a"]},"maxItems":\#($0)}"#) },
        Ladder(construct: "object maxProperties", points: [1, 4, 16, 64, 256]) { wrap(#"{"type":"object","additionalProperties":{"type":"string"},"maxProperties":\#($0)}"#) },
        Ladder(construct: "object minProperties", points: [1, 4, 16, 64, 256]) { wrap(#"{"type":"object","additionalProperties":{"type":"string"},"minProperties":\#($0)}"#) },
        Ladder(construct: "integer maximum 10^n", points: [1, 3, 6, 9, 12, 15]) { wrap(#"{"type":"integer","minimum":0,"maximum":\#(String(repeating: "9", count: $0))}"#) },
        Ladder(construct: "number maximum 10^n", points: [1, 3, 6, 9, 12]) { wrap(#"{"type":"number","minimum":0,"maximum":\#(String(repeating: "9", count: $0))}"#) },
        Ladder(construct: "pattern [a-z]{1,n}", points: [1, 5, 10, 20, 50, 100, 200]) { wrap(#"{"type":"string","pattern":"^[a-z]{1,\#($0)}$"}"#) },
        Ladder(construct: "pattern [a-z]{n}", points: [1, 5, 10, 20, 50, 100, 200]) { wrap(#"{"type":"string","pattern":"^[a-z]{\#($0)}$"}"#) },
        Ladder(construct: "properties (strings)", points: [1, 10, 50, 100, 200]) {
            n in #"{"type":"object","properties":{"#
                + (0 ..< n).map { #""p\#($0)":{"type":"string"}"# }.joined(separator: ",") + #"},"required":["p0"]}"# },
        Ladder(construct: "nesting depth", points: [1, 3, 5, 10, 20]) {
            n in String(repeating: #"{"type":"object","properties":{"c":"#, count: n) + #"{"type":"string"}"#
                + String(repeating: #"},"required":["c"]}"#, count: n) },
        Ladder(construct: "enum values (strings)", points: [10, 100, 1000, 5000]) {
            n in wrap(#"{"enum":["# + (0 ..< n).map { #""value-\#($0)""# }.joined(separator: ",") + "]}") },
        Ladder(construct: "anyOf branches (objects)", points: [2, 8, 32, 128]) {
            n in wrap(#"{"anyOf":["# + (0 ..< n).map { #"{"type":"object","properties":{"k\#($0)":{"type":"string"}},"required":["k\#($0)"]}"# }.joined(separator: ",") + "]}") },
        Ladder(construct: "string format", points: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) {
            let formats = ["email", "date", "time", "date-time", "duration", "ipv4", "ipv6", "hostname", "uuid", "uri", "uri-reference", "uri-template", "json-pointer", "relative-json-pointer"]
            return wrap(#"{"type":"string","format":"\#(formats[$0])"}"#) },
    ]

    @Test func ladders() async throws {
        let directory = try #require(grammarMeasurementDirectory)
        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        let vocabulary = try ServeGrammarVocabulary(directory: directory, tokenizer: tokenizer, eosTokenID: 106)
        // a throwaway compile so the first rung doesn't pay one-time costs
        _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(Self.wrap(#"{"type":"string"}"#)))
        for ladder in Self.ladders {
            for point in ladder.points {
                let schema = ladder.schema(point)
                let start = ContinuousClock.now
                var failure = ""
                do { _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(schema)) } catch { failure = " error=\(error)" }
                let d = ContinuousClock.now - start
                let ms = Int((Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000)
                print("BOUND construct=\"\(ladder.construct)\" n=\(point) ms=\(ms)\(failure)")
                if ms > 2000 || !failure.isEmpty { break }
            }
        }
    }
}

/// WW4-1-2 / ruling AR, second pass: the exact boundaries the ladders bracketed, whether bounded fields add up inside one
/// schema (the ruling is about the *compile*, not about a field), and the constructs the ladders stopped before. Prints
/// `BOUND2 label=… ms=…`. Opt-in like the ladders.
@Suite(.serialized, .enabled(if: grammarMeasurementDirectory != nil && ProcessInfo.processInfo.environment["MLXUI_MEASURE_COMPILE"] != nil))
struct ServeSchemaBoundRefinementTests {
    private static func obj(_ props: [(String, String)], required: [String]? = nil) -> String {
        let body = props.map { #""\#($0.0)":\#($0.1)"# }.joined(separator: ",")
        let req = (required ?? props.map(\.0)).map { #""\#($0)""# }.joined(separator: ",")
        return #"{"type":"object","properties":{\#(body)},"required":[\#(req)]}"#
    }
    private static func s(_ extra: String) -> String { #"{"type":"string"\#(extra)}"# }

    private static let cases: [(String, String)] = {
        var c: [(String, String)] = []
        // exact boundaries
        for n in [0, 5] { c.append(("maxLength \(n)", obj([("t", s(#","maxLength":\#(n)"#))]))) }
        for n in [0] { c.append(("minLength \(n)", obj([("t", s(#","minLength":\#(n)"#))]))) }
        for (lo, hi) in [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (4, 5), (5, 5), (1, 1), (2, 2)] {
            c.append(("minLength \(lo) maxLength \(hi)", obj([("t", s(#","minLength":\#(lo),"maxLength":\#(hi)"#))])))
        }
        // do bounded fields add up in one schema?
        c.append(("4 props maxLength 1,2,3,4 (distinct)", obj((1 ... 4).map { ("p\($0)", s(#","maxLength":\#($0)"#)) })))
        c.append(("4 props maxLength 4 (identical)", obj((1 ... 4).map { ("p\($0)", s(#","maxLength":4"#)) })))
        c.append(("1 prop maxLength 4 + 1 free string", obj([("a", s(#","maxLength":4"#)), ("b", s(""))])))
        c.append(("array items maxLength 4, maxItems 100", obj([("t", #"{"type":"array","items":\#(s(#","maxLength":4"#)),"maxItems":100}"#)])))
        c.append(("2 props maxLength 2", obj((1 ... 2).map { ("p\($0)", s(#","maxLength":2"#)) })))
        c.append(("8 props maxLength 1", obj((1 ... 8).map { ("p\($0)", s(#","maxLength":1"#)) })))
        // pattern quantifier boundaries
        for n in [25, 30, 35, 40] { c.append(("pattern [a-z]{1,\(n)}", obj([("t", s(#","pattern":"^[a-z]{1,\#(n)}$""#))]))) }
        for n in [60, 70, 80] { c.append(("pattern [a-z]{\(n)}", obj([("t", s(#","pattern":"^[a-z]{\#(n)}$""#))]))) }
        c.append(("pattern 2x [a-z]{20}", obj([("a", s(#","pattern":"^[a-z]{20}$""#)), ("b", s(#","pattern":"^[0-9]{20}$""#))])))
        c.append(("pattern \\d{4}-\\d{2}-\\d{2}", obj([("d", s(#","pattern":"^\\d{4}-\\d{2}-\\d{2}$""#))])))
        c.append(("pattern ^[A-Z]{2}-[0-9]{4}$", obj([("d", s(#","pattern":"^[A-Z]{2}-[0-9]{4}$""#))])))
        c.append(("pattern with * +", obj([("d", s(#","pattern":"^[a-z]+@[a-z]+\\.[a-z]{2,6}$""#))])))
        // property counts
        for n in [100, 150, 175, 300, 500] {
            c.append(("properties \(n) strings", obj((0 ..< n).map { ("p\($0)", s("")) }, required: ["p0"])))
        }
        // enums
        for n in [7500, 10000, 20000] { c.append(("enum \(n) values", obj([("e", #"{"enum":[\#((0 ..< n).map { "\"value-\($0)\"" }.joined(separator: ","))]}"#)]))) }
        // counts the ladders stopped early on, and absurd ones
        for n in [4096, 100_000, 1_000_000_000] { c.append(("array maxItems \(n) integers", obj([("l", #"{"type":"array","items":{"type":"integer"},"maxItems":\#(n)}"#)]))) }
        for n in [100_000, 1_000_000_000] { c.append(("array maxItems \(n) strings", obj([("l", #"{"type":"array","items":{"type":"string"},"maxItems":\#(n)}"#)]))) }
        for n in [1_000_000] { c.append(("array minItems \(n) strings", obj([("l", #"{"type":"array","items":{"type":"string"},"minItems":\#(n)}"#)]))) }
        for n in [1024, 4096] {
            c.append(("object maxProperties \(n)", obj([("o", #"{"type":"object","additionalProperties":{"type":"string"},"maxProperties":\#(n)}"#)])))
            c.append(("object minProperties \(n)", obj([("o", #"{"type":"object","additionalProperties":{"type":"string"},"minProperties":\#(n)}"#)])))
        }
        return c
    }()

    @Test func refinements() async throws {
        let directory = try #require(grammarMeasurementDirectory)
        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        let vocabulary = try ServeGrammarVocabulary(directory: directory, tokenizer: tokenizer, eosTokenID: 106)
        _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(Self.obj([("t", Self.s(""))])))
        for (label, schema) in Self.cases {
            let start = ContinuousClock.now
            var failure = ""
            do { _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(schema)) } catch { failure = " error=\(String(describing: error).prefix(160))" }
            let d = ContinuousClock.now - start
            print("BOUND2 label=\"\(label)\" ms=\(Int((Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000)) bytes=\(schema.utf8.count)\(failure)")
        }
    }
}


/// The cases the second pass never reached (the host crashed on a deeply nested schema first — see `ServeSchemaDepthProbeTests`).
@Suite(.serialized, .enabled(if: grammarMeasurementDirectory != nil && ProcessInfo.processInfo.environment["MLXUI_MEASURE_COMPILE"] != nil))
struct ServeSchemaBoundTailTests {
    private static func obj(_ props: [(String, String)], required: [String]? = nil) -> String {
        let body = props.map { #""\#($0.0)":\#($0.1)"# }.joined(separator: ",")
        let req = (required ?? props.map(\.0)).map { #""\#($0)""# }.joined(separator: ",")
        return #"{"type":"object","properties":{\#(body)},"required":[\#(req)]}"#
    }
    private static func s(_ extra: String) -> String { #"{"type":"string"\#(extra)}"# }

    private static let cases: [(String, String)] = {
        var c: [(String, String)] = []
        for n in [512, 2048] {
            c.append(("anyOf \(n) branches", obj([("a", #"{"anyOf":[\#((0 ..< n).map { #"{"type":"object","properties":{"k\#($0)":{"type":"string"}},"required":["k\#($0)"]}"# }.joined(separator: ","))]}"#)])))
        }
        c.append(("recursive tree via $ref #", ##"{"type":"object","properties":{"name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#"}}},"required":["name"]}"##))
        c.append(("recursive tree via $defs", ##"{"$defs":{"node":{"type":"object","properties":{"v":{"type":"string"},"next":{"anyOf":[{"$ref":"#/$defs/node"},{"type":"null"}]}},"required":["v"]}},"$ref":"#/$defs/node"}"##))
        c.append(("prefixItems 32 strings", obj([("t", #"{"type":"array","prefixItems":[\#(Array(repeating: #"{"type":"string"}"#, count: 32).joined(separator: ","))],"items":false}"#)])))
        c.append(("allOf of 3 objects", obj([("t", #"{"allOf":[{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},{"type":"object","properties":{"b":{"type":"integer"}},"required":["b"]},{"type":"object","properties":{"c":{"type":"boolean"}},"required":["c"]}]}"#)])))
        c.append(("type [string,null]", obj([("t", #"{"type":["string","null"]}"#)])))
        // formats the ladder stopped before
        for format in ["json-pointer", "relative-json-pointer", "uri-template", "uri", "email"] {
            c.append(("format \(format)", obj([("f", s(#","format":"\#(format)""#))])))
        }
        // regex repetition applied to groups, and open-ended counts (the ladders measured only a repeated character class)
        c.append(("pattern (ab){30}", obj([("t", s(#","pattern":"^(ab){30}$""#))])))
        c.append(("pattern ([a-z]{3}-){10}", obj([("t", s(#","pattern":"^([a-z]{3}-){10}$""#))])))
        c.append(("pattern (a|b|c){20}", obj([("t", s(#","pattern":"^(a|b|c){20}$""#))])))
        c.append(("pattern [a-z]{40,}", obj([("t", s(#","pattern":"^[a-z]{40,}$""#))])))
        c.append(("pattern [a-z]{20,}", obj([("t", s(#","pattern":"^[a-z]{20,}$""#))])))
        // every construct at its bound in ONE schema, to see whether bounds add up
        c.append(("composite at the bounds", obj([
            ("a", s(#","maxLength":5"#)), ("b", s(#","pattern":"^[a-z]{1,35}$""#)), ("c", s(#","pattern":"^[a-z]{60}$""#)),
            ("d", #"{"enum":[\#((0 ..< 5000).map { "\"value-\($0)\"" }.joined(separator: ","))]}"#),
            ("e", #"{"type":"array","items":{"type":"string"},"maxItems":100000}"#), ("f", s(#","format":"email""#)),
        ] + (0 ..< 170).map { ("p\($0)", s("")) })))
        return c
    }()

    @Test func tail() async throws {
        let directory = try #require(grammarMeasurementDirectory)
        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        let vocabulary = try ServeGrammarVocabulary(directory: directory, tokenizer: tokenizer, eosTokenID: 106)
        _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(Self.obj([("t", Self.s(""))])))
        for (label, schema) in Self.cases {
            let start = ContinuousClock.now
            var failure = ""
            do { _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(schema)) } catch { failure = " error=\(String(describing: error).prefix(160))" }
            let d = ContinuousClock.now - start
            print("BOUND3 label=\"\(label)\" ms=\(Int((Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000)) bytes=\(schema.utf8.count)\(failure)")
        }
    }
}

/// One nesting depth per process (`MLXUI_NEST_DEPTH=n`, n × 2 JSON containers, plus the root): xgrammar's JSON copy recurses on the stack, and a
/// depth that overflows it kills the host, so the edge is found by running this once per depth.
@Suite(.serialized, .enabled(if: grammarMeasurementDirectory != nil && ProcessInfo.processInfo.environment["MLXUI_NEST_DEPTH"] != nil))
struct ServeSchemaDepthProbeTests {
    @Test func probe() async throws {
        let directory = try #require(grammarMeasurementDirectory)
        let depth = try #require(Int(ProcessInfo.processInfo.environment["MLXUI_NEST_DEPTH"] ?? ""))
        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        let vocabulary = try ServeGrammarVocabulary(directory: directory, tokenizer: tokenizer, eosTokenID: 106)
        let schema = String(repeating: #"{"type":"object","properties":{"c":"#, count: depth) + #"{"type":"string"}"#
            + String(repeating: #"},"required":["c"]}"#, count: depth)
        print("DEPTH n=\(depth) jsonContainers=\(depth * 2 + 1) starting")
        let start = ContinuousClock.now
        // On the compile thread (ruling AV): the edge is the one the shipping path has, not a pool thread's.
        _ = try await ServeGrammarCompileThread.shared.run { _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(schema)) }
        let d = ContinuousClock.now - start
        print("DEPTH n=\(depth) jsonContainers=\(depth * 2 + 1) ms=\(Int((Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000)) ok")
    }
}

/// Third pass (ruling AR): the boundaries the tail bracketed, and combinations — the composite of every bound took 4.5 s, so the
/// check also estimates the whole schema, and these rows test that estimate against real compiles. Prints `BOUND4`.
@Suite(.serialized, .enabled(if: grammarMeasurementDirectory != nil && ProcessInfo.processInfo.environment["MLXUI_MEASURE_COMPILE"] != nil))
struct ServeSchemaBoundCombinationTests {
    private static func obj(_ props: [(String, String)]) -> String {
        #"{"type":"object","properties":{\#(props.map { #""\#($0.0)":\#($0.1)"# }.joined(separator: ","))},"required":["\#(props[0].0)"]}"#
    }
    private static func s(_ extra: String) -> String { #"{"type":"string"\#(extra)}"# }
    private static func enumOf(_ n: Int) -> String { #"{"enum":[\#((0 ..< n).map { "\"value-\($0)\"" }.joined(separator: ","))]}"# }
    private static func anyOf(_ n: Int) -> String {
        #"{"anyOf":[\#((0 ..< n).map { #"{"type":"object","properties":{"k\#($0)":{"type":"string"}},"required":["k\#($0)"]}"# }.joined(separator: ","))]}"#
    }
    private static func props(_ n: Int, from: Int = 0) -> [(String, String)] { (from ..< from + n).map { ("p\($0)", s("")) } }

    private static let cases: [(String, String)] = [
        ("anyOf 256", obj([("a", anyOf(256))])),
        ("anyOf 400", obj([("a", anyOf(400))])),
        ("pattern [a-z]{50,}", obj([("t", s(#","pattern":"^[a-z]{50,}$""#))])),
        ("pattern [a-z]{60,}", obj([("t", s(#","pattern":"^[a-z]{60,}$""#))])),
        ("enum 5000 + 175 props", obj([("e", enumOf(5000))] + props(174, from: 1))),
        ("maxLength 5 + pattern {1,35}", obj([("a", s(#","maxLength":5"#)), ("b", s(#","pattern":"^[a-z]{1,35}$""#))])),
        ("pattern {1,35} + pattern {60}", obj([("a", s(#","pattern":"^[a-z]{1,35}$""#)), ("b", s(#","pattern":"^[0-9]{60}$""#))])),
        ("anyOf 128 + enum 2500", obj([("a", anyOf(128)), ("e", enumOf(2500))])),
        ("email + json-pointer + uri-reference", obj([("a", s(#","format":"email""#)), ("b", s(#","format":"json-pointer""#)), ("c", s(#","format":"uri-reference""#))])),
        ("maxLength 5 + enum 2500 + 90 props", obj([("a", s(#","maxLength":5"#)), ("e", enumOf(2500))] + props(88, from: 2))),
        ("maxLength 3 + maxLength 5 (two fields)", obj([("a", s(#","maxLength":3"#)), ("b", s(#","maxLength":5"#))])),
        ("composite of half of every bound", obj([
            ("a", s(#","maxLength":2"#)), ("b", s(#","pattern":"^[a-z]{1,17}$""#)), ("c", s(#","pattern":"^[a-z]{30}$""#)),
            ("d", enumOf(2500)), ("f", s(#","format":"email""#)), ("g", anyOf(64)),
        ] + props(85, from: 10))),
    ]

    @Test func combinations() async throws {
        let directory = try #require(grammarMeasurementDirectory)
        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        let vocabulary = try ServeGrammarVocabulary(directory: directory, tokenizer: tokenizer, eosTokenID: 106)
        _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(Self.obj([("t", Self.s(""))])))
        for (label, schema) in Self.cases {
            let start = ContinuousClock.now
            var failure = ""
            do { _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(schema)) } catch { failure = " error=\(String(describing: error).prefix(160))" }
            let d = ContinuousClock.now - start
            print("BOUND4 label=\"\(label)\" ms=\(Int((Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000)) bytes=\(schema.utf8.count)\(failure)")
        }
    }
}

/// Ruling AW: a self-referential schema compiles on the real Gemma 4 vocabulary without crashing and inside the 1 s budget.
@Suite(.serialized, .enabled(if: grammarMeasurementDirectory != nil))
struct ServeRecursiveRefMeasurementTests {
    @Test func recursiveRefsCompileInsideTheBudgetOnTheRealVocabulary() async throws {
        let directory = try #require(grammarMeasurementDirectory)
        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        let vocabulary = try ServeGrammarVocabulary(directory: directory, tokenizer: tokenizer, eosTokenID: 106)
        let schemas = [
            ("tree", ##"{"type":"object","properties":{"name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#"}}},"required":["name","children"]}"##),
            ("defs", ##"{"$defs":{"node":{"type":"object","properties":{"next":{"anyOf":[{"$ref":"#/$defs/node"},{"type":"null"}]}},"required":["next"]}},"$ref":"#/$defs/node"}"##),
        ]
        for (label, schema) in schemas {
            let start = ContinuousClock.now
            _ = try await ServeGrammarCompileThread.shared.run { _ = try ServeGrammarCache.compile(vocabulary: vocabulary, format: .jsonSchema(schema)) }
            let d = ContinuousClock.now - start
            let ms = Int((Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000)
            print("RECURSIVE-REF label=\(label) cold_ms=\(ms)")
            #expect(ms < 1000, "\(label) compiled in \(ms) ms")
        }
    }
}
