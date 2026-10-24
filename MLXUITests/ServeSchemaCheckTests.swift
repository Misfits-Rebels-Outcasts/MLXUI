import Foundation
import MLXGuidedGeneration
import MLXLMCommon
import Testing

@testable import MLXUI

// WW4-1-2 — ServeSchemaCheck and the structured-output 400s. Pure: no model, no server, no dialect parsing.

struct ServeSchemaCheckTests {
    private typealias Bounds = ServeSchemaCheck.Bounds

    /// The findings of a schema that must be refused (fails the test, with the outcome, if it isn't).
    private static func findings(_ schema: String, sourceLocation: SourceLocation = #_sourceLocation) -> [ServeSchemaCheck.Finding] {
        guard case .refused(let findings) = ServeSchemaCheck.check(schema) else {
            Issue.record("expected \(schema.prefix(120)) to be refused, got \(ServeSchemaCheck.check(schema))", sourceLocation: sourceLocation)
            return []
        }
        return findings
    }

    private static func isOK(_ schema: String) -> Bool { ServeSchemaCheck.check(schema) == .ok }

    private static func prop(_ name: String, _ body: String) -> String {
        #"{"type":"object","properties":{"\#(name)":\#(body)}}"#
    }

    // MARK: - what is fine

    @Test func anOrdinarySchemaIsAccepted() {
        #expect(Self.isOK(#"""
        {"type":"object","title":"Person","description":"x","properties":{"name":{"type":"string","description":"full name"},
         "age":{"type":"integer","minimum":0,"maximum":150},"email":{"type":"string","format":"email"},
         "tags":{"type":"array","items":{"type":"string"},"maxItems":10,"minItems":1},
         "kind":{"enum":["a","b"]},"note":{"anyOf":[{"type":"string"},{"type":"null"}]}},
         "required":["name","age"],"additionalProperties":false,"$schema":"https://json-schema.org/draft/2020-12/schema"}
        """#))
        #expect(Self.isOK("{}"))
        #expect(Self.isOK("true"))
    }

    @Test func aPropertyNamedLikeAnIgnoredKeywordIsNotAKeyword() {
        #expect(Self.isOK(Self.prop("not", #"{"type":"string"}"#)))
        #expect(Self.isOK(Self.prop("multipleOf", #"{"type":"integer"}"#)))
        #expect(Self.isOK(#"{"type":"object","properties":{"if":{"type":"string"},"then":{"type":"string"}},"required":["if"]}"#))
    }

    @Test func enumAndConstValuesAreDataNotSchemas() {
        #expect(Self.isOK(#"{"enum":[{"multipleOf":3,"not":1}],"default":{"uniqueItems":true},"examples":[{"contains":1}]}"#))
        #expect(Self.isOK(#"{"const":{"if":1}}"#))
    }

    @Test func boundedConstructsMeasuredFlatAreNotBounded() {
        #expect(Self.isOK(Self.prop("l", #"{"type":"array","items":{"type":"string"},"maxItems":1000000000,"minItems":1000000}"#)))
        #expect(Self.isOK(Self.prop("o", #"{"type":"object","additionalProperties":{"type":"string"},"maxProperties":4096,"minProperties":4096}"#)))
        #expect(Self.isOK(Self.prop("n", #"{"type":"integer","minimum":0,"maximum":999999999999999}"#)))
    }

    // MARK: - keywords xgrammar silently ignores

    @Test(arguments: ["not", "if", "then", "else", "dependentRequired", "dependentSchemas", "multipleOf", "uniqueItems", "contains",
                      "minContains", "maxContains", "dependencies", "additionalItems", "contentEncoding", "contentMediaType",
                      "contentSchema", "$anchor", "$dynamicRef", "$recursiveRef"])
    func everyIgnoredKeywordIsNamedWithItsPointer(keyword: String) {
        let found = Self.findings(Self.prop("qty", #"{"type":"integer","\#(keyword)":1}"#))
        #expect(found.count == 1)
        #expect(found.first?.pointer == "/properties/qty/\(keyword)")
        #expect(found.first?.keyword == keyword)
        #expect(found.first?.kind == .ignoredKeyword)
    }

    @Test func theIgnoredKeywordListMatchesTheVendoredSource() throws {
        // The list is read from `json_schema_converter.cc` (xgrammar 0.1.30). If the package moves, re-read the source: this pins the list
        // so a bump is a decision, not a drift.
        #expect(ServeSchemaCheck.warnedAndIgnored == ["not", "if", "then", "else", "dependentRequired", "dependentSchemas",
                                                       "multipleOf", "uniqueItems", "contains", "minContains", "maxContains"])
        #expect(ServeSchemaCheck.neverRead == ["dependencies", "additionalItems", "contentEncoding", "contentMediaType",
                                               "contentSchema", "$anchor", "$dynamicRef", "$recursiveRef"])
        let source = Self.vendoredConverterSource()
        guard let source else { return }                       // not readable from a sandboxed test host: the pins above still hold
        for keyword in ServeSchemaCheck.warnedAndIgnored.subtracting(["if", "then", "else", "not", "dependentRequired", "dependentSchemas"]) {
            #expect(source.contains("\"\(keyword)\""), "\(keyword) no longer appears in the vendored converter")
        }
        for keyword in ServeSchemaCheck.neverRead {
            #expect(!source.contains("\"\(keyword)\""), "the converter now reads \(keyword): remove it from `neverRead`")
        }
    }

    @Test func ignoredKeywordsAreFoundInEveryPlaceASchemaNests() {
        let schema = #"""
        {"$defs":{"d":{"type":"object","multipleOf":2}},
         "type":"object",
         "properties":{"a/b":{"type":"array","items":{"type":"integer","multipleOf":3},"uniqueItems":true},
                       "c~d":{"anyOf":[{"type":"string"},{"type":"integer","multipleOf":4}]},
                       "e":{"oneOf":[{"not":{"type":"null"}}]},
                       "f":{"allOf":[{"type":"integer","multipleOf":5}]},
                       "g":{"type":"array","prefixItems":[{"type":"integer","multipleOf":6}]},
                       "h":{"type":"object","additionalProperties":{"type":"integer","multipleOf":7}}}}
        """#
        let pointers = Set(Self.findings(schema).map(\.pointer))
        #expect(pointers == [
            "/$defs/d/multipleOf",
            "/properties/a~1b/items/multipleOf", "/properties/a~1b/uniqueItems",
            "/properties/c~0d/anyOf/1/multipleOf",
            "/properties/e/oneOf/0/not",
            "/properties/f/allOf/0/multipleOf",
            "/properties/g/prefixItems/0/multipleOf",
            "/properties/h/additionalProperties/multipleOf",
        ])
    }

    @Test func everyFindingIsReportedNotJustTheFirst() {
        let found = Self.findings(#"{"type":"object","properties":{"a":{"type":"integer","multipleOf":2},"b":{"not":{}}},"if":{}}"#)
        #expect(found.map(\.pointer) == ["/if", "/properties/a/multipleOf", "/properties/b/not"], "deterministic order: keyword order within a node, then members sorted")
    }

    // MARK: - $ref, pattern, format

    @Test func onlyLocalReferencesAreAccepted() {
        #expect(Self.isOK(##"{"$defs":{"n":{"type":"string"}},"type":"object","properties":{"a":{"$ref":"#/$defs/n"},"b":{"$ref":"#"}}}"##))
        for ref in ["http://example.com/s.json", "other.json#/x", "#foo", "urn:x"] {
            let found = Self.findings(Self.prop("a", #"{"$ref":"\#(ref)"}"#))
            #expect(found.first?.kind == .unresolvableReference(ref), "\(ref)")
            #expect(found.first?.pointer == "/properties/a/$ref")
        }
    }

    @Test func minLengthMaxLengthAndFormatBesidePatternAreRefused() {
        let found = Self.findings(Self.prop("z", #"{"type":"string","pattern":"^[A-Z]+$","minLength":1,"maxLength":3,"format":"email"}"#))
        #expect(Set(found.map(\.pointer)) == ["/properties/z/minLength", "/properties/z/maxLength", "/properties/z/format"])
        #expect(found.allSatisfy { $0.kind == .ignoredBesidePattern })
    }

    @Test func formatsAreCheckedOnStringsOnly() {
        for format in ServeSchemaCheck.supportedFormats {
            #expect(Self.isOK(Self.prop("f", #"{"type":"string","format":"\#(format)"}"#)), "\(format)")
        }
        #expect(Self.findings(Self.prop("f", #"{"type":"string","format":"int32"}"#)).first?.kind == .unsupportedFormat("int32"))
        #expect(Self.isOK(Self.prop("f", #"{"type":"integer","format":"int32"}"#)), "xgrammar reads `format` on strings only")
        #expect(Self.isOK(Self.prop("f", #"{"type":["string","null"],"format":"uuid"}"#)))
        #expect(Self.findings(Self.prop("f", #"{"type":["string","null"],"format":"made-up"}"#)).first?.kind == .unsupportedFormat("made-up"))
    }

    @Test func aMeasuredSlowFormatIsRefusedOutright() {
        let found = Self.findings(Self.prop("f", #"{"type":"string","format":"uri-template"}"#))
        guard case .refusedOutright(let reason)? = found.first?.kind else { Issue.record("expected refusedOutright"); return }
        #expect(reason.contains("uri-template"))
    }

    // MARK: - the bounds, both sides of each

    @Test func maxLengthIsAllowedUpToItsBoundAndRefusedAbove() {
        #expect(Self.isOK(Self.prop("t", #"{"type":"string","maxLength":\#(Bounds.maxLength)}"#)))
        #expect(Self.isOK(Self.prop("t", #"{"type":"string","maxLength":1}"#)))
        let found = Self.findings(Self.prop("t", #"{"type":"string","maxLength":\#(Bounds.maxLength + 1)}"#))
        #expect(found.map(\.pointer) == ["/properties/t/maxLength"])
        #expect(found.first?.kind == .overBound(value: Bounds.maxLength + 1, allowed: Bounds.maxLength))
    }

    @Test func minLengthNeedsAMaxLengthWithinTheBound() {
        #expect(Self.isOK(Self.prop("t", #"{"type":"string","minLength":0}"#)), "minLength 0 is free")
        let alone = Self.findings(Self.prop("t", #"{"type":"string","minLength":1}"#))
        guard case .refusedOutright? = alone.first?.kind else { Issue.record("a bare minLength must be refused outright"); return }
        #expect(alone.first?.pointer == "/properties/t/minLength")
        #expect(Self.isOK(Self.prop("t", #"{"type":"string","minLength":1,"maxLength":\#(Bounds.maxLength)}"#)))
        #expect(Self.isOK(Self.prop("t", #"{"type":"string","minLength":\#(Bounds.maxLength),"maxLength":\#(Bounds.maxLength)}"#)))
        // beside a too-large maxLength only the maxLength is reported (one finding per cause)
        let both = Self.findings(Self.prop("t", #"{"type":"string","minLength":2,"maxLength":\#(Bounds.maxLength + 1)}"#))
        #expect(both.map(\.pointer) == ["/properties/t/maxLength"])
    }

    @Test func regexQuantifiersAreAllowedUpToTheirBounds() {
        func pattern(_ p: String) -> String { Self.prop("t", #"{"type":"string","pattern":"\#(p)"}"#) }
        #expect(Self.isOK(pattern("^[a-z]{1,\(Bounds.patternRangeQuantifier)}$")))
        #expect(Self.isOK(pattern("^[a-z]{\(Bounds.patternExactQuantifier)}$")))
        #expect(Self.isOK(pattern("^[A-Z]{2}-[0-9]{4}$")))
        #expect(Self.isOK(pattern("^[a-z]+@[a-z]+\\\\.[a-z]{2,6}$")))
        let range = Self.findings(pattern("^[a-z]{1,\(Bounds.patternRangeQuantifier + 1)}$"))
        #expect(range.first?.kind == .overBound(value: Bounds.patternRangeQuantifier + 1, allowed: Bounds.patternRangeQuantifier))
        #expect(range.first?.pointer == "/properties/t/pattern")
        let exact = Self.findings(pattern("^[a-z]{\(Bounds.patternExactQuantifier + 1)}$"))
        #expect(exact.first?.kind == .overBound(value: Bounds.patternExactQuantifier + 1, allowed: Bounds.patternExactQuantifier))
        #expect(Self.isOK(pattern("^[a-z]{\(Bounds.patternOpenQuantifier),}$")))
        let open = Self.findings(pattern("^[a-z]{\(Bounds.patternOpenQuantifier + 1),}$"))
        #expect(open.first?.kind == .overBound(value: Bounds.patternOpenQuantifier + 1, allowed: Bounds.patternOpenQuantifier))
    }

    @Test func theQuantifierReaderIgnoresEscapedBracesAndOtherBraces() {
        #expect(ServeSchemaCheck.quantifiers(in: #"^\{100\}$"#).isEmpty)
        #expect(ServeSchemaCheck.quantifiers(in: "a{3}b{2,5}c{4,}") == [
            .init(weight: 3, form: .exact), .init(weight: 5, form: .range), .init(weight: 4, form: .open)])
        #expect(ServeSchemaCheck.quantifiers(in: "{x}{,3}{3,1x}").isEmpty)
        #expect(ServeSchemaCheck.quantifiers(in: "a{2,2}") == [.init(weight: 2, form: .exact)])
    }

    @Test func propertyCountIsBounded() {
        func schema(_ count: Int) -> String {
            #"{"type":"object","properties":{"# + (0 ..< count).map { #""p\#($0)":{"type":"string"}"# }.joined(separator: ",") + "}}"
        }
        #expect(Self.isOK(schema(Bounds.properties)))
        let found = Self.findings(schema(Bounds.properties + 1))
        #expect(found.first?.kind == .overBound(value: Bounds.properties + 1, allowed: Bounds.properties))
        #expect(found.first?.pointer == "/properties")
    }

    @Test func enumSizeIsBounded() {
        func schema(_ count: Int) -> String { #"{"enum":["# + (0 ..< count).map { "\($0)" }.joined(separator: ",") + "]}" }
        #expect(Self.isOK(schema(Bounds.enumValues)))
        #expect(Self.findings(schema(Bounds.enumValues + 1)).first?.kind == .overBound(value: Bounds.enumValues + 1, allowed: Bounds.enumValues))
    }

    @Test func branchCountIsBounded() {
        func schema(_ keyword: String, _ count: Int) -> String {
            #"{"\#(keyword)":["# + Array(repeating: #"{"type":"string"}"#, count: count).joined(separator: ",") + "]}"
        }
        for keyword in ["anyOf", "oneOf"] {
            #expect(Self.isOK(schema(keyword, Bounds.branches)))
            #expect(Self.findings(schema(keyword, Bounds.branches + 1)).first?.kind == .overBound(value: Bounds.branches + 1, allowed: Bounds.branches))
        }
    }

    @Test func nestingDepthIsBoundedBecauseXGrammarOverflowsItsStack() {
        // Each `items` level is one JSON container; the innermost schema's depth is `levels + 1`.
        func nested(_ levels: Int) -> String {
            String(repeating: #"{"type":"array","items":"#, count: levels) + #"{"type":"string"}"# + String(repeating: "}", count: levels)
        }
        #expect(Self.isOK(nested(Bounds.maxJSONDepth - 1)), "depth \(Bounds.maxJSONDepth) is allowed")
        let found = Self.findings(nested(Bounds.maxJSONDepth))
        #expect(found.count == 1, "reported once, not once per level")
        #expect(found.first?.kind == .tooDeep(allowed: Bounds.maxJSONDepth))
    }

    @Test func aHostileNestingDoesNotRecurseUnbounded() {
        let evil = String(repeating: "[", count: 100_000) + String(repeating: "]", count: 100_000)
        if case .ok = ServeSchemaCheck.check(evil) { Issue.record("must not be accepted") }
        let deepObjects = String(repeating: #"{"items":"#, count: 5000) + "{}" + String(repeating: "}", count: 5000)
        _ = ServeSchemaCheck.check(deepObjects)             // must return (Foundation's parser has its own depth limit)
    }

    @Test func theSchemaSizeIsBoundedBeforeItIsParsed() {
        let big = #"{"description":""# + String(repeating: "x", count: Bounds.schemaBytes) + #""}"#
        let found = Self.findings(big)
        #expect(found.first?.kind == .tooLarge(bytes: big.utf8.count, allowed: Bounds.schemaBytes))
        #expect(Self.isOK(#"{"description":""# + String(repeating: "x", count: Bounds.schemaBytes - 100) + #""}"#))
    }

    @Test func notJSONAndNotASchemaAreMalformed() {
        for text in ["", "not json", "{", "[1,2]", "42", "\"str\"", "null"] {
            guard case .malformed = ServeSchemaCheck.check(text) else { Issue.record("\(text) should be malformed"); continue }
        }
    }

    @Test func theBoundsArePinnedToTheirMeasurements() {
        // Gemma 4 e4b (262,144 tokens), cold compile, Debug build, 2026-10-24 (journal 2026-572). Changing a number here means
        // re-running ServeSchemaBound*MeasurementTests (TEST_RUNNER_MLXUI_MEASURE_COMPILE=1) and saying so in the journal.
        #expect(Bounds.maxLength == 5)                  // 872 ms at 5, 1033 ms at 6
        #expect(Bounds.patternRangeQuantifier == 35)    // 971 ms at {1,35}, 1163 ms at {1,40}
        #expect(Bounds.patternExactQuantifier == 60)    // 858 ms at {60}, 1016 ms at {70}
        #expect(Bounds.patternOpenQuantifier == 50)     // 956 ms at {50,}, 1091 ms at {60,}
        #expect(Bounds.properties == 175)               // 946 ms at 175, 2041 ms at 300
        #expect(Bounds.enumValues == 5000)              // 773 ms at 5000, 1143 ms at 7500
        #expect(Bounds.branches == 400)                 // 884 ms at 400, 1012 ms at 512
        #expect(Bounds.maxJSONDepth == 60)              // compiled at 81 containers, SIGBUS at 101
        #expect(Bounds.schemaBytes == 80_000)
        #expect(Bounds.estimatedCompileMs == 2000)      // slowest accepted measured schema: 1830 ms
        #expect(Bounds.refusedFormats == ["uri-template"])    // 2503 ms
    }

    // MARK: - the whole-schema estimate (the bounds do not add up to a safe whole)

    private static func object(_ properties: [(String, String)]) -> String {
        #"{"type":"object","properties":{\#(properties.map { #""\#($0.0)":\#($0.1)"# }.joined(separator: ","))}}"#
    }
    private static func string(_ extra: String = "") -> String { #"{"type":"string"\#(extra)}"# }
    private static func enumOf(_ count: Int) -> String { #"{"enum":[\#((0 ..< count).map { "\"value-\($0)\"" }.joined(separator: ","))]}"# }
    private static func anyOf(_ count: Int) -> String {
        #"{"anyOf":[\#((0 ..< count).map { #"{"type":"object","properties":{"k\#($0)":{"type":"string"}},"required":["k\#($0)"]}"# }.joined(separator: ","))]}"#
    }
    private static func strings(_ count: Int, from start: Int = 0) -> [(String, String)] { (start ..< start + count).map { ("p\($0)", string()) } }

    /// The schemas measured in `ServeSchemaBound*MeasurementTests`, with the measured cold compile (ms).
    private static let measured: [(label: String, schema: String, ms: Double)] = [
        ("enum 5000 + 175 properties", object([("e", enumOf(5000))] + strings(174, from: 1)), 1830),
        ("maxLength 5 + pattern {1,35}", object([("a", string(#","maxLength":5"#)), ("b", string(#","pattern":"^[a-z]{1,35}$""#))]), 1501),
        ("pattern {1,35} + pattern {60}", object([("a", string(#","pattern":"^[a-z]{1,35}$""#)), ("b", string(#","pattern":"^[0-9]{60}$""#))]), 1092),
        ("anyOf 128 + enum 2500", object([("a", anyOf(128)), ("e", enumOf(2500))]), 1009),
        ("email + json-pointer + uri-reference", object([("a", string(#","format":"email""#)), ("b", string(#","format":"json-pointer""#)), ("c", string(#","format":"uri-reference""#))]), 1084),
        ("maxLength 5 + enum 2500 + 90 properties", object([("a", string(#","maxLength":5"#)), ("e", enumOf(2500))] + strings(88, from: 2)), 1664),
        ("half of every bound", object([("a", string(#","maxLength":2"#)), ("b", string(#","pattern":"^[a-z]{1,17}$""#)), ("c", string(#","pattern":"^[a-z]{30}$""#)),
                                       ("d", enumOf(2500)), ("f", string(#","format":"email""#)), ("g", anyOf(64))] + strings(85, from: 10)), 2676),
        ("every bound at once (176 properties)", object([("a", string(#","maxLength":5"#)), ("b", string(#","pattern":"^[a-z]{1,35}$""#)), ("c", string(#","pattern":"^[a-z]{60}$""#)),
                                       ("d", enumOf(5000)), ("e", #"{"type":"array","items":{"type":"string"},"maxItems":100000}"#), ("f", string(#","format":"email""#))]
                                       + strings(170, from: 10)), 4492),
    ]

    @Test func theEstimateNeverUnderstatesAMeasuredSchemaByMuch() {
        for row in Self.measured {
            guard let estimate = ServeSchemaCheck.estimatedCompileMs(row.schema) else { Issue.record("\(row.label) did not parse"); continue }
            #expect(estimate >= row.ms * 0.9, "\(row.label): estimated \(Int(estimate)) ms against a measured \(Int(row.ms)) ms")
        }
    }

    @Test func aSchemaTheEstimateAcceptsWasMeasuredUnderTheBudgetPlusMargin() {
        for row in Self.measured where ServeSchemaCheck.estimatedCompileMs(row.schema).map({ $0 <= Double(Bounds.estimatedCompileMs) }) == true {
            #expect(row.ms <= Double(Bounds.estimatedCompileMs) * 1.1, "\(row.label) was accepted but took \(Int(row.ms)) ms")
        }
    }

    @Test func theCompositeOfEveryBoundIsRefusedAsAWholeAndNamesItsLargestParts() {
        // Every construct within its own bound (100 properties, under 175), and together ~4 s: refused as one finding.
        let atBounds = Self.object([("a", Self.string(#","maxLength":5"#)), ("b", Self.string(#","pattern":"^[a-z]{1,35}$""#)),
                                    ("c", Self.string(#","pattern":"^[a-z]{60}$""#)), ("d", Self.enumOf(5000)), ("f", Self.string(#","format":"email""#))]
                                   + Self.strings(100, from: 10))
        let found = Self.findings(atBounds)
        #expect(found.count == 1, "only the whole-schema finding: no single construct is over its bound")
        guard case .tooExpensive(let estimate, let allowed, let largest)? = found.first?.kind else { Issue.record("expected tooExpensive, got \(found)"); return }
        #expect(estimate > allowed)
        #expect(allowed == Bounds.estimatedCompileMs)
        #expect(largest.count == 3)
        #expect(largest.allSatisfy { $0.contains("ms") })
        #expect(found.first?.pointer == "")
        #expect(found.first?.sentence(param: "text.format.schema").contains("as a whole") == true)
    }

    @Test func eachConstructAloneAtItsBoundIsStillAccepted() {
        #expect(Self.isOK(Self.object([("a", Self.string(#","maxLength":\#(Bounds.maxLength)"#))])))
        #expect(Self.isOK(Self.object([("a", Self.string(#","pattern":"^[a-z]{1,\#(Bounds.patternRangeQuantifier)}$""#))])))
        #expect(Self.isOK(Self.object([("a", Self.string(#","pattern":"^[a-z]{\#(Bounds.patternExactQuantifier)}$""#))])))
        #expect(Self.isOK(Self.object([("a", Self.string(#","pattern":"^[a-z]{\#(Bounds.patternOpenQuantifier),}$""#))])))
        #expect(Self.isOK(Self.object([("a", Self.enumOf(Bounds.enumValues))])))
        #expect(Self.isOK(Self.object([("a", Self.anyOf(Bounds.branches))])))
        #expect(Self.isOK(Self.object(Self.strings(Bounds.properties))))
        #expect(Self.isOK(Self.object([("a", Self.string(#","format":"email""#))])))
    }

    // MARK: - the sentences

    @Test func aSentenceNamesTheParamThePointerTheKeywordAndTheBound() {
        let found = Self.findings(Self.prop("title", #"{"type":"string","maxLength":50}"#))
        let sentence = found[0].sentence(param: "response_format.json_schema.schema")
        #expect(sentence.contains("response_format.json_schema.schema/properties/title/maxLength"))
        #expect(sentence.contains("`maxLength` is 50"))
        #expect(sentence.contains("maximum of \(Bounds.maxLength)"))
    }

    // MARK: - helpers

    /// The vendored converter source, when the test host can read it (it can't in the sandbox).
    private static func vendoredConverterSource() -> String? {
        let root = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Developer/Xcode/DerivedData")
        guard let derived = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return nil }
        for directory in derived where directory.lastPathComponent.hasPrefix("MLXUI-") {
            let file = directory.appendingPathComponent("SourcePackages/checkouts/mlx-swift-lm/Libraries/MLXCXGrammar/xgrammar/cpp/json_schema_converter.cc")
            if let text = try? String(contentsOf: file, encoding: .utf8) { return text }
        }
        return nil
    }
}
