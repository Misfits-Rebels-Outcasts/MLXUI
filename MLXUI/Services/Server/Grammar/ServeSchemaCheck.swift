import Foundation

/// WW4-1-2 — what a JSON Schema may ask of this server, decided **before** it reaches xgrammar (design note
/// `RSI/notes/claude-code/wwdc26/ww4-design.md`; owner rulings AO, AR — 2026-10-24).
///
/// xgrammar fails in three ways a server must close itself:
/// 1. **It silently drops keywords** (`WarnUnsupportedKeywords` logs a warning nobody reads and carries on): a client that sends
///    `multipleOf: 5` gets a reply the server would call schema-valid and that is not. Ruling AR: *refused, never silently
///    dropped* — every such keyword is named, with its JSON pointer.
/// 2. **A few constructs compile for a long time** and a compile can't be cancelled (it runs before the response head, so the
///    client waits with nothing on the wire). Ruling AR: a construct is allowed only up to the bound at which its cold compile
///    stays ≤ 1 s on this Mac (Gemma 4's 262 k vocabulary, the largest installed), and refused beyond it, naming the
///    construct, its pointer and the allowed bound. The table is `Bounds`; each number is a measurement
///    (`ServeSchemaBoundMeasurementTests`, `…RefinementTests`, `…TailTests`, journal 2026-572), not a guess.
/// 3. **A deeply nested schema overflows xgrammar's stack and kills the process** (found by that measurement: the test host
///    died with SIGBUS in `picojson::object_with_ordered_keys`'s copy constructor at 150 JSON levels). That is a crash vector,
///    not a slowness; `Bounds.maxJSONDepth` closes it.
///
/// Pure: text in, findings out. A walk that follows the schema's own structure (`properties`, `items`, `anyOf`, `$defs`, …),
/// not a scan for key names, so a *property called* `not` is not a keyword.
nonisolated enum ServeSchemaCheck {

    // MARK: - The table

    /// Every number here is a measurement on Gemma 4's tokenizer (262,144 tokens), cold compile, this Mac, Debug build.
    /// "ok at" is the largest value measured ≤ 1000 ms, "over at" the smallest measured above it.
    enum Bounds {
        /// string `maxLength` — ok at 5 (872 ms), over at 6 (1033 ms); the cost is ~170 ms per character, and a free string
        /// costs ~480 ms. Fields do not add: four distinct bounded strings (1…4) took 773 ms against 678 ms for the largest.
        static let maxLength = 5
        /// string `minLength` **alone** is over 1 s at 1 (1473 ms) and grows, so it is refused unless a `maxLength` ≤ `maxLength`
        /// sits beside it; with one, every measured pair up to (5, 5) was ≤ 543 ms (cheaper than the `maxLength` alone).
        /// `minLength: 0` is free (50 ms) and always allowed.
        static let minLengthNeedsMaxLength = true
        /// regex quantifier `{m,n}` — ok at n = 35 (971 ms), over at 40 (1163 ms).
        static let patternRangeQuantifier = 35
        /// regex quantifier `{n}` — ok at 60 (858 ms), over at 70 (1016 ms).
        static let patternExactQuantifier = 60
        /// regex quantifier `{m,}` — ok at 50 (956 ms), over at 60 (1091 ms).
        static let patternOpenQuantifier = 50
        /// properties declared in one object — ok at 175 (946 ms), over at 300 (2041 ms).
        static let properties = 175
        /// `enum` values — ok at 5000 (773 ms), over at 7500 (1143 ms).
        static let enumValues = 5000
        /// the schema text, in UTF-8 bytes. The 5000-value enum measured is ~64 KB; 7500 values is 96 KB (over).
        static let schemaBytes = 80_000
        /// `anyOf` / `oneOf` branches — ok at 400 (884 ms), over at 512 (1012 ms).
        static let branches = 400
        /// JSON nesting depth (`{`/`[` containers). The test host compiled a schema 81 containers deep (551 ms) and died with
        /// SIGBUS (stack overflow in xgrammar's recursive copy) at 101, in a Debug build on a cooperative-pool thread, the same
        /// stack the server compiles on. 60 leaves a margin under the deepest one measured to work.
        static let maxJSONDepth = 60
        /// The whole schema's estimated cold compile (`CostModel`), in milliseconds. Bounds are per construct and **do not
        /// add up to a safe whole**: every construct at its own bound, in one schema, took 4,492 ms, and half of each took 2,676 ms.
        /// The worst measured schema the estimate accepts took 1,830 ms (enum 5000 + 175 properties).
        static let estimatedCompileMs = 2000

        /// Measured **flat** (≤ 0.75 s up to 10⁹), so unbounded: `maxItems`, `minItems`, `minProperties`, `maxProperties`
        /// (strings: 574–645 ms at 10⁶–10⁹; integers 250 ms), integer/number `minimum`/`maximum` (≤ 11 ms at 10¹²).
        static let flat = ["maxItems", "minItems", "minProperties", "maxProperties", "minimum", "maximum"]
        /// `format`s that were measured and are over 1 s: refused outright. (`uri-template`: 2503 ms.)
        static let refusedFormats: Set<String> = ["uri-template"]
    }

    /// Keywords xgrammar reads and **warns about and ignores** (vendored v0.1.30, `json_schema_converter.cc`:
    /// `WarnUnsupportedKeywords` at the schema entry, for integers, numbers and arrays).
    static let warnedAndIgnored: Set<String> = [
        "not", "if", "then", "else", "dependentRequired", "dependentSchemas",
        "multipleOf", "uniqueItems", "contains", "minContains", "maxContains",
    ]
    /// Keywords the converter **never reads at all** (no occurrence in the source): they cannot constrain anything. Annotations
    /// (`title`, `description`, `default`, `examples`, `$comment`, `$schema`, `$id`, `readOnly`, `deprecated`) are allowed: they
    /// are not meant to.
    static let neverRead: Set<String> = [
        "dependencies", "additionalItems", "contentEncoding", "contentMediaType", "contentSchema",
        "$anchor", "$dynamicRef", "$recursiveRef",
    ]
    /// `format` values xgrammar turns into a grammar; any other `format` on a string is silently a plain string.
    static let supportedFormats: Set<String> = [
        "email", "date", "time", "date-time", "duration", "ipv4", "ipv6", "hostname", "uuid", "uri", "uri-reference",
        "json-pointer", "relative-json-pointer",
    ]

    // MARK: - Result

    enum Outcome: Equatable, Sendable {
        case ok
        /// Not JSON, or not a schema (the root must be an object or a boolean).
        case malformed(String)
        case refused([Finding])
    }

    struct Finding: Equatable, Sendable {
        enum Kind: Equatable, Sendable {
            /// xgrammar would drop the keyword.
            case ignoredKeyword
            /// `minLength` / `maxLength` / `format` beside `pattern`: xgrammar keeps the pattern and drops these.
            case ignoredBesidePattern
            case unsupportedFormat(String)
            /// A `$ref` that isn't `#` or `#/…`: xgrammar turns it into "any value".
            case unresolvableReference(String)
            case overBound(value: Int, allowed: Int)
            /// Refused at every value (the reason says what to do instead).
            case refusedOutright(String)
            case tooDeep(allowed: Int)
            case tooLarge(bytes: Int, allowed: Int)
            /// Every construct is within its own bound but together they would take too long (`CostModel`).
            case tooExpensive(estimateMs: Int, allowedMs: Int, largest: [String])
        }

        /// JSON pointer to the keyword (`/properties/qty/multipleOf`); `""` for the whole schema.
        let pointer: String
        let keyword: String
        let kind: Kind

        func sentence(param: String) -> String {
            let at = "\(param)\(pointer)"
            switch kind {
            case .ignoredKeyword:
                return "\(at): `\(keyword)` can't be enforced by this server, so the schema is refused rather than silently weakened."
            case .ignoredBesidePattern:
                return "\(at): `\(keyword)` is ignored when `pattern` is present; remove it or put the rule in the pattern."
            case .unsupportedFormat(let format):
                return "\(at): format \"\(format)\" isn't supported; supported formats are "
                    + ServeSchemaCheck.supportedFormats.sorted().joined(separator: ", ") + "."
            case .unresolvableReference(let ref):
                return "\(at): `$ref` \"\(ref)\" must be \"#\" or start with \"#/\" (remote and anchor references aren't supported)."
            case .overBound(let value, let allowed):
                return "\(at): `\(keyword)` is \(value), above the allowed maximum of \(allowed) "
                    + "(a larger value makes the schema take more than a second to prepare)."
            case .refusedOutright(let reason):
                return "\(at): `\(keyword)` isn't supported: \(reason)"
            case .tooDeep(let allowed):
                return "\(at): the schema is nested more than \(allowed) levels deep, which this server refuses."
            case .tooLarge(let bytes, let allowed):
                return "\(at): the schema is \(bytes) bytes, above the allowed maximum of \(allowed)."
            case .tooExpensive(let estimate, let allowed, let largest):
                return "\(at): the schema as a whole would take about \(estimate) ms to prepare, above the allowed \(allowed) ms "
                    + "(largest parts: \(largest.joined(separator: ", "))). Simplify it."
            }
        }
    }

    // MARK: - Check

    static func check(_ text: String) -> Outcome {
        let bytes = text.utf8.count
        if bytes > Bounds.schemaBytes {
            return .refused([Finding(pointer: "", keyword: "(schema)", kind: .tooLarge(bytes: bytes, allowed: Bounds.schemaBytes))])
        }
        guard let data = text.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else {
            return .malformed("it is not valid JSON")
        }
        guard root is [String: Any] || root is Bool else {
            return .malformed("the root of a JSON Schema must be an object (or true/false)")
        }
        var walker = Walker()
        walker.visit(root, pointer: "", depth: 1)
        if !walker.findings.isEmpty { return .refused(walker.findings) }
        let estimate = walker.cost.total
        if estimate > Double(Bounds.estimatedCompileMs) {
            return .refused([Finding(pointer: "", keyword: "(schema)",
                                     kind: .tooExpensive(estimateMs: Int(estimate.rounded()), allowedMs: Bounds.estimatedCompileMs,
                                                         largest: walker.cost.largest(3)))])
        }
        return .ok

    }

    /// The estimated cold compile of a schema that passed `check`, in milliseconds (for tests and the journal).
    static func estimatedCompileMs(_ text: String) -> Double? {
        guard let data = text.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else { return nil }
        var walker = Walker()
        walker.visit(root, pointer: "", depth: 1)
        return walker.cost.total
    }

    // MARK: - The walk

    private struct Walker {
        var findings: [Finding] = []
        var cost = CostModel()
        private var tooDeepReported = false

        mutating func add(_ pointer: String, _ keyword: String, _ kind: Finding.Kind) {
            findings.append(Finding(pointer: pointer, keyword: keyword, kind: kind))
        }

        /// `depth` counts JSON containers from the root, the thing xgrammar's recursive copy pays for.
        mutating func visit(_ node: Any, pointer: String, depth: Int) {
            guard let object = node as? [String: Any] else { return }          // `true` / `false` constrain nothing we need to check
            if depth > Bounds.maxJSONDepth {
                if !tooDeepReported {
                    tooDeepReported = true
                    add(pointer, "(nesting)", .tooDeep(allowed: Bounds.maxJSONDepth))
                }
                return
            }

            for keyword in object.keys.sorted() {
                if warnedAndIgnored.contains(keyword) || neverRead.contains(keyword) {
                    add(Self.child(pointer, keyword), keyword, .ignoredKeyword)
                }
            }

            if let ref = object["$ref"] as? String, ref != "#", !ref.hasPrefix("#/") {
                add(Self.child(pointer, "$ref"), "$ref", .unresolvableReference(ref))
            }

            checkString(object, pointer: pointer)
            checkCounts(object, pointer: pointer)

            // Descend by the schema's own structure.
            for keyword in ["properties", "patternProperties", "$defs", "definitions"] {
                guard let members = object[keyword] as? [String: Any] else { continue }
                for name in members.keys.sorted() {
                    visit(members[name] as Any, pointer: Self.child(Self.child(pointer, keyword), name), depth: depth + 2)
                }
            }
            for keyword in ["items", "additionalProperties", "unevaluatedProperties", "unevaluatedItems", "propertyNames"] {
                guard let sub = object[keyword] else { continue }
                if let list = sub as? [Any] {                                    // draft-4 style tuple `items: [..]`
                    for (index, element) in list.enumerated() { visit(element, pointer: Self.child(Self.child(pointer, keyword), String(index)), depth: depth + 2) }
                } else {
                    visit(sub, pointer: Self.child(pointer, keyword), depth: depth + 1)
                }
            }
            for keyword in ["prefixItems", "anyOf", "oneOf", "allOf"] {
                guard let list = object[keyword] as? [Any] else { continue }
                for (index, element) in list.enumerated() {
                    visit(element, pointer: Self.child(Self.child(pointer, keyword), String(index)), depth: depth + 2)
                }
            }
        }

        private mutating func checkString(_ object: [String: Any], pointer: String) {
            let isString = (object["type"] as? String) == "string" || ((object["type"] as? [Any])?.contains { ($0 as? String) == "string" } ?? false)
            guard isString else { return }
            let hasPattern = object["pattern"] is String

            if hasPattern {
                for keyword in ["minLength", "maxLength", "format"] where object[keyword] != nil {
                    add(Self.child(pointer, keyword), keyword, .ignoredBesidePattern)
                }
                if let pattern = object["pattern"] as? String { checkPattern(pattern, pointer: Self.child(pointer, "pattern")) }
                return
            }

            let maxLength = Self.integer(object["maxLength"])
            if let maxLength { cost.add("maxLength", Double(max(0, maxLength)) * CostModel.msPerMaxLengthCharacter) }
            if let maxLength, maxLength > Bounds.maxLength {
                add(Self.child(pointer, "maxLength"), "maxLength", .overBound(value: maxLength, allowed: Bounds.maxLength))
            }
            if let minLength = Self.integer(object["minLength"]), minLength > 0 {
                if maxLength == nil || (maxLength ?? 0) > Bounds.maxLength {
                    // Over-large `maxLength` is already reported above; without one, `minLength` alone is the slow case.
                    if maxLength == nil {
                        add(Self.child(pointer, "minLength"), "minLength",
                            .refusedOutright("on its own it takes more than a second to prepare; give a `maxLength` of at most "
                                + "\(Bounds.maxLength) beside it, or drop it."))
                    }
                }
            }
            if let format = object["format"] as? String {
                cost.add("format", CostModel.formatMs[format] ?? CostModel.otherFormatMs)
                if Bounds.refusedFormats.contains(format) {
                    add(Self.child(pointer, "format"), "format",
                        .refusedOutright("format \"\(format)\" takes more than a second to prepare."))
                } else if !supportedFormats.contains(format) {
                    add(Self.child(pointer, "format"), "format", .unsupportedFormat(format))
                }
            }
        }

        /// Regex repetition counts expand into grammar states: `{m,n}` costs ~27 ms per unit, `{n}` ~14 ms.
        private mutating func checkPattern(_ pattern: String, pointer: String) {
            for quantifier in ServeSchemaCheck.quantifiers(in: pattern) {
                cost.add("pattern", Double(quantifier.weight) * quantifier.form.msPerUnit)
            }
            for quantifier in ServeSchemaCheck.quantifiers(in: pattern) {
                let limit = quantifier.form.bound
                if quantifier.weight > limit {
                    add(pointer, "pattern", .overBound(value: quantifier.weight, allowed: limit))
                    return
                }
            }
        }

        private mutating func checkCounts(_ object: [String: Any], pointer: String) {
            if let properties = object["properties"] as? [String: Any] {
                cost.add("properties", Double(properties.count) * CostModel.msPerProperty)
                if properties.count > Bounds.properties {
                    add(Self.child(pointer, "properties"), "properties", .overBound(value: properties.count, allowed: Bounds.properties))
                }
            }
            if let values = object["enum"] as? [Any] {
                cost.add("enum", Double(values.count) * CostModel.msPerEnumValue)
                if values.count > Bounds.enumValues {
                    add(Self.child(pointer, "enum"), "enum", .overBound(value: values.count, allowed: Bounds.enumValues))
                }
            }
            for keyword in ["anyOf", "oneOf"] {
                guard let branches = object[keyword] as? [Any] else { continue }
                cost.add("anyOf/oneOf", Double(branches.count) * CostModel.msPerBranch)
                if branches.count > Bounds.branches {
                    add(Self.child(pointer, keyword), keyword, .overBound(value: branches.count, allowed: Bounds.branches))
                }
            }
        }

        static func child(_ pointer: String, _ token: String) -> String {
            pointer + "/" + token.replacingOccurrences(of: "~", with: "~0").replacingOccurrences(of: "/", with: "~1")
        }

        /// A JSON integer (an `NSNumber` that is not a boolean and has no fractional part), else `nil`.
        static func integer(_ value: Any?) -> Int? {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
            let double = number.doubleValue
            guard double.isFinite, double == double.rounded(), abs(double) < 9e15 else { return nil }
            return Int(double)
        }
    }

    // MARK: - The whole-schema estimate

    /// An additive estimate of a schema's cold compile on Gemma 4's vocabulary, from the ladders (journal 2026-572). Per-unit
    /// costs are the slopes measured on one construct alone; `baseline` is a free string's cost (~480 ms), added once.
    ///
    /// Calibration (`ServeSchemaCheckTests.theEstimateNeverUnderstatesAMeasuredSchemaByMuch`): against eight measured schemas
    /// the estimate is 0.95–2.0× the measured time — it over-states most combinations (strings of different shapes share work)
    /// and under-states one by 5 %. With a 2,000 ms budget the slowest schema it accepts that was measured took 1,830 ms.
    struct CostModel {
        static let baseline = 500.0
        static let msPerEnumValue = 0.155
        static let msPerProperty = 2.7
        static let msPerBranch = 1.0
        static let msPerMaxLengthCharacter = 170.0
        /// Measured alone (ms): email 749, json-pointer 522, relative-json-pointer 520, uri-reference 311, uri 272, hostname 180,
        /// ipv6 113; the rest are ≤ 14.
        static let formatMs: [String: Double] = [
            "email": 750, "json-pointer": 520, "relative-json-pointer": 520, "uri-reference": 310, "uri": 270,
            "hostname": 180, "ipv6": 115, "date": 10, "time": 10, "date-time": 10, "duration": 10, "ipv4": 10, "uuid": 15,
        ]
        static let otherFormatMs = 750.0

        private var parts: [String: Double] = [:]
        mutating func add(_ construct: String, _ ms: Double) { parts[construct, default: 0] += ms }
        var total: Double { Self.baseline + parts.values.reduce(0, +) }
        func largest(_ count: Int) -> [String] {
            parts.sorted { $0.value > $1.value }.prefix(count).filter { $0.value >= 1 }
                .map { "\($0.key) ≈ \(Int($0.value.rounded())) ms" }
        }
    }

    // MARK: - Regex repetition counts

    struct Quantifier: Equatable {
        enum Form: Equatable {
            /// `{n}` (and `{n,n}`).
            case exact
            /// `{m,n}`, `n > m`.
            case range
            /// `{m,}`.
            case open

            var bound: Int {
                switch self {
                case .exact: return Bounds.patternExactQuantifier
                case .range: return Bounds.patternRangeQuantifier
                case .open: return Bounds.patternOpenQuantifier
                }
            }
            /// Slopes measured per unit of the count: exact ~14, range ~27, open ~16 (ms).
            var msPerUnit: Double {
                switch self {
                case .exact: return 14
                case .range: return 27
                case .open: return 16
                }
            }
        }
        /// The count that expands: `n` of `{m,n}`; `n` of `{n}`; `m` of `{m,}`.
        let weight: Int
        let form: Form
    }

    /// The `{n}`, `{m,n}` and `{m,}` quantifiers in a pattern (an escaped `\{` is a literal brace, not one).
    static func quantifiers(in pattern: String) -> [Quantifier] {
        var found: [Quantifier] = []
        let characters = Array(pattern)
        var index = 0
        while index < characters.count {
            if characters[index] == "\\" { index += 2; continue }
            if characters[index] == "{" {
                var end = index + 1
                while end < characters.count, characters[end] != "}" { end += 1 }
                if end < characters.count {
                    let body = String(characters[(index + 1) ..< end])
                    let parts = body.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                    if parts.count == 1, let n = Int(parts[0]) {
                        found.append(Quantifier(weight: n, form: .exact))
                    } else if parts.count == 2, let m = Int(parts[0]) {
                        if parts[1].isEmpty { found.append(Quantifier(weight: m, form: .open)) }
                        else if let n = Int(parts[1]) { found.append(Quantifier(weight: n, form: n > m ? .range : .exact)) }
                    }
                    index = end
                }
            }
            index += 1
        }
        return found
    }
}
