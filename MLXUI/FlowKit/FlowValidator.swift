import Foundation

/// The check-21 opaque-asset kind descriptor — ported from
/// `catflow-mlx/src/catflow/catalog/opaque_assets.py`. `vector` is the only registered
/// entry v0.8 ships; the validator reads every task/field name off it, never a literal.
nonisolated struct ShapeField: Sendable, Equatable {
    var name: String
    var defaultValue: String
}

nonisolated struct OpaqueAssetKind: Sendable {
    var kind: Kind
    var identityField: String
    var containerIdentityPlaceholder: String
    var queryIdentityPlaceholder: String
    var producerTasks: Set<String>
    var containerTask: String
    var containerKind: Kind
    var containerPayloadRef: Int
    var loaderTask: String
    var consumerTask: String
    var consumerContainerRef: Int
    var consumerPayloadRef: Int
    var shapeFields: [ShapeField]
    var errorCode: String
}

nonisolated enum OpaqueAssets {
    static let vectorKind = OpaqueAssetKind(
        kind: .vector,
        identityField: "embedder",
        containerIdentityPlaceholder: "index-embedder",
        queryIdentityPlaceholder: "row-embedder",
        producerTasks: ["Embed"],
        containerTask: "Store Index",
        containerKind: .index,
        containerPayloadRef: 1,
        loaderTask: "Read Index",
        consumerTask: "Retrieve",
        consumerContainerRef: 0,
        consumerPayloadRef: 1,
        shapeFields: [ShapeField(name: "normalization", defaultValue: "none")],
        errorCode: "E209"
    )

    /// Check 21's dispatch: consumer task -> the registered kind's spec.
    static let consumerTasks: [String: OpaqueAssetKind] = [vectorKind.consumerTask: vectorKind]
}

/// The curated model registry the validator consults — a minimal port of
/// `catalog/registry.py::Registry` (display-name resolution + id lookup). `resolveDisplay`
/// returns the manifest the E104 exemption / E209 canonicalization / F010 need.
nonisolated protocol FlowRegistry: Sendable {
    func resolveDisplay(_ display: String) -> FlowManifest?
    func get(_ id: String) -> FlowManifest?
}

/// The subset of `Registry Manifest` the validator reads — ported from
/// `catalog/registry.py::Manifest` (id/display/kind/engine/tasks + capabilities).
nonisolated struct FlowManifest: Sendable, Equatable {
    var id: String
    var display: String
    var kind: String
    var engine: String
    var tasks: [String]
    var capabilities: [String: String]

    func capabilityBool(_ name: String) -> Bool? {
        guard let raw = capabilities[name] else { return nil }
        return raw == "true"
    }
}

/// A registry loaded from `Fixtures/CatFlow/registry/*.json` (or the app bundle's
/// `CatFlow/models/`), mirroring `load_registry`.
nonisolated struct DirectoryFlowRegistry: FlowRegistry {
    var byDisplay: [String: FlowManifest]
    var byID: [String: FlowManifest]

    init(manifests: [FlowManifest]) {
        var byDisplay: [String: FlowManifest] = [:]
        var byID: [String: FlowManifest] = [:]
        for m in manifests {
            byDisplay[m.display] = m
            byID[m.id] = m
        }
        self.byDisplay = byDisplay
        self.byID = byID
    }

    static func load(directory: URL) throws -> DirectoryFlowRegistry {
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".json") }
            .sorted()
        var manifests: [FlowManifest] = []
        for f in files {
            let data = try Data(contentsOf: directory.appendingPathComponent(f))
            guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            manifests.append(FlowManifest(
                id: obj["id"] as? String ?? "",
                display: obj["display"] as? String ?? "",
                kind: obj["kind"] as? String ?? "",
                engine: obj["engine"] as? String ?? "",
                tasks: obj["tasks"] as? [String] ?? [],
                capabilities: Self.stringCapabilities(obj["capabilities"])
            ))
        }
        return DirectoryFlowRegistry(manifests: manifests)
    }

    private static func stringCapabilities(_ raw: Any?) -> [String: String] {
        guard let dict = raw as? [String: Any] else { return [:] }
        var out: [String: String] = [:]
        for (k, v) in dict {
            if let s = v as? String { out[k] = s }
            else if let b = v as? Bool { out[k] = b ? "true" : "false" }
            else if let n = v as? NSNumber { out[k] = n.stringValue }
        }
        return out
    }

    func resolveDisplay(_ display: String) -> FlowManifest? { byDisplay[display] }
    func get(_ id: String) -> FlowManifest? { byID[id] }
}

/// One validation issue — ported from `core/validator.py::Issue`. `row` is the
/// scope-relative dotted path ("3", "2.1"); `message` is the ⚠-free wording.
nonisolated struct FlowIssue: Equatable, Sendable, CustomStringConvertible {
    var row: String
    var code: String
    var message: String

    var description: String { "⚠ row \(row): \(message)" }
}

/// A non-blocking style lint — ported from `core/validator.py::Lint`.
nonisolated struct FlowLint: Equatable, Sendable, CustomStringConvertible {
    var row: String
    var code: String
    var message: String

    var description: String { "lint row \(row): \(message)" }
}

/// Settings-tokenizer — ported from `tools/_settings.py::Settings` (the shared
/// tokenizer the validator's `_parse_settings_kv` and `Range` checks build on).
nonisolated struct CatFlowSettings {
    var raw: String
    var pairs: [String: String]
    var bare: [String]

    init(_ raw: String?) {
        self.raw = raw ?? ""
        var pairs: [String: String] = [:]
        var bare: [String] = []
        for token in Self.tokenize(self.raw) {
            if token.hasPrefix("\"") {
                bare.append(Self.unquote(token))
                continue
            }
            if let eq = token.firstIndex(of: "=") {
                let key = String(token[..<eq])
                if Self.validKey(key) {
                    pairs[key] = Self.unquote(String(token[token.index(after: eq)...]))
                    continue
                }
            }
            bare.append(Self.unquote(token))
        }
        self.pairs = pairs
        self.bare = bare
    }

    func get(_ key: String, default dflt: String? = nil) -> String? { pairs[key] ?? dflt }
    func has(_ key: String) -> Bool { pairs[key] != nil }
    func firstBare() -> String? { bare.first }

    /// `_TOKEN_RE`: adapter keys (`lora=`/`controlnet=`) greedy to `;`,
    /// `key="quoted"`, bare `"quoted"`, else a bare word. `;` excluded from the
    /// bare-word class (v0.8 tight-left settings).
    static func tokenize(_ raw: String) -> [String] {
        var tokens: [String] = []
        let chars = Array(raw)
        var i = 0
        let n = chars.count
        while i < n {
            let c = chars[i]
            if isIdentStart(c) {
                var j = i
                while j < n, isIdentChar(chars[j]) { j += 1 }
                let word = String(chars[i..<j])
                if (word == "lora" || word == "controlnet"), j < n, chars[j] == "=" {
                    var k = j + 1
                    while k < n, chars[k] != ";" { k += 1 }
                    tokens.append(String(chars[i..<k]))
                    i = k
                    continue
                }
                if j < n, chars[j] == "=", j + 1 < n, chars[j + 1] == "\"" {
                    var k = j + 2
                    var tokenEnd = k
                    while k < n {
                        if chars[k] == "\\", k + 1 < n { k += 2; continue }
                        if chars[k] == "\"" { tokenEnd = k + 1; break }
                        k += 1
                    }
                    if tokenEnd > j + 2 {
                        tokens.append(String(chars[i..<tokenEnd]))
                        i = tokenEnd
                        continue
                    }
                }
                var k = i
                while k < n, !chars[k].isWhitespace, chars[k] != ";" { k += 1 }
                tokens.append(String(chars[i..<k]))
                i = k
                continue
            }
            if c == "\"" {
                var k = i + 1
                var tokenEnd = n
                while k < n {
                    if chars[k] == "\\", k + 1 < n { k += 2; continue }
                    if chars[k] == "\"" { tokenEnd = k + 1; break }
                    k += 1
                }
                tokens.append(String(chars[i..<tokenEnd]))
                i = tokenEnd
                continue
            }
            if c.isNumber || c == "-" {
                var k = i
                while k < n, !chars[k].isWhitespace, chars[k] != ";" { k += 1 }
                tokens.append(String(chars[i..<k]))
                i = k
                continue
            }
            i += 1
        }
        return tokens
    }

    private static func isIdentStart(_ c: Character) -> Bool { c.isLetter || c == "_" }
    private static func isIdentChar(_ c: Character) -> Bool { c.isLetter || c.isNumber || c == "_" }

    private static func validKey(_ key: String) -> Bool {
        guard let first = key.first else { return false }
        return (first.isLetter || first == "_") && key.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
    }

    static func unquote(_ token: String) -> String {
        if token.count >= 2, token.hasPrefix("\""), token.hasSuffix("\"") {
            return String(token.dropFirst().dropLast())
                .replacingOccurrences(of: "\\n", with: "\n")
                .replacingOccurrences(of: "\\t", with: "\t")
                .replacingOccurrences(of: "\\\"", with: "\"")
                .replacingOccurrences(of: "\\\\", with: "\\")
        }
        return token
    }
}

// MARK: - `catflow check` — the validator (port of core/validator.py)

/// The full validity checker — ported from `catflow-mlx/src/catflow/core/validator.py`
/// (3,300 lines) with the same messages, the same dotted row paths, and the same
/// registry-optional semantics. CFM-R5-3.
nonisolated enum FlowValidator {

    static let maxRefs = 4

    static let ordinals = [1: "first", 2: "second", 3: "third", 4: "fourth"]
    static func ordinal(_ n: Int) -> String { ordinals[n] ?? "\(n)th" }

    static let broadcastSuggestions: [Kind: String] = [.text: "Join Text", .video: "Join Video"]

    static func broadcastSuggestion(_ kind: Kind, pool: Set<String>?) -> String {
        if let s = broadcastSuggestions[kind], pool == nil || pool!.contains(s) { return s }
        return "a task that combines the list into one"
    }

    static func tierAwareSuggestionPool(_ rows: [ParsedRow]) -> Set<String> {
        let hidden = Set(["Generate", "Decide"])
        var usesHidden = false
        for (_, row) in iterFlowRows(rows) where row.blockKind == nil {
            if let t = row.task, hidden.contains(t) { usesHidden = true }
        }
        let all = Set(TaskCatalog.allTasks().map(\.name))
        return usesHidden ? all : all.subtracting(hidden)
    }

    /// `_iter_flow_rows` — (dotted path, row) pairs, recursing into block children only.
    static func iterFlowRows(_ rows: [ParsedRow], prefix: String = "") -> [(String, ParsedRow)] {
        var out: [(String, ParsedRow)] = []
        for (i, row) in rows.enumerated() {
            let path = "\(prefix)\(i + 1)"
            out.append((path, row))
            if row.blockKind != nil {
                out.append(contentsOf: iterFlowRows(row.children, prefix: "\(path)."))
            }
        }
        return out
    }

    static func describe(_ shape: Shape) -> String { shape.signatureText }

    static func refDisplay(_ ref: ParsedRef) -> String {
        switch ref {
        case .row(let n): return String(n)
        case .input(let p): return "input:\(p)"
        case .param(let name): return "param:\(name)"
        }
    }

    static func findBundleMismatch(accepts: Shape, bundle: [Shape], rk: RefKind?) -> (Int, String)? {
        if rk == .frame {
            for (i, g) in bundle.enumerated() where Shape.baseKind(g) != .text { return (i, "text") }
            return nil
        }
        if bundle.count == 1 {
            return (0, describe(accepts))
        }
        switch accepts {
        case .anyKind:
            return nil
        case .tupleOf(let kinds):
            if bundle.count != kinds.count { return (0, describe(accepts)) }
            for (i, (g, k)) in zip(bundle, kinds).enumerated() where Shape.baseKind(g) != k { return (i, k.rawValue) }
            return nil
        case .listOf(let acceptsKind):
            for (i, g) in bundle.enumerated() where Shape.baseKind(g) != acceptsKind { return (i, acceptsKind.rawValue) }
            return nil
        case .single, .unionOf, .sameAsInput:
            return (0, describe(accepts))
        }
    }

    // MARK: - Composite calling-row checks (E404/E406)

    private static let reKV = NSRegularExpression.compiled("^([\\w-]+)\\s*=\\s*(.+)$")
    private static let reInternalCap = NSRegularExpression.compiled("[a-z][A-Z]")

    static func looksLikeCompositeName(_ task: String) -> Bool {
        !task.contains(" ") && reInternalCap.firstMatch(in: task) != nil
    }

    static func splitOnDotQuoted(_ text: String, sep: String) -> [String] {
        var parts: [String] = []
        var current = ""
        var inQuotes = false
        for c in text {
            if c == "\"" {
                inQuotes.toggle()
                current.append(c)
            } else if String(c) == sep && !inQuotes {
                parts.append(current)
                current = ""
            } else {
                current.append(c)
            }
        }
        parts.append(current)
        return parts
    }

    static func parseSettingsKV(_ settings: String, isV08: Bool) -> [String: String] {
        var result: [String: String] = [:]
        for part in splitOnDotQuoted(settings, sep: isV08 ? ";" : "·") {
            let stripped = part.trimmingCharacters(in: .whitespaces)
            if let m = reKV.firstMatch(in: stripped) {
                let ns = stripped as NSString
                result[ns.substring(with: m.range(at: 1))] = ns.substring(with: m.range(at: 2))
            }
        }
        return result
    }

    static func checkCompositeCall(row: ParsedRow, path: String, comp: CompositeDef, issues: inout [FlowIssue], isV08: Bool) {
        var bound = Set<String>()
        let parts = splitOnDotQuoted(row.settings ?? "", sep: isV08 ? ";" : "·")
        let firstPart = parts.first?.trimmingCharacters(in: .whitespaces) ?? ""
        let positionalBound = row.model != nil || (!firstPart.isEmpty && reKV.firstMatch(in: firstPart) == nil)
        if !comp.params.isEmpty && positionalBound { bound.insert(comp.params[0].name) }
        let declared = Set(comp.params.map(\.name))
        for key in parseSettingsKV(row.settings ?? "", isV08: isV08).keys {
            if !declared.contains(key) {
                issues.append(FlowIssue(row: path, code: "E406", message: (try? ErrorCatalog.fill(
                    code: "E406", values: ["n": path, "key": key, "composite": row.task ?? "", "params": declared.joined(separator: ", ")])) ?? ""))
            } else {
                bound.insert(key)
            }
        }
        for p in comp.params where !bound.contains(p.name) && p.defaultValue == nil {
            issues.append(FlowIssue(row: path, code: "E404", message: (try? ErrorCatalog.fill(
                code: "E404", values: ["n": path, "composite": row.task ?? "", "param": p.name])) ?? ""))
        }
    }

    // MARK: - E207 placeholder check

    private static let reTemplateToken = NSRegularExpression.compiled("\\{(input(?::(\\d+))?|\\d+|[A-Za-z_][A-Za-z0-9_-]*)\\}")

    static func unquotePattern(_ raw: String) -> String {
        let s = raw.trimmingCharacters(in: .whitespaces)
        if s.count >= 2, s.hasPrefix("\""), s.hasSuffix("\"") { return String(s.dropFirst().dropLast()) }
        return s
    }

    static func bundlePositionCount(_ shapes: [Shape]) -> Int? {
        var total = 0
        for shape in shapes {
            switch shape {
            case .tupleOf(let ks): total += ks.count
            case .single: total += 1
            default: return nil
            }
        }
        return total
    }

    static func checkTemplatePlaceholders(row: ParsedRow, path: String, given: [Shape], issues: inout [FlowIssue], inEach: Bool) {
        guard let count = bundlePositionCount(given) else { return }
        let positions = count > 0 ? (1...count).map(String.init).joined(separator: ", ") : "(none)"
        let pattern = unquotePattern(row.settings ?? "")
        let ns = pattern as NSString
        for m in reTemplateToken.matches(in: pattern, range: NSRange(location: 0, length: ns.length)) {
            let token = ns.substring(with: m.range(at: 1))
            if token == "input" { continue }
            if (token == "index" || token == "item") && inEach { continue }
            let k: Int
            if token.hasPrefix("input:") {
                k = Int(token.split(separator: ":").dropFirst().first ?? "") ?? 0
            } else if token.allSatisfy(\.isNumber) {
                k = Int(token) ?? 0
            } else {
                issues.append(FlowIssue(row: path, code: "E207", message: (try? ErrorCatalog.fill(
                    code: "E207", values: ["n": path, "ph": token, "count": String(count), "list": positions])) ?? ""))
                continue
            }
            if count < 1 || !(1...count).contains(k) {
                issues.append(FlowIssue(row: path, code: "E207", message: (try? ErrorCatalog.fill(
                    code: "E207", values: ["n": path, "ph": token, "count": String(count), "list": positions])) ?? ""))
            }
        }
    }

    // MARK: - Signature helpers

    /// `_task_signature` — block inference or catalog lookup (+ controlnet widening,
    /// Save/Count Context override).
    static func taskSignature(_ row: ParsedRow, isV04: Bool) -> (Shape, Shape)? {
        if row.blockKind != nil {
            guard let last = row.children.last else { return nil }
            guard let lastSig = taskSignature(last, isV04: isV04) else { return nil }
            let gives = lastSig.1
            guard let accepts = blockAcceptsForCheck(row) else { return nil }
            var a = accepts
            var g = gives
            if row.blockKind == .each {
                if case .single(let k) = a { a = .listOf(k) }
                if case .single(let k) = g { g = .listOf(k) }
            }
            return (a, g)
        }
        guard let task = row.task else { return nil }
        var desc = TaskCatalog.catalog[task]
        if desc == nil && isV04 { desc = TaskCatalog.deciderTasks[task] }
        guard let desc else { return nil }
        if task == "Save Context" || task == "Count Context" {
            return (.anyKind, desc.gives)
        }
        var accepts = desc.accepts
        if let settings = row.settings, settings.contains("controlnet="),
           task == "Generate Image" || task == "Edit Image" {
            let base: [Kind]
            if case .single(let k) = accepts { base = [k] }
            else if case .tupleOf(let ks) = accepts { base = ks }
            else { base = [] }
            accepts = .tupleOf([.image] + base)
        }
        return (accepts, desc.gives)
    }

    /// `_block_accepts_for_check` (SPEC-Q118's narrowing).
    static func blockAcceptsForCheck(_ row: ParsedRow) -> Shape? {
        guard let first = row.children.first else { return nil }
        let firstReferencesInput = first.refs.contains { ref in
            if case .input(let p) = ref { return p == 1 }
            return false
        }
        if !firstReferencesInput && anyChildReferencesBlockInput(Array(row.children.dropFirst())) {
            return .anyKind
        }
        guard let firstSig = taskSignature(first, isV04: true) else { return nil }
        return firstSig.0
    }

    static func anyChildReferencesBlockInput(_ children: [ParsedRow]) -> Bool {
        for c in children {
            if c.refs.contains(where: { if case .input(let p) = $0 { return p == 1 }; return false }) { return true }
        }
        return false
    }

    /// `_task_ref_kind` — CATALOG only (deciders return nil under v0.4).
    static func taskRefKind(_ row: ParsedRow, isV04: Bool) -> RefKind? {
        guard let task = row.task else { return nil }
        if let desc = TaskCatalog.catalog[task] { return desc.refKind }
        return nil
    }

    // MARK: - Embedder mismatch (E707) + opaque assets (E209)

    static func traceEmbedderModel(_ ref: ParsedRef, rows: [ParsedRow]) -> String? {
        guard case .row(let n) = ref else { return nil }
        let idx = n - 1
        guard (0..<rows.count).contains(idx) else { return nil }
        let target = rows[idx]
        if target.task == "Embed" { return target.model }
        if target.task == "Store Index", target.refs.count >= 2 {
            return traceEmbedderModel(target.refs[1], rows: rows)
        }
        return nil
    }

    static func checkEmbedderMismatch(row: ParsedRow, rows: [ParsedRow], path: String, issues: inout [FlowIssue]) {
        let indexModel = traceEmbedderModel(row.refs[0], rows: rows)
        let queryModel = traceEmbedderModel(row.refs[1], rows: rows)
        if let im = indexModel, let qm = queryModel, im.trimmingCharacters(in: .whitespaces) != qm.trimmingCharacters(in: .whitespaces) {
            issues.append(FlowIssue(row: path, code: "E707", message: (try? ErrorCatalog.fill(
                code: "E707", values: ["n": path, "index": "row \(refDisplay(row.refs[0]))", "m": refDisplay(row.refs[1]), "embedder-a": im.trimmingCharacters(in: .whitespaces), "embedder-b": qm.trimmingCharacters(in: .whitespaces)])) ?? ""))
        }
    }

    static func traceOpaqueIdentity(_ ref: ParsedRef, rows: [ParsedRow], spec: OpaqueAssetKind) -> String? {
        guard case .row(let n) = ref else { return nil }
        let idx = n - 1
        guard (0..<rows.count).contains(idx) else { return nil }
        let target = rows[idx]
        if spec.producerTasks.contains(target.task ?? "") { return target.model }
        if target.task == spec.containerTask && target.refs.count > spec.containerPayloadRef {
            return traceOpaqueIdentity(target.refs[spec.containerPayloadRef], rows: rows, spec: spec)
        }
        return nil
    }

    static func containerManifest(_ ref: ParsedRef, rows: [ParsedRow], spec: OpaqueAssetKind) -> [String: String]? {
        guard case .row(let n) = ref else { return nil }
        let idx = n - 1
        guard (0..<rows.count).contains(idx) else { return nil }
        let target = rows[idx]
        if target.task == spec.containerTask && target.refs.count > spec.containerPayloadRef {
            guard let identity = traceOpaqueIdentity(target.refs[spec.containerPayloadRef], rows: rows, spec: spec) else { return nil }
            var info = [spec.identityField: identity.trimmingCharacters(in: .whitespaces)]
            let settings = CatFlowSettings(target.settings)
            for field in spec.shapeFields {
                info[field.name] = settings.get(field.name) ?? field.defaultValue
            }
            return info
        }
        return nil
    }

    static func resolveCanonical(_ name: String, registry: (any FlowRegistry)?) -> String {
        guard let registry else { return name }
        return registry.resolveDisplay(name)?.id ?? name
    }

    static func checkOpaqueAssetCompatibility(
        row: ParsedRow, rows: [ParsedRow], path: String, registry: (any FlowRegistry)?,
        issues: inout [FlowIssue], spec: OpaqueAssetKind
    ) {
        guard let containerInfo = containerManifest(row.refs[spec.consumerContainerRef], rows: rows, spec: spec),
              let queryIdentity = traceOpaqueIdentity(row.refs[spec.consumerPayloadRef], rows: rows, spec: spec) else { return }
        let containerIdentity = containerInfo[spec.identityField]
        let qIdentity = queryIdentity.trimmingCharacters(in: .whitespaces)
        guard let containerIdentity, !containerIdentity.isEmpty, containerIdentity != "unknown" else { return }

        func flag() {
            issues.append(FlowIssue(row: path, code: spec.errorCode, message: (try? ErrorCatalog.fill(
                code: spec.errorCode, values: ["n": path, "m": refDisplay(row.refs[spec.consumerPayloadRef]),
                                               spec.containerIdentityPlaceholder: containerIdentity,
                                               spec.queryIdentityPlaceholder: qIdentity], isV08: true)) ?? ""))
        }

        if resolveCanonical(containerIdentity, registry: registry) != resolveCanonical(qIdentity, registry: registry) {
            flag()
        }
    }

    // MARK: - Declared signature (R12, "signature-mismatch")

    static func parseDeclaredType(_ text: String) -> Shape? {
        let t = text.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("[") && t.hasSuffix("]") {
            let parts = String(t.dropFirst().dropLast()).split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            let kinds = parts.compactMap { Kind(rawValue: $0) }
            if kinds.isEmpty { return nil }
            return kinds.count == 1 ? .listOf(kinds[0]) : .tupleOf(kinds)
        }
        guard let kind = Kind(rawValue: t) else { return nil }
        return .single(kind)
    }

    static func parseDeclaredSignature(_ raw: String) -> (Shape?, Shape?) {
        for sep in ["→", "->"] where raw.contains(sep) {
            let parts = raw.components(separatedBy: sep)
            return (parseDeclaredType(parts[0]), parseDeclaredType(parts.dropFirst().joined(separator: sep)))
        }
        return (nil, parseDeclaredType(raw))
    }

    static func shapesLooselyMatch(_ a: Shape, _ b: Shape) -> Bool {
        if a == b { return true }
        let ka = Shape.baseKind(a)
        let kb = Shape.baseKind(b)
        return ka != nil && ka == kb
    }

    static func checkDeclaredSignature(row: ParsedRow, path: String, accepts: Shape, resolvedGives: Shape?, issues: inout [FlowIssue]) {
        guard let raw = row.declaredSignature else { return }
        let (dAccept, dGive) = parseDeclaredSignature(raw)
        if row.blockKind == nil, let dAccept, case .anyKind = accepts {} else {
            if row.blockKind == nil, let dAccept, !shapesLooselyMatch(dAccept, accepts) {
                issues.append(FlowIssue(row: path, code: "signature-mismatch",
                    message: "Row \(path)'s declared signature says the input is \(describe(dAccept)), but the inferred type is \(describe(accepts)). Update the annotation, or fix the row."))
            }
        }
        if let dGive, let resolvedGives, !shapesLooselyMatch(dGive, resolvedGives) {
            issues.append(FlowIssue(row: path, code: "signature-mismatch",
                message: "Row \(path)'s declared signature says the output is \(describe(dGive)), but the inferred type is \(describe(resolvedGives)). Update the annotation, or fix the row."))
        }
    }
}

extension FlowValidator {

    // MARK: - checkFlow entry

    /// `check_flow(flow, registry)` — run every check; issues in row order.
    static func checkFlow(_ flow: ParsedFlow, registry: (any FlowRegistry)? = nil,
                          workspace: FlowWorkspace? = nil, flowID: String? = nil,
                          rootFile: URL? = nil) -> [FlowIssue] {
        var issues: [FlowIssue] = []
        let isV08 = flow.version == "0.8"
        let isV04 = ["0.4", "0.7", "0.8"].contains(flow.version)

        _ = checkRows(
            flow.rows, pathPrefix: "", blockInputs: nil, issues: &issues,
            isV04: isV04, definitions: flow.definitions, isV08: isV08,
            uses: flow.uses, registry: registry, transforms: flow.transforms
        )

        if flow.version == "0.7" || flow.version == "0.8" {
            checkHumanRows(flow.rows, issues: &issues, isV08: isV08)
            checkTriggerPlacement(flow.rows, issues: &issues)
            checkEventsFlag(flow.rows, flags: flow.flags, issues: &issues, isV08: isV08)
            checkRateBudget(flow.rows, issues: &issues, isV08: isV08, transforms: flow.transforms)
            checkPipelineRules(flow, issues: &issues)
            checkModelsPinned(flow.rows, models: flow.models, registry: registry, issues: &issues)
            checkDoors(flow, issues: &issues, workspace: workspace, flowID: flowID,
                       registry: registry, rootFile: rootFile)
        }
        if isV08, !flow.uses.isEmpty {
            checkUsesAllNamed(flow.rows, uses: flow.uses, issues: &issues)
        }
        return issues
    }

    // MARK: - The door checks (CFM-R10-FIX-3; E109–E112, E118, E407–E409, E114–E117, E119)

    /// Emits the thirteen door checks that existed in `ErrorCatalog` and never fired
    /// (`core/validator.py:2710-2810`, `core/uses.py::_check_capability_propagation`).
    /// The `uses:` checks (E114–E117) need a workspace + flowID to resolve sibling flows;
    /// without them they are skipped (the editor always passes both).
    static func checkDoors(_ flow: ParsedFlow, issues: inout [FlowIssue],
                           workspace: FlowWorkspace?, flowID: String?,
                           registry: (any FlowRegistry)? = nil, rootFile: URL? = nil) {
        let flags = Set(flow.flags)

        // E109: an `Improvise` row without the `improvise` header flag.
        for (path, row) in iterFlowRows(flow.rows) where row.task == "Improvise" {
            if !flags.contains("improvise") {
                issues.append(FlowIssue(row: path, code: "E109",
                                        message: (try? ErrorCatalog.fill(code: "E109", values: ["n": path], isV08: true)) ?? ""))
            }
        }
        // E110: an `Improvise` row without `max_actions=N` and `timeout=<duration>`.
        for (path, row) in iterFlowRows(flow.rows) where row.task == "Improvise" {
            let s = FlowSettings(row.settings)
            let bounded = s.value(for: "max_actions") != nil && s.value(for: "timeout") != nil
            if !bounded {
                issues.append(FlowIssue(row: path, code: "E110",
                                        message: (try? ErrorCatalog.fill(code: "E110", values: ["n": path], isV08: true)) ?? ""))
            }
        }
        // E111: an `Improvise` row without `workdir=<dir>`.
        for (path, row) in iterFlowRows(flow.rows) where row.task == "Improvise" {
            if FlowSettings(row.settings).value(for: "workdir") == nil {
                issues.append(FlowIssue(row: path, code: "E111",
                                        message: (try? ErrorCatalog.fill(code: "E111", values: ["n": path], isV08: true)) ?? ""))
            }
        }
        // E112: `improvise` and `events` flags together (§14.4f).
        if flags.contains("improvise"), flags.contains("events") {
            issues.append(FlowIssue(row: "1", code: "E112",
                                    message: (try? ErrorCatalog.fill(code: "E112", isV08: true)) ?? ""))
        }
        // E118: `transforms:` declared but no `code` flag.
        if !flow.transforms.isEmpty, !flags.contains("code") {
            issues.append(FlowIssue(row: "1", code: "E118",
                                    message: (try? ErrorCatalog.fill(code: "E118", isV08: true)) ?? ""))
        }
        // E407/E408/E119: a calling row's transform must declare `run:`/`timeout:`/`workdir:`.
        for (path, row) in iterFlowRows(flow.rows) {
            guard let transform = flow.transforms[row.task ?? ""] else { continue }
            if transform.run == nil {
                issues.append(FlowIssue(row: path, code: "E407",
                                        message: (try? ErrorCatalog.fill(code: "E407", values: ["n": path, "transform": row.task ?? ""], isV08: true)) ?? ""))
            }
            if transform.timeout == nil {
                issues.append(FlowIssue(row: path, code: "E408",
                                        message: (try? ErrorCatalog.fill(code: "E408", values: ["n": path, "transform": row.task ?? ""], isV08: true)) ?? ""))
            }
            if transform.workdir == nil {
                issues.append(FlowIssue(row: path, code: "E119",
                                        message: (try? ErrorCatalog.fill(code: "E119", values: ["n": path, "transform": row.task ?? ""], isV08: true)) ?? ""))
            }
            // E409: the `run:` script must exist and be executable (needs the workspace).
            if let run = transform.run, let workspace, let flowID,
               let url = try? workspace.resolve(run, flowID: flowID),
               !FileManager.default.isExecutableFile(atPath: url.path) {
                issues.append(FlowIssue(row: path, code: "E409",
                                        message: (try? ErrorCatalog.fill(code: "E409", values: ["n": path, "transform": row.task ?? "", "path": url.lastPathComponent], isV08: true)) ?? ""))
            }
        }
        // E114–E117: the `uses:` checks (sibling-flow resolution; needs the workspace).
        if let workspace, let flowID, !flow.uses.isEmpty {
            checkUsesDoors(flow, issues: &issues, workspace: workspace, flowID: flowID,
                           registry: registry, rootFile: rootFile)
        }
    }

    /// E114–E117 — a port of `core/uses.py::_resolve_level` (R13-1). A broken entry
    /// reports its E114/E115/E116 and `continue`s to the next entry (a second `uses:`
    /// entry is still checked after a first escape), the anchor comes from
    /// `_first_row_naming` — the lowest-numbered top-level row that calls the entry,
    /// "1" when nothing does — and every entry that reads, parses and checks clean is
    /// recursed into before its reachable capabilities propagate to this flow (E117).
    static func checkUsesDoors(_ flow: ParsedFlow, issues: inout [FlowIssue],
                               workspace: FlowWorkspace, flowID: String,
                               registry: (any FlowRegistry)? = nil, rootFile: URL? = nil) {
        let flowDir = workspace.directory(for: flowID)
        _ = resolveUsesLevel(flow, flowDir: flowDir, workspace: workspace, flowID: flowID,
                             stack: rootFile.map { [$0] } ?? [],
                             registry: registry, issues: &issues)
    }

    /// One resolved `uses:` entry — mirrors `core/uses.py::UsedFlow`. `flow` is nil for a
    /// broken entry (E114/E115/E116 already reported it); `nested` is the entry's own
    /// resolved graph, which E117's capability reach walks.
    private struct ResolvedUsed {
        var rawPath: String
        var flow: ParsedFlow?
        var nested: [String: ResolvedUsed] = [:]
    }

    /// `_first_row_naming` — the lowest-numbered top-level row whose task names the
    /// entry; an entry nobody calls anchors at "1" instead (same as E112 for a
    /// flow-level fact with no row of its own).
    private static func firstRowNaming(_ flow: ParsedFlow, _ name: String) -> String? {
        for (i, row) in flow.rows.enumerated() where row.blockKind == nil && row.task == name {
            return String(i + 1)
        }
        return nil
    }

    @discardableResult
    private static func resolveUsesLevel(_ flow: ParsedFlow, flowDir: URL,
                                         workspace: FlowWorkspace, flowID: String,
                                         stack: [URL], registry: (any FlowRegistry)?,
                                         issues: inout [FlowIssue]) -> [String: ResolvedUsed] {
        var entries: [String: ResolvedUsed] = [:]
        for (name, rawPath) in flow.uses.sorted(by: { $0.key < $1.key }) {
            let anchor = firstRowNaming(flow, name) ?? "1"

            // E114: an absolute path or any escape (`..`, a symlink leaving the tree) —
            // `workspace.resolve` throws `escapesFlow` for exactly the two conditions
            // `uses.py` refuses at `_resolve_level`.
            let candidate: URL
            do {
                candidate = try workspace.resolve(rawPath, flowID: flowID)
            } catch {
                issues.append(FlowIssue(row: anchor, code: "E114",
                                        message: (try? ErrorCatalog.fill(code: "E114", values: ["path": rawPath], isV08: true)) ?? ""))
                entries[name] = ResolvedUsed(rawPath: rawPath)
                continue
            }

            // E115: the candidate is already an ancestor in the stack — the edge that
            // closes the loop. `_report_cycle` names the files, not the entry keys.
            if let idx = stack.firstIndex(of: candidate) {
                reportCycle(stack: stack, at: idx, issues: &issues)
                entries[name] = ResolvedUsed(rawPath: rawPath)
                continue
            }

            // E116: the file isn't there. Python's `is_file()` — a directory is not one.
            var isDir: ObjCBool = false
            let exists = FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDir)
            if !exists || isDir.boolValue {
                issues.append(FlowIssue(row: anchor, code: "E116",
                                        message: (try? ErrorCatalog.fill(code: "E116", values: ["n": anchor, "path": rawPath, "first-error": "no such file"], isV08: true)) ?? ""))
                entries[name] = ResolvedUsed(rawPath: rawPath)
                continue
            }

            // E116: the file doesn't read or doesn't parse — the first-error is the
            // parser's own sentence (Python: `str(exc).lstrip("⚠ ").strip()`).
            let text: String
            do {
                text = try String(contentsOf: candidate, encoding: .utf8)
            } catch {
                issues.append(FlowIssue(row: anchor, code: "E116",
                                        message: (try? ErrorCatalog.fill(code: "E116", values: ["n": anchor, "path": rawPath, "first-error": cleanedError(error)], isV08: true)) ?? ""))
                entries[name] = ResolvedUsed(rawPath: rawPath)
                continue
            }
            let used: ParsedFlow
            do {
                used = try CatParser.parseForValidation(text)
            } catch {
                issues.append(FlowIssue(row: anchor, code: "E116",
                                        message: (try? ErrorCatalog.fill(code: "E116", values: ["n": anchor, "path": rawPath, "first-error": cleanedError(error)], isV08: true)) ?? ""))
                entries[name] = ResolvedUsed(rawPath: rawPath)
                continue
            }

            // E116: the used flow must pass its own checks; the first finding is the
            // first-error (Python: `f"row {first.row}: {first.message}"`).
            let childIssues = checkFlow(used, registry: registry)
            if let first = childIssues.first {
                let firstError = "row \(first.row): \(first.message)"
                issues.append(FlowIssue(row: anchor, code: "E116",
                                        message: (try? ErrorCatalog.fill(code: "E116", values: ["n": anchor, "path": rawPath, "first-error": firstError], isV08: true)) ?? ""))
                entries[name] = ResolvedUsed(rawPath: rawPath)
                continue
            }

            // Clean child — recurse, then propagate its reachable capabilities (E117).
            // The app resolves every uses: path against the flow's own directory (the
            // sandbox model `FlowWorkspace` documents); the recursion resolves the same
            // way, so the stack is the same set of resolved URLs at every depth.
            let nested = resolveUsesLevel(used, flowDir: flowDir, workspace: workspace,
                                          flowID: flowID, stack: stack + [candidate],
                                          registry: registry, issues: &issues)
            entries[name] = ResolvedUsed(rawPath: rawPath, flow: used, nested: nested)
            checkCapabilityPropagation(flow, rawPath: rawPath, child: used, nested: nested,
                                       anchor: anchor, issues: &issues)
        }
        return entries
    }

    /// `_report_cycle` — SPEC-Q151's binding: `a` is the file whose own `uses:` entry is
    /// the direct edge that closes the loop, `b` is the ancestor it names, and `chain`
    /// lists whatever sits strictly between `b` and `a` along the original path — empty
    /// for a direct two-file cycle.
    private static func reportCycle(stack: [URL], at idx: Int, issues: inout [FlowIssue]) {
        guard let aName = stack.last?.lastPathComponent else { return }
        let bName = stack[idx].lastPathComponent
        let between = stack[(idx + 1)..<(stack.count - 1)].map { $0.lastPathComponent }
        let chain = between.joined(separator: " -> ")
        issues.append(FlowIssue(row: "1", code: "E115",
                                message: (try? ErrorCatalog.fill(code: "E115", values: ["a": aName, "b": bName, "chain": chain], isV08: true)) ?? ""))
    }

    /// The Python's error sentence with the `⚠ ` voice stripped (`str(exc).lstrip("⚠ ")`)
    /// — used as an E116 `first-error`. For an I/O failure the Cocoa message is shown
    /// instead (the fixture corpus never reads a non-UTF-8 sibling; that divergence is
    /// documented in the R13-1 journal).
    private static func cleanedError(_ error: Error) -> String {
        var s = String(describing: error)
        if s.hasPrefix("⚠ ") { s.removeFirst(2) }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The capability flags E117 propagates (Spec §1.4b, R25/R28) — `core/uses.py`'s
    /// `_CAPABILITY_FLAGS`.
    private static let capabilityFlags = ["improvise", "network", "code", "offdevice"]

    /// `_capability_reach` — every capability flag reachable from *flow* at any depth
    /// through *nested*, mapped to the chain of as-declared raw paths strictly between
    /// *flow* and whichever descendant first declares it natively. An empty chain means
    /// *flow* declares the flag itself.
    private static func capabilityReach(_ flow: ParsedFlow, _ nested: [String: ResolvedUsed]) -> [String: [String]] {
        var reach: [String: [String]] = [:]
        for flag in capabilityFlags where flow.flags.contains(flag) {
            reach[flag] = []
        }
        for used in nested.values.sorted(by: { $0.rawPath < $1.rawPath }) {
            guard let usedFlow = used.flow else { continue }  // broken entry — already reported
            for (flag, subChain) in capabilityReach(usedFlow, used.nested) where reach[flag] == nil {
                reach[flag] = [used.rawPath] + subChain
            }
        }
        return reach
    }

    /// `_check_capability_propagation` — check 20 (E117): every capability flag reachable
    /// through this entry, at any depth, must also be declared by *this* flow. Runs once
    /// per entry, at every level of the graph (each flow reaching a capability is
    /// independently required to say so). `improvise` renders the catalog template; the
    /// other three flags carry their own verbatim templates from `core/uses.py`.
    private static func checkCapabilityPropagation(_ flow: ParsedFlow, rawPath: String,
                                                   child: ParsedFlow, nested: [String: ResolvedUsed],
                                                   anchor: String, issues: inout [FlowIssue]) {
        let flags = Set(flow.flags)
        for (flag, chainFiles) in capabilityReach(child, nested).sorted(by: { $0.key < $1.key })
        where !flags.contains(flag) {
            let chain = chainFiles.map { "`\($0)`" }.joined(separator: " -> ")
            let chainPhrase = chain.isEmpty ? "" : "through \(chain)"
            let message: String
            switch flag {
            case "code":
                message = "This flow uses `\(rawPath)`, which \(chainPhrase) runs external code — but this header doesn't say `; code`. Add it. What a flow can do has to be readable on line one, even when it happens two files away."
            case "offdevice":
                message = "This flow uses `\(rawPath)`, which \(chainPhrase) sends data off this machine — but this header doesn't say `; offdevice`. Add it. What a flow can do has to be readable on line one, even when it happens two files away."
            case "network":
                message = "This flow uses `\(rawPath)`, which \(chainPhrase) talks to the internet — but this header doesn't say `; network`. Add it. What a flow can do has to be readable on line one, even when it happens two files away."
            default:
                message = (try? ErrorCatalog.fill(code: "E117", values: ["path": rawPath, "chain": chainPhrase], isV08: true)) ?? ""
            }
            issues.append(FlowIssue(row: anchor, code: "E117", message: message))
        }
    }

    // MARK: - _check_rows

    static func checkRows(
        _ rows: [ParsedRow], pathPrefix: String, blockInputs: [Shape?]?, issues: inout [FlowIssue],
        isV04: Bool, blockPath: String? = nil, parentRowCount: Int? = nil,
        definitions: [String: CompositeDef]? = nil, isV08: Bool = false,
        uses: [String: String]? = nil, registry: (any FlowRegistry)? = nil, inEach: Bool = false,
        transforms: [String: TransformDef]? = nil
    ) -> [Int: Shape?] {
        var resolved: [Int: Shape?] = [:]

        var controlIssuesByPosition: [Int: [FlowIssue]] = [:]
        if isV04 {
            controlIssuesByPosition = controlFlowIssues(rows, pathPrefix: pathPrefix, isV08: isV08)
        }

        var edgeTargets: Set<Int> = []
        for r in rows {
            for t in clauseTargets(r.clause) {
                switch t {
                case .row(let number), .call(let number): edgeTargets.insert(number)
                case .resume, .done: break
                }
            }
        }

        for (i, row) in rows.enumerated() {
            let position = i + 1
            let path = "\(pathPrefix)\(position)"
            issues.append(contentsOf: controlIssuesByPosition[position] ?? [])

            var accepts: Shape = .anyKind
            var gives: Shape = .anyKind
            var refKind: RefKind?

            if row.blockKind == nil, let task = row.task, TaskCatalog.catalog[task] == nil,
               !(isV04 && TaskCatalog.deciderTasks[task] != nil) {
                if let refusal = refusalRegistry[task] {
                    issues.append(FlowIssue(row: path, code: "refused-task", message: refusal))
                    resolved[position] = nil
                    continue
                }
                if let comp = definitions?[task] {
                    checkCompositeCall(row: row, path: path, comp: comp, issues: &issues, isV08: isV08)
                    let declared = comp.signature.flatMap(parseDeclaredSignature)
                    accepts = declared?.0 ?? .anyKind
                    gives = declared?.1 ?? .anyKind
                    refKind = nil
                } else if let uses, uses[task] != nil {
                    accepts = .anyKind
                    gives = .anyKind
                    refKind = nil
                } else if let uses, !uses.isEmpty {
                    issues.append(FlowIssue(row: path, code: "E113", message: (try? ErrorCatalog.fill(
                        code: "E113", values: ["n": path, "task": task], isV08: isV08)) ?? ""))
                    resolved[position] = nil
                    continue
                } else if let transform = transforms?[task] {
                    // A row naming a `transforms:` entry is a transform call — no catalog
                    // entry, resolved by its declared signature (CFM-R10-FIX-3).
                    let pair = transform.signature.flatMap(parseDeclaredSignature) ?? (nil, nil)
                    accepts = pair.0 ?? .anyKind
                    gives = pair.1 ?? .anyKind
                } else if looksLikeCompositeName(task) {
                    issues.append(FlowIssue(row: path, code: "E405", message: (try? ErrorCatalog.fill(
                        code: "E405", values: ["n": path, "composite": task])) ?? ""))
                    resolved[position] = nil
                    continue
                } else {
                    let redirect = refusalRegistry[task]
                    issues.append(FlowIssue(
                        row: path,
                        code: redirect != nil ? "refused-task" : "unknown-task",
                        message: redirect ?? "\"\(task)\" isn't a recognized task -- check spelling against `mlxflow tasks`."
                    ))
                    resolved[position] = nil
                    continue
                }
            } else {
                if let sig = taskSignature(row, isV04: isV04) {
                    accepts = sig.0
                    gives = sig.1
                }
                refKind = taskRefKind(row, isV04: isV04)
            }

            if row.refs.count > maxRefs {
                issues.append(FlowIssue(row: path, code: "E205", message: (try? ErrorCatalog.fill(
                    code: "E205", values: ["n": path, "count": String(row.refs.count)])) ?? ""))
            }

            let boundInputCount = blockInputs?.count ?? 0
            let (bundle, incomplete) = resolveRefs(
                row.refs, resolved: resolved, myPosition: position, path: path,
                blockInputs: blockInputs, blockPath: blockPath, parentRowCount: parentRowCount,
                boundInputCount: boundInputCount, issues: &issues
            )

            var effectiveAccepts: Shape?
            let isMultiInputBlock = row.blockKind != nil && row.refs.count > 1
            if !row.refs.isEmpty {
                if let bundle, !incomplete {
                    effectiveAccepts = bundle.count == 1 ? bundle[0] : nil
                    if !isMultiInputBlock && !Shape.bundleCompatible(accepts: accepts, given: bundle, rk: refKind) {
                        if bundle.count == 1, case .single(let acceptsKind) = accepts,
                           case .listOf(let bundleKind) = bundle[0], bundleKind == acceptsKind {
                            issues.append(FlowIssue(row: path, code: "E206", message: (try? ErrorCatalog.fill(
                                code: "E206", values: ["n": path, "task": row.task ?? "", "kind": acceptsKind.rawValue,
                                                       "count": "\(acceptsKind.rawValue)s",
                                                       "suggestion": broadcastSuggestion(acceptsKind, pool: tierAwareSuggestionPool(rows))],
                                isV08: isV08)) ?? ""))
                        } else {
                            let mismatch = findBundleMismatch(accepts: accepts, bundle: bundle, rk: refKind)
                            let idx = mismatch?.0 ?? 0
                            let wanted = mismatch?.1 ?? accepts.signatureText
                            let mRef = idx < row.refs.count ? row.refs[idx] : (row.refs.first ?? .input(position: 1))
                            let got = idx < bundle.count ? bundle[idx].signatureText : bundle.map(\.signatureText).joined(separator: ", ")
                            issues.append(FlowIssue(row: path, code: "E202", message: (try? ErrorCatalog.fill(
                                code: "E202", values: ["n": path, "m": refDisplay(mRef), "position": ordinal(idx + 1), "wanted": wanted, "got": got],
                                isV08: isV08)) ?? ""))
                        }
                    }
                }
            } else if !row.chainBreak && i > 0, let prevGives = resolved[position - 1], prevGives != nil {
                if Shape.singleCompatible(accepts: accepts, given: prevGives!, rk: refKind) {
                    effectiveAccepts = prevGives
                }
            }

            if row.blockKind == nil && position > 1 && !edgeTargets.contains(position)
                && row.refs.isEmpty && effectiveAccepts == nil && row.settings == nil {
                let nMinus1 = String(position - 1)
                let got: String
                if let prev = resolved[position - 1], prev != nil {
                    got = prev!.signatureText
                } else {
                    got = "nothing"
                }
                issues.append(FlowIssue(row: path, code: "E201", message: (try? ErrorCatalog.fill(
                    code: "E201", values: ["n": path, "task": row.task ?? "", "wanted": accepts.signatureText, "got": got,
                                           "n-1": nMinus1, "suggestion": "a reference or your own settings"],
                    isV08: isV08)) ?? ""))
            }

            checkAmbiguousAutoChain(row: row, i: i, path: path, accepts: accepts, resolved: resolved, issues: &issues)

            if isV08, let task = row.task, let spec = OpaqueAssets.consumerTasks[task],
               row.refs.count > max(spec.consumerContainerRef, spec.consumerPayloadRef), !incomplete {
                checkOpaqueAssetCompatibility(row: row, rows: rows, path: path, registry: registry, issues: &issues, spec: spec)
            } else if row.task == "Retrieve", row.refs.count == 2, !incomplete {
                checkEmbedderMismatch(row: row, rows: rows, path: path, issues: &issues)
            }

            if row.task == "Template" && row.blockKind == nil {
                var givenShapes: [Shape]?
                if !row.refs.isEmpty, let bundle, !incomplete {
                    givenShapes = bundle
                } else if row.refs.isEmpty, let effectiveAccepts {
                    givenShapes = [effectiveAccepts]
                }
                if let givenShapes {
                    checkTemplatePlaceholders(row: row, path: path, given: givenShapes, issues: &issues, inEach: inEach)
                }
            }

            var resolvedGives: Shape? = gives
            if case .sameAsInput = gives {
                if let effectiveAccepts {
                    switch effectiveAccepts {
                    case .single, .listOf: resolvedGives = effectiveAccepts
                    default: resolvedGives = nil
                    }
                } else {
                    resolvedGives = nil
                }
            }

            if row.blockKind != nil {
                var childBlockInputs: [Shape?] = []
                if !row.refs.isEmpty {
                    for ref in row.refs {
                        switch ref {
                        case .input(let position):
                            if let blockInputs, (1...blockInputs.count).contains(position) {
                                childBlockInputs.append(blockInputs[position - 1])
                            } else {
                                childBlockInputs.append(nil)
                            }
                        case .row(let n):
                            childBlockInputs.append((position > 1 && (1...(position - 1)).contains(n)) ? (resolved[n] ?? nil) : nil)
                        case .param:
                            childBlockInputs.append(nil)
                        }
                    }
                    if row.blockKind == .each, !childBlockInputs.isEmpty, case .listOf(let k)? = childBlockInputs[0] {
                        childBlockInputs[0] = .single(k)
                    }
                } else {
                    let childInput = effectiveAccepts ?? accepts
                    if row.blockKind == .each, case .listOf(let k) = childInput {
                        childBlockInputs = [.single(k)]
                    } else {
                        childBlockInputs = [childInput]
                    }
                }
                let childResolved = checkRows(
                    row.children, pathPrefix: "\(path).", blockInputs: childBlockInputs, issues: &issues,
                    isV04: isV04, blockPath: path, parentRowCount: rows.count,
                    definitions: definitions, isV08: isV08, uses: uses, registry: registry,
                    inEach: inEach || row.blockKind == .each
                )
                if row.blockKind == .parallel {
                    resolvedGives = parallelBundleGives(row: row, path: path, resolved: childResolved, issues: &issues)
                }
                checkBlockReturnType(row: row, path: path, resolved: childResolved, issues: &issues)
            }

            resolved[position] = resolvedGives

            checkDeclaredSignature(row: row, path: path, accepts: accepts, resolvedGives: resolvedGives, issues: &issues)

            if isV04 {
                checkTagStructure(row: row, path: path, issues: &issues)
                checkDecideEdgeTypes(row: row, path: path, resolvedGives: resolvedGives, rows: rows, isV04: isV04, issues: &issues)
                checkBudgetPairing(row: row, path: path, issues: &issues, isV08: isV08)
                FlowRangeBounds.checkRangeBounds(row: row, path: path, issues: &issues, isV08: isV08)
            }
        }
        return resolved
    }

    // MARK: - _resolve_refs

    static func resolveRefs(
        _ refs: [ParsedRef], resolved: [Int: Shape?], myPosition: Int, path: String,
        blockInputs: [Shape?]?, blockPath: String?, parentRowCount: Int?,
        boundInputCount: Int, issues: inout [FlowIssue]
    ) -> ([Shape]?, Bool) {
        if refs.isEmpty { return (nil, false) }
        var shapes: [Shape] = []
        var incomplete = false
        for ref in refs {
            switch ref {
            case .input(let position):
                if blockInputs == nil {
                    issues.append(FlowIssue(
                        row: path, code: "bad-input-ref",
                        message: "(input:\(position)) only makes sense inside a <list>/<each> block -- this row isn't inside one."
                    ))
                    incomplete = true
                    continue
                }
                if let blockInputs, !(1...blockInputs.count).contains(position) {
                    issues.append(FlowIssue(row: path, code: "E401", message: (try? ErrorCatalog.fill(
                        code: "E401", values: ["block": blockPath ?? "?", "n": path, "k": String(position), "count": String(blockInputs.count)])) ?? ""))
                    incomplete = true
                    continue
                }
                if let slot = blockInputs?[position - 1] {
                    shapes.append(slot)
                } else {
                    incomplete = true
                }
            case .param:
                shapes.append(.anyKind)
            case .row(let n):
                if myPosition <= 1 || !(1...(myPosition - 1)).contains(n) {
                    if let blockPath, let parentRowCount, (1...parentRowCount).contains(n) {
                        issues.append(FlowIssue(row: path, code: "E403", message: (try? ErrorCatalog.fill(
                            code: "E403", values: ["block": blockPath, "n": path, "m": String(n), "next": String(boundInputCount + 1)])) ?? ""))
                    } else {
                        issues.append(FlowIssue(row: path, code: "E203", message: (try? ErrorCatalog.fill(
                            code: "E203", values: ["n": path, "m": String(n)])) ?? ""))
                    }
                    incomplete = true
                    continue
                }
                if let targetGives = resolved[n], targetGives != nil {
                    shapes.append(targetGives!)
                } else {
                    incomplete = true
                }
            }
        }
        return (shapes, incomplete)
    }

    static func checkAmbiguousAutoChain(row: ParsedRow, i: Int, path: String, accepts: Shape, resolved: [Int: Shape?], issues: inout [FlowIssue]) {
        if row.chainBreak || i == 0 { return }
        if case .anyKind = accepts { return }
        guard row.refs.count == 1, case .row(let refN) = row.refs[0] else { return }
        let position = i + 1
        if refN == position - 1 { return }
        guard let prevGives = resolved[position - 1], prevGives != nil else { return }
        let rk = taskRefKind(row, isV04: true)
        if Shape.singleCompatible(accepts: accepts, given: prevGives!, rk: rk) {
            issues.append(FlowIssue(
                row: path, code: "ambiguous-auto-chain",
                message: "this row could auto-chain from the row above (same type), but it references a different row instead -- add a blank line above to make the branch explicit."
            ))
        }
    }
}

extension FlowValidator {

    // MARK: - Control flow (reachability, cycles, dominance; E301/E305/E306/E309/E203/E204)

    static func clauseTargets(_ clause: Clause?) -> [ClauseTarget] {
        guard let clause else { return [] }
        switch clause {
        case .goto(let t):
            if case .row = t { return [t] }
            return []
        case .fork(let targets):
            return targets.filter { if case .row = $0 { return true }; return false }
        case .decide(let edges):
            var out: [ClauseTarget] = []
            for e in edges {
                if case .row = e.target { out.append(e.target) }
                if case .call = e.target { out.append(e.target) }
            }
            return out
        case .call(let number):
            return [.row(number: number)]
        case .resume:
            return []
        }
    }

    static func clauseInvolvesResume(_ clause: Clause?) -> Bool {
        guard let clause else { return false }
        switch clause {
        case .resume: return true
        case .decide(let edges): return edges.contains { if case .resume = $0.target { return true }; return false }
        default: return false
        }
    }

    static func clauseIsDone(_ clause: Clause?) -> Bool {
        guard let clause else { return false }
        switch clause {
        case .goto(let t): return t == .done
        case .fork(let ts): return ts.contains(.done)
        case .decide(let edges): return edges.contains { $0.target == .done }
        default: return false
        }
    }

    static func callTargetPositions(_ rows: [ParsedRow]) -> Set<Int> {
        var targets = Set<Int>()
        for row in rows {
            switch row.clause {
            case .call(let n): targets.insert(n)
            case .decide(let edges):
                for e in edges { if case .call(let n) = e.target { targets.insert(n) } }
            default: break
            }
        }
        return targets
    }

    static func buildControlGraph(_ rows: [ParsedRow]) -> ([Int: [Int]], [(Int, Int)]) {
        let n = rows.count
        var edges: [Int: [Int]] = [:]
        if n > 0 { for p in 1...n { edges[p] = [] } }
        var dangling: [(Int, Int)] = []
        for (i, row) in rows.enumerated() {
            let position = i + 1
            if row.clause == nil {
                if position < n { edges[position]!.append(position + 1) }
                continue
            }
            for target in clauseTargets(row.clause) {
                let number: Int
                switch target {
                case .row(let n): number = n
                case .call(let n): number = n
                case .resume, .done: continue
                }
                if (1...n).contains(number) {
                    edges[position]!.append(number)
                } else {
                    dangling.append((position, number))
                }
            }
        }
        return (edges, dangling)
    }

    static func bfsReachable(_ edges: [Int: [Int]], root: Int) -> Set<Int> {
        var seen: Set<Int> = [root]
        var stack = [root]
        while let v = stack.popLast() {
            for w in edges[v] ?? [] where !seen.contains(w) {
                seen.insert(w)
                stack.append(w)
            }
        }
        return seen
    }

    static func tarjanSCC(_ edges: [Int: [Int]], n: Int) -> [Set<Int>] {
        var indexCounter = 0
        var stack: [Int] = []
        var onStack = Set<Int>()
        var indices: [Int: Int] = [:]
        var lowlink: [Int: Int] = [:]
        var result: [Set<Int>] = []

        func strongconnect(_ v: Int) {
            indices[v] = indexCounter
            lowlink[v] = indexCounter
            indexCounter += 1
            stack.append(v)
            onStack.insert(v)
            for w in edges[v] ?? [] {
                if indices[w] == nil {
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v]!, lowlink[w]!)
                } else if onStack.contains(w) {
                    lowlink[v] = min(lowlink[v]!, indices[w]!)
                }
            }
            if lowlink[v] == indices[v] {
                var scc = Set<Int>()
                while true {
                    let w = stack.popLast()!
                    onStack.remove(w)
                    scc.insert(w)
                    if w == v { break }
                }
                result.append(scc)
            }
        }

        for v in 1...n where indices[v] == nil {
            strongconnect(v)
        }
        return result
    }

    static func controlFlowIssues(_ rows: [ParsedRow], pathPrefix: String, isV08: Bool) -> [Int: [FlowIssue]] {
        let n = rows.count
        var byPosition: [Int: [FlowIssue]] = [:]
        if n == 0 { return byPosition }
        for p in 1...n { byPosition[p] = [] }   // 1...0 would trap — guarded above (H1)

        let (edges, dangling) = buildControlGraph(rows)

        for (position, bad) in dangling {
            let path = "\(pathPrefix)\(position)"
            byPosition[position]!.append(FlowIssue(row: path, code: "E203", message: (try? ErrorCatalog.fill(
                code: "E203", values: ["n": path, "m": String(bad)])) ?? ""))
        }

        let reachable = bfsReachable(edges, root: 1)
        for (position, row) in rows.enumerated() where !reachable.contains(position + 1) {
            let path = "\(pathPrefix)\(position + 1)"
            if clauseInvolvesResume(row.clause) {
                byPosition[position + 1]!.append(FlowIssue(row: path, code: "E309", message: (try? ErrorCatalog.fill(
                    code: "E309", values: ["n": path, "first-of-chain": String(position + 1)])) ?? ""))
            } else {
                byPosition[position + 1]!.append(FlowIssue(row: path, code: "E301", message: (try? ErrorCatalog.fill(
                    code: "E301", values: ["n": path])) ?? ""))
            }
        }

        for scc in tarjanSCC(edges, n: n) {
            let rep = scc.min() ?? 1
            let isSelfLoop = scc.count == 1 && (edges[rep]?.contains(rep) ?? false)
            if scc.count <= 1 && !isSelfLoop { continue }
            var hasExit = false
            var hasBudget = false
            var hasWaitForever = false
            for position in scc {
                let row = rows[position - 1]
                if row.visitsLeq != nil { hasBudget = true }
                if isHumanRow(row), parseSettingsKV(row.settings ?? "", isV08: isV08)["wait"] == "forever" {
                    hasWaitForever = true
                }
                if case .decide(let edgesList)? = row.clause {
                    for e in edgesList {
                        switch e.target {
                        case .resume, .done: hasExit = true
                        case .row(let n), .call(let n):
                            if !scc.contains(n) { hasExit = true }
                        }
                    }
                }
            }
            let repPath = "\(pathPrefix)\(rep)"
            let cycleStr = scc.sorted().map(String.init).joined(separator: ", ")
            if !hasExit {
                byPosition[rep]!.append(FlowIssue(row: repPath, code: "E305", message: (try? ErrorCatalog.fill(
                    code: "E305", values: ["cycle": cycleStr])) ?? ""))
            }
            if !hasBudget && !hasWaitForever {
                let deciderInSCC = scc.sorted().first { p in
                    if case .decide = rows[p - 1].clause { return true }
                    return false
                } ?? rep
                byPosition[rep]!.append(FlowIssue(row: repPath, code: "E306", message: (try? ErrorCatalog.fill(
                    code: "E306", values: ["cycle": cycleStr, "suggested": String(deciderInSCC)], isV08: isV08)) ?? ""))
            }
        }

        checkReferenceDominance(rows, edges: edges, reachable: reachable, pathPrefix: pathPrefix, byPosition: &byPosition)
        return byPosition
    }

    static func computeDominators(_ edges: [Int: [Int]], reachable: Set<Int>, root: Int) -> [Int: Set<Int>] {
        var preds: [Int: [Int]] = [:]
        for n in reachable { preds[n] = [] }
        for (u, targets) in edges where reachable.contains(u) {
            for v in targets where reachable.contains(v) {
                preds[v]!.append(u)
            }
        }
        var dom: [Int: Set<Int>] = [:]
        for n in reachable { dom[n] = reachable }
        dom[root] = [root]
        var changed = true
        while changed {
            changed = false
            for n in reachable where n != root {
                let new: Set<Int>
                if let ps = preds[n], !ps.isEmpty {
                    new = ps.dropFirst().reduce(into: dom[ps[0]]!) { $0.formIntersection(dom[$1]!) }.union([n])
                } else {
                    new = [n]
                }
                if new != dom[n] {
                    dom[n] = new
                    changed = true
                }
            }
        }
        return dom
    }

    static func forkSiblings(_ rows: [ParsedRow]) -> [Int: Set<Int>] {
        var siblings: [Int: Set<Int>] = [:]
        for row in rows {
            if case .fork(let targets)? = row.clause {
                var ts = Set<Int>()
                for t in targets { if case .row(let n) = t { ts.insert(n) } }
                for t in ts {
                    siblings[t, default: []].formUnion(ts.subtracting([t]))
                }
            }
        }
        return siblings
    }

    static func checkReferenceDominance(_ rows: [ParsedRow], edges: [Int: [Int]], reachable: Set<Int>, pathPrefix: String, byPosition: inout [Int: [FlowIssue]]) {
        guard reachable.contains(1) else { return }
        let dom = computeDominators(edges, reachable: reachable, root: 1)
        let forkSib = forkSiblings(rows)
        for (position, row) in rows.enumerated() where reachable.contains(position + 1) {
            for ref in row.refs {
                guard case .row(let target) = ref else { continue }
                if !reachable.contains(target) { continue }
                if dom[position + 1]?.contains(target) ?? false { continue }
                if let sib = forkSib[target], sib.intersection(dom[position + 1] ?? []).count > 0 { continue }
                let path = "\(pathPrefix)\(position + 1)"
                let culprit = findSkipCulprit(rows, edges: edges, dom: dom, target: target, referencer: position + 1)
                let deciderRow = culprit?.0 ?? target
                let tag = culprit?.1 ?? "?"
                byPosition[position + 1]!.append(FlowIssue(row: path, code: "E204", message: (try? ErrorCatalog.fill(
                    code: "E204", values: ["n": path, "m": String(target), "decider-row": String(deciderRow), "tag": tag])) ?? ""))
            }
        }
    }

    static func findSkipCulprit(_ rows: [ParsedRow], edges: [Int: [Int]], dom: [Int: Set<Int>], target: Int, referencer: Int) -> (Int, String)? {
        let candidates = dom[referencer]?.filter { $0 != referencer }.sorted(by: >) ?? []
        for d in candidates {
            guard case .decide(let edgesList)? = rows[d - 1].clause else { continue }
            for e in edgesList {
                guard case .row(let n) = e.target else { continue }
                var seen: Set<Int> = [n]
                var stack = [n]
                var found = false
                while let v = stack.popLast() {
                    if v == target { continue }
                    if v == referencer { found = true; break }
                    for w in edges[v] ?? [] where !seen.contains(w) {
                        seen.insert(w)
                        stack.append(w)
                    }
                }
                if found { return (d, e.tag) }
            }
        }
        return nil
    }
}

extension FlowValidator {

    // MARK: - Block return type (E402) + parallel bundle (E504)

    static func scopeExitPoints(_ rows: [ParsedRow], resolved: [Int: Shape?]) -> [(Int, Shape?)] {
        let n = rows.count
        let (edges, _) = buildControlGraph(rows)
        let reachable = n > 0 ? bfsReachable(edges, root: 1) : []
        var calledSubgraph: Set<Int> = Set()
        for t in callTargetPositions(rows) where (1...n).contains(t) {
            calledSubgraph.formUnion(bfsReachable(edges, root: t))
        }
        var exits: [(Int, Shape?)] = []
        for (i, row) in rows.enumerated() {
            let position = i + 1
            if !reachable.contains(position) { continue }
            let inCall = calledSubgraph.contains(position)
            let isExit = clauseIsDone(row.clause) || (
                !inCall && ((row.clause == nil && position == n) || clauseInvolvesResume(row.clause))
            )
            if isExit {
                exits.append((position, resolved[position] ?? nil))
            }
        }
        return exits
    }

    static func checkBlockReturnType(row: ParsedRow, path: String, resolved: [Int: Shape?], issues: inout [FlowIssue]) {
        let exits = scopeExitPoints(row.children, resolved: resolved)
            .filter { $0.1 != nil }
            .filter { if case .anyKind = $0.1! { return false }; return true }
        if exits.count < 2 { return }
        let first = exits[0]
        for (pos, shape) in exits.dropFirst() {
            if shape != first.1 {
                issues.append(FlowIssue(row: path, code: "E402", message: (try? ErrorCatalog.fill(
                    code: "E402", values: ["block": path,
                                           "a": "\(path).\(first.0)", "type-a": describe(first.1!),
                                           "b": "\(path).\(pos)", "type-b": describe(shape!)])) ?? ""))
                return
            }
        }
    }

    static func parallelChains(_ children: [ParsedRow]) -> [[Int]] {
        var chains: [[Int]] = []
        var current: [Int] = []
        for (i, child) in children.enumerated() {
            let position = i + 1
            if child.chainBreak && !current.isEmpty {
                chains.append(current)
                current = []
            }
            current.append(position)
        }
        if !current.isEmpty { chains.append(current) }
        return chains
    }

    static func parallelBundleGives(row: ParsedRow, path: String, resolved: [Int: Shape?], issues: inout [FlowIssue]) -> Shape? {
        let chains = parallelChains(row.children)
        var kinds: [Kind] = []
        var ok = true
        for (k, chain) in chains.enumerated() {
            let tailShape = resolved[chain.last!] ?? nil
            let base = tailShape.flatMap(Shape.baseKind)
            if let base {
                kinds.append(base)
            } else {
                ok = false
                issues.append(FlowIssue(row: path, code: "E504", message: (try? ErrorCatalog.fill(
                    code: "E504", values: ["k": String(k + 1), "block": path,
                                           "got": tailShape.map(describe) ?? "an unresolved type"])) ?? ""))
            }
        }
        return ok && !kinds.isEmpty ? .tupleOf(kinds) : nil
    }

    // MARK: - Tag structure (E302/E303)

    /// Closest match over the sorted declared tags — Python's difflib without the
    /// cutoff: the validator's fallback (`sorted(declared)[0]`) only ever fires when
    /// nothing is close, matching the golden.
    static func closestTag(_ tag: String, declared: [String]) -> String {
        if declared.isEmpty { return "?" }
        return declared.sorted()[0]
    }

    static func checkTagStructure(row: ParsedRow, path: String, issues: inout [FlowIssue]) {
        guard case .decide(let edgesList)? = row.clause, let tags = row.tags, !tags.isEmpty else { return }
        let declared = Set(tags)
        let declaredStr = tags.joined(separator: ", ")
        let edgeTags = edgesList.map(\.tag)
        let edgeTagSet = Set(edgeTags)
        for tag in tags where !edgeTagSet.contains(tag) {
            issues.append(FlowIssue(row: path, code: "E302", message: (try? ErrorCatalog.fill(
                code: "E302", values: ["n": path, "tags": declaredStr, "tag": tag])) ?? ""))
        }
        var seen = Set<String>()
        for tag in edgeTags where !declared.contains(tag) && !seen.contains(tag) {
            seen.insert(tag)
            let closest = closestTag(tag, declared: Array(declared))
            issues.append(FlowIssue(row: path, code: "E303", message: (try? ErrorCatalog.fill(
                code: "E303", values: ["n": path, "tag": tag, "tags": declaredStr, "closest": closest])) ?? ""))
        }
    }

    static func checkDecideEdgeTypes(row: ParsedRow, path: String, resolvedGives: Shape?, rows: [ParsedRow], isV04: Bool, issues: inout [FlowIssue]) {
        guard case .decide(let edgesList)? = row.clause, let resolvedGives else { return }
        for e in edgesList {
            guard case .row(let n) = e.target else { continue }
            if !(1...rows.count).contains(n) { continue }
            let targetRow = rows[n - 1]
            if !targetRow.refs.isEmpty { continue }
            guard let targetSig = taskSignature(targetRow, isV04: isV04) else { continue }
            let targetAccepts = targetSig.0
            let targetRefKind = taskRefKind(targetRow, isV04: isV04)
            if !Shape.singleCompatible(accepts: targetAccepts, given: resolvedGives, rk: targetRefKind) {
                issues.append(FlowIssue(row: path, code: "E304", message: (try? ErrorCatalog.fill(
                    code: "E304", values: ["n": path, "tag": e.tag, "got": describe(resolvedGives), "m": String(n), "wanted": describe(targetAccepts)])) ?? ""))
            }
        }
    }

    static func checkBudgetPairing(row: ParsedRow, path: String, issues: inout [FlowIssue], isV08: Bool) {
        guard let visits = row.visitsLeq, case .decide(let edgesList)? = row.clause else { return }
        if let onBudget = row.onBudget {
            if onBudget == "fail" { return }
            var validTags = Set(row.tags ?? [])
            for e in edgesList { validTags.insert(e.tag) }
            if !validTags.contains(onBudget) {
                issues.append(FlowIssue(row: path, code: "E308", message: (try? ErrorCatalog.fill(
                    code: "E308", values: ["n": path, "tag": onBudget, "tags": (row.tags ?? []).joined(separator: ", ")])) ?? ""))
            }
        } else {
            let exampleTag = row.tags?.first ?? (edgesList.first?.tag ?? "tag")
            issues.append(FlowIssue(row: path, code: "E307", message: (try? ErrorCatalog.fill(
                code: "E307", values: ["n": path, "N": String(visits), "tag": exampleTag], isV08: isV08)) ?? ""))
        }
    }

    // MARK: - Row classification helpers

    static func isHumanRow(_ row: ParsedRow) -> Bool {
        guard let task = row.task, let desc = TaskCatalog.catalog[task] else { return false }
        return desc.taskClass == .human
    }

    static func isTriggerRow(_ row: ParsedRow) -> Bool {
        guard let task = row.task, let desc = TaskCatalog.catalog[task] else { return false }
        return desc.taskClass == .trigger
    }

    static func isImproviseRow(_ row: ParsedRow) -> Bool {
        guard let task = row.task, let desc = TaskCatalog.catalog[task] else { return false }
        return desc.taskClass == .agent
    }

    static func namesAModel(_ row: ParsedRow) -> Bool {
        guard let model = row.model, let task = row.task else { return false }
        if model.count >= 2, model.hasPrefix("{"), model.hasSuffix("}") { return false }
        let desc = TaskCatalog.catalog[task] ?? TaskCatalog.deciderTasks[task]
        return desc?.taskClass == .model
    }

    /// AFM-1 (the trap the plan called out by name): "remote" means *leaves this Mac*, never
    /// "the display name contains ` @ `" — `apple-foundation @ system` uses the same `name @
    /// provider` grammar but runs on-device. A `" @ "` name is remote unless the system
    /// registry (kind-based — `TaskModels.systemDisplayNames`, sourced from the curated
    /// manifest's own `kind: "system"`, never a string test on the display) claims it.
    /// `TaskModels.providerDisplayNames` is checked too, symmetrically, though every `" @ "`
    /// name that isn't a system ref is remote by construction until Phase RM's registry has
    /// entries of its own.
    static func isRemoteRow(_ row: ParsedRow) -> Bool {
        guard namesAModel(row), let model = row.model, model.contains(" @ ") else { return false }
        if TaskModels.systemDisplayNames.contains(model) { return false }
        return true
    }

    static func joinAnd(_ items: [String]) -> String {
        if items.count == 1 { return items[0] }
        if items.count == 2 { return "\(items[0]) and \(items[1])" }
        return items.dropLast().joined(separator: ", ") + ", and \(items.last!)"
    }

    // MARK: - Human rows (E501/E502)

    static func checkHumanRows(_ rows: [ParsedRow], issues: inout [FlowIssue], isV08: Bool) {
        for (path, row) in iterFlowRows(rows) where isHumanRow(row) {
            let kv = parseSettingsKV(row.settings ?? "", isV08: isV08)
            let hasWaitForever = kv["wait"] == "forever"
            let hasTimeoutDefault = kv["timeout"] != nil && kv["default"] != nil
            if !hasWaitForever && !hasTimeoutDefault {
                issues.append(FlowIssue(row: path, code: "E501", message: (try? ErrorCatalog.fill(
                    code: "E501", values: ["n": path, "task": row.task ?? ""], isV08: isV08)) ?? ""))
                continue
            }
            if row.task == "Ask Human", let dflt = kv["default"] {
                let tags = row.tags ?? []
                if !tags.contains(dflt) {
                    issues.append(FlowIssue(row: path, code: "E502", message: (try? ErrorCatalog.fill(
                        code: "E502", values: ["n": path, "value": dflt, "tags": tags.joined(separator: ", ")])) ?? ""))
                }
            }
        }
    }

    // MARK: - Trigger placement (E601/E602/E603)

    static func checkTriggerPlacement(_ rows: [ParsedRow], issues: inout [FlowIssue]) {
        for (i, row) in rows.enumerated() {
            let position = i + 1
            if position == 1 || !isTriggerRow(row) { continue }
            issues.append(FlowIssue(row: String(position), code: "E601", message: (try? ErrorCatalog.fill(
                code: "E601", values: ["trigger": row.task ?? "", "n": String(position)])) ?? ""))
        }

        func scanBlocks(_ children: [ParsedRow], blockPath: String) {
            for (i, child) in children.enumerated() {
                let childPath = "\(blockPath).\(i + 1)"
                if isTriggerRow(child) {
                    issues.append(FlowIssue(row: childPath, code: "E602", message: (try? ErrorCatalog.fill(
                        code: "E602", values: ["block": blockPath, "trigger": child.task ?? ""])) ?? ""))
                }
                if child.blockKind != nil {
                    scanBlocks(child.children, blockPath: childPath)
                }
            }
        }
        for (i, row) in rows.enumerated() where row.blockKind != nil {
            scanBlocks(row.children, blockPath: String(i + 1))
        }

        if let first = rows.first, isTriggerRow(first) {
            for (i, row) in rows.enumerated() {
                let position = i + 1
                for target in clauseTargets(row.clause) where target == .row(number: 1) {
                    issues.append(FlowIssue(row: String(position), code: "E603", message: (try? ErrorCatalog.fill(
                        code: "E603", values: ["n": String(position)])) ?? ""))
                }
            }
        }
    }

    // MARK: - Events flag (E604)

    static func checkEventsFlag(_ rows: [ParsedRow], flags: [String], issues: inout [FlowIssue], isV08: Bool) {
        if let first = rows.first, isTriggerRow(first), !flags.contains("events") {
            issues.append(FlowIssue(row: "1", code: "E604", message: (try? ErrorCatalog.fill(
                code: "E604", isV08: isV08)) ?? ""))
        }
    }

    // MARK: - Rate budget (E605)

    private static let reRunsLeq = NSRegularExpression.compiled("\\bruns\\s*(≤|<=)")
    private static let reMaxRuns = NSRegularExpression.compiled("\\bmax_runs\\s*=")

    static func checkRateBudget(_ rows: [ParsedRow], issues: inout [FlowIssue], isV08: Bool, transforms: [String: TransformDef]) {
        guard let first = rows.first, isTriggerRow(first) else { return }
        let settings = first.settings ?? ""
        if reRunsLeq.firstMatch(in: settings) != nil || reMaxRuns.firstMatch(in: settings) != nil { return }
        var costly: [String] = []
        for (path, row) in iterFlowRows(rows) {
            let desc = TaskCatalog.catalog[row.task ?? ""]
            let isStaged = desc?.taskClass == .staged
            let isNet = desc?.taskClass == .net
            let isTransform = row.task.map { transforms[$0] != nil } ?? false
            if isStaged || isNet || isTransform || isRemoteRow(row) {
                costly.append("\(row.task ?? "?") (row \(path))")
            }
        }
        if !costly.isEmpty {
            issues.append(FlowIssue(row: "1", code: "E605", message: (try? ErrorCatalog.fill(
                code: "E605", values: ["trigger": first.task ?? "", "costly-rows": joinAnd(costly)], isV08: isV08)) ?? ""))
        }
    }

    // MARK: - Models pinned (E104)

    static func checkModelsPinned(_ rows: [ParsedRow], models: [String: String], registry: (any FlowRegistry)?, issues: inout [FlowIssue]) {
        for (path, row) in iterFlowRows(rows) {
            guard namesAModel(row), let model = row.model, models[model] == nil else { continue }
            if let registry, !isRemoteRow(row), registry.resolveDisplay(model) != nil { continue }
            issues.append(FlowIssue(row: path, code: "E104", message: (try? ErrorCatalog.fill(
                code: "E104", values: ["n": path, "display": model])) ?? ""))
        }
    }

    // MARK: - Uses all named (unused-uses-entry)

    static func collectTaskNames(_ rows: [ParsedRow]) -> Set<String> {
        var names = Set<String>()
        for row in rows {
            if row.blockKind == nil, let task = row.task {
                names.insert(task)
            } else {
                names.formUnion(collectTaskNames(row.children))
            }
        }
        return names
    }

    static func checkUsesAllNamed(_ rows: [ParsedRow], uses: [String: String], issues: inout [FlowIssue]) {
        let named = collectTaskNames(rows)
        for (name, rawPath) in uses where !named.contains(name) {
            issues.append(FlowIssue(
                row: "1", code: "unused-uses-entry",
                message: "`uses:` names \"\(name)\" (`\(rawPath)`), but no row calls it. Remove the entry, or add the row that calls it."
            ))
        }
    }
}

extension FlowValidator {

    // MARK: - Refusal registry + pipeline rules

    static let refusalRegistry: [String: String] = [
        "Split Sigmas": "There's no Split Sigmas row — schedule shaping is a setting here: `scheduler=karras` on the Denoise row.",
        "KSamplerSelect": "Samplers are a setting, not a row: `sampler=euler` on Denoise.",
        "Denoise Step": "Denoising steps aren't rows — `steps=30` on Denoise. Rows are decisions, not iterations.",
        "Run Shell": "Arbitrary code runs one declared way: an `Improvise` row, or a `transforms:` entry.",
    ]

    static let pipelineRefused: [String: String] = [
        "Improvise": "a pipeline is a declared-deterministic library unit, and a .cat calling it would inherit nondeterminism it was told was reproducible",
        "On File": "a pipeline is invoked, never armed — a trigger row is row 1 of a flow",
        "On Schedule": "a pipeline is invoked, never armed — a trigger row is row 1 of a flow",
        "On Flow": "a pipeline is invoked, never armed — a trigger row is row 1 of a flow",
        "Stage Send": "committing an outbound effect is a human decision at flow level",
        "Stage Post": "committing an outbound effect is a human decision at flow level",
    ]

    static func shapeTouchesModel(_ shapes: [Shape]) -> Bool {
        for shape in shapes {
            switch shape {
            case .single(let k), .listOf(let k):
                if k == .model { return true }
            case .tupleOf(let ks), .unionOf(let ks):
                if ks.contains(.model) { return true }
            default: break
            }
        }
        return false
    }

    static func checkPipelineRules(_ flow: ParsedFlow, issues: inout [FlowIssue]) {
        if flow.fileKind == .catpipeline {
            if flow.gives == "model" && flow.accepts != nil {
                issues.append(FlowIssue(
                    row: "", code: "E711",
                    message: (try? ErrorCatalog.fill(
                        code: "E711", values: ["name": "unnamed pipeline"], isV08: true)) ?? ""
                ))
            }
            for (path, row) in iterFlowRows(flow.rows) {
                if let reason = row.task.flatMap({ pipelineRefused[$0] }) {
                    issues.append(FlowIssue(row: path, code: "E710", message: (try? ErrorCatalog.fill(
                        code: "E710", values: ["task": row.task ?? "", "reason": reason], isV08: true)) ?? ""))
                }
            }
        } else {
            for (path, row) in iterFlowRows(flow.rows) {
                if let task = row.task, let desc = TaskCatalog.catalog[task],
                   shapeTouchesModel([desc.accepts, desc.gives]) {
                    issues.append(FlowIssue(row: path, code: "E713", message: (try? ErrorCatalog.fill(
                        code: "E713", values: ["task": task], isV08: true)) ?? ""))
                }
            }
            if !flow.presets.isEmpty {
                issues.append(FlowIssue(row: "", code: "E716", message: (try? ErrorCatalog.fill(
                    code: "E716", isV08: true)) ?? ""))
            }
        }
    }
}

// MARK: - Lints (core/validator.py::lint_flow)

extension FlowValidator {

    /// `lint_flow(flow, registry)` — non-blocking style checks.
    static func lintFlow(_ flow: ParsedFlow, registry: (any FlowRegistry)? = nil) -> [FlowLint] {
        var lints: [FlowLint] = []
        if !["0.4", "0.7", "0.8"].contains(flow.version) { return lints }
        for (path, row) in iterFlowRows(flow.rows) {
            lintCallFromOneShot(row: row, path: path, lints: &lints)
            lintForkSuggestsParallel(row: row, path: path, lints: &lints)
            lintRangeLarge(row: row, path: path, lints: &lints)
            lintRemoteDeciderUnconstrained(row: row, path: path, flow: flow, registry: registry, lints: &lints)
        }
        return lints
    }

    static func lintCallFromOneShot(row: ParsedRow, path: String, lints: inout [FlowLint]) {
        guard let task = row.task, TaskCatalog.deciderTasks[task] != nil, task != "Think" else { return }
        let hasCall: Bool
        if case .call = row.clause {
            hasCall = true
        } else if case .decide(let edges)? = row.clause {
            hasCall = edges.contains { if case .call = $0.target { return true }; return false }
        } else {
            hasCall = false
        }
        if hasCall {
            lints.append(FlowLint(
                row: path, code: "call-from-one-shot",
                message: "one-shot routing uses plain edges that converge, not `call` -- only a transcript-keeping decider (Think) can call a tool and get its result back."
            ))
        }
    }

    static func lintForkSuggestsParallel(row: ParsedRow, path: String, lints: inout [FlowLint]) {
        guard case .fork(let targets)? = row.clause else { return }
        let parts = targets.map { t -> String in
            if t == .done { return "done" }
            if case .row(let n) = t { return String(n) }
            return "?"
        }
        lints.append(FlowLint(
            row: path, code: "fork-suggests-parallel",
            message: "Row \(path)'s fork edge (→ \(parts.joined(separator: " & "))) pushes the same output to multiple chains -- a <parallel> block is the user-facing form for this now."
        ))
    }
}

// MARK: - Range bounds (E311) + range-large lint (P6-LC-03)

/// `tools/range_.py` bounds parsing — shared between the validator's E311/range-large
/// and (later) the Range tool, so validate-time and run-time can't disagree.
nonisolated enum FlowRangeBounds {
    static let largeRangeThreshold = 10_000

    struct Bounds: Sendable {
        var start: Int
        var stop: Int
        var step: Int
        func count() -> Int { (stop - start) / step + 1 }
    }

    struct BoundsError: Error, Sendable {
        var bounds: String
        var detail: String
    }

    private static let reBounds = NSRegularExpression.compiled("^(-?\\d+)\\.\\.(-?\\d+)$")
    private static let reInt = NSRegularExpression.compiled("^-?\\d+$")

    /// `parse_bounds(settings_raw)` — the shared parse; throws `BoundsError` on a
    /// malformed/descending range.
    static func parse(_ settingsRaw: String?) throws -> Bounds {
        let s = CatFlowSettings(settingsRaw)
        let literal = s.firstBare() ?? ""
        guard let m = reBounds.firstMatch(in: literal) else {
            throw BoundsError(
                bounds: literal.isEmpty ? "(none)" : literal,
                detail: "isn't `a..b` -- write an inclusive integer range, e.g. `0..99`"
            )
        }
        let start = Int((literal as NSString).substring(with: m.range(at: 1))) ?? 0
        let stop = Int((literal as NSString).substring(with: m.range(at: 2))) ?? 0
        let stepRaw = s.get("step") ?? "1"
        guard reInt.firstMatch(in: stepRaw) != nil else {
            throw BoundsError(bounds: literal, detail: "`step=\(stepRaw)` isn't an integer")
        }
        let step = Int(stepRaw) ?? 1
        if step <= 0 {
            throw BoundsError(bounds: literal, detail: "`step=\(step)` must be positive -- Range never counts down")
        }
        if start > stop {
            throw BoundsError(
                bounds: literal,
                detail: "descends (\(start) > \(stop)) -- Range never infers a negative step; write the ascending range you meant"
            )
        }
        return Bounds(start: start, stop: stop, step: step)
    }

    static func checkRangeBounds(row: ParsedRow, path: String, issues: inout [FlowIssue], isV08: Bool) {
        guard row.task == "Range" else { return }
        do {
            _ = try parse(row.settings)
        } catch let err as BoundsError {
            issues.append(FlowIssue(row: path, code: "E311", message: (try? ErrorCatalog.fill(
                code: "E311", values: ["n": path, "bounds": err.bounds, "detail": err.detail], isV08: isV08)) ?? ""))
        } catch {}
    }
}

extension FlowValidator {
    static func lintRangeLarge(row: ParsedRow, path: String, lints: inout [FlowLint]) {
        guard row.task == "Range" else { return }
        guard let bounds = try? FlowRangeBounds.parse(row.settings) else { return }
        let count = bounds.count()
        if count > FlowRangeBounds.largeRangeThreshold {
            lints.append(FlowLint(
                row: path, code: "range-large",
                message: "Row \(path)'s `Range` produces \(count) items, past the \(FlowRangeBounds.largeRangeThreshold)-item guideline -- if a downstream <each> runs a model per item, that's \(count) model calls. Probably intentional; worth a second look if not."
            ))
        }
    }

    static func lintRemoteDeciderUnconstrained(row: ParsedRow, path: String, flow: ParsedFlow, registry: (any FlowRegistry)?, lints: inout [FlowLint]) {
        guard let registry, let task = row.task, TaskCatalog.deciderTasks[task] != nil, namesAModel(row) else { return }
        let pinnedID = row.model.flatMap { flow.models[$0] }
        let manifest: FlowManifest?
        if let pinnedID {
            manifest = registry.get(pinnedID)
        } else {
            manifest = row.model.flatMap { registry.resolveDisplay($0) }
        }
        guard let manifest, manifest.capabilityBool("constrained_decoding") == false else { return }
        lints.append(FlowLint(row: path, code: "F010", message: (try? ErrorCatalog.fill(
            code: "F010", values: ["n": path, "task": task, "provider": manifest.display], isV08: true)) ?? ""))
    }
}
