import Foundation

/// CAT Flow `.cat` parser — ported from `catflow-mlx/src/catflow/core/parser.py`
/// (1,742 lines) with the same regexes and the same heuristics, so a parsed tree
/// matches the Python byte-for-byte (CFM-R5-2). The parse-tree shape is that of
/// `tests/conftest.py::flow_to_dict` / `conformance/*.expect.json`.
///
/// Parsing is two-phase: the grammar pass produces `ParsedRow`s whose references are
/// still **numeric** (the wire shape), then `resolve` mints a UUID per row in each
/// scope and rewrites refs to ids — the same conversion the pre-parsed-JSON decode
/// path already did (`FlowDocument.resolve`), so the R5 parser and the JSON loader
/// share one identity contract.
///
/// **Version.** `0.8` is the only accepted header version — anything else is E101.
/// Headerless files parse with the v0.7 `·`-separated grammar (a headerless flow has
/// `version == ""`), which is why the v0.7 branches survive here.
nonisolated enum CatParser {

    // MARK: - Task-name lexicon (greedy, longest-first)

    /// Ported verbatim from `core/parser.py::TASK_NAMES`; a row's leading task name is
    /// the longest prefix in this list, else a run of Title-Case words.
    static let taskNames: [String] = [
        // multi-word tasks (before single-word for greedy match)
        "Read Audio", "Read Text", "Read Image", "Read Video",
        "Read Index", "Read Files", "Read Images", "Read PDF", "Read CSV", "Read JSON",
        "Save Audio", "Save Text", "Save Image", "Save Video", "Save Images",
        "Store Index", "Keyword Search", "Join Video", "Join Text",
        "Query Table", "Table to Text", "Text to Table",
        "Store Query", "Store Read", "Store Write",
        "Set Field", "Append Row", "Merge Record",
        "Extract Structured", "Extract Frame", "Extract Audio",
        "Describe Image", "Overlay Text",
        "Generate Image",
        "Edit Image", "Instruct Edit",
        "Inpaint", "Upscale",
        "Contact Sheet",
        "Detect Edges", "Estimate Depth", "Detect Pose",
        "Generate Video", "Generate Sound", "Animate", "Mux",
        "Segment",
        "Load Checkpoint", "Blend", "Bake LoRA", "Pin Model",
        "Init Latent", "Encode Latent", "Decode Latent",
        "Denoise",
        "Split Sigmas", "KSamplerSelect", "Denoise Step", "Run Shell",
        "Ask Human", "Human Input", "Save Context", "Read Context", "Count Context",
        "Stage Send", "Stage Post", "On File", "On Schedule", "On Flow",
        // single-word tasks
        "Transcribe", "Translate", "Summarize", "Rewrite", "Answer", "Ask",
        "Title", "Critique", "Verify", "Revise", "Merge", "Draft",
        "Speak", "OCR", "Watermark", "Resize", "Crop", "Convert", "Trim",
        "Split", "Template", "Filter", "Sort", "Dedupe",
        "Extract", "Count", "Diff",
        "Embed", "Retrieve", "Rerank", "Calculate",
        "Generate", "Decide",
        "Classify", "Gate", "Grade", "Score", "Judge", "Think",
        "Chart",
        "Improvise",
    ]

    static let taskNamesByLen: [String] = taskNames.sorted { $0.count > $1.count }

    /// Known header capability flags — `core/parser.py::_KNOWN_HEADER_FLAGS`.
    static let knownHeaderFlags: Set<String> = ["network", "events", "improvise", "code", "offdevice"]

    private static let reHeader = NSRegularExpression.compiled("^(catflow|catpipeline|mlxflow|mlxpipeline)\\s+(\\S+)((?:\\s*·\\s*\\S+)*)\\s*$")
    private static let reHeaderV08 = NSRegularExpression.compiled("^(catflow|catpipeline|mlxflow|mlxpipeline)\\s+(\\S+)((?:\\s*;\\s*\\S+)*)\\s*$")

    /// `parser.py:346` — the header keyword the reference normalizes to the internal
    /// two-value vocabulary. The raw keyword (pre-normalization) is kept as
    /// `FlowDocument.headerKeyword` / `ParsedFlow.headerKeyword` so `CatSerializer` can
    /// preserve the family the source file wrote (CFM-R18-1).
    private static let headerKeywordAliases: [String: String] = ["mlxflow": "catflow", "mlxpipeline": "catpipeline"]
    private static let reNumbered = NSRegularExpression.compiled("^(\\s*)(\\d+)\\.\\s+(.+?)\\s*$")
    private static let reBlockHeader = NSRegularExpression.compiled("^<(list|each|parallel)(?:\\s+([\\w_-]+))?\\s*(?:·\\s*(.+?))?\\s*>(.*)$", options: [.caseInsensitive])
    private static let reBlockHeaderV08 = NSRegularExpression.compiled("^<(list|each|parallel)(?:\\s+([\\w_-]+))?\\s*(?:;\\s*(.+?))?\\s*(?<!-)>(.*)$", options: [.caseInsensitive])
    private static let reRefs = NSRegularExpression.compiled("\\(([^)]+)\\)")
    private static let reInputRef = NSRegularExpression.compiled("^input:(\\d+)$", options: [.caseInsensitive])
    private static let reParamRef = NSRegularExpression.compiled("^param:([\\w_-]+)$", options: [.caseInsensitive])
    private static let reActivationRefFull = NSRegularExpression.compiled("^(\\d+)\\s*@\\s*(\\d+)$")
    private static let reBlank = NSRegularExpression.compiled("^\\s*$")
    private static let reVisitsLeq = NSRegularExpression.compiled("visits≤(\\d+)")
    private static let reOnBudget = NSRegularExpression.compiled("on_budget=(\\w+)")
    private static let reMaxVisits = NSRegularExpression.compiled("max_visits=(\\d+)")
    private static let reTagsTrailing = NSRegularExpression.compiled("\\s*tags:\\s*(\\w+(?:\\s*,\\s*\\w+)*)\\s*$")
    private static let reQuotedSpan = NSRegularExpression.compiled("\"[^\"]*\"")

    private static let rePresetsHeader = NSRegularExpression.compiled("^presets:\\s*$")
    private static let rePresetEntry = NSRegularExpression.compiled("^(\\S+?)\\s*=\\s*(.*)$")
    private static let reCompositeLine = NSRegularExpression.compiled("^composite\\s+([\\w_-]+)\\s*(?:\\s{2,}(.+))?$")
    private static let reParamToken = NSRegularExpression.compiled("^([\\w_-]+)(?:\\s*\\(([\\w_-]+)\\))?(?:\\s*=\\s*(.+))?$")
    private static let reModelLine = NSRegularExpression.compiled("^(.+?)\\s*=\\s*(\\S.*)$")
    private static let reTransformLine = NSRegularExpression.compiled("^(\\S.*?)\\s{2,}(\\S.*)$")
    private static let reTransformField = NSRegularExpression.compiled("^(run|timeout|workdir|params):\\s*(.*)$")

    private static let reCallWhole = NSRegularExpression.compiled("^call\\s+(\\d+)$")
    private static let reCallLead = NSRegularExpression.compiled("^call\\b")
    private static let reDigits = NSRegularExpression.compiled("^\\d+$")
    private static let reModelAtProvider = NSRegularExpression.compiled("^(\\S+(?:\\s+\\S+)*?)\\s+@\\s+(\\S+)\\s*(.*)$")

    /// Type-signature trail — `core/parser.py::_RE_SIG_TRAIL`. Kind names are the
    /// closed vocabulary from `core/kinds.py` plus `tagged` (SameAsInput, not a Kind).
    private static let kindWord = Kind.allCases.map(\.rawValue).joined(separator: "|") + "|tagged"
    private static let kindList = "(?:\(kindWord))(?:\\s*,\\s*(?:\(kindWord)))*"
    private static let typeRE = "\\[?(?:\(kindList))\\]?"
    private static let reSigTrail = NSRegularExpression.compiled(
        "("
            + "\\s+\(typeRE)\\s*(?:→|->)\\s*\(typeRE)"
            + "|\\s{3,}\\[\(kindList)\\]"
            + ")\\s*$"
    )

    // MARK: - Public API

    /// Parse *text* as a CAT Flow. `0.8` is the only accepted header version (E101
    /// otherwise). Throws `CatParserError` for parse-time failures.
    static func parse(_ text: String) throws -> FlowDocument {
        let lines = text.components(separatedBy: "\n")
        var start = 0
        var version = ""
        var fileKind = "catflow"
        var headerKeyword = "catflow"
        var flags: [String] = []
        var pipelineName: String?
        var accepts: [Kind]?
        var gives: String?
        var params: [ParamDecl] = []
        var presets: [String: PresetDecl] = [:]

        // Optional header on the very first non-blank line.
        for (i, line) in lines.enumerated() {
            if reBlank.firstMatch(in: line) != nil { continue }
            var m = headerMatch(line)
            var sep = "·"
            if m == nil {
                if let m08 = reHeaderV08.firstMatch(in: line),
                   group(m08, in: line, index: 2) == "0.8" {
                    m = m08
                    sep = ";"
                }
            }
            if let m {
                let kind = group(m, in: line, index: 1) ?? ""
                let v = group(m, in: line, index: 2) ?? ""
                if v != "0.8" {
                    throw CatParserError.needsVersion(token: "\(kind) \(v)", version: v, line: i + 1)
                }
                let flagsStr = (group(m, in: line, index: 3) ?? "").trimmingCharacters(in: .whitespaces)
                if !flagsStr.isEmpty {
                    for f in splitOnSeparator(flagsStr, sep: sep) {
                        let f = f.trimmingCharacters(in: .whitespaces)
                        if f.isEmpty { continue }
                        if !knownHeaderFlags.contains(f) {
                            throw CatParserError.unknownHeaderFlag(flag: f, line: i + 1)
                        }
                        flags.append(f)
                    }
                }
                version = v
                headerKeyword = kind
                fileKind = headerKeywordAliases[kind] ?? kind
                start = i + 1
            }
            break  // first non-blank line processed
        }

        let isV08 = version == "0.8"

        // `.catpipeline` declaration block: `pipeline <Name>`, `accepts:`, `gives:`, `params:`.
        if isV08 && fileKind == "catpipeline" {
            while start < lines.count {
                let stripped = lines[start].trimmingCharacters(in: .whitespaces)
                if stripped.hasPrefix("pipeline ") {
                    pipelineName = String(stripped.dropFirst("pipeline ".count)).trimmingCharacters(in: .whitespaces)
                    start += 1
                    continue
                }
                if stripped.hasPrefix("accepts:") {
                    accepts = try parseAcceptsList(String(stripped.dropFirst("accepts:".count)), lineNum: start + 1)
                    start += 1
                    continue
                }
                if stripped.hasPrefix("gives:") {
                    gives = String(stripped.dropFirst("gives:".count)).trimmingCharacters(in: .whitespaces)
                    start += 1
                    continue
                }
                if stripped.hasPrefix("params:") {
                    params = try parseParamsLine(stripped, lineNum: start + 1, isV08: true)
                    start += 1
                    continue
                }
                break
            }
        }

        // v0.8: a used flow's own `params:` line, directly after the header.
        if isV08 && start < lines.count && lines[start].hasPrefix("params:") {
            params = try parseParamsLine(lines[start], lineNum: start + 1, isV08: true)
            start += 1
        }

        // v0.8: `presets:` block, scanned in the pre-rows region.
        var presetsOrder: [String] = []
        if isV08 {
            var j = start
            while j < lines.count {
                let l = lines[j]
                let t = l.trimmingCharacters(in: .whitespaces)
                if t.isEmpty || t.hasPrefix("#") { j += 1 } else { break }
            }
            if j < lines.count, rePresetsHeader.firstMatch(in: lines[j].trimmingCharacters(in: .whitespaces)) != nil {
                (presets, presetsOrder, start) = try parsePresetsBlock(lines, start: j + 1)
            }
        }

        // Truncate the row-list view at the first section header so _parse_rows never
        // has to know sections exist. (Sections are v0.7/v0.8 only.)
        var boundary = lines.count
        if isV08 {
            boundary = findSectionBoundary(lines, start: start)
        }
        let rowLines = Array(lines[..<boundary])

        let parsedRows = try parseRowsRaw(rowLines, start: start, indent: 0, isV08: isV08)

        var definitions: [String: CompositeDef] = [:]
        var uses: [String: String] = [:]
        var models: [String: String] = [:]
        var transforms: [String: TransformDef] = [:]
        var definitionsOrder: [String] = []
        var usesOrder: [String] = []
        var modelsOrder: [String] = []
        var transformsOrder: [String] = []

        if isV08 && boundary < lines.count {
            (definitions, uses, models, transforms,
             definitionsOrder, usesOrder, modelsOrder, transformsOrder) =
                try parseSections(lines, start: boundary, isV08: true)
        }

        return FlowDocument(
            version: version,
            fileKind: fileKind == "catpipeline" ? .catpipeline : .catflow,
            headerKeyword: headerKeyword,
            rows: try resolve(parsedRows),
            flags: Set(flags.compactMap { CapabilityFlag(rawValue: $0) }),
            flagsOrder: flags.compactMap { CapabilityFlag(rawValue: $0) },
            uses: uses,
            models: models,
            transforms: transforms,
            definitions: definitions,
            accepts: accepts,
            gives: gives,
            params: params,
            presets: presets,
            pipelineName: pipelineName,
            modelsOrder: modelsOrder,
            usesOrder: usesOrder,
            definitionsOrder: definitionsOrder,
            transformsOrder: transformsOrder,
            presetsOrder: presetsOrder
        )
    }

    /// Parse *text* and return the raw, **numeric-ref** parse tree plus the document
    /// metadata — the shape the validator (and the golden comparisons) operate on,
    /// before id-resolution. `FlowDocument` is produced from this by `resolve`.
    static func parseForValidation(_ text: String) throws -> ParsedFlow {
        let lines = text.components(separatedBy: "\n")
        var start = 0
        var version = ""
        var fileKind = "catflow"
        var headerKeyword = "catflow"
        var flags: [String] = []
        var accepts: [Kind]?
        var gives: String?
        var params: [ParamDecl] = []
        var presets: [String: PresetDecl] = [:]

        for (i, line) in lines.enumerated() {
            if reBlank.firstMatch(in: line) != nil { continue }
            var m = headerMatch(line)
            var sep = "·"
            if m == nil {
                if let m08 = reHeaderV08.firstMatch(in: line),
                   group(m08, in: line, index: 2) == "0.8" {
                    m = m08
                    sep = ";"
                }
            }
            if let m {
                let kind = group(m, in: line, index: 1) ?? ""
                let v = group(m, in: line, index: 2) ?? ""
                if v != "0.8" {
                    throw CatParserError.needsVersion(token: "\(kind) \(v)", version: v, line: i + 1)
                }
                let flagsStr = (group(m, in: line, index: 3) ?? "").trimmingCharacters(in: .whitespaces)
                if !flagsStr.isEmpty {
                    for f in splitOnSeparator(flagsStr, sep: sep) {
                        let f = f.trimmingCharacters(in: .whitespaces)
                        if f.isEmpty { continue }
                        if !knownHeaderFlags.contains(f) {
                            throw CatParserError.unknownHeaderFlag(flag: f, line: i + 1)
                        }
                        flags.append(f)
                    }
                }
                version = v
                headerKeyword = kind
                fileKind = headerKeywordAliases[kind] ?? kind
                start = i + 1
            }
            break
        }

        let isV08 = version == "0.8"

        if isV08 && fileKind == "catpipeline" {
            while start < lines.count {
                let stripped = lines[start].trimmingCharacters(in: .whitespaces)
                if stripped.hasPrefix("pipeline ") {
                    start += 1
                    continue
                }
                if stripped.hasPrefix("accepts:") {
                    accepts = try parseAcceptsList(String(stripped.dropFirst("accepts:".count)), lineNum: start + 1)
                    start += 1
                    continue
                }
                if stripped.hasPrefix("gives:") {
                    gives = String(stripped.dropFirst("gives:".count)).trimmingCharacters(in: .whitespaces)
                    start += 1
                    continue
                }
                if stripped.hasPrefix("params:") {
                    params = try parseParamsLine(stripped, lineNum: start + 1, isV08: true)
                    start += 1
                    continue
                }
                break
            }
        }

        if isV08 && start < lines.count && lines[start].hasPrefix("params:") {
            params = try parseParamsLine(lines[start], lineNum: start + 1, isV08: true)
            start += 1
        }

        if isV08 {
            var j = start
            while j < lines.count {
                let l = lines[j]
                let t = l.trimmingCharacters(in: .whitespaces)
                if t.isEmpty || t.hasPrefix("#") { j += 1 } else { break }
            }
            if j < lines.count, rePresetsHeader.firstMatch(in: lines[j].trimmingCharacters(in: .whitespaces)) != nil {
                (presets, _, start) = try parsePresetsBlock(lines, start: j + 1)
            }
        }

        var boundary = lines.count
        if isV08 {
            boundary = findSectionBoundary(lines, start: start)
        }
        let rowLines = Array(lines[..<boundary])

        let parsedRows = try parseRowsRaw(rowLines, start: start, indent: 0, isV08: isV08)

        var definitions: [String: CompositeDef] = [:]
        var uses: [String: String] = [:]
        var models: [String: String] = [:]
        var transforms: [String: TransformDef] = [:]

        if isV08 && boundary < lines.count {
            (definitions, uses, models, transforms, _, _, _, _) =
                try parseSections(lines, start: boundary, isV08: true)
        }

        return ParsedFlow(
            version: version,
            fileKind: fileKind == "catpipeline" ? .catpipeline : .catflow,
            headerKeyword: headerKeyword,
            flags: flags,
            rows: parsedRows,
            uses: uses,
            models: models,
            transforms: transforms,
            definitions: definitions,
            accepts: accepts,
            gives: gives,
            params: params,
            presets: presets
        )
    }

    /// Convert a `ParsedFlow` into the id-resolved `FlowDocument` (used by `parse`
    /// and by callers that already hold a `ParsedFlow`).
    static func resolveDocument(_ parsed: ParsedFlow) throws -> FlowDocument {
        FlowDocument(
            version: parsed.version,
            fileKind: parsed.fileKind,
            headerKeyword: parsed.headerKeyword,
            rows: try resolve(parsed.rows),
            flags: Set(parsed.flags.compactMap { CapabilityFlag(rawValue: $0) }),
            uses: parsed.uses,
            models: parsed.models,
            transforms: parsed.transforms,
            definitions: parsed.definitions,
            accepts: parsed.accepts,
            gives: parsed.gives,
            params: parsed.params,
            presets: parsed.presets
        )
    }

    // MARK: - Identity resolution (numbers → ids, per scope)

    /// Mint a UUID per row and rewrite numeric refs to ids, per scope — the same
    /// contract `FlowDocument.resolve` applies to the pre-parsed JSON.
    static func resolve(_ parsed: [ParsedRow], scopeLabel: String = "the flow") throws -> [Row] {
        let ids = parsed.enumerated().map { (number: $0.offset + 1, id: UUID()) }
        let byNumber = Dictionary(ids.map { ($0.number, $0.id) }, uniquingKeysWith: { a, _ in a })
        return try parsed.enumerated().map { index, p in
            let refs = try p.refs.map { ref -> Ref in
                switch ref {
                case .row(let number):
                    guard let targetID = byNumber[number] else {
                        throw FlowDocumentError.missingReferenceTarget(number, scope: scopeLabel)
                    }
                    return .rowRef(targetID)
                case .input(let position):
                    return .inputRef(position)
                case .param(let name):
                    return .paramRef(name)
                }
            }
            let childRows = try resolve(p.children, scopeLabel: "the block '\(p.blockName ?? "")'")
            return Row(
                id: ids[index].id,
                task: p.task,
                blockKind: p.blockKind,
                blockName: p.blockName,
                model: p.model,
                settings: p.settings,
                refs: refs,
                chainBreak: p.chainBreak,
                children: childRows,
                clause: p.clause,
                tags: p.tags,
                visitsLeq: p.visitsLeq,
                onBudget: p.onBudget,
                declaredSignature: p.declaredSignature,
                comment: p.comment,
                leadingComments: p.leadingComments
            )
        }
    }

    // MARK: - Header helpers

    private static func headerMatch(_ line: String) -> NSTextCheckingResult? {
        reHeader.firstMatch(in: line)
    }

    private static func group(_ m: NSTextCheckingResult, in text: String, index: Int) -> String? {
        let r = m.range(at: index)
        if r.location == NSNotFound { return nil }
        return (text as NSString).substring(with: r)
    }

    /// Quote-aware split on *sep* (SPEC-Q8): a default value's own quoted string never
    /// gets split on a separator it happens to contain.
    private static func splitOnSeparator(_ text: String, sep: String) -> [String] {
        let scan = stripQuoted(text)
        var parts: [String] = []
        var last = scan.startIndex
        var i = scan.startIndex
        while i < scan.endIndex {
            if String(scan[i]) == sep {
                parts.append(String(text[text.index(text.startIndex, offsetBy: scan.distance(from: scan.startIndex, to: last))..<text.index(text.startIndex, offsetBy: scan.distance(from: scan.startIndex, to: i))]))
                last = scan.index(after: i)
            }
            i = scan.index(after: i)
        }
        parts.append(String(text[text.index(text.startIndex, offsetBy: scan.distance(from: scan.startIndex, to: last))..<text.endIndex]))
        return parts
    }

    /// Blank out double-quoted spans, preserving length so columns stay valid.
    static func stripQuoted(_ line: String) -> String {
        var chars = Array(line)
        let ns = line as NSString
        for match in reQuotedSpan.matches(in: line, range: NSRange(location: 0, length: ns.length)) {
            for i in match.range.location..<(match.range.location + match.range.length) where i < chars.count {
                chars[i] = " "
            }
        }
        return String(chars)
    }

    /// True if *content* has an unclosed quoted span (SPEC §2.4).
    static func quoteParityOdd(_ content: String) -> Bool {
        var inQuote = false
        let chars = Array(content)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c == "\\", i + 1 < chars.count, chars[i + 1] == "\"" {
                i += 2
                continue
            }
            if c == "\"" { inQuote.toggle() }
            i += 1
        }
        return inQuote
    }

    /// Split off a trailing `# comment` — quote-aware (a `#` in quotes is content).
    private static func splitTrailingComment(_ content: String) -> (String, String?) {
        let scan = stripQuoted(content)
        if let idx = scan.firstIndex(of: "#") {
            let comment = String(content[content.index(after: idx)...]).trimmingCharacters(in: .whitespaces)
            let rest = String(content[..<idx]).trimmingCharacters(in: .whitespaces)
            return (rest, comment.isEmpty ? nil : comment)
        }
        return (content, nil)
    }

    // MARK: - Sections

    private static func findSectionBoundary(_ lines: [String], start: Int) -> Int {
        for i in start..<lines.count where sectionHeaderName(lines[i]) != nil {
            return i
        }
        return lines.count
    }

    private static func sectionHeaderName(_ line: String) -> String? {
        let t = line.trimmingCharacters(in: .whitespaces)
        for name in ["definitions", "uses", "models", "transforms"] {
            if t == "\(name):" { return name }
        }
        return nil
    }

    private static func parseSections(
        _ lines: [String], start: Int, isV08: Bool
    ) throws -> ([String: CompositeDef], [String: String], [String: String], [String: TransformDef],
                 [String], [String], [String], [String]) {
        var definitions: [String: CompositeDef] = [:]
        var uses: [String: String] = [:]
        var models: [String: String] = [:]
        var transforms: [String: TransformDef] = [:]
        var definitionsOrder: [String] = []
        var usesOrder: [String] = []
        var modelsOrder: [String] = []
        var transformsOrder: [String] = []
        var i = start
        while i < lines.count {
            let line = lines[i]
            if line.trimmingCharacters(in: .whitespaces).isEmpty || line.ltrimmedPrefix("#") != nil {
                i += 1
                continue
            }
            switch sectionHeaderName(line) {
            case "definitions":
                i += 1
                (definitions, definitionsOrder, i) = try parseDefinitionsSection(lines, start: i, isV08: isV08)
            case "uses" where isV08:
                i += 1
                (uses, usesOrder, i) = try parseUsesSection(lines, start: i, isV08: isV08)
            case "models":
                i += 1
                (models, modelsOrder, i) = try parseModelsSection(lines, start: i, isV08: isV08)
            case "transforms" where isV08:
                i += 1
                (transforms, transformsOrder, i) = try parseTransformsSection(lines, start: i, isV08: isV08)
            default:
                let detail = isV08
                    ? "expected `definitions:`, `uses:`, `models:`, or `transforms:`"
                    : "expected `definitions:` or `models:`"
                throw CatParserError.unparseableRow(
                    line: i + 1, fragment: String(line.prefix(40)), detail: detail, isV08: isV08
                )
            }
        }
        return (definitions, uses, models, transforms,
                definitionsOrder, usesOrder, modelsOrder, transformsOrder)
    }

    private static func parseDefinitionsSection(
        _ lines: [String], start: Int, isV08: Bool
    ) throws -> ([String: CompositeDef], [String], Int) {
        var definitions: [String: CompositeDef] = [:]
        var order: [String] = []
        var i = start
        while i < lines.count {
            let line = lines[i]
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                i += 1
                continue
            }
            if !line.hasPrefix(" ") && !line.hasPrefix("\t") { break }
            let stripped = line.trimmingCharacters(in: .whitespaces)
            if stripped.hasPrefix("#") {
                i += 1
                continue
            }
            if isV08 { try checkPreV08Chars(line, lineno: i + 1, section: "definitions") }
            let (content, _) = splitTrailingComment(stripped)
            guard let cm = reMatch(reCompositeLine, text: content.trimmingCharacters(in: .whitespaces)) else {
                throw CatParserError.unparseableRow(
                    line: i + 1, fragment: String(stripped.prefix(40)),
                    detail: "expected `composite Name` inside definitions:", isV08: isV08
                )
            }
            let name = cm[1] ?? ""
            let signature = cm[2]?.trimmingCharacters(in: .whitespaces)
            i += 1

            var params: [ParamDecl] = []
            if i < lines.count && lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("params:") {
                params = try parseParamsLine(lines[i], lineNum: i + 1, isV08: isV08)
                i += 1
            }

            let (_, bodyLines, consumed) = collectBlockBody(lines, start: i, isV08: isV08)
            let bodyRows = bodyLines.isEmpty ? [] : try resolve(try parseRowsRaw(bodyLines, start: 0, indent: 0, isV08: isV08))
            i += consumed

            if definitions[name] == nil { order.append(name) }
            definitions[name] = CompositeDef(name: name, signature: signature, params: params, rows: bodyRows)
        }
        return (definitions, order, i)
    }

    private static func parsePresetsBlock(_ lines: [String], start: Int) throws -> ([String: PresetDecl], [String], Int) {
        var presets: [String: PresetDecl] = [:]
        var order: [String] = []
        var i = start
        while i < lines.count {
            let stripped = lines[i].trimmingCharacters(in: .whitespaces)
            if stripped.isEmpty || stripped.hasPrefix("#") { break }
            try checkPreV08Chars(lines[i], lineno: i + 1, section: "presets")
            let (content, _) = splitTrailingComment(stripped)
            guard let m = reMatch(rePresetEntry, text: content.trimmingCharacters(in: .whitespaces)) else { break }
            let name = (m[1] ?? "").trimmingCharacters(in: .whitespaces)
            let rhs = (m[2] ?? "").trimmingCharacters(in: .whitespaces)
            var bindings: [PresetBinding] = []
            for part in splitOnSeparator(rhs, sep: ";") {
                let part = part.trimmingCharacters(in: .whitespaces)
                if part.isEmpty { continue }
                guard let eq = part.firstIndex(of: "=") else {
                    throw CatParserError.unparseableRow(
                        line: i + 1, fragment: String(part.prefix(40)),
                        detail: "expected `key=value` inside a preset entry", isV08: true
                    )
                }
                let key = String(part[..<eq]).trimmingCharacters(in: .whitespaces)
                let value = String(part[part.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
                bindings.append(PresetBinding(key: key, value: value))
            }
            if presets[name] == nil { order.append(name) }
            presets[name] = PresetDecl(name: name, bindings: bindings)
            i += 1
        }
        return (presets, order, i)
    }

    private static func parseAcceptsList(_ text: String, lineNum: Int) throws -> [Kind] {
        let text = text.trimmingCharacters(in: .whitespaces)
        guard text.hasPrefix("[") && text.hasSuffix("]") else {
            throw CatParserError.parse(line: lineNum, message: "`accepts:` must be a bracketed list like `[image, text]`")
        }
        let inner = String(text.dropFirst().dropLast())
        let names = inner.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        return try names.map { name in
            guard let kind = Kind(rawValue: name) else {
                throw CatParserError.parse(line: lineNum, message: "unknown kind in `accepts:` -- \(inner)")
            }
            return kind
        }
    }

    private static func parseParamsLine(_ line: String, lineNum: Int, isV08: Bool) throws -> [ParamDecl] {
        if isV08 { try checkPreV08Chars(line, lineno: lineNum, section: "params") }
        let (content, _) = splitTrailingComment(line.trimmingCharacters(in: .whitespaces))
        let contentS = content.trimmingCharacters(in: .whitespaces)
        guard contentS.hasPrefix("params:") else {
            throw CatParserError.unparseableRow(
                line: lineNum, fragment: String(contentS.prefix(40)),
                detail: "expected a `params:` line", isV08: isV08
            )
        }
        let rest = String(contentS.dropFirst("params:".count)).trimmingCharacters(in: .whitespaces)
        var params: [ParamDecl] = []
        for part in splitOnSeparator(rest, sep: isV08 ? ";" : "·") {
            let part = part.trimmingCharacters(in: .whitespaces)
            if part.isEmpty { continue }
            guard let pm = reMatch(reParamToken, text: part) else {
                throw CatParserError.unparseableRow(
                    line: lineNum, fragment: String(part.prefix(40)), detail: "malformed params: entry", isV08: isV08
                )
            }
            params.append(ParamDecl(name: pm[1] ?? "", kind: pm[2], defaultValue: pm[3]))
        }
        return params
    }

    private static func parseModelsSection(_ lines: [String], start: Int, isV08: Bool) throws -> ([String: String], [String], Int) {
        var models: [String: String] = [:]
        var order: [String] = []
        var i = start
        while i < lines.count {
            let line = lines[i]
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                i += 1
                continue
            }
            if !line.hasPrefix(" ") && !line.hasPrefix("\t") { break }
            let stripped = line.trimmingCharacters(in: .whitespaces)
            if stripped.hasPrefix("#") {
                i += 1
                continue
            }
            if isV08 { try checkPreV08Chars(line, lineno: i + 1, section: "models") }
            let (content, _) = splitTrailingComment(stripped)
            guard let m = reMatch(reModelLine, text: content.trimmingCharacters(in: .whitespaces)) else {
                throw CatParserError.unparseableRow(
                    line: i + 1, fragment: String(stripped.prefix(40)),
                    detail: "expected `display name = pinned-id` inside models:", isV08: isV08
                )
            }
            let name = (m[1] ?? "").trimmingCharacters(in: .whitespaces)
            if models[name] == nil { order.append(name) }
            models[name] = (m[2] ?? "").trimmingCharacters(in: .whitespaces)
            i += 1
        }
        return (models, order, i)
    }

    private static func parseUsesSection(_ lines: [String], start: Int, isV08: Bool) throws -> ([String: String], [String], Int) {
        var uses: [String: String] = [:]
        var order: [String] = []
        var i = start
        while i < lines.count {
            let line = lines[i]
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                i += 1
                continue
            }
            if !line.hasPrefix(" ") && !line.hasPrefix("\t") { break }
            let stripped = line.trimmingCharacters(in: .whitespaces)
            if stripped.hasPrefix("#") {
                i += 1
                continue
            }
            try checkPreV08Chars(line, lineno: i + 1, section: "uses")
            let (content, _) = splitTrailingComment(stripped)
            guard let m = reMatch(reModelLine, text: content.trimmingCharacters(in: .whitespaces)) else {
                throw CatParserError.unparseableRow(
                    line: i + 1, fragment: String(stripped.prefix(40)),
                    detail: "expected `Name = ./path.cat` inside uses:", isV08: isV08
                )
            }
            let name = (m[1] ?? "").trimmingCharacters(in: .whitespaces)
            if uses[name] == nil { order.append(name) }
            uses[name] = (m[2] ?? "").trimmingCharacters(in: .whitespaces)
            i += 1
        }
        return (uses, order, i)
    }

    private static func parseTransformsSection(_ lines: [String], start: Int, isV08: Bool) throws -> ([String: TransformDef], [String], Int) {
        var transforms: [String: TransformDef] = [:]
        var order: [String] = []
        var current: TransformDef?
        var i = start
        while i < lines.count {
            let line = lines[i]
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                i += 1
                continue
            }
            if !line.hasPrefix(" ") && !line.hasPrefix("\t") { break }
            let stripped = line.trimmingCharacters(in: .whitespaces)
            if stripped.hasPrefix("#") {
                i += 1
                continue
            }
            try checkPreV08Chars(line, lineno: i + 1, section: "transforms")
            let (content, _) = splitTrailingComment(stripped)
            let contentS = content.trimmingCharacters(in: .whitespaces)

            if let fm = reMatch(reTransformField, text: contentS) {
                guard var cur = current else {
                    throw CatParserError.unparseableRow(
                        line: i + 1, fragment: String(stripped.prefix(40)),
                        detail: "a `transforms:` field needs a preceding `Name  in -> out` entry line", isV08: isV08
                    )
                }
                let key = fm[1] ?? ""
                let value = (fm[2] ?? "").trimmingCharacters(in: .whitespaces)
                switch key {
                case "run": cur.run = value
                case "timeout": cur.timeout = value
                case "workdir": cur.workdir = value
                default: cur.params = try parseParamsLine("params: \(value)", lineNum: i + 1, isV08: isV08)
                }
                current = cur
                transforms[cur.name] = cur
                i += 1
                continue
            }

            guard let m = reMatch(reTransformLine, text: contentS) else {
                throw CatParserError.unparseableRow(
                    line: i + 1, fragment: String(stripped.prefix(40)),
                    detail: "expected `Name  in -> out` inside transforms:", isV08: isV08
                )
            }
            let name = (m[1] ?? "").trimmingCharacters(in: .whitespaces)
            let signature = (m[2] ?? "").trimmingCharacters(in: .whitespaces)
            current = TransformDef(name: name, signature: signature, params: [])
            if transforms[name] == nil { order.append(name) }
            transforms[name] = current
            i += 1
        }
        return (transforms, order, i)
    }

    // MARK: - Row parsing

    private static func parseRowsRaw(_ lines: [String], start: Int, indent: Int, isV08: Bool) throws -> [ParsedRow] {
        var rows: [ParsedRow] = []
        var pendingBreak = false
        var pendingComments: [String] = []
        var seenPositions: Set<Int> = []
        var lastRowNumber: Int?
        var i = start

        while i < lines.count {
            let line = lines[i]

            // Whole-line comment: never touches pending_break, collected for the next row.
            if line.ltrimmedPrefix("#") != nil {
                pendingComments.append(line.trimmingCharacters(in: .whitespaces).dropFirst().trimmingCharacters(in: .whitespaces))
                i += 1
                continue
            }

            // Blank line -> mark next row as chain_break.
            if reBlank.firstMatch(in: line) != nil {
                pendingBreak = true
                i += 1
                continue
            }

            guard let m = reNumbered.firstMatch(in: line) else {
                // Non-numbered, non-blank: under v0.8, the E108 scan runs before the
                // clause-line attempt.
                if isV08 {
                    try checkPreV08Chars(line, lineno: i + 1, row: lastRowNumber)
                }
                if isV08 && !rows.isEmpty {
                    if let clause = try tryTryParseClauseLine(line, lineNum: i + 1) {
                        rows[rows.count - 1].clause = clause
                        i += 1
                        continue
                    }
                }
                throw CatParserError.unparseableRow(
                    line: i + 1, fragment: String(line.trimmingCharacters(in: .whitespaces).prefix(40)),
                    detail: "this doesn't start with a row number", isV08: isV08
                )
            }

            let lineIndent = group(m, in: line, index: 1)?.count ?? 0

            // If we're inside a block and this line's indent is less, the block ended.
            if indent > 0 && lineIndent < indent { break }
            if indent == 0 && lineIndent > 0 {
                throw CatParserError.unparseableRow(
                    line: i + 1, fragment: String(line.trimmingCharacters(in: .whitespaces).prefix(40)),
                    detail: "this row is indented but no block is open here", isV08: isV08
                )
            }

            let position = Int(group(m, in: line, index: 2) ?? "") ?? 0

            if isV08 {
                try checkPreV08Chars(line, lineno: i + 1, row: position)
            }

            if seenPositions.contains(position) {
                throw CatParserError.duplicatePosition(number: position, line: i + 1)
            }
            seenPositions.insert(position)
            lastRowNumber = position

            let rowStartLine = i + 1
            var (rowContent, comment) = splitTrailingComment(group(m, in: line, index: 3) ?? "")
            var comments: [String] = comment.map { [$0] } ?? []
            i += 1

            // R13: merge wrap-continuation lines onto rowContent.
            if isV08 {
                var consumedWrap = 0
                (rowContent, comments, consumedWrap) = try consumeWrappedContinuation(
                    lines, start: i, rowIndent: lineIndent, content: rowContent, comments: comments,
                    isV08: isV08, row: position
                )
                i += consumedWrap
            }

            if quoteParityOdd(rowContent) {
                throw CatParserError.unclosedQuote(line: rowStartLine)
            }

            var row = try parseRowContent(rowContent, lineNum: rowStartLine, isV08: isV08)
            row.comment = comments.isEmpty ? nil : comments.joined(separator: " ")
            row.leadingComments = pendingComments
            pendingComments = []
            row.chainBreak = pendingBreak && !rows.isEmpty
            pendingBreak = false

            // If this is a block row, collect its indented body.
            if row.blockKind != nil {
                let (_, bodyLines, consumed) = collectBlockBody(lines, start: i, isV08: isV08)
                if !bodyLines.isEmpty {
                    row.children = try parseRowsRaw(bodyLines, start: 0, indent: 0, isV08: isV08)
                }
                i += consumed
            }

            rows.append(row)
        }

        return rows
    }

    private static func parseRowContent(_ content: String, lineNum: Int, isV08: Bool) throws -> ParsedRow {
        var row = ParsedRow()
        let sep = isV08 ? ";" : "·"

        // --- Block row? ---
        let blockRe = isV08 ? reBlockHeaderV08 : reBlockHeader
        if let bm = blockRe.firstMatch(in: content) {
            row.blockKind = BlockKind(rawValue: group(bm, in: content, index: 1)!.lowercased())
            row.blockName = group(bm, in: content, index: 2)
            let insideMarkers = (group(bm, in: content, index: 3) ?? "").trimmingCharacters(in: .whitespaces)
            let tailText = group(bm, in: content, index: 4) ?? ""
            let (strippedTail, sig) = extractSigTrail(tailText)
            row.declaredSignature = sig
            var trailing = strippedTail.trimmingCharacters(in: .whitespaces)
            if !trailing.isEmpty {
                let (refs, rest) = try extractRefs(trailing, lineNum: lineNum, isV08: isV08, anchorEnd: 0)
                row.refs = refs
                trailing = rest.trimmingCharacters(in: .whitespaces)
                if trailing.hasPrefix(sep) {
                    trailing = String(trailing.dropFirst()).trimmingCharacters(in: .whitespaces)
                }
            }
            let parts = [insideMarkers, trailing].filter { !$0.isEmpty }
            row.settings = parts.isEmpty ? nil : parts.joined(separator: " \(sep) ")
            return row
        }

        // --- Regular row: strip type signature from the end. ---
        var (contentNoSig, sig) = extractSigTrail(content)
        row.declaredSignature = sig
        contentNoSig = contentNoSig.trimmingCharacters(in: .whitespaces)

        // --- Extract refs (prefer the group right after the task name). ---
        let taskEnd = taskNameEnd(contentNoSig, isV08: isV08)
        let (refs, contentNoRefs) = try extractRefs(contentNoSig, lineNum: lineNum, isV08: isV08, anchorEnd: taskEnd)
        row.refs = refs
        var rest = contentNoRefs

        // --- Extract row-level options: max_visits=N / on_budget=X (v0.8). ---
        if isV08 {
            var visits: Int?
            var onBudget: String?
            (rest, visits, onBudget) = extractRowOptions(rest, isV08: isV08)
            row.visitsLeq = visits
            row.onBudget = onBudget
        }

        // --- Split task / model / settings. ---
        let (task, model, settings) = try splitTaskModelSettings(rest, lineNum: lineNum, isV08: isV08)
        row.task = task
        row.model = model.isEmpty ? nil : model

        // --- Lift a declared tag set out of settings (v0.8). ---
        if isV08, !settings.isEmpty {
            var tags: [String]?
            var settingsOut = settings
            (settingsOut, tags) = extractTags(settingsOut, isV08: isV08)
            if let tags, !tags.isEmpty {
                row.tags = tags
            }
            row.settings = settingsOut.isEmpty ? nil : settingsOut
        } else {
            row.settings = settings.isEmpty ? nil : settings
        }

        return row
    }

    /// Split *text*'s optional trailing type signature off the end (R12, §3.3).
    private static func extractSigTrail(_ text: String) -> (String, String?) {
        let ns = text as NSString
        guard let match = reSigTrail.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) else {
            return (text, nil)
        }
        let start = match.range.location
        let sig = group(match, in: text, index: 1)?.trimmingCharacters(in: .whitespaces)
        let rest = (ns.substring(with: NSRange(location: 0, length: start)))
        return (rest, sig)
    }

    private static func taskNameEnd(_ content: String, isV08: Bool) -> Int {
        let lstripped = content.trimmingCharacters(in: .whitespaces)
        let leadWS = content.count - lstripped.count
        let sep = isV08 ? ";" : "·"

        for name in taskNamesByLen {
            if lstripped == name
                || lstripped.hasPrefix(name + " ")
                || lstripped.hasPrefix(name + "\t") {
                return leadWS + name.count
            }
        }

        // Fall back: a run of leading Title-Case words.
        var taskWords: [String] = []
        for w in lstripped.split(separator: " ") {
            let stripped = String(w).trimmingCharacters(in: CharacterSet(charactersIn: sep)).trimmingCharacters(in: .whitespaces)
            if let first = stripped.first, first.isUppercase {
                taskWords.append(String(w))
            } else {
                break
            }
        }
        if !taskWords.isEmpty {
            return leadWS + taskWords.joined(separator: " ").count
        }
        return 0
    }

    private static func splitTaskModelSettings(_ content: String, lineNum: Int, isV08: Bool) throws -> (String, String, String) {
        let sep = isV08 ? ";" : "·"
        var content = content.trimmingCharacters(in: .whitespaces)

        let taskEnd = taskNameEnd(content, isV08: isV08)
        let task: String
        if taskEnd > 0 {
            task = String(content.prefix(taskEnd))
            content = String(content.dropFirst(taskEnd)).trimmingCharacters(in: .whitespaces)
        } else {
            throw CatParserError.unparseableRow(
                line: lineNum, fragment: String(content.prefix(40)),
                detail: "no task name (a Title Case word or a known task) at the start", isV08: isV08
            )
        }

        if content.hasPrefix(sep) {
            content = String(content.dropFirst()).trimmingCharacters(in: .whitespaces)
        }

        let model: String
        let settings: String
        if content.hasPrefix("\"") {
            model = ""
            settings = content
        } else if isV08, let idx = stripQuoted(content).firstIndex(of: ";") {
            let modelPart = String(content[..<idx])
            let settingsPart = String(content[content.index(after: idx)...])
            let probe = splitModelSettings(modelPart.trimmingCharacters(in: .whitespaces))
            if !probe.model.isEmpty && probe.settings.isEmpty {
                model = probe.model
                settings = settingsPart.trimmingCharacters(in: .whitespaces)
            } else {
                model = ""
                settings = content
            }
        } else if !isV08, let idx = content.firstIndex(of: "·") {
            let modelPart = String(content[..<idx])
            let settingsPart = String(content[content.index(after: idx)...])
            let probe = splitModelSettings(modelPart.trimmingCharacters(in: .whitespaces))
            if !probe.model.isEmpty && probe.settings.isEmpty {
                model = probe.model
                settings = settingsPart.trimmingCharacters(in: .whitespaces)
            } else {
                model = ""
                settings = content
            }
        } else if content.contains("=") && !(content.first?.isUppercase ?? false) {
            model = ""
            settings = content
        } else {
            let probe = splitModelSettings(content)
            model = probe.model
            settings = probe.settings
        }

        return (task, model.trimmingCharacters(in: .whitespaces), settings.trimmingCharacters(in: .whitespaces))
    }

    /// Split 'content' into (model, settings) by heuristic — `core/parser.py:1499–1586`,
    /// the exact rules the backlog names as the place this goes wrong.
    private static func splitModelSettings(_ content: String) -> (model: String, settings: String) {
        if content.isEmpty { return ("", "") }

        // `name @ provider` — the provider is unconditionally part of the model spec.
        if let atMatch = reModelAtProvider.firstMatch(in: content),
           !(group(atMatch, in: content, index: 1)?.hasPrefix("\"") ?? true) {
            let model = "\(group(atMatch, in: content, index: 1) ?? "") @ \(group(atMatch, in: content, index: 2) ?? "")"
            return (model, group(atMatch, in: content, index: 3) ?? "")
        }

        let tokens = content.split(separator: " ").map(String.init)
        var modelWords: [String] = []
        for tok in tokens {
            if tok.hasPrefix("\"") { break }
            if tok.contains("=") { break }
            if tok.hasPrefix("/") { break }
            if tok.range(of: "^\\w+://", options: .regularExpression) != nil { break }
            if tok.range(of: "^-?\\d+\\.\\.-?\\d+$", options: .regularExpression) != nil { break }
            if tok.range(of: "\\.\\w{2,4}$", options: .regularExpression) != nil
                && tok.range(of: "^v\\d", options: [.regularExpression, .caseInsensitive]) == nil
                && tok.range(of: "^\\d+\\.\\d", options: .regularExpression) == nil {
                break
            }
            if tok.contains("/") && !tok.hasSuffix("/") {
                modelWords.append(tok)
                continue
            }
            if let first = tok.first, first.isLowercase,
               tok.range(of: "^v\\d", options: [.regularExpression, .caseInsensitive]) == nil {
                break
            }
            modelWords.append(tok)
        }

        let model = modelWords.joined(separator: " ")
        let settings = tokens.dropFirst(modelWords.count).joined(separator: " ")
        return (model, settings)
    }

    /// Pull `max_visits=N` (v0.8) and `on_budget=X` out of *content* — quote-aware.
    private static func extractRowOptions(_ content: String, isV08: Bool) -> (String, Int?, String?) {
        let scan = stripQuoted(content)
        let sep = isV08 ? ";" : "·"
        var visits: Int?
        var onBudget: String?
        var removals: [(Int, Int)] = []

        let visitsPattern = isV08 ? reMaxVisits : reVisitsLeq
        if let m = visitsPattern.firstMatch(in: scan) {
            visits = Int(group(m, in: scan, index: 1) ?? "")
            removals.append((extendLeftOverDot(scan, start: m.range.location, sep: sep), m.range.location + m.range.length))
        }

        if let m = reOnBudget.firstMatch(in: scan) {
            onBudget = group(m, in: scan, index: 1)
            removals.append((extendLeftOverDot(scan, start: m.range.location, sep: sep), m.range.location + m.range.length))
        }

        if removals.isEmpty {
            return (content, nil, nil)
        }

        var out = content
        for (rstart, rend) in removals.sorted(by: { $0.0 > $1.0 }) {
            let startIndex = out.index(out.startIndex, offsetBy: rstart)
            let endIndex = out.index(out.startIndex, offsetBy: rend)
            out.removeSubrange(startIndex..<endIndex)
        }
        return (out.trimmingCharacters(in: .whitespaces), visits, onBudget)
    }

    /// Extend a removal span's start leftward over an immediately preceding *sep*
    /// decoration plus whitespace, so stripping never leaves a dangling separator.
    private static func extendLeftOverDot(_ text: String, start: Int, sep: String) -> Int {
        guard start > 0 else { return start }
        let chars = Array(text)
        var dotPos = start - 1
        var found = false
        while dotPos >= 0 {
            if String(chars[dotPos]) == sep {
                found = true
                break
            }
            if !chars[dotPos].isWhitespace { return start }
            dotPos -= 1
        }
        if !found { return start }
        return dotPos
    }

    /// Lift a declared `tags: a, b` clause out of *settings* — trailing inside the LAST
    /// quoted span, or a bare trailing settings token (SPEC-Q2, SPEC-Q29).
    private static func extractTags(_ settings: String, isV08: Bool) -> (String, [String]?) {
        // Form (a): trailing, inside the LAST quoted span.
        let ns = settings as NSString
        let matches = reQuotedSpan.matches(in: settings, range: NSRange(location: 0, length: ns.length))
        if let last = matches.last {
            let inner = ns.substring(with: NSRange(location: last.range.location + 1, length: last.range.length - 2))
            if let tm = reTagsTrailing.firstMatch(in: inner) {
                let tags = (group(tm, in: inner, index: 1) ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                let newInner = inner.prefix(inner.count - tm.range.length).trimmingCharacters(in: .whitespaces)
                let newQuoted = "\"\(newInner)\""
                var out = ns.substring(with: NSRange(location: 0, length: last.range.location))
                out += newQuoted
                out += ns.substring(from: last.range.location + last.range.length)
                return (out, tags)
            }
        }

        // Form (b): bare trailing token outside any quotes.
        let scan = stripQuoted(settings)
        if let tm = reTagsTrailing.firstMatch(in: scan) {
            let tags = (group(tm, in: scan, index: 1) ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            let rstart = extendLeftOverDot(scan, start: tm.range.location, sep: isV08 ? ";" : "·")
            var out = settings
            let startIndex = out.index(out.startIndex, offsetBy: rstart)
            let endIndex = out.index(out.startIndex, offsetBy: tm.range.location + tm.range.length)
            out.removeSubrange(startIndex..<endIndex)
            return (out.trimmingCharacters(in: .whitespaces), tags)
        }

        return (settings, nil)
    }

    /// Pull the refs group from content — `core/parser.py::_extract_refs` (P6-RF-02's
    /// boundary + anchor_end rules). Returns (refs, content_without_refs).
    private static func extractRefs(_ content: String, lineNum: Int, isV08: Bool, anchorEnd: Int) throws -> ([ParsedRef], String) {
        let scan = stripQuoted(content)
        let sep = isV08 ? ";" : "·"
        let ns = content as NSString

        let allMatches = reRefs.matches(in: scan, range: NSRange(location: 0, length: ns.length))
        if allMatches.isEmpty {
            return ([], content)
        }

        func isBoundary(_ pos: Int) -> Bool {
            if pos < 0 || pos >= content.count { return true }
            let ch = Array(content)[pos]
            return ch.isWhitespace || String(ch) == sep
        }

        var candidates: [(NSRange, [ParsedRef])] = []
        for m in allMatches {
            let inner = ns.substring(with: NSRange(location: m.range.location + 1, length: m.range.length - 2))
            if let refs = try parseRefList(inner, lineNum: lineNum, isV08: isV08) {
                if isBoundary(m.range.location - 1) && isBoundary(m.range.location + m.range.length) {
                    candidates.append((m.range, refs))
                }
            }
        }

        if candidates.isEmpty {
            return ([], content)
        }

        let chosen: NSRange
        let refs: [ParsedRef]
        if candidates.count == 1 {
            chosen = candidates[0].0
            refs = candidates[0].1
        } else {
            let leading = candidates.filter { c in
                let gapStart = max(anchorEnd, 0)
                let gapLen = c.0.location - gapStart
                if gapLen <= 0 { return true }
                let gap = ns.substring(with: NSRange(location: gapStart, length: gapLen))
                return gap.trimmingCharacters(in: .whitespaces).isEmpty
            }
            if !leading.isEmpty {
                throw CatParserError.unparseableRow(
                    line: lineNum,
                    fragment: String(content.prefix(40)),
                    detail: "two reference lists -- one right after the task name, one later in the row. Keep references in one place, immediately after the task name",
                    isV08: isV08
                )
            }
            chosen = candidates[candidates.count - 1].0
            refs = candidates[candidates.count - 1].1
        }

        let left = ns.substring(with: NSRange(location: 0, length: chosen.location)).trimmingCharacters(in: .whitespaces)
        let right = ns.substring(from: chosen.location + chosen.length)
        let contentOut: String
        if !left.isEmpty && !right.isEmpty && !right.hasPrefix(" ") && !right.hasPrefix("\t") {
            contentOut = left + " " + right
        } else {
            contentOut = left + right
        }
        return (refs, contentOut.trimmingCharacters(in: .whitespaces))
    }

    /// Parse 'inner' as a comma-separated list of refs; nil if not a ref list.
    private static func parseRefList(_ inner: String, lineNum: Int, isV08: Bool) throws -> [ParsedRef]? {
        let parts = inner.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        var refs: [ParsedRef] = []
        for part in parts {
            if reActivationRefFull.firstMatch(in: part) != nil {
                throw CatParserError.parse(
                    line: lineNum,
                    message: "`(N@k)` activation references were removed from CAT Flow (v0.6) — `@k` now only addresses `inspect ROW@k` on the command line."
                )
            }
            if let im = reInputRef.firstMatch(in: part) {
                refs.append(.input(position: Int(group(im, in: part, index: 1) ?? "") ?? 1))
                continue
            }
            if let pm = reParamRef.firstMatch(in: part) {
                refs.append(.param(name: group(pm, in: part, index: 1) ?? ""))
                continue
            }
            if part.range(of: "^\\d+$", options: .regularExpression) != nil {
                refs.append(.row(number: Int(part) ?? 0))
                continue
            }
            return nil
        }
        return refs.isEmpty ? nil : refs
    }

    // MARK: - Block bodies + wrapped continuation (R13)

    /// Collect a block's indented body lines, dedented, returning
    /// (body_indent, body_lines_dedented, lines_consumed).
    private static func collectBlockBody(_ lines: [String], start: Int, isV08: Bool) -> (Int, [String], Int) {
        var body: [String] = []
        var i = start
        var bodyIndent: Int?

        while i < lines.count {
            let line = lines[i]
            if reBlank.firstMatch(in: line) != nil {
                body.append(line)
                i += 1
                continue
            }
            guard let m = reNumbered.firstMatch(in: line) else {
                if isV08, let bi = bodyIndent {
                    let rawIndent = line.count - line.trimmingCharacters(in: .whitespaces).count
                    if rawIndent >= bi {
                        body.append(String(line.dropFirst(bi)))
                        i += 1
                        continue
                    }
                }
                break
            }
            let lineIndent = group(m, in: line, index: 1)?.count ?? 0
            if lineIndent == 0 { break }
            if bodyIndent == nil { bodyIndent = lineIndent }
            if lineIndent < (bodyIndent ?? 0) { break }
            body.append(String(line.dropFirst(bodyIndent ?? 0)))
            i += 1
        }
        return (bodyIndent ?? 0, body, i - start)
    }

    /// R13: merge wrap-continuation lines onto *content* (SPEC §2.5).
    private static func consumeWrappedContinuation(
        _ lines: [String], start: Int, rowIndent: Int, content: String, comments: [String],
        isV08: Bool, row: Int?
    ) throws -> (String, [String], Int) {
        var content = content
        var comments = comments
        let prefixes = isV08 ? [";", "\"", "("] : ["·", "\"", "("]
        var i = start
        var consumed = 0
        while i < lines.count {
            let nxt = lines[i]
            if quoteParityOdd(content) {
                let (piece, comment) = splitTrailingComment(nxt)
                content = content.trimmingCharacters(in: .whitespaces) + " " + piece.trimmingCharacters(in: .whitespaces)
                if let comment { comments.append(comment) }
                i += 1
                consumed += 1
                continue
            }
            if reBlank.firstMatch(in: nxt) != nil { break }
            let stripped = nxt.trimmingCharacters(in: .whitespaces)
            let thisIndent = nxt.count - stripped.count
            if thisIndent <= rowIndent { break }
            if let first = stripped.first, !prefixes.contains(String(first)) { break }
            if isV08 {
                try checkPreV08Chars(nxt, lineno: i + 1, row: row)
            }
            let (piece, comment) = splitTrailingComment(stripped)
            content = content.trimmingCharacters(in: .whitespaces) + " " + piece.trimmingCharacters(in: .whitespaces)
            if let comment { comments.append(comment) }
            i += 1
            consumed += 1
        }
        return (content, comments, consumed)
    }

    // MARK: - Continuation clauses

    private static func tryTryParseClauseLine(_ line: String, lineNum: Int) throws -> Clause? {
        let stripped = line.trimmingCharacters(in: .whitespaces)
        if stripped.isEmpty { return nil }

        let normalized = stripped.replacingOccurrences(of: "->", with: "→")

        if normalized == "resume" {
            return .resume
        }

        if reCallLead.firstMatch(in: normalized) != nil {
            if let cm = reCallWhole.firstMatch(in: normalized) {
                return .call(number: Int(group(cm, in: normalized, index: 1) ?? "") ?? 0)
            }
            throw CatParserError.parse(
                line: lineNum, message: "malformed `call` clause: \(stripped) — expected `call N`."
            )
        }

        if !normalized.hasPrefix("→") {
            return nil
        }

        let rest = String(normalized.dropFirst()).trimmingCharacters(in: .whitespaces)
        if rest.isEmpty {
            throw CatParserError.parse(
                line: lineNum,
                message: "`→` needs a target: a row number, `done`, `{ tag: target | ... }`, or `N & M`."
            )
        }

        if rest == "done" {
            return .goto(target: .done)
        }

        if rest.hasPrefix("{") {
            return try parseDecideClause(rest, lineNum: lineNum)
        }

        if rest.contains("&") {
            return try parseForkClause(rest, lineNum: lineNum)
        }

        if reDigits.firstMatch(in: rest) != nil {
            return .goto(target: .row(number: Int(rest) ?? 0))
        }

        throw CatParserError.parse(line: lineNum, message: "malformed `→` clause: \(stripped).")
    }

    private static func parseForkClause(_ rest: String, lineNum: Int) throws -> Clause {
        let parts = rest.split(separator: "&").map { $0.trimmingCharacters(in: .whitespaces) }
        if parts.count < 2 || parts.contains(where: { $0.isEmpty }) {
            throw CatParserError.parse(line: lineNum, message: "malformed fork clause: \(rest) — expected `N & M`.")
        }
        var targets: [ClauseTarget] = []
        for p in parts {
            if p == "done" {
                targets.append(.done)
            } else if reDigits.firstMatch(in: p) != nil {
                targets.append(.row(number: Int(p) ?? 0))
            } else {
                throw CatParserError.parse(line: lineNum, message: "malformed fork target: \(p) — expected a row number or `done`.")
            }
        }
        return .fork(targets: targets)
    }

    private static func parseDecideClause(_ rest: String, lineNum: Int) throws -> Clause {
        guard rest.hasSuffix("}") else {
            throw CatParserError.parse(line: lineNum, message: "unbalanced `{` in decide clause: \(rest).")
        }
        let inner = String(rest.dropFirst().dropLast()).trimmingCharacters(in: .whitespaces)
        if inner.isEmpty {
            throw CatParserError.parse(line: lineNum, message: "decide clause has no tags: `{ }`.")
        }

        var edges: [ClauseEdge] = []
        var seenTags: Set<String> = []
        for part in inner.split(separator: "|") {
            let partS = part.trimmingCharacters(in: .whitespaces)
            guard let colon = partS.firstIndex(of: ":") else {
                throw CatParserError.parse(line: lineNum, message: "malformed tag edge: \(partS) — expected `tag: target`.")
            }
            let tag = String(partS[..<colon]).trimmingCharacters(in: .whitespaces)
            if tag.isEmpty {
                throw CatParserError.parse(line: lineNum, message: "malformed tag edge: \(partS) — missing a tag name.")
            }
            if seenTags.contains(tag) {
                throw CatParserError.parse(line: lineNum, message: "duplicate tag \(tag) in decide clause.")
            }
            seenTags.insert(tag)
            let targetStr = String(partS[partS.index(after: colon)...])
            edges.append(ClauseEdge(tag: tag, target: try parseTarget(targetStr, lineNum: lineNum)))
        }
        return .decide(edges: edges)
    }

    private static func parseTarget(_ s: String, lineNum: Int) throws -> ClauseTarget {
        let s = s.trimmingCharacters(in: .whitespaces)
        if s == "done" { return .done }
        if s == "resume" { return .resume }
        if let cm = reCallWhole.firstMatch(in: s) {
            return .call(number: Int(group(cm, in: s, index: 1) ?? "") ?? 0)
        }
        if reDigits.firstMatch(in: s) != nil {
            return .row(number: Int(s) ?? 0)
        }
        throw CatParserError.parse(
            line: lineNum,
            message: "unknown continuation target: \(s) — expected a row number, `done`, `resume`, or `call N`."
        )
    }

    // MARK: - E108 pre-0.8 character check

    /// Under a `catflow 0.8` header, a pre-0.8 character (`·`, `→`, `≤`) outside a
    /// quoted string raises E108. `row`/`section` name the message's subject.
    private static func checkPreV08Chars(_ line: String, lineno: Int, row: Int? = nil, section: String? = nil) throws {
        let scan = stripQuoted(line)
        // Fixed order: · fires before → before ≤ (mirrors PRE_V08_CHECKS).
        if scan.contains("·") {
            throw CatParserError.preV08Character(char: "·", ascii: ";", line: lineno, row: row, section: section)
        }
        if scan.contains("→") {
            throw CatParserError.preV08Character(char: "→", ascii: "->", line: lineno, row: row, section: section)
        }
        if scan.contains("≤") {
            throw CatParserError.preV08Character(char: "≤", ascii: "<=", line: lineno, row: row, section: section)
        }
    }

    // MARK: - Regex helpers

    private static func reMatch(_ rx: NSRegularExpression, text: String) -> ReMatch? {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let result = rx.firstMatch(in: text, range: range), result.range.location == 0 else {
            return nil
        }
        let groups = (0..<result.numberOfRanges).map { i -> String? in
            let r = result.range(at: i)
            if r.location == NSNotFound { return nil }
            return ns.substring(with: r)
        }
        return ReMatch(range: result.range, groups: groups)
    }
}

/// A `re.match`-style result with captured groups (0 = whole match).
struct ReMatch {
    var range: NSRange
    var groups: [String?]

    subscript(_ i: Int) -> String? {
        i < groups.count ? groups[i] : nil
    }
}

/// The grammar pass's intermediate row: references are still numeric (the wire
/// shape), to be rewritten to ids by `CatParser.resolve`.
struct ParsedRow {
    var task: String?
    var blockKind: BlockKind?
    var blockName: String?
    var model: String?
    var settings: String?
    var refs: [ParsedRef]
    var chainBreak: Bool
    var children: [ParsedRow]
    var clause: Clause?
    var tags: [String]?
    var visitsLeq: Int?
    var onBudget: String?
    var declaredSignature: String?
    var comment: String?
    var leadingComments: [String]

    init() {
        self.task = nil
        self.blockKind = nil
        self.blockName = nil
        self.model = nil
        self.settings = nil
        self.refs = []
        self.chainBreak = false
        self.children = []
        self.clause = nil
        self.tags = nil
        self.visitsLeq = nil
        self.onBudget = nil
        self.declaredSignature = nil
        self.comment = nil
        self.leadingComments = []
    }
}

/// A numeric reference in the grammar pass — `row N`, `input:K`, or `param:name`.
enum ParsedRef: Equatable {
    case row(number: Int)
    case input(position: Int)
    case param(name: String)
}

/// The raw parse result the validator operates on: numeric refs throughout, plus the
/// document metadata (version/flags/models/…). `CatParser.resolveDocument` converts it
/// to the id-based `FlowDocument` for the app.
nonisolated struct ParsedFlow {
    var version: String
    var fileKind: FileKind
    /// The raw header keyword as written — see `FlowDocument.headerKeyword`. Defaults to
    /// `"catflow"` (`model.py:211`) for callers that build a `ParsedFlow` without a parsed
    /// header (e.g. tests constructing one directly).
    var headerKeyword: String = "catflow"
    var flags: [String]
    var rows: [ParsedRow]
    var uses: [String: String]
    var models: [String: String]
    var transforms: [String: TransformDef]
    var definitions: [String: CompositeDef]
    var accepts: [Kind]?
    var gives: String?
    var params: [ParamDecl]
    var presets: [String: PresetDecl]

    var flowDocument: FlowDocument? {
        try? CatParser.resolveDocument(self)
    }
}

/// Parse-time errors for the CAT Flow grammar — ported from `core/errors.py`. Every
/// message is `⚠ `-prefixed, byte-for-byte the documented wording (via `ErrorCatalog`).
nonisolated enum CatParserError: Error, Equatable, CustomStringConvertible {
    case parse(line: Int?, message: String)
    case needsVersion(token: String, version: String, line: Int)
    case unknownHeaderFlag(flag: String, line: Int)
    case unparseableRow(line: Int?, fragment: String, detail: String, isV08: Bool)
    case unclosedQuote(line: Int)
    case duplicatePosition(number: Int, line: Int)
    case preV08Character(char: String, ascii: String, line: Int, row: Int?, section: String?)

    var description: String {
        switch self {
        case .parse(let line, let message):
            return line.map { "⚠ row \($0): \(message)" } ?? "⚠ \(message)"
        case .needsVersion(_, let version, _):
            let rendered = (try? ErrorCatalog.fill(code: "E101", values: ["version": version], isV08: true)) ?? version
            return "⚠ \(rendered)"
        case .unknownHeaderFlag(let flag, _):
            let rendered = (try? ErrorCatalog.fill(code: "E102", values: ["flag": flag], isV08: true)) ?? flag
            return "⚠ \(rendered)"
        case .unparseableRow(let line, let fragment, let detail, let isV08):
            let n = line.map(String.init) ?? ""
            let rendered = (try? ErrorCatalog.fill(
                code: "E105", values: ["n": n, "fragment": fragment, "detail": detail], isV08: isV08
            )) ?? ("Row couldn't be read: \(detail)")
            return "⚠ \(rendered)"
        case .unclosedQuote(let line):
            let rendered = (try? ErrorCatalog.fill(code: "E106", values: ["n": String(line)])) ?? "unclosed quote"
            return "⚠ \(rendered)"
        case .duplicatePosition(let number, _):
            let rendered = (try? ErrorCatalog.fill(code: "E107", values: ["n": String(number)])) ?? "duplicate position"
            return "⚠ \(rendered)"
        case .preV08Character(let char, let ascii, let line, let row, let section):
            let subject: String
            if let row {
                subject = "Row \(row)"
            } else if let section {
                subject = "The \(section): section"
            } else {
                subject = "Line \(line)"
            }
            let rendered = (try? ErrorCatalog.fill(
                code: "E108", values: ["subject": subject, "char": char, "ascii": ascii], isV08: true
            )) ?? "pre-0.8 character"
            return "⚠ \(rendered)"
        }
    }
}

extension String {
    /// If the string starts with *prefix*, returns the remainder; else nil.
    func ltrimmedPrefix(_ prefix: String) -> String? {
        hasPrefix(prefix) ? String(dropFirst(prefix.count)) : nil
    }
}

extension NSRegularExpression {
    /// `re.search`-style: first match anywhere in *text*.
    func firstMatch(in text: String) -> NSTextCheckingResult? {
        let ns = text as NSString
        return firstMatch(in: text, range: NSRange(location: 0, length: ns.length))
    }
}
