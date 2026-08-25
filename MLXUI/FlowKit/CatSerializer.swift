import Foundation

/// The canonical text serializer — a faithful port of `catflow-mlx/src/catflow/core/fmt.py`
/// `render` (CFM-R6-1). Renders a `FlowDocument` back to `.cat` text with canonical column
/// alignment: the "N. Task" label padded to (max non-block sibling width + 3), the ref column
/// sized to the scope, quoted-settings wrapping at 74 with continuation lines indented to the
/// settings column, block bodies indented 5 spaces, and the section blocks (`models:` etc.)
/// column-aligned and in source order.
///
/// The same "rows read like the file" guarantee the owner asked for is exactly `catflow fmt`'s
/// output — this port is what makes the UI show the canonical line and what `Save` will write.
nonisolated enum CatSerializer {

    private static let gap = 3
    private static let blockIndent = "     "
    private static let clauseIndent = "   "
    private static let wrapWidth = 74

    /// `format_flow`'s text half — `render(parse(text))`'s `render`. Ends with a single
    /// trailing newline (or `""` for an empty flow).
    static func serialize(_ doc: FlowDocument, deadRefNumbers: [UUID: Int] = [:]) -> String {
        let (lines, _, _) = serializeLines(doc, deadRefNumbers: deadRefNumbers)
        return lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n"
    }

    /// `render(flow)` as a line list, plus each top-level row's line range in the output —
    /// the R6-2 API, so the view can render one cell per row without re-deriving the layout.
    /// `deadRefNumbers` (CFM-R8-3) lets a *display* rendering show a reference to a deleted
    /// row as `(?N)` — the number it held — instead of the read-only path's `-1`. The
    /// editor's save gate never writes a broken reference, so `(?N)` never reaches a file.
    static func serializeLines(_ doc: FlowDocument, deadRefNumbers: [UUID: Int] = [:]) -> (lines: [String], lineRanges: [UUID: Range<Int>], clauseRanges: [UUID: Range<Int>]) {
        let isV08 = doc.version == "0.8"
        var lines: [String] = []
        var ranges: [UUID: Range<Int>] = [:]
        // QR12R2-1: a block's clause line is emitted *after* its children, so the header-only
        // range never covers it — record it separately so the flattened list can draw it.
        var clauseRanges: [UUID: Range<Int>] = [:]

        if !doc.version.isEmpty {
            let token = doc.fileKind == .catpipeline ? "catpipeline" : "catflow"
            var header = "\(token) \(doc.version)"
            let joiner = isV08 ? "; " : " · "
            // Source order when the parser saw it; a stable order otherwise (Set is unordered).
            let ordered = doc.flagsOrder.isEmpty
                ? CapabilityFlag.allCases.filter { doc.flags.contains($0) }
                : doc.flagsOrder.filter { doc.flags.contains($0) }
            for flag in ordered {
                header += joiner + flag.rawValue
            }
            lines.append(header)
        }

        if doc.fileKind == .catpipeline {
            if let name = doc.pipelineName {
                lines.append("  pipeline \(name)")
            }
            if let accepts = doc.accepts {
                lines.append("  accepts: [" + accepts.map(\.rawValue).joined(separator: ", ") + "]")
            }
            if let gives = doc.gives {
                lines.append("  gives:   \(gives)")
            }
        }

        if !doc.params.isEmpty {
            let joiner = isV08 ? "; " : " · "
            let parts = doc.params.map { paramText($0) }
            lines.append("params: " + parts.joined(separator: joiner))
        }

        if !doc.presets.isEmpty {
            lines.append("presets:")
            let names = order(of: doc.presets, ordered: doc.presetsOrder)
            let width = names.map { $0.unicodeScalars.count }.max() ?? 0
            for name in names {
                guard let preset = doc.presets[name] else { continue }
                let rhs = preset.bindings.map { "\($0.key)=\($0.value)" }.joined(separator: "; ")
                lines.append("  \(ljust(name, width)) = \(rhs)")
            }
        }

        let rows = renderRows(doc.rows, wrap: isV08, isV08: isV08, baseLine: lines.count,
                              deadRefNumbers: deadRefNumbers)
        lines.append(contentsOf: rows.lines)
        ranges.merge(rows.ranges) { a, _ in a }
        clauseRanges.merge(rows.clauseRanges) { a, _ in a }

        if !doc.definitions.isEmpty {
            lines.append("")
            lines.append("definitions:")
            for name in order(of: doc.definitions, ordered: doc.definitionsOrder) {
                guard let comp = doc.definitions[name] else { continue }
                lines.append(contentsOf: renderComposite(name, comp, wrap: isV08, isV08: isV08))
            }
        }
        if !doc.uses.isEmpty {
            lines.append("")
            lines.append("uses:")
            let names = order(of: doc.uses, ordered: doc.usesOrder)
            let width = names.map { $0.unicodeScalars.count }.max() ?? 0
            for name in names {
                guard let path = doc.uses[name] else { continue }
                lines.append("  \(ljust(name, width)) = \(path)")
            }
        }
        if !doc.models.isEmpty {
            lines.append("")
            lines.append("models:")
            let names = order(of: doc.models, ordered: doc.modelsOrder)
            let width = names.map { $0.unicodeScalars.count }.max() ?? 0
            for display in names {
                guard let pinned = doc.models[display] else { continue }
                lines.append("  \(ljust(display, width)) = \(pinned)")
            }
        }
        if !doc.transforms.isEmpty {
            lines.append("")
            lines.append("transforms:")
            for name in order(of: doc.transforms, ordered: doc.transformsOrder) {
                guard let transform = doc.transforms[name] else { continue }
                lines.append(contentsOf: renderTransform(name, transform, isV08: isV08))
            }
        }

        return (lines, ranges, clauseRanges)
    }

    // MARK: - Section helpers

    private static func order<K, V>(of dict: [K: V], ordered: [K]) -> [K] {
        ordered.isEmpty ? Array(dict.keys).sorted { "\($0)" < "\($1)" } : ordered
    }

    private static func ljust(_ s: String, _ width: Int) -> String {
        // Python's `str.ljust` never truncates — it pads to the width and is a no-op when the
        // string already exceeds it. `String.padding(toLength:)` is NSString-backed (UTF-16
        // units) and *removes* trailing characters, so it can silently drop an NFD "é" etc.
        // (CFM-R6-FIX-2).
        let padCount = max(0, width - s.unicodeScalars.count)
        return s + String(repeating: " ", count: padCount)
    }

    private static func paramText(_ p: ParamDecl) -> String {
        var piece = p.name
        if let kind = p.kind { piece += " (\(kind))" }
        if let d = p.defaultValue { piece += " = \(d)" }
        return piece
    }

    private static func renderComposite(_ name: String, _ comp: CompositeDef,
                                        wrap: Bool, isV08: Bool) -> [String] {
        var lines: [String] = []
        let sig = comp.signature.map { "   \($0)" } ?? ""
        lines.append("  composite \(name)\(sig)")
        if !comp.params.isEmpty {
            lines.append("  params: " + comp.params.map { paramText($0) }.joined(separator: isV08 ? "; " : " · "))
        }
        for child in renderRows(comp.rows, wrap: wrap, isV08: isV08, baseLine: 0).lines {
            lines.append(child.isEmpty ? child : "\(blockIndent)\(child)")
        }
        return lines
    }

    private static func renderTransform(_ name: String, _ t: TransformDef, isV08: Bool) -> [String] {
        let sig = t.signature.map { "   \($0)" } ?? ""
        var lines = ["  \(name)\(sig)"]
        if let run = t.run { lines.append("    run:     \(run)") }
        if let timeout = t.timeout { lines.append("    timeout: \(timeout)") }
        if let workdir = t.workdir { lines.append("    workdir: \(workdir)") }
        if !t.params.isEmpty {
            lines.append("    params:  " + t.params.map { paramText($0) }.joined(separator: isV08 ? "; " : " · "))
        }
        return lines
    }

    // MARK: - Rows (the column-alignment core)

    /// `_render_rows` — one pass over a scope so the "N. Task" label column and the ref
    /// column are sized to that scope. Returns each row's absolute line range.
    static func renderRows(_ rows: [Row], wrap: Bool, isV08: Bool,
                           baseLine: Int, deadRefNumbers: [UUID: Int] = [:]) -> (lines: [String], ranges: [UUID: Range<Int>], clauseRanges: [UUID: Range<Int>]) {
        let labels = rows.enumerated().map { label(number: $0.offset + 1, row: $0.element, isV08: isV08) }
        let tailWidths = zip(labels, rows).compactMap { $1.blockKind == nil ? $0.unicodeScalars.count : nil }
        let col = tailWidths.isEmpty ? 0 : (tailWidths.max()! + gap)
        let scope = Dictionary(rows.enumerated().map { ($1.id, $0 + 1) }, uniquingKeysWith: { a, _ in a })
        let refStrs = rows.map { refsString($0.refs, scope: scope, deadRefNumbers: deadRefNumbers) }
        let refWidths = zip(refStrs, rows).compactMap { ($1.blockKind == nil && !$0.isEmpty) ? $0.unicodeScalars.count : nil }
        let refCol = refWidths.isEmpty ? 0 : refWidths.max()!

        var lines: [String] = []
        var ranges: [UUID: Range<Int>] = [:]
        var clauseRanges: [UUID: Range<Int>] = [:]
        var current = baseLine
        for (row, label, refsStr) in zip(rows, zip(labels, refStrs)).map({ ($0, $1.0, $1.1) }) {
            if row.chainBreak {
                lines.append(""); current += 1
            }
            // Range starts after the chain-break blank so the blank isn't part of this row —
            // the view draws one `Divider` for the break, and the blank is emitted only in
            // the serialized bytes (CFM-R6-FIX-6).
            let start = current
            for comment in row.leadingComments {
                lines.append("# \(comment)"); current += 1
            }
            if row.blockKind != nil {
                let rest = blockTail(row)
                let prefix = refsStr.isEmpty ? label : "\(label)\(pad(gap))\(refsStr)"
                let emitted = emitRowLines(label: prefix, tail: rest, col: prefix.unicodeScalars.count + gap, wrap: wrap)
                lines.append(contentsOf: emitted); current += emitted.count
                // CFM-R12-FIX-2: a block's range covers only its *header* lines — the
                // flattened list renders children as their own rows, so the full span would
                // print the body twice. Children get their own ranges via the merge below.
                ranges[row.id] = start..<current
                let children = renderRows(row.children, wrap: wrap, isV08: isV08, baseLine: current,
                                          deadRefNumbers: deadRefNumbers)
                for child in children.lines {
                    lines.append(child.isEmpty ? child : "\(blockIndent)\(child)")
                    current += 1
                }
                ranges.merge(children.ranges) { a, _ in a }
                clauseRanges.merge(children.clauseRanges) { a, _ in a }
            } else {
                let rest = tail(row: row, isV08: isV08)
                let prefix: String
                let tailCol: Int
                if refCol > 0 && !rest.isEmpty {
                    prefix = "\(label)\(pad(col - label.unicodeScalars.count))\(ljust(refsStr, refCol))"
                    tailCol = col + refCol + gap
                } else if !refsStr.isEmpty {
                    prefix = "\(label)\(pad(col - label.unicodeScalars.count))\(refsStr)"
                    tailCol = col
                } else {
                    prefix = label
                    tailCol = col
                }
                let emitted = emitRowLines(label: prefix, tail: rest, col: tailCol, wrap: wrap)
                lines.append(contentsOf: emitted); current += emitted.count
                // QR12R2-1: record every row's range *before* the clause line, so the clause
                // lives only in clauseRanges — a decider row's `-> {…}` must not render twice.
                ranges[row.id] = start..<current
            }
            if let clause = row.clause {
                lines.append("\(clauseIndent)\(renderClause(clause, isV08: isV08))")
                clauseRanges[row.id] = current..<(current + 1)
                current += 1
            }
            // Non-block rows land here with no range yet; a block's header range was already
            // recorded (FIX-2) and must not be overwritten by the full span.
            if ranges[row.id] == nil {
                ranges[row.id] = start..<current
            }
        }
        return (lines, ranges, clauseRanges)
    }

    private static func pad(_ n: Int) -> String {
        String(repeating: " ", count: max(0, n))
    }

    private static func emitRowLines(label: String, tail: String, col: Int, wrap: Bool) -> [String] {
        if tail.isEmpty { return [label] }
        let full = "\(label)\(pad(col - label.unicodeScalars.count))\(tail)"
        if !wrap || full.unicodeScalars.count <= wrapWidth { return [full] }
        if let wrapped = wrapTail(label: label, tail: tail, col: col) { return wrapped }
        return [full]
    }

    /// `_wrap_tail` — only the documented common case word-wraps: an overlong quoted settings
    /// string, continuation lines indented to the tail column.
    private static let quotedSpan = NSRegularExpression.compiled(#""(?:[^"\\]|\\.)*""#)

    private static func wrapTail(label: String, tail: String, col: Int) -> [String]? {
        let ns = NSRange(tail.startIndex..<tail.endIndex, in: tail)
        guard let match = quotedSpan.firstMatch(in: tail, range: ns),
              let range = Range(match.range, in: tail) else { return nil }
        let before = String(tail[..<range.lowerBound])
        let inner = String(tail[range]).dropFirst().dropLast()
        let after = String(tail[range.upperBound...])
        let words = inner.split(separator: " ", omittingEmptySubsequences: false).map(String.init)

        var lines: [String] = []
        var cur = "\(label)\(pad(col - label.unicodeScalars.count))\(before)\""
        var started = false
        for word in words {
            let candidate = started ? "\(cur) \(word)" : "\(cur)\(word)"
            if started && candidate.unicodeScalars.count > wrapWidth {
                lines.append(cur)
                cur = pad(col) + word
            } else {
                cur = candidate
            }
            started = true
        }
        lines.append(cur + "\"" + after)
        return lines
    }

    // MARK: - Per-row text

    private static func label(number: Int, row: Row, isV08: Bool) -> String {
        if let blockKind = row.blockKind {
            let name = row.blockName.map { " \($0)" } ?? ""
            let markers: String
            if let settings = row.settings, !settings.isEmpty {
                markers = isV08 ? "; \(settings)" : " · \(settings)"
            } else {
                markers = ""
            }
            return "\(number). <\(blockKind.rawValue)\(name)\(markers)>"
        }
        return "\(number). \(row.task ?? "")"
    }

    private static func tail(row: Row, isV08: Bool) -> String {
        var settings = row.settings
        if let tags = row.tags, !tags.isEmpty {
            settings = reliftTags(settings, tags: tags, isV08: isV08, task: row.task)
        }
        var t: String
        if let model = row.model, let settings, !settings.isEmpty {
            t = joinField(model, settings, isV08)
        } else {
            t = row.model ?? settings ?? ""
        }
        if let v = row.visitsLeq {
            t = joinField(t, isV08 ? "max_visits=\(v)" : "visits≤\(v)", isV08)
        }
        if let onBudget = row.onBudget {
            t = joinField(t, "on_budget=\(onBudget)", isV08)
        }
        return appendSigComment(t, row)
    }

    private static func blockTail(_ row: Row) -> String {
        appendSigComment("", row)
    }

    private static func appendSigComment(_ tail: String, _ row: Row) -> String {
        var t = tail
        if let sig = row.declaredSignature {
            t = t.isEmpty ? sig : "\(t)   \(sig)"
        }
        if let comment = row.comment {
            let c = "# \(comment)"
            t = t.isEmpty ? c : "\(t)   \(c)"
        }
        return t
    }

    private static func joinField(_ tail: String, _ field: String, _ isV08: Bool) -> String {
        tail.isEmpty ? field : (isV08 ? "\(tail); \(field)" : "\(tail) · \(field)")
    }

    /// `_relift_tags` — put a declared tag set back into its canonical position (inside the
    /// quoted criterion when there is one — except `Compare`, whose quoted value is arithmetic).
    private static func reliftTags(_ settings: String?, tags: [String], isV08: Bool, task: String?) -> String {
        let tagsStr = "tags: " + tags.joined(separator: ", ")
        if task != "Compare", let settings, !settings.isEmpty,
           settings.hasPrefix("\""), settings.hasSuffix("\"") {
            let inner = String(settings.dropFirst().dropLast())
            return inner.isEmpty ? "\"\(tagsStr)\"" : "\"\(inner) \(tagsStr)\""
        }
        if let settings { return joinField(settings, tagsStr, isV08) }
        return tagsStr
    }

    /// `_refs_str` — refs render as `(1,2)` / `(input:1)` / `(param:name)`, row refs resolved
    /// to their number **in this scope** (the Swift stores ids, the file shows numbers). A
    /// row ref whose id is missing from scope renders `(?N)` when the editor knows the
    /// number the deleted row held (CFM-R8-3), else `-1` (the read-only fallback).
    private static func refsString(_ refs: [Ref], scope: [UUID: Int],
                                   deadRefNumbers: [UUID: Int] = [:]) -> String {
        guard !refs.isEmpty else { return "" }
        var parts: [String] = []
        for ref in refs {
            switch ref {
            case .rowRef(let id):
                if let number = scope[id] {
                    parts.append(String(number))
                } else if let dead = deadRefNumbers[id] {
                    parts.append("?" + String(dead))
                } else {
                    parts.append("-1")
                }
            case .inputRef(let position):
                parts.append("input:\(position)")
            case .paramRef(let name):
                parts.append("param:\(name)")
            }
        }
        return "(" + parts.joined(separator: ",") + ")"
    }

    // MARK: - Continuation clauses

    private static func renderClause(_ clause: Clause, isV08: Bool) -> String {
        let arrow = isV08 ? "-> " : "→ "
        switch clause {
        case .goto(let target):
            return "\(arrow)\(targetText(target))"
        case .fork(let targets):
            return arrow + targets.map { targetText($0) }.joined(separator: " & ")
        case .decide(let edges):
            let joined = edges.map { "\($0.tag): \(targetText($0.target))" }.joined(separator: " | ")
            return "\(arrow){ \(joined) }"
        case .call(let number):
            return "call \(number)"
        case .resume:
            return "resume"
        }
    }

    private static func targetText(_ target: ClauseTarget) -> String {
        switch target {
        case .row(let number): return String(number)
        case .call(let number): return "call \(number)"
        case .resume: return "resume"
        case .done: return "done"
        }
    }
}
