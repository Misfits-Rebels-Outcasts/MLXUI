import Foundation

/// The §2 text instant tools — the Swift port of `catflow-mlx/src/catflow/tools/text.py`
/// (CFM-R4-1/2), as `AssetStage`s. Pure text transforms: no engine, no filesystem, no
/// model imports (the Python's isolation rule's "tool" side). All operate on the input
/// items' inline text.
///
/// Several defaults / micro-syntax choices are the Python's own (SPEC-Q13): `Split`'s
/// `by=tokens` default with a 200-unit sliding window, `Filter`'s `length=N` op, `Count`'s
/// `items` vs `words`, `Template`'s `{1}`/`{input:K}` placeholders, `Join Text`'s default
/// `\n` separator, `Diff`'s `inline` (ndiff) vs `unified`.
nonisolated enum TextTools {

    /// `_unquote_whole` — strip one surrounding pair of double quotes and process
    /// `\n \t \" \\` escapes (used for settings that are themselves one value).
    static func unquoteWhole(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.count >= 2, s.first == "\"", s.last == "\"" {
            s = String(s.dropFirst().dropLast())
        }
        return s.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
    }

    /// The flat inline texts of every input asset, in ref/chain order.
    static func flatTexts(_ inputs: [Asset]) -> [String] {
        inputs.flatMap { asset in asset.items.compactMap { $0.value } }
    }

    /// The first input item's inline text, or "".
    static func singleText(_ inputs: [Asset]) -> String {
        inputs.first?.items.first?.value ?? ""
    }
}

// MARK: - Split

nonisolated struct SplitTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .single(.text) }
    var produces: Shape { .listOf(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let text = TextTools.singleText([input])
        let s = FlowSettings(settings)
        let by = s.value(for: "by", default: "tokens") ?? "tokens"
        let units = splitUnits(text, by: by)
        guard !units.isEmpty else { return Asset(items: [Item(kind: .text, value: "", path: nil, sourceText: nil)]) }
        // lines/sentences/headings are document boundaries — one item per unit (SPEC-Q121).
        if by == "lines" || by == "sentences" || by == "headings" {
            return Asset(items: units.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
        }
        // tokens — sliding window by chunk_size (default 200) with overlap.
        let rawSize = s.value(for: "chunk_size") ?? s.pathValue()
        let chunkSize = rawSize.flatMap(Int.init).map { max($0, 1) } ?? 200
        let overlap = Int(s.value(for: "overlap", default: "0") ?? "0") ?? 0
        let stride = max(chunkSize - overlap, 1)
        var chunks: [String] = []
        var i = 0
        while i < units.count {
            let window = Array(units[i..<min(i + chunkSize, units.count)])
            chunks.append(joinUnits(window, by: by))
            if i + chunkSize >= units.count { break }
            i += stride
        }
        return Asset(items: chunks.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    private func splitUnits(_ text: String, by: String) -> [String] {
        switch by {
        case "lines":
            return text.split(separator: "\n").map(String.init).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        case "sentences":
            return text.split(whereSeparator: { $0 == "." || $0 == "!" || $0 == "?" })
                .map(String.init).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        case "headings":
            return splitHeadings(text)
        default:
            return text.split(whereSeparator: \.isWhitespace).map(String.init)
        }
    }

    private func splitHeadings(_ text: String) -> [String] {
        let lines = text.split(separator: "\n").map(String.init)
        var sections: [String] = []
        var current: [String] = []
        for line in lines {
            if line.hasPrefix("#"), !current.isEmpty {
                sections.append(current.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))
                current = [line]
            } else {
                current.append(line)
            }
        }
        if !current.isEmpty {
            sections.append(current.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return sections.filter { !$0.isEmpty }
    }

    private func joinUnits(_ units: [String], by: String) -> String {
        if by == "lines" { return units.joined(separator: "\n") }
        return units.joined(separator: " ")
    }
}

// MARK: - Filter

nonisolated struct FilterTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .listOf(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        let s = FlowSettings(settings)
        let contains = s.value(for: "contains")
        let regex = s.value(for: "regex")
        let lengthMatch = parseLength(settings)

        func keep(_ text: String) -> Bool {
            if let contains, !text.contains(contains) { return false }
            if let regex, text.range(of: regex, options: .regularExpression) == nil { return false }
            if let lengthMatch {
                let (op, n) = lengthMatch
                if !applyLength(op, length: text.count, n: n) { return false }
            }
            return true
        }
        return Asset(items: texts.filter(keep).map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    private func parseLength(_ settings: String) -> (String, Int)? {
        // length (>=|<=|>|<|=)? N
        let pattern = try! NSRegularExpression(pattern: #"length\s*(>=|<=|>|<|=)?\s*(\d+)"#)
        guard let match = pattern.firstMatch(in: settings, range: NSRange(settings.startIndex..<settings.endIndex, in: settings)),
              match.numberOfRanges >= 3,
              let numRange = Range(match.range(at: 2), in: settings),
              let n = Int(settings[numRange]) else { return nil }
        let opRange = match.range(at: 1)
        let op = opRange.location == NSNotFound ? "=" : String(settings[Range(opRange, in: settings)!])
        return (op, n)
    }

    private func applyLength(_ op: String, length: Int, n: Int) -> Bool {
        switch op {
        case ">":  return length > n
        case ">=": return length >= n
        case "<":  return length < n
        case "<=": return length <= n
        default:   return length == n
        }
    }
}

// MARK: - Dedupe

nonisolated struct DedupeTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .listOf(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        let s = FlowSettings(settings)
        let by = s.value(for: "by", default: "exact") ?? "exact"
        let direction = s.value(for: "direction", default: "first") ?? "first"

        func key(_ t: String) -> String {
            by == "normalized" ? t.casefoldWords : t
        }

        // Swift `Dictionary` has no insertion order, so both directions build an explicit
        // `order` array and map through it (the Python's `dict.values()` preserves insertion
        // order; the Swift port must not rely on dictionary ordering).
        var firstValue: [String: String] = [:]
        var order: [String] = []
        for t in texts {
            let k = key(t)
            if firstValue[k] == nil {
                firstValue[k] = t
                order.append(k)
            }
        }
        var result: [String]
        if direction == "last" {
            // Python: keep the LAST occurrence, ordered by that occurrence's index.
            var lastValue: [String: String] = [:]
            var lastIndex: [String: Int] = [:]
            for (i, t) in texts.enumerated() {
                let k = key(t)
                lastValue[k] = t
                lastIndex[k] = i
            }
            result = lastIndex.keys.sorted { lastIndex[$0]! < lastIndex[$1]! }.compactMap { lastValue[$0] }
        } else {
            result = order.compactMap { firstValue[$0] }
        }
        return Asset(items: result.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }
}

private extension String {
    /// `" ".join(text.casefold().split())` — the Python's normalized dedupe key.
    var casefoldWords: String {
        lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}

// MARK: - Count

nonisolated struct CountTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        let mode = FlowSettings(settings).pathValue() ?? "items"
        let n = mode == "items" ? texts.count : texts.reduce(0) { $0 + $1.split(whereSeparator: \.isWhitespace).count }
        return Asset(items: [Item(kind: .text, value: String(n), path: nil, sourceText: nil)])
    }
}

// MARK: - Join Text

nonisolated struct JoinTextTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        let s = FlowSettings(settings)
        let sep: String
        if let value = s.value(for: "separator") {
            sep = value
        } else if !settings.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            sep = TextTools.unquoteWhole(settings)
        } else {
            sep = "\n"
        }
        return Asset(items: [Item(kind: .text, value: texts.joined(separator: sep), path: nil, sourceText: nil)])
    }
}

// MARK: - Template

nonisolated struct TemplateTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let pattern = TextTools.unquoteWhole(settings)
        let texts = TextTools.flatTexts([input])
        let joined = texts.joined(separator: "\n")

        let regex = try! NSRegularExpression(pattern: #"\{((input(?::(\d+))?)|\d+)\}"#)
        let nsRange = NSRange(pattern.startIndex..<pattern.endIndex, in: pattern)
        let matches = regex.matches(in: pattern, range: nsRange)
        var out = pattern
        for match in matches.reversed() {
            let token = String(out[Range(match.range, in: out)!])
            let inner = token.dropFirst().dropLast()   // strip { }
            var idx: Int?
            if inner.hasPrefix("input") {
                let sub = inner.dropFirst("input".count)
                if sub.hasPrefix(":") { idx = Int(sub.dropFirst()) }
            } else {
                idx = Int(inner)
            }
            let value: String
            if let idx {
                let i = idx - 1
                value = (i >= 0 && i < texts.count) ? texts[i] : ""
            } else {
                value = joined
            }
            out.replaceSubrange(Range(match.range, in: out)!, with: value)
        }
        return Asset(items: [Item(kind: .text, value: out, path: nil, sourceText: nil)])
    }
}

// MARK: - Diff

nonisolated struct DiffTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .tupleOf([.text, .text]) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard input.items.count >= 2 else {
            throw FlowError.badInputCardinality(row: "Diff", expected: "two text inputs", got: input.items.count)
        }
        let a = input.items[0].value ?? ""
        let b = input.items[1].value ?? ""
        let fmt = FlowSettings(settings).value(for: "format", default: "inline") ?? "inline"
        let aLines = a.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let bLines = b.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let out = fmt == "unified" ? unifiedDiff(aLines, bLines) : inlineDiff(aLines, bLines)
        return Asset(items: [Item(kind: .text, value: out, path: nil, sourceText: nil)])
    }

    /// `difflib.ndiff` — `+`/`-`/` ` per line via LCS alignment (the Python's `inline`).
    private func inlineDiff(_ a: [String], _ b: [String]) -> String {
        ops(a, b).map { op in
            switch op {
            case .equal(let line): return "  " + line
            case .delete(let line): return "- " + line
            case .insert(let line): return "+ " + line
            case .replace(let d, let i): return "- \(d)\n+ \(i)"
            }
        }.joined(separator: "\n")
    }

    /// `difflib.unified_diff` with `lineterm=""` (the Python's `format=unified`). Context 3,
    /// clamped to the file. Built from the same `ops()` sequence the inline diff uses, with
    /// the `@@` range derived from the change extents.
    private func unifiedDiff(_ a: [String], _ b: [String]) -> String {
        var out = ["--- ", "+++ "]
        let ops = self.ops(a, b)

        // Compute the changed region: first/last a-index and first/last b-index that
        // participate in a non-equal op.
        var firstA = -1, lastA = -1, firstB = -1, lastB = -1
        var aIdx = 0, bIdx = 0
        for op in ops {
            switch op {
            case .equal:
                aIdx += 1; bIdx += 1
            case .delete:
                if firstA < 0 { firstA = aIdx }
                lastA = aIdx; aIdx += 1
            case .insert:
                if firstB < 0 { firstB = bIdx }
                lastB = bIdx; bIdx += 1
            case .replace:
                if firstA < 0 { firstA = aIdx }
                if firstB < 0 { firstB = bIdx }
                lastA = aIdx; lastB = bIdx
                aIdx += 1; bIdx += 1
            }
        }
        if firstA < 0 {   // no changes
            out.append("@@ -0,0 +0,0 @@")
            return out.joined(separator: "\n")
        }
        // Expand by context 3, clamped to the files.
        let lo = max(0, firstA - 3)
        let hi = min(a.count, lastA + 1 + 3)
        let aStart = lo + 1
        let aLen = hi - lo
        let bLo = max(0, firstB - 3)
        let bHi = min(b.count, lastB + 1 + 3)
        let bStart = bLo + 1
        let bLen = bHi - bLo
        out.append("@@ -\(aStart),\(aLen) +\(bStart),\(bLen) @@")

        // Emit the expanded region: reproduce the ops but only within [lo, hi) for a and
        // [bLo, bHi) for b, with leading/trailing context lines.
        var aIdx2 = 0, bIdx2 = 0
        for op in ops {
            switch op {
            case .equal(let line):
                if aIdx2 >= lo && aIdx2 < hi { out.append(" " + line) }
                aIdx2 += 1; bIdx2 += 1
            case .delete(let line):
                if aIdx2 >= lo && aIdx2 < hi { out.append("-" + line) }
                aIdx2 += 1
            case .insert(let line):
                if bIdx2 >= bLo && bIdx2 < bHi { out.append("+" + line) }
                bIdx2 += 1
            case .replace(let d, let i):
                if aIdx2 >= lo && aIdx2 < hi { out.append("-" + d) }
                if bIdx2 >= bLo && bIdx2 < bHi { out.append("+" + i) }
                aIdx2 += 1; bIdx2 += 1
            }
        }
        return out.joined(separator: "\n")
    }

    /// A diff operation on a single line.
    private enum Op {
        case equal(String)
        case delete(String)
        case insert(String)
        case replace(String, String)
    }

    /// Line-based diff via LCS. Equal lines are kept aligned; a changed line becomes a
    /// `replace`; a run of deletes/inserts between equal lines yields adjacent ops in
    /// delete-then-insert order (matching `ndiff`'s output shape).
    private func ops(_ a: [String], _ b: [String]) -> [Op] {
        let n = a.count, m = b.count
        // DP table: LCS length.
        var dp = Array(repeating: Array(repeating: 0, count: m + 1), count: n + 1)
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in stride(from: m - 1, through: 0, by: -1) {
                dp[i][j] = a[i] == b[j] ? dp[i + 1][j + 1] + 1 : max(dp[i + 1][j], dp[i][j + 1])
            }
        }
        var result: [Op] = []
        var i = 0, j = 0
        while i < n && j < m {
            if a[i] == b[j] {
                result.append(.equal(a[i])); i += 1; j += 1
            } else if dp[i + 1][j] >= dp[i][j + 1] {
                result.append(.delete(a[i])); i += 1
            } else {
                result.append(.insert(b[j])); j += 1
            }
        }
        while i < n { result.append(.delete(a[i])); i += 1 }
        while j < m { result.append(.insert(b[j])); j += 1 }
        return result
    }
}
