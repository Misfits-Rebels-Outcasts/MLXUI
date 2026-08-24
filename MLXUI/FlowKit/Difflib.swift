import Foundation

/// A faithful Swift port of the parts of Python's `difflib` that `Diff` needs (CFM-FIX-1):
/// `SequenceMatcher` (`__chain_b` → `find_longest_match` → `get_matching_blocks` →
/// `get_opcodes` → `get_grouped_opcodes`), `unified_diff`, and `Differ.compare` (ndiff with
/// intraline `? ` hints). `Fixtures/CatFlow/tools/diff_golden.json` pins the output against
/// Python 3.13's difflib byte-for-byte.
///
/// Only the behavior `DiffTool` exercises is ported: `isjunk=nil` for the line-level matcher,
/// `IS_CHARACTER_JUNK` (space/tab) for the char-level one, `autojunk` left at its default
/// (it only engages for ≥200 lines, which diffs don't reach).
nonisolated enum SwiftDifflib {

    // MARK: - Ratio

    /// `difflib._calculate_ratio` — length 0 yields 1.0.
    private static func calculateRatio(matches: Int, length: Int) -> Double {
        length == 0 ? 1.0 : 2.0 * Double(matches) / Double(length)
    }

    // MARK: - Opcode / match shapes

    /// `difflib` opcode tags.
    enum Tag: Sendable {
        case replace, delete, insert, equal
    }

    /// A `(i, j, k)` matching block.
    struct Match: Equatable, Sendable {
        let i: Int
        let j: Int
        let k: Int
    }

    /// A `(tag, i1, i2, j1, j2)` opcode.
    struct Opcode: Sendable {
        let tag: Tag
        let i1: Int
        let i2: Int
        let j1: Int
        let j2: Int
    }

    // MARK: - SequenceMatcher

    /// `difflib.SequenceMatcher` for a `Hashable` element type (lines are `[String]`, the
    /// intraline cruncher uses `[UnicodeScalar]` so code points match Python).
    final class Matcher<T: Hashable & Sendable>: @unchecked Sendable {
        private let isJunk: (T) -> Bool
        private let autojunk: Bool
        private var a: [T]
        private var b: [T]
        private var b2j: [T: [Int]] = [:]
        private var bjunk: Set<T> = []
        private var bpopular: Set<T> = []
        private var fullbcount: [T: Int]?
        private var matchingBlocksCache: [Match]?
        private var opcodesCache: [Opcode]?

        init(isJunk: @escaping (T) -> Bool = { _ in false },
             a: [T] = [], b: [T] = [], autojunk: Bool = true) {
            self.isJunk = isJunk
            self.autojunk = autojunk
            self.a = a
            self.b = b
            chainB()
        }

        /// `set_seq1` — marks the matcher dirty, keeps b's chain.
        func setSeq1(_ a: [T]) {
            self.a = a
            matchingBlocksCache = nil
            opcodesCache = nil
        }

        /// `set_seq2` — re-chains b.
        func setSeq2(_ b: [T]) {
            self.b = b
            matchingBlocksCache = nil
            opcodesCache = nil
            fullbcount = nil
            chainB()
        }

        /// `set_seqs`.
        func setSeqs(_ a: [T], _ b: [T]) {
            self.a = a
            self.b = b
            matchingBlocksCache = nil
            opcodesCache = nil
            fullbcount = nil
            chainB()
        }

        /// `__chain_b` — b2j with junk and (autojunk) popular elements removed.
        private func chainB() {
            var b2j: [T: [Int]] = [:]
            for (i, elt) in b.enumerated() {
                b2j[elt, default: []].append(i)
            }
            var bjunk: Set<T> = []
            for key in b2j.keys where isJunk(key) {
                bjunk.insert(key)
            }
            for elt in bjunk { b2j.removeValue(forKey: elt) }
            var bpopular: Set<T> = []
            let n = b.count
            if autojunk && n >= 200 {
                let ntest = n / 100 + 1
                for (elt, idxs) in b2j where idxs.count > ntest {
                    bpopular.insert(elt)
                }
                for elt in bpopular { b2j.removeValue(forKey: elt) }
            }
            self.b2j = b2j
            self.bjunk = bjunk
            self.bpopular = bpopular
        }

        /// `find_longest_match(alo, ahi, blo, bhi)`.
        func findLongestMatch(alo: Int, ahi: Int, blo: Int, bhi: Int) -> Match {
            let a = self.a
            let b = self.b
            let b2j = self.b2j
            let isbjunk = bjunk.contains
            var besti = alo
            var bestj = blo
            var bestsize = 0
            var j2len: [Int: Int] = [:]
            for i in alo..<ahi {
                var newj2len: [Int: Int] = [:]
                for j in b2j[a[i]] ?? [] {
                    if j < blo { continue }
                    if j >= bhi { break }
                    let k = (j2len[j - 1] ?? 0) + 1
                    newj2len[j] = k
                    if k > bestsize {
                        besti = i - k + 1
                        bestj = j - k + 1
                        bestsize = k
                    }
                }
                j2len = newj2len
            }
            while besti > alo && bestj > blo && !isbjunk(b[bestj - 1]) && a[besti - 1] == b[bestj - 1] {
                besti -= 1; bestj -= 1; bestsize += 1
            }
            while besti + bestsize < ahi && bestj + bestsize < bhi && !isbjunk(b[bestj + bestsize]) && a[besti + bestsize] == b[bestj + bestsize] {
                bestsize += 1
            }
            while besti > alo && bestj > blo && isbjunk(b[bestj - 1]) && a[besti - 1] == b[bestj - 1] {
                besti -= 1; bestj -= 1; bestsize += 1
            }
            while besti + bestsize < ahi && bestj + bestsize < bhi && isbjunk(b[bestj + bestsize]) && a[besti + bestsize] == b[bestj + bestsize] {
                bestsize += 1
            }
            return Match(i: besti, j: bestj, k: bestsize)
        }

        /// `get_matching_blocks` — recursion over a queue, then adjacent-block collapse.
        func getMatchingBlocks() -> [Match] {
            if let cached = matchingBlocksCache { return cached }
            let la = a.count
            let lb = b.count
            var queue: [(alo: Int, ahi: Int, blo: Int, bhi: Int)] = [(0, la, 0, lb)]
            var blocks: [Match] = []
            while let block = queue.popLast() {
                let x = findLongestMatch(alo: block.alo, ahi: block.ahi, blo: block.blo, bhi: block.bhi)
                if x.k != 0 {
                    blocks.append(x)
                    if block.alo < x.i && block.blo < x.j {
                        queue.append((block.alo, x.i, block.blo, x.j))
                    }
                    if x.i + x.k < block.ahi && x.j + x.k < block.bhi {
                        queue.append((x.i + x.k, block.ahi, x.j + x.k, block.bhi))
                    }
                }
            }
            blocks.sort { lhs, rhs in
                lhs.i == rhs.i ? (lhs.j == rhs.j ? lhs.k < rhs.k : lhs.j < rhs.j) : lhs.i < rhs.i
            }
            var nonAdjacent: [Match] = []
            var i1 = 0, j1 = 0, k1 = 0
            for m in blocks {
                if i1 + k1 == m.i && j1 + k1 == m.j {
                    k1 += m.k
                } else {
                    if k1 != 0 { nonAdjacent.append(Match(i: i1, j: j1, k: k1)) }
                    i1 = m.i; j1 = m.j; k1 = m.k
                }
            }
            if k1 != 0 { nonAdjacent.append(Match(i: i1, j: j1, k: k1)) }
            nonAdjacent.append(Match(i: la, j: lb, k: 0))
            matchingBlocksCache = nonAdjacent
            return nonAdjacent
        }

        /// `get_opcodes`.
        func getOpcodes() -> [Opcode] {
            if let cached = opcodesCache { return cached }
            var i = 0, j = 0
            var answer: [Opcode] = []
            for m in getMatchingBlocks() {
                let tag: Tag
                if i < m.i && j < m.j { tag = .replace }
                else if i < m.i { tag = .delete }
                else if j < m.j { tag = .insert }
                else { tag = .equal }   // Python's empty tag — skipped below
                if i < m.i || j < m.j {
                    answer.append(Opcode(tag: tag, i1: i, i2: m.i, j1: j, j2: m.j))
                }
                i = m.i + m.k
                j = m.j + m.k
                if m.k != 0 {
                    answer.append(Opcode(tag: .equal, i1: m.i, i2: i, j1: m.j, j2: j))
                }
            }
            opcodesCache = answer
            return answer
        }

        /// `get_grouped_opcodes(n)` — change clusters with up to `n` lines of context.
        func getGroupedOpcodes(n: Int = 3) -> [[Opcode]] {
            var codes = getOpcodes()
            if codes.isEmpty {
                codes = [Opcode(tag: .equal, i1: 0, i2: 1, j1: 0, j2: 1)]
            }
            if codes[0].tag == .equal {
                let c = codes[0]
                codes[0] = Opcode(tag: c.tag, i1: max(c.i1, c.i2 - n), i2: c.i2,
                                  j1: max(c.j1, c.j2 - n), j2: c.j2)
            }
            if codes[codes.count - 1].tag == .equal {
                let c = codes[codes.count - 1]
                codes[codes.count - 1] = Opcode(tag: c.tag, i1: c.i1, i2: min(c.i2, c.i1 + n),
                                                j1: c.j1, j2: min(c.j2, c.j1 + n))
            }
            let nn = n + n
            var group: [Opcode] = []
            var result: [[Opcode]] = []
            for code in codes {
                var code = code
                if code.tag == .equal && code.i2 - code.i1 > nn {
                    group.append(Opcode(tag: code.tag, i1: code.i1, i2: min(code.i2, code.i1 + n),
                                        j1: code.j1, j2: min(code.j2, code.j1 + n)))
                    result.append(group)
                    group = []
                    code = Opcode(tag: code.tag, i1: max(code.i1, code.i2 - n), i2: code.i2,
                                  j1: max(code.j1, code.j2 - n), j2: code.j2)
                }
                group.append(code)
            }
            if !group.isEmpty && !(group.count == 1 && group[0].tag == .equal) {
                result.append(group)
            }
            return result
        }

        /// `ratio`.
        func ratio() -> Double {
            let matches = getMatchingBlocks().reduce(0) { $0 + $1.k }
            return SwiftDifflib.calculateRatio(matches: matches, length: a.count + b.count)
        }

        /// `quick_ratio` — multiset intersection bound.
        func quickRatio() -> Double {
            if fullbcount == nil {
                var fullbcount: [T: Int] = [:]
                for elt in b { fullbcount[elt, default: 0] += 1 }
                self.fullbcount = fullbcount
            }
            let fullbcount = self.fullbcount!
            var avail: [T: Int] = [:]
            var matches = 0
            for elt in a {
                let numb = avail[elt] ?? fullbcount[elt] ?? 0
                avail[elt] = numb - 1
                if numb > 0 { matches += 1 }
            }
            return SwiftDifflib.calculateRatio(matches: matches, length: a.count + b.count)
        }

        /// `real_quick_ratio` — the absolute bound.
        func realQuickRatio() -> Double {
            SwiftDifflib.calculateRatio(matches: min(a.count, b.count), length: a.count + b.count)
        }
    }

    // MARK: - unified_diff

    /// `_format_range_unified(start, stop)`.
    static func formatRangeUnified(start: Int, stop: Int) -> String {
        let beginning = start + 1
        let length = stop - start
        if length == 1 { return "\(beginning)" }
        if length == 0 { return "\(beginning - 1),0" }
        return "\(beginning),\(length)"
    }

    /// `unified_diff(a, b, n=3, lineterm="")` — the `fromfile`/`tofile`/date slots are empty
    /// in every `Diff` call, so the header is the bare `--- `/`+++ ` pair.
    static func unifiedDiff(_ a: [String], _ b: [String], n: Int = 3, lineterm: String = "") -> [String] {
        var out: [String] = []
        var started = false
        let matcher = Matcher<String>(a: a, b: b)
        for group in matcher.getGroupedOpcodes(n: n) {
            if !started {
                started = true
                out.append("--- \(lineterm)")
                out.append("+++ \(lineterm)")
            }
            let first = group[0]
            let last = group[group.count - 1]
            let file1Range = formatRangeUnified(start: first.i1, stop: last.i2)
            let file2Range = formatRangeUnified(start: first.j1, stop: last.j2)
            out.append("@@ -\(file1Range) +\(file2Range) @@\(lineterm)")
            for op in group {
                switch op.tag {
                case .equal:
                    for k in op.i1..<op.i2 { out.append(" " + a[k]) }
                case .replace:
                    for k in op.i1..<op.i2 { out.append("-" + a[k]) }
                    for k in op.j1..<op.j2 { out.append("+" + b[k]) }
                case .delete:
                    for k in op.i1..<op.i2 { out.append("-" + a[k]) }
                case .insert:
                    for k in op.j1..<op.j2 { out.append("+" + b[k]) }
                }
            }
        }
        return out
    }

    // MARK: - ndiff / Differ

    /// `IS_CHARACTER_JUNK` — space or tab.
    private static func isCharacterJunk(_ scalar: UnicodeScalar) -> Bool {
        scalar == " " || scalar == "\t"
    }

    /// `Differ.compare` — the ndiff delta with intraline `? ` hints.
    static func ndiff(_ a: [String], _ b: [String]) -> [String] {
        var out: [String] = []
        let cruncher = Matcher<String>(a: a, b: b)
        for op in cruncher.getOpcodes() {
            switch op.tag {
            case .replace:
                out.append(contentsOf: fancyReplace(a, op.i1, op.i2, b, op.j1, op.j2))
            case .delete:
                out.append(contentsOf: dump(tag: "-", x: a, lo: op.i1, hi: op.i2))
            case .insert:
                out.append(contentsOf: dump(tag: "+", x: b, lo: op.j1, hi: op.j2))
            case .equal:
                out.append(contentsOf: dump(tag: " ", x: a, lo: op.i1, hi: op.i2))
            }
        }
        return out
    }

    /// `Differ._dump(tag, x, lo, hi)`.
    private static func dump(tag: String, x: [String], lo: Int, hi: Int) -> [String] {
        (lo..<hi).map { "\(tag) \(x[$0])" }
    }

    /// `Differ._plain_replace` — dump the shorter block first.
    private static func plainReplace(_ a: [String], _ alo: Int, _ ahi: Int,
                                    _ b: [String], _ blo: Int, _ bhi: Int) -> [String] {
        if bhi - blo < ahi - alo {
            return dump(tag: "+", x: b, lo: blo, hi: bhi) + dump(tag: "-", x: a, lo: alo, hi: ahi)
        }
        return dump(tag: "-", x: a, lo: alo, hi: ahi) + dump(tag: "+", x: b, lo: blo, hi: bhi)
    }

    /// `Differ._fancy_helper`.
    private static func fancyHelper(_ a: [String], _ alo: Int, _ ahi: Int,
                                   _ b: [String], _ blo: Int, _ bhi: Int) -> [String] {
        if alo < ahi {
            if blo < bhi {
                return fancyReplace(a, alo, ahi, b, blo, bhi)
            }
            return dump(tag: "-", x: a, lo: alo, hi: ahi)
        } else if blo < bhi {
            return dump(tag: "+", x: b, lo: blo, hi: bhi)
        }
        return []
    }

    /// `Differ._fancy_replace` — find a similar (ratio > 0.75) synch pair and emit the
    /// intraline `- / ? / + / ?` quad; otherwise a straight plain replace.
    private static func fancyReplace(_ a: [String], _ alo: Int, _ ahi: Int,
                                    _ b: [String], _ blo: Int, _ bhi: Int) -> [String] {
        var bestRatio = 0.74
        let cutoff = 0.75
        let cruncher = Matcher<UnicodeScalar>(isJunk: { SwiftDifflib.isCharacterJunk($0) })
        var eqi: Int?
        var eqj: Int?
        var bestI = 0
        var bestJ = 0
        for j in blo..<bhi {
            let bj = b[j]
            cruncher.setSeq2(Array(bj.unicodeScalars))
            for i in alo..<ahi {
                let ai = a[i]
                if ai == bj {
                    if eqi == nil { eqi = i; eqj = j }
                    continue
                }
                cruncher.setSeq1(Array(ai.unicodeScalars))
                if cruncher.realQuickRatio() > bestRatio
                    && cruncher.quickRatio() > bestRatio
                    && cruncher.ratio() > bestRatio {
                    bestRatio = cruncher.ratio()
                    bestI = i
                    bestJ = j
                }
            }
        }
        if bestRatio < cutoff {
            if eqi == nil {
                return plainReplace(a, alo, ahi, b, blo, bhi)
            }
            bestI = eqi!
            bestJ = eqj!
            bestRatio = 1.0
        } else {
            eqi = nil
        }
        var out: [String] = []
        out.append(contentsOf: fancyHelper(a, alo, bestI, b, blo, bestJ))
        let aelt = a[bestI]
        let belt = b[bestJ]
        if eqi == nil {
            var atags = ""
            var btags = ""
            let c = Matcher<UnicodeScalar>(isJunk: { SwiftDifflib.isCharacterJunk($0) })
            c.setSeqs(Array(aelt.unicodeScalars), Array(belt.unicodeScalars))
            for op in c.getOpcodes() {
                let la = op.i2 - op.i1
                let lb = op.j2 - op.j1
                switch op.tag {
                case .replace:
                    atags += String(repeating: "^", count: la)
                    btags += String(repeating: "^", count: lb)
                case .delete:
                    atags += String(repeating: "-", count: la)
                case .insert:
                    btags += String(repeating: "+", count: lb)
                case .equal:
                    atags += String(repeating: " ", count: la)
                    btags += String(repeating: " ", count: lb)
                }
            }
            out.append(contentsOf: qformat(aelt, belt, atags, btags))
        } else {
            out.append("  " + aelt)
        }
        out.append(contentsOf: fancyHelper(a, bestI + 1, ahi, b, bestJ + 1, bhi))
        return out
    }

    /// `_keep_original_ws` — keep the original whitespace character wherever the tag is a
    /// space and the original is whitespace.
    private static func keepOriginalWS(_ line: [UnicodeScalar], _ tags: String) -> String {
        var result = String.UnicodeScalarView()
        let tagScalars = Array(tags.unicodeScalars)
        for (idx, c) in line.enumerated() {
            let tag = tagScalars[idx]
            if tag == " " && c.properties.isWhitespace {
                result.append(c)
            } else {
                result.append(tag)
            }
        }
        return String(result)
    }

    /// `str.rstrip()` — strip trailing whitespace only (leading spaces are significant in a
    /// `? ` hint line).
    private static func rstrip(_ s: String) -> String {
        var scalars = Array(s.unicodeScalars)
        while let last = scalars.last, last.properties.isWhitespace {
            scalars.removeLast()
        }
        return String(String.UnicodeScalarView(scalars))
    }

    /// `Differ._qformat` — the `- / ? / + / ?` quad. The `? ` lines carry their own trailing
    /// `\n` (exactly as the Python, which the caller's `"\n".join` then turns into a blank
    /// line between the two `? ` lines).
    private static func qformat(_ aline: String, _ bline: String,
                                _ atags: String, _ btags: String) -> [String] {
        let at = rstrip(keepOriginalWS(Array(aline.unicodeScalars), atags))
        let bt = rstrip(keepOriginalWS(Array(bline.unicodeScalars), btags))
        var out = ["- " + aline]
        if !at.isEmpty { out.append("? \(at)\n") }
        out.append("+ " + bline)
        if !bt.isEmpty { out.append("? \(bt)\n") }
        return out
    }
}
