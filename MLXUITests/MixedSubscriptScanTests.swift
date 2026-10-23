import Foundation
import Testing

// WW2-3-6 — a static scan for the pattern behind the Kokoro crash (journal 2026-557/558): an `MLXArray` index that
// FOLLOWS another index in a subscript (`x[0..., 0..., idxArray]`, `x[0..., idxArray]`, `x[0 ..< 1, idxArray]`,
// `x[.ellipsis, idxArray]`, `x[0, idxArray]`) records an op on a thread-bound stream under mlx-swift 0.32, so its result
// can only be evaluated on the building thread. Measured on 0.32.3 (and all fine on 0.31.4): fine are `x[idxArray]`,
// `x[idxArray, 0..., 0...]` (array FIRST), `x[idxA, idxB]`, `Int` subscripts, slice + `Int`, `take`, and assignment.
// App code (`MLXUI/`, `Vendor/`) must not contain it; the packages are scanned for a report only.
//
// A text scan, not a type check: an index counts as an array when the name is declared in the same file as an
// `MLXArray` (`: MLXArray`, `= MLXArray(…)`, `= MLX.…`, `.asType(…)`, `argMax`/`argSort`, …) or is written inline as
// `MLXArray(…)`. `.item(…)` results and loop counters are `Int`. It can miss an array whose declaration is in another
// file; it is a net, not a proof.

enum MixedSubscriptScanner {
    struct Finding: Equatable { let line: Int; let text: String }

    private static let arrayDeclarationHints = [
        "MLXArray", "MLX.", "asType(", "argMax", "argSort", "argPartition", "floor(", "ceil(", "round(",
        "MLXRandom", "take(", "where(", "stacked(", "concatenated(", "minimum(", "maximum(", "clip(",
    ]

    static func scan(_ source: String) -> [Finding] {
        let lines = source.components(separatedBy: "\n")
        let code = lines.map { line -> String in
            if let r = line.range(of: "//") { return String(line[..<r.lowerBound]) }
            return line
        }
        var findings: [Finding] = []
        // `name[ … ]` — a subscript directly after an identifier, `)` or `]`; array literals (`= [a, b]`) don't match.
        let pattern = try! NSRegularExpression(pattern: #"([A-Za-z0-9_\)\]]+)\s*\[([^\[\]\n]*)\]"#)
        let keywords: Set<String> = ["return", "throw", "in", "case", "else", "if", "while", "guard", "where", "as", "is", "try", "await"]
        for (index, line) in code.enumerated() {
            let ns = line as NSString
            for match in pattern.matches(in: line, range: NSRange(location: 0, length: ns.length)) {
                let receiver = ns.substring(with: match.range(at: 1))
                if keywords.contains(receiver) { continue }                            // an array literal after a keyword
                // an assignment target (`x[0..., idx] = v`) measured fine on 0.32.3 (scatter, not a gather): skip it
                let after = ns.substring(from: match.range.location + match.range.length)
                if after.range(of: #"^\s*=[^=]"#, options: .regularExpression) != nil { continue }
                let inner = ns.substring(with: match.range(at: 2))
                let elements = topLevelElements(inner)
                guard elements.count >= 2 else { continue }
                let flags = elements.map { isArrayIndex($0.trimmingCharacters(in: .whitespaces), before: index, in: code) }
                // an array index at position ≥ 1 with anything non-array before it
                var risky = false
                for position in 1 ..< elements.count where flags[position] && flags[0 ..< position].contains(false) { risky = true }
                if risky {
                    findings.append(Finding(line: index + 1, text: lines[index].trimmingCharacters(in: .whitespaces)))
                    break
                }
            }
        }
        return findings
    }

    private static func topLevelElements(_ inner: String) -> [String] {
        var parts: [String] = []
        var depth = 0
        var current = ""
        for ch in inner {
            if ch == "(" { depth += 1 } else if ch == ")" { depth -= 1 }
            if ch == "," && depth == 0 { parts.append(current); current = "" } else { current.append(ch) }
        }
        parts.append(current)
        return parts
    }

    /// Whether `element` is an `MLXArray` index. A bare identifier is classified by its **nearest preceding** binding in
    /// the file (so a loop counter `k` is not mistaken for an unrelated `let k = project(…)` further down).
    private static func isArrayIndex(_ element: String, before lineIndex: Int, in code: [String]) -> Bool {
        if element.hasPrefix("MLXArray(") { return true }
        guard element.range(of: #"^[a-z_][A-Za-z0-9_]*$"#, options: .regularExpression) != nil else { return false }
        let name = NSRegularExpression.escapedPattern(for: element)
        for line in code[0 ... lineIndex].reversed() {
            if line.range(of: "\\bfor\\s+\(name)\\s+in\\b", options: .regularExpression) != nil { return false }        // a counter
            if line.range(of: "\\b\(name)\\s*:\\s*MLXArray\\b", options: .regularExpression) != nil { return true }   // typed parameter
            if line.range(of: "\\b\(name)\\s*:\\s*(Int|Int32|Int64)\\b", options: .regularExpression) != nil { return false }
            if let decl = line.range(of: "\\b(let|var)\\s+\(name)\\b[^=]*=(.*)$", options: .regularExpression) {
                let text = String(line[decl])
                if text.contains(".item(") || text.contains("Int(") || text.contains(".count") { return false }
                return arrayDeclarationHints.contains { text.contains($0) }
            }
        }
        return false
    }

    static func swiftFiles(under directory: URL) -> [URL] {
        guard let walker = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil) else { return [] }
        return walker.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
    }
}

struct MixedSubscriptScanTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    private static var sourcesReadable: Bool {
        FileManager.default.isReadableFile(atPath: repoRoot.appendingPathComponent("MLXUI").path)
    }

    // MARK: the scanner itself

    @Test func itFlagsTheKokoroPattern() {
        let source = """
        let xLow = MLX.floor(xCoords).asType(.int32)
        let yLow = input[0..., 0..., xLow]
        let z = x[0..., MLXArray(positions)]
        func f(_ y: MLXArray, idx: MLXArray) { _ = y[.ellipsis, idx]; _ = y[0, idx] }
        """
        #expect(MixedSubscriptScanner.scan(source).map(\.line) == [2, 3, 4])
    }

    @Test func itLeavesIntSubscriptsAndPureArrayIndicesAlone() {
        let source = """
        let k = 3
        let eosIndex = tokens.firstIndex(of: 2) ?? 0
        let i = Int(round(x))
        let a = ids[.ellipsis, k]
        let b = hidden[0 ..< 1, eosIndex]
        let c = emb[0..., 0..., layer]
        let n = scores.max()!.item(Int.self)
        let d = masks[0, 0..., 0..., n]
        let idx = MLXArray([1, 2])
        let e = weights[idx]
        let list = [idx, idx]
        let first = y[idx, 0..., 0...]
        let both = y[idx, idx]
        """
        #expect(MixedSubscriptScanner.scan(source).isEmpty)
    }

    @Test func itIgnoresComments() {
        #expect(MixedSubscriptScanner.scan("// let y = input[0..., 0..., MLXArray([1])]").isEmpty)
    }

    @Test func itClassifiesTheNearestBindingNotAnUnrelatedLaterOne() {
        let source = """
        for k in 0 ..< n { let emb = table[k](tokens[0..., 0..., k]) }
        let k = project(x, kProj)
        """
        #expect(MixedSubscriptScanner.scan(source).isEmpty)
    }

    // MARK: the repository

    /// App code must not contain the pattern (`Vendor/PaddleOCRVL` is ours too). Skipped where the sources are not
    /// readable (a sandboxed test host).
    @Test(.enabled(if: MixedSubscriptScanTests.sourcesReadable))
    func appSourcesHaveNoMixedArrayIndexSubscripts() {
        // Reviewed and accepted, with the reason. Each entry must keep being true; remove it when the code changes.
        let knownBenign: [String: String] = [
            "MLXUI/Modules/StableDiffusion/SDOpenCLIPEncoder.swift":
                "`hidden[0 ..< 1, eosIndex]` with `eosIndex = argMax(…)[0]` (a 0-d MLXArray). Built and `eval`'d in one "
                + "synchronous section of `SDXLTurboEngine.generate` (no `await` between the encode and `eval(…, pooled)`), so it "
                + "never crosses a thread today. Journal 2026-558; swap for `take` if that ever changes.",
        ]
        var hits: [String] = []
        for dir in ["MLXUI", "Vendor"] {
            for file in MixedSubscriptScanner.swiftFiles(under: Self.repoRoot.appendingPathComponent(dir)) {
                guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
                let relative = file.path.replacingOccurrences(of: Self.repoRoot.path + "/", with: "")
                for finding in MixedSubscriptScanner.scan(text) where knownBenign[relative] == nil {
                    hits.append("\(relative):\(finding.line): \(finding.text)")
                }
            }
        }
        #expect(hits.isEmpty, "mixed slice + MLXArray-index subscript (thread-bound stream under mlx-swift 0.32):\n\(hits.joined(separator: "\n"))")
    }

    /// Report only: the resolved packages' sources, when a checkouts directory is given
    /// (`TEST_RUNNER_MLXUI_SCAN_PACKAGES=<DerivedData>/SourcePackages/checkouts`). Prints, never fails — a library
    /// finding is an upstream issue (`RSI/notes/ww2/upstream-issues.md`), not something this repo can fix.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLXUI_SCAN_PACKAGES"] != nil))
    func libraryScanReport() {
        let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MLXUI_SCAN_PACKAGES"] ?? "")
        var reported = 0
        for package in ["mlx-swift-lm", "mlx-audio-swift", "mlx-swift"] {
            for file in MixedSubscriptScanner.swiftFiles(under: root.appendingPathComponent(package)) {
                guard !file.path.contains("/Tests/"), let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
                for finding in MixedSubscriptScanner.scan(text) {
                    print("MIXEDSUBSCRIPT \(package)/\(file.path.components(separatedBy: "/\(package)/").last ?? ""):\(finding.line): \(finding.text)")
                    reported += 1
                }
            }
        }
        print("MIXEDSUBSCRIPT total \(reported)")
    }
}
