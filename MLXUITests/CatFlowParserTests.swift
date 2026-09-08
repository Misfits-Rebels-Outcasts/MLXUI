import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R5-2: `CatParser` (port of `core/parser.py`) reproduces the Python
/// parse trees for every conformance flow, byte-for-byte against the `tree` object of
/// `conformance/*.expect.json` / `tests/goldens/parse/*.json`. Also pins the parse-time
/// error messages (E101/E102/E105/E106/E107/E108) against the adversarial fixtures.
struct CatFlowParserTests {

    // MARK: - Fixtures dir (repo-relative, mirroring the other CatFlow tests)

    private var fixturesDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow")
    }

    private func parseTreeJSON(from data: Data) throws -> [String: Any] {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TestError("expected a JSON object")
        }
        return obj
    }

    private struct TestError: Error { let message: String; init(_ m: String) { self.message = m } }

    // MARK: - All 41 conformance flows reproduce their golden parse trees

    @Test func allConformanceFlowsMatchGoldenTrees() throws {
        let conformanceDir = fixturesDir.appendingPathComponent("conformance")
        let cats = try FileManager.default.contentsOfDirectory(atPath: conformanceDir.path)
            .filter { $0.hasSuffix(".cat") }
            .sorted()
        #expect(cats.count == 41)

        var checked = 0
        for cat in cats {
            let name = String(cat.dropLast(".cat".count))
            let catURL = conformanceDir.appendingPathComponent(cat)
            let catText = try String(contentsOf: catURL, encoding: .utf8)

            let doc = try CatParser.parse(catText)
            let actual = try parseTreeJSON(from: try doc.toParseTreeJSON())

            let expectURL = conformanceDir.appendingPathComponent("\(name).expect.json")
            let expectObj = try parseTreeJSON(from: try Data(contentsOf: expectURL))
            guard let expected = expectObj["tree"] as? [String: Any] else {
                throw TestError("\(name): expect file has no `tree`")
            }

            if !anyEquals(actual, expected) {
                let actualJSON = String(data: try JSONSerialization.data(withJSONObject: actual, options: [.sortedKeys]), encoding: .utf8) ?? "?"
                let expectedJSON = String(data: try JSONSerialization.data(withJSONObject: expected, options: [.sortedKeys]), encoding: .utf8) ?? "?"
                Issue.record("\(name): parse tree mismatch.\nSWIFT: \(actualJSON)\nPYTHON: \(expectedJSON)")
            }
            checked += 1
        }
        #expect(checked == 41)
    }

    // MARK: - Golden parse trees also match (tests/goldens/parse/*.json)

    @Test func allGoldenParseTreesMatch() throws {
        let parseDir = fixturesDir.appendingPathComponent("goldens/parse")
        let cats = try FileManager.default.contentsOfDirectory(atPath: parseDir.path)
            .filter { $0.hasSuffix(".cat") }
            .sorted()
            .filter { FileManager.default.fileExists(atPath: parseDir.appendingPathComponent(String($0.dropLast(".cat".count)) + ".json").path) }
        // The goldens/parse corpus predates the conformance corpus; not every .cat
        // carries a .json sibling. Assert the complete pairs that exist.
        #expect(cats.count >= 15)

        var checked = 0
        for cat in cats {
            let name = String(cat.dropLast(".cat".count))
            let catText = try String(contentsOf: parseDir.appendingPathComponent(cat), encoding: .utf8)
            let doc = try CatParser.parse(catText)
            let actual = try parseTreeJSON(from: try doc.toParseTreeJSON())
            let expected = try parseTreeJSON(from: try Data(contentsOf: parseDir.appendingPathComponent("\(name).json")))

            if !anyEquals(actual, expected) {
                let actualJSON = String(data: try JSONSerialization.data(withJSONObject: actual, options: [.sortedKeys]), encoding: .utf8) ?? "?"
                let expectedJSON = String(data: try JSONSerialization.data(withJSONObject: expected, options: [.sortedKeys]), encoding: .utf8) ?? "?"
                Issue.record("\(name): parse tree mismatch.\nSWIFT: \(actualJSON)\nPYTHON: \(expectedJSON)")
            }
            checked += 1
        }
        #expect(checked == cats.count)
    }

    // MARK: - Version gating (E101)

    @Test func headerlessIsAccepted() throws {
        let doc = try CatParser.parse("1. Read Audio  talk.m4a\n")
        #expect(doc.version == "")
        #expect(doc.rows.count == 1)
        #expect(doc.rows.first?.task == "Read Audio")
    }

    @Test func nonV08HeadersRaiseE101() {
        #expect(throws: CatParserError.self) {
            _ = try CatParser.parse("mlxflow 0.4\n1. Read Audio  talk.m4a\n")
        }
        #expect(throws: CatParserError.self) {
            _ = try CatParser.parse("mlxflow 0.9\n1. Read Audio  talk.m4a\n")
        }
    }

    @Test func e101MessageQuotesTheVersion() throws {
        do {
            _ = try CatParser.parse("mlxflow 0.4\n1. Read Audio  talk.m4a\n")
            Issue.record("expected E101")
        } catch let err as CatParserError {
            let msg = String(describing: err)
            #expect(msg == "⚠ This flow says `mlxflow 0.4`. This runtime speaks 0.8 only, and there is no converter. Nothing was run.")
        }
    }

    // MARK: - Unknown header flag (E102)

    @Test func unknownHeaderFlagRaisesE102() throws {
        do {
            _ = try CatParser.parse("mlxflow 0.8 · nope\n1. Read Audio  talk.m4a\n")
            Issue.record("expected E102")
        } catch let err as CatParserError {
            // E102's v0.8 catalog always renders the `; ` separator, even for a `·` header.
            #expect(String(describing: err) == "⚠ This flow declares `; nope`, which this runtime doesn't support. Nothing was run.")
        }
    }

    // MARK: - E108 (adversarial fixtures)

    private struct E108Case: Decodable {
        var name: String
        var source: String
        var char: String
        var ascii: String
        var line: Int
        var row: Int?
        var section: String?
    }

    @Test func preV08CharactersRaiseE108() throws {
        let url = fixturesDir.appendingPathComponent("goldens/errors/adversarial_v08.json")
        let data = try Data(contentsOf: url)
        let cases = try JSONDecoder().decode([E108Case].self, from: data)

        var checked = 0
        for c in cases {
            do {
                _ = try CatParser.parse(c.source)
                Issue.record("\(c.name): expected E108 but parsed cleanly")
            } catch let err as CatParserError {
                guard case .preV08Character(let char, let ascii, let line, let row, let section) = err else {
                    Issue.record("\(c.name): expected preV08Character, got \(err)")
                    continue
                }
                #expect(char == c.char, "\(c.name): char")
                #expect(ascii == c.ascii, "\(c.name): ascii")
                #expect(line == c.line, "\(c.name): line")
                #expect(row == c.row, "\(c.name): row")
                #expect(section == c.section, "\(c.name): section")
                let subject = row.map { "Row \($0)" } ?? section.map { "The \($0): section" } ?? "Line \(line)"
                #expect(String(describing: err) == "⚠ \(subject) uses `\(c.char)`, which mlx-workflow 0.8 replaced with `\(c.ascii)`. Run `mlxflow fmt --upgrade` to convert this file. Nothing was run.", "\(c.name): message")
                checked += 1
            }
        }
        #expect(checked == cases.count)
    }

    // MARK: - CFM-R18-1: `mlxflow`/`mlxpipeline` header aliases (SPEC-Q — `mlxflow 0.8`,
    // the header keyword the reference already writes; `parser.py:219/230`'s four-spelling
    // alias + `:346`'s normalization to the two-value internal vocabulary). CFM-R18-6 has
    // flipped every other header literal in `MLXUITests/` to `mlxflow`, so this block — plus
    // `fmtOfACatflowHeadedFileStillEmitsCatflow` below — is now the *whole* owned regression
    // that a pre-rename `catflow 0.8` file still parses. Do not delete it as "redundant" with
    // the fixture corpus; the 30 old-spelling conformance inputs are regenerated from
    // upstream and are not ours to keep.

    @Test(arguments: ["catflow", "mlxflow"])
    func flowFamilyAliasesParseIdenticalDocuments(_ keyword: String) throws {
        let doc = try CatParser.parse("\(keyword) 0.8; code\n1. Read Audio  talk.m4a\n")
        #expect(doc.version == "0.8")
        #expect(doc.headerKeyword == keyword)
        #expect(doc.fileKind == .catflow)
        #expect(doc.flags.contains(.code))
        #expect(doc.rows.count == 1)
        #expect(doc.rows.first?.task == "Read Audio")
    }

    @Test(arguments: ["catpipeline", "mlxpipeline"])
    func pipelineFamilyAliasesParseIdenticalDocuments(_ keyword: String) throws {
        let doc = try CatParser.parse("\(keyword) 0.8\n  pipeline Test Pipeline\n  accepts: [text]\n  gives:   text\n1. Read Text  a.txt\n")
        #expect(doc.version == "0.8")
        #expect(doc.headerKeyword == keyword)
        #expect(doc.fileKind == .catpipeline)
        #expect(doc.pipelineName == "Test Pipeline")
        #expect(doc.accepts == [.text])
        #expect(doc.gives == "text")
        #expect(doc.rows.count == 1)
    }

    /// A pre-0.8 version under the `mlxflow` spelling still raises E101, its `token` quotes
    /// exactly what was written (`"mlxflow 0.7"`), and — since `CFM-R18-2` rebranded the E101
    /// template — the rendered `.description` now says `mlxflow` too (the E101 wording is a
    /// fixed `mlxflow {version}`, not an echo of the written keyword, matching the reference).
    @Test func mlxflowInvalidVersionQuotesTheWrittenToken() throws {
        do {
            _ = try CatParser.parse("mlxflow 0.7\n1. Read Audio  talk.m4a\n")
            Issue.record("expected E101")
        } catch let err as CatParserError {
            guard case .needsVersion(let token, let version, let line) = err else {
                Issue.record("expected needsVersion, got \(err)")
                return
            }
            #expect(token == "mlxflow 0.7")
            #expect(version == "0.7")
            #expect(line == 1)
            #expect(String(describing: err) == "⚠ This flow says `mlxflow 0.7`. This runtime speaks 0.8 only, and there is no converter. Nothing was run.")
        }
    }

    /// Every accepted spelling still refuses a non-0.8 version.
    @Test(arguments: ["catflow", "catpipeline", "mlxflow", "mlxpipeline"])
    func everyAliasSpellingStillGatesOnVersion(_ keyword: String) {
        #expect(throws: CatParserError.self) {
            _ = try CatParser.parse("\(keyword) 0.7\n1. Read Audio  talk.m4a\n")
        }
    }

    /// `parse(fmt(x)) == parse(x)` for both families — round-tripping through the
    /// serializer never changes the parse tree, whichever header family wrote it.
    @Test(arguments: ["catflow", "mlxflow", "catpipeline", "mlxpipeline"])
    func fmtRoundTripPreservesTheParseTreeForBothFamilies(_ keyword: String) throws {
        let isPipeline = keyword.hasSuffix("pipeline")
        let source = isPipeline
            ? "\(keyword) 0.8\n  pipeline Test Pipeline\n  gives:   text\n1. Read Text  a.txt\n"
            : "\(keyword) 0.8; code\n1. Read Audio  talk.m4a\n"

        let original = try CatParser.parse(source)
        let formatted = CatSerializer.serialize(original)
        let reparsed = try CatParser.parse(formatted)

        let originalTree = try parseTreeJSON(from: try original.toParseTreeJSON())
        let reparsedTree = try parseTreeJSON(from: try reparsed.toParseTreeJSON())
        #expect(anyEquals(originalTree, reparsedTree), "\(keyword): parse(fmt(x)) != parse(x)")
        #expect(reparsed.headerKeyword == keyword, "\(keyword): fmt round-trip lost the header family")
    }

    /// The regression this whole item exists to catch: `fmt` of a `catflow`-headed file
    /// must still emit `catflow` — `fmt.py:215-216` preserves the source family, it does
    /// not canonicalize to whatever the reference's own default is. Mirrors the fact that
    /// all eight `tests/goldens/fmt/*.fmt.cat` still say `catflow 0.8`. CFM-R18-6 flipped
    /// every other header literal in this file to `mlxflow`; these three `catflow` uses are
    /// load-bearing — do not "tidy" them.
    @Test func fmtOfACatflowHeadedFileStillEmitsCatflow() throws {
        let doc = try CatParser.parse("catflow 0.8\n1. Read Audio  talk.m4a\n")
        let out = CatSerializer.serialize(doc)
        #expect(out.hasPrefix("catflow 0.8\n"))
    }

    /// And the mirror: `fmt` of an `mlxflow`-headed file emits `mlxflow`, not `catflow`.
    @Test func fmtOfAnMlxflowHeadedFileStillEmitsMlxflow() throws {
        let doc = try CatParser.parse("mlxflow 0.8\n1. Read Audio  talk.m4a\n")
        let out = CatSerializer.serialize(doc)
        #expect(out.hasPrefix("mlxflow 0.8\n"))
    }

    // MARK: - E105 / E106 / E107 parse errors

    @Test func nonRowLineRaisesE105() throws {
        do {
            _ = try CatParser.parse("mlxflow 0.8\nblah\n")
            Issue.record("expected E105")
        } catch let err as CatParserError {
            #expect(String(describing: err).hasPrefix("⚠ Row 2 couldn't be read from \"blah…\". A row is: `N. Task  (refs)  model; settings`."))
        }
    }

    @Test func unclosedQuoteRaisesE106() throws {
        do {
            _ = try CatParser.parse("mlxflow 0.8\n1. Summarize  Qwen3 8B; \"never closed\n")
            Issue.record("expected E106")
        } catch let err as CatParserError {
            #expect(String(describing: err) == "⚠ Row 2's quoted text opens with `\"` but never closes. If the text should contain a quote, write `\\\"`.")
        }
    }

    @Test func duplicatePositionRaisesE107() throws {
        do {
            _ = try CatParser.parse("mlxflow 0.8\n1. Read Audio  a\n1. Read Text  b\n")
            Issue.record("expected E107")
        } catch let err as CatParserError {
            #expect(String(describing: err) == "⚠ Two rows are numbered 1 in the same scope. Positions must be sequential — running fmt will renumber and re-aim references.")
        }
    }

    // MARK: - Deep equality helper (dictionaries/arrays/scalars)

    private func anyEquals(_ a: Any, _ b: Any) -> Bool {
        if let ad = a as? [String: Any], let bd = b as? [String: Any] {
            guard ad.count == bd.count else { return false }
            for (k, v) in ad {
                guard let bv = bd[k] else { return false }
                if !anyEquals(v, bv) { return false }
            }
            return true
        }
        if let aa = a as? [Any], let ba = b as? [Any] {
            guard aa.count == ba.count else { return false }
            for (x, y) in zip(aa, ba) where !anyEquals(x, y) {
                return false
            }
            return true
        }
        if let as_ = a as? NSNumber, let bs = b as? NSNumber {
            return as_.isEqual(to: bs) || as_.doubleValue == bs.doubleValue
        }
        if let an = a as? NSNull, let bn = b as? NSNull {
            return an === bn
        }
        return String(describing: a) == String(describing: b)
    }
}
