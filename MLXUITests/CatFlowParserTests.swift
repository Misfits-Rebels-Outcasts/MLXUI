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
            _ = try CatParser.parse("catflow 0.4\n1. Read Audio  talk.m4a\n")
        }
        #expect(throws: CatParserError.self) {
            _ = try CatParser.parse("catflow 0.9\n1. Read Audio  talk.m4a\n")
        }
    }

    @Test func e101MessageQuotesTheVersion() throws {
        do {
            _ = try CatParser.parse("catflow 0.4\n1. Read Audio  talk.m4a\n")
            Issue.record("expected E101")
        } catch let err as CatParserError {
            let msg = String(describing: err)
            #expect(msg == "⚠ This flow says `catflow 0.4`. This runtime speaks 0.8 only, and there is no converter. Nothing was run.")
        }
    }

    // MARK: - Unknown header flag (E102)

    @Test func unknownHeaderFlagRaisesE102() throws {
        do {
            _ = try CatParser.parse("catflow 0.8 · nope\n1. Read Audio  talk.m4a\n")
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
                #expect(String(describing: err) == "⚠ \(subject) uses `\(c.char)`, which CAT Flow 0.8 replaced with `\(c.ascii)`. Run `catflow fmt --upgrade` to convert this file. Nothing was run.", "\(c.name): message")
                checked += 1
            }
        }
        #expect(checked == cases.count)
    }

    // MARK: - E105 / E106 / E107 parse errors

    @Test func nonRowLineRaisesE105() throws {
        do {
            _ = try CatParser.parse("catflow 0.8\nblah\n")
            Issue.record("expected E105")
        } catch let err as CatParserError {
            #expect(String(describing: err).hasPrefix("⚠ Row 2 couldn't be read from \"blah…\". A row is: `N. Task  (refs)  model; settings`."))
        }
    }

    @Test func unclosedQuoteRaisesE106() throws {
        do {
            _ = try CatParser.parse("catflow 0.8\n1. Summarize  Qwen3 8B; \"never closed\n")
            Issue.record("expected E106")
        } catch let err as CatParserError {
            #expect(String(describing: err) == "⚠ Row 2's quoted text opens with `\"` but never closes. If the text should contain a quote, write `\\\"`.")
        }
    }

    @Test func duplicatePositionRaisesE107() throws {
        do {
            _ = try CatParser.parse("catflow 0.8\n1. Read Audio  a\n1. Read Text  b\n")
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
