import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R5-3: `FlowValidator` (port of `core/validator.py`) reproduces the check
/// goldens — `[Issue{row, code, message}]` with the dotted-path row name and the exact
/// ⚠-free wording, against the synced curated registry (the E104 display-name exemption,
/// E209 identity canonicalization, and F010 all resolve through it, exactly as the Python
/// harness loads `models/curated`).
struct CatFlowValidatorTests {

    var fixturesDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow")
    }

    func loadRegistry() throws -> DirectoryFlowRegistry {
        try DirectoryFlowRegistry.load(directory: fixturesDir.appendingPathComponent("registry"))
    }

    struct CaseFixture: Decodable {
        var name: String
        var cat: String
        var issues: [IssueFixture]?
        var lints: [IssueFixture]?
    }

    struct IssueFixture: Decodable, Equatable {
        var row: String
        var code: String
        var message: String
    }

    // MARK: - fixtures.json — every case matches the golden

    /// SPEC-Q223: MLXUI's `Web Search` requires `; offdevice` even though the Python
    /// reference's own validator never checks for it there — the reference's `Web
    /// Search` (a DuckDuckGo scrape) has no manifest/credential concept at all to hang a
    /// provider check on, so `_check_offdevice_flag` never even looks at `.net`-class
    /// rows by name. This is a deliberate MLXUI-only strengthening (see the entry), not
    /// a bug — but it means the one fixture in this Python-synced corpus that exercises
    /// `Web Search` without `offdevice` now diverges from its golden by design.
    /// **Never edit the golden** (it's a synced, sha-pinned Python artifact) — instead
    /// this case is checked precisely against the *expected* divergence (exactly one new
    /// `E120`, nothing else different) rather than blindly skipped, so any *other*,
    /// unexpected drift in it would still fail loudly.
    private let knownGoldenDivergences: Set<String> = ["tools-doc-research-block-networked-tools-valid"]

    @Test func checkFixturesMatchGolden() throws {
        let url = fixturesDir.appendingPathComponent("goldens/check/fixtures.json")
        let cases = try JSONDecoder().decode([CaseFixture].self, from: try Data(contentsOf: url))
        let registry = try loadRegistry()
        #expect(cases.count == 99)

        var checked = 0
        for c in cases {
            let parsed = try CatParser.parseForValidation(c.cat)
            let issues = FlowValidator.checkFlow(parsed, registry: registry)
                .map { IssueFixture(row: $0.row, code: $0.code, message: $0.message) }
            let expected = (c.issues ?? []).map { IssueFixture(row: $0.row, code: $0.code, message: $0.message) }
            if knownGoldenDivergences.contains(c.name) {
                let withoutOffdevice = issues.filter { $0.code != "E120" }
                #expect(withoutOffdevice == expected, "\(c.name): unexpected non-E120 divergence")
                #expect(issues.contains { $0.code == "E120" && $0.row == "1" },
                        "\(c.name): expected exactly the SPEC-Q223 E120 addition on row 1")
            } else if issues != expected {
                Issue.record("\(c.name): mismatch.\nSWIFT: \(issues)\nGOLDEN: \(expected)")
            }
            checked += 1
        }
        #expect(checked == 99)
    }

    // MARK: - lint_fixtures.json

    @Test func lintFixturesMatchGolden() throws {
        let url = fixturesDir.appendingPathComponent("goldens/check/lint_fixtures.json")
        let cases = try JSONDecoder().decode([CaseFixture].self, from: try Data(contentsOf: url))
        let registry = try loadRegistry()
        #expect(cases.count == 7)

        var checked = 0
        for c in cases {
            let parsed = try CatParser.parseForValidation(c.cat)
            let lints = FlowValidator.lintFlow(parsed, registry: registry)
                .map { IssueFixture(row: $0.row, code: $0.code, message: $0.message) }
            let expected = (c.lints ?? []).map { IssueFixture(row: $0.row, code: $0.code, message: $0.message) }
            if lints != expected {
                Issue.record("\(c.name): mismatch.\nSWIFT: \(lints)\nGOLDEN: \(expected)")
            }
            checked += 1
        }
        #expect(checked == 7)
    }

    // MARK: - Conformance flows validate to their `check` expectations

    @Test func conformanceFlowsMatchCheckExpectations() throws {
        let conformanceDir = fixturesDir.appendingPathComponent("conformance")
        let cats = try FileManager.default.contentsOfDirectory(atPath: conformanceDir.path)
            .filter { $0.hasSuffix(".cat") }
            .sorted()
        let registry = try loadRegistry()

        var checked = 0
        for cat in cats {
            let name = String(cat.dropLast(".cat".count))
            let expectData = try Data(contentsOf: conformanceDir.appendingPathComponent("\(name).expect.json"))
            guard let expectObj = try JSONSerialization.jsonObject(with: expectData) as? [String: Any],
                  let check = expectObj["check"] as? [String: Any] else { continue }
            let expectedIssues = (check["issues"] as? [[String: Any]] ?? [])
                .map { (row: $0["row"] as? String ?? "", code: $0["code"] as? String ?? "", message: $0["message"] as? String ?? "") }

            let parsed = try CatParser.parseForValidation(
                try String(contentsOf: conformanceDir.appendingPathComponent(cat), encoding: .utf8)
            )
            let issues = FlowValidator.checkFlow(parsed, registry: registry)
                .map { IssueFixture(row: $0.row, code: $0.code, message: $0.message) }
            if issues != expectedIssues.map({ IssueFixture(row: $0.row, code: $0.code, message: $0.message) }) {
                Issue.record("\(name): check mismatch.\nSWIFT: \(issues)\nGOLDEN: \(expectedIssues)")
            }
            checked += 1
        }
        #expect(checked == cats.count)
    }
}
