import Testing
import Foundation
@testable import MLXUI

/// CFM-R13-1 — the `uses:` graph walk is a port of `core/uses.py::_resolve_level`, not the
/// old reach-and-swallow chain. The golden comes from `catflow check --json` (regenerate via
/// `Fixtures/CatFlow/validator/uses/gen.py`); the fixture exercises all four codes in one file
/// — E114 (a `..` escape and an absolute path), E116 (no-such-file, parse-error, child that
/// fails its own check), E117 (an inherited `improvise`), and E115 (a two-file cycle) — each
/// anchored to the row that names the entry.
struct CatFlowUsesChecksTests {

    var fixturesDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow")
    }

    @Test func usesFixtureMatchesPythonGolden() throws {
        let srcDir = fixturesDir.appendingPathComponent("validator/uses")
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-uses-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let flowDir = root.appendingPathComponent("t", isDirectory: true)
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)

        let fm = FileManager.default
        for file in try fm.contentsOfDirectory(at: srcDir, includingPropertiesForKeys: nil)
        where file.pathExtension == "cat" {
            try fm.copyItem(at: file, to: flowDir.appendingPathComponent(file.lastPathComponent))
        }

        let ws = FlowWorkspace(root: root)
        let flowID = "t"
        let rootText = try String(contentsOf: flowDir.appendingPathComponent("root.cat"), encoding: .utf8)
        let parsed = try CatParser.parseForValidation(rootText)
        let issues = FlowValidator.checkFlow(parsed, workspace: ws, flowID: flowID,
                                             rootFile: flowDir.appendingPathComponent("root.cat"))
        let actual = issues.map { "\($0.row)|\($0.code)|\($0.message)" }.sorted()

        let goldenURL = fixturesDir.appendingPathComponent("goldens/check/uses_golden.json")
        let goldenObj = try JSONSerialization.jsonObject(with: try Data(contentsOf: goldenURL)) as? [[String: Any]] ?? []
        let expected = goldenObj.map {
            "\($0["row"] as? String ?? "")|\($0["code"] as? String ?? "")|\($0["message"] as? String ?? "")"
        }.sorted()

        #expect(actual == expected,
                "uses checks differ from the Python golden\nSWIFT: \(actual)\nGOLDEN: \(expected)")
    }
}
