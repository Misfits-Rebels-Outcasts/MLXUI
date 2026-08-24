import Testing
import Foundation
@testable import MLXUI

/// CFM-R10-FIX-3 — the thirteen door checks (E109–E112, E118, E407–E409, E114–E117, E119)
/// that were defined in `ErrorCatalog` and never emitted. The Swift validator is pinned
/// against the Python-derived golden (`Fixtures/CatFlow/tools/door_checks_golden.json`), and
/// the E117 `uses:`-inherited `code` flag is refused under `APPSTORE_BUILD`.
struct CatFlowDoorChecksTests {

    private func parse(_ text: String) throws -> ParsedFlow {
        try CatParser.parseForValidation(text)
    }

    private func golden() throws -> [String: [[Any]]] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Fixtures/CatFlow/tools/door_checks_golden.json")
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as? [String: [[Any]]] ?? [:]
    }

    private let flows: [String: String] = [
        "improvise_no_flag": """
        catflow 0.8
        1. Read Text   memo.txt
        2. Improvise   workdir=scratch; max_actions=5; timeout=30s
        """,
        "improvise_unbounded": """
        catflow 0.8; improvise
        1. Improvise   workdir=scratch
        """,
        "improvise_no_workdir": """
        catflow 0.8; improvise
        1. Improvise   max_actions=5; timeout=30s
        """,
        "improvise_events": """
        catflow 0.8; improvise; events
        1. On File   inbox/
        2. Improvise   workdir=scratch; max_actions=5; timeout=30s
        """,
        "transforms_no_code": """
        catflow 0.8
        1. Tidy
        transforms:
          Tidy  text -> text
            run: script.sh
        """,
        "transform_incomplete": """
        catflow 0.8; code
        1. Tidy
        transforms:
          Tidy  text -> text
            run: script.sh
        """,
    ]

    @Test func doorChecksMatchPythonGolden() throws {
        let golden = try golden()
        for (name, text) in flows {
            let parsed = try parse(text)
            let issues = FlowValidator.checkFlow(parsed)
            let actual = issues.map { "\($0.row)|\($0.code)|\($0.message)" }.sorted()
            let expected = (golden[name] ?? []).map { triple in
                let row = triple[0] as? String ?? "\(triple[0])"
                let code = triple[1] as? String ?? ""
                let message = triple[2] as? String ?? ""
                return "\(row)|\(code)|\(message)"
            }.sorted()
            #expect(actual == expected, "\(name): door checks differ from the Python golden\n\(actual)")
        }
    }

    // MARK: - E117: the uses:-inherited capability is refused under APPSTORE_BUILD

    @Test func usesInheritedCodeFlagIsRefusedUnderAppStore() throws {
        // A flow that `uses:` a sibling declaring `code` while the header doesn't — the
        // header lies and the gate must not believe it (R28 / E117).
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-e117-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: root)
        let flowID = "t"
        let flowDir = ws.directory(for: flowID)
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        try "catflow 0.8; code\n1. Read Text   a.txt\n".write(
            to: flowDir.appendingPathComponent("lib.cat"), atomically: true, encoding: .utf8)
        let doc = try CatParser.parse("catflow 0.8\n1. Lib\nuses:\n  Lib = lib.cat\n")

        // The declared flags alone say clean; the effective flags (uses-inherited `code`)
        // must refuse under the App Store tier.
        #expect(CapabilityGate.effectiveFlags(doc, workspace: ws, flowID: flowID).contains("code"))
        if CapabilityGate.isAppStoreBuild {
            guard case .notRunnable(let reason) = FlowRunner.canRun(doc, flowID: flowID, workspace: ws) else {
                Issue.record("expected the uses-inherited code flag to refuse")
                return
            }
            #expect(reason.contains("code"))
        }
    }
}
