import Testing
import Foundation
@testable import MLXUI

/// FH-4 (`RSI/DelegateOutputViewerBacklog.md`) — one case per `FlowFlagStatus`, built from
/// real parsed flows where the backlog names one, a synthetic minimal case otherwise.
struct CatFlowFlagInventoryTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // PipelineStudio
    }

    private func tempWorkspace() -> (workspace: FlowWorkspace, cleanup: () -> Void) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-flaginventory-\(UUID().uuidString)")
        return (FlowWorkspace(root: root), { try? FileManager.default.removeItem(at: root) })
    }

    // MARK: - requiredAndDeclared — flow 76 (Basic Gallery, Extract Web Page Links)

    /// The concrete example the backlog names: flow 76 declares `network` on line one and its
    /// `Web Fetch` row (row 3) genuinely needs it.
    @Test func flow76DeclaresNetworkAndNeedsIt() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/BasicGallery/5-ExtractWebPageLinks.cat")
        let text = try String(contentsOf: url, encoding: .utf8)
        let doc = try CatParser.parse(text)
        #expect(doc.flags.contains(.network))

        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        let status = FlowFlagInventory.status(of: .network, in: doc, workspace: workspace, flowID: "flow-76")
        guard case .requiredAndDeclared(let rowNumber, let task) = status else {
            Issue.record("expected .requiredAndDeclared, got \(status)")
            return
        }
        #expect(rowNumber == "3")
        #expect(task == "Web Fetch")
    }

    // MARK: - requiredButMissing — the same flow, with `network` stripped

    /// A copy of flow 76 with `network` stripped: the same `Web Fetch` row still needs it, but
    /// now nothing declares it — `requiredButMissing`, carrying `E103`'s own (golden-tested,
    /// never-raised-until-now) template.
    @Test func flow76WithNetworkStrippedIsRequiredButMissing() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/BasicGallery/5-ExtractWebPageLinks.cat")
        let text = try String(contentsOf: url, encoding: .utf8)
        let doc = try CatParser.parse(text)
        let stripped = FlowHeaderRepair.remove(.network, from: doc)
        #expect(!stripped.flags.contains(.network))

        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        let status = FlowFlagInventory.status(of: .network, in: stripped, workspace: workspace, flowID: "flow-76")
        guard case .requiredButMissing(let code, let message) = status else {
            Issue.record("expected .requiredButMissing, got \(status)")
            return
        }
        #expect(code == "E103")
        #expect(message.contains("Row 3"))
        #expect(message.contains("Web Fetch"))
        #expect(message.contains("network"))
    }

    // MARK: - declaredNotRequired

    @Test func aDeclaredFlagNoRowNeedsIsDeclaredNotRequired() throws {
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow",
                               rows: [Row(id: UUID(), task: "Read Text", settings: "memo.txt")],
                               flags: [.network], flagsOrder: [.network])
        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        #expect(FlowFlagInventory.status(of: .network, in: doc, workspace: workspace, flowID: "flow-1") == .declaredNotRequired)
    }

    @Test func anUndeclaredFlagNoRowNeedsIsAlsoDeclaredNotRequired() throws {
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow",
                               rows: [Row(id: UUID(), task: "Read Text", settings: "memo.txt")])
        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        #expect(FlowFlagInventory.status(of: .network, in: doc, workspace: workspace, flowID: "flow-1") == .declaredNotRequired)
    }

    // MARK: - inherited — a synthetic `uses:` caller/callee, mirroring `uses_example`'s shape

    /// `uses_example`'s own bundled files declare no capability flag today (checked against
    /// the shipped fixture before writing this), so this pins the mechanism with a minimal
    /// caller/callee pair that does — same shape (`uses: X = ./X.cat`), a flag on the callee.
    /// Uses `offdevice`, not `code`/`improvise`: those two are always `.refusedByChannel` on
    /// this (App Store) test host regardless of need or inheritance, which would make this
    /// test pass for the wrong reason.
    @Test func aUsesSiblingDeclaringOffdeviceGivesInherited() throws {
        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        let flowID = "caller-flow"
        let dir = workspace.directory(for: flowID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try """
        mlxflow 0.8; offdevice
        1. Read Text   context.txt
        """.write(to: dir.appendingPathComponent("Callee.cat"), atomically: true, encoding: .utf8)

        let callerText = """
        mlxflow 0.8
        1. Read Text   notes.txt
        2. Callee      (1)
        3. Save Text   out.md

        uses:
          Callee = ./Callee.cat
        """
        let doc = try CatParser.parse(callerText)
        #expect(!doc.flags.contains(.offdevice))

        let status = FlowFlagInventory.status(of: .offdevice, in: doc, workspace: workspace, flowID: flowID)
        #expect(status == .inherited(path: "./Callee.cat"))
    }

    /// A flag the caller already declares itself is never "inherited" — the doc's own
    /// declaration takes precedence over anything a `uses:` sibling also happens to declare.
    @Test func aFlagTheCallerAlreadyDeclaresIsNeverInherited() throws {
        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        let flowID = "caller-flow"
        let dir = workspace.directory(for: flowID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try """
        mlxflow 0.8; offdevice
        1. Read Text   context.txt
        """.write(to: dir.appendingPathComponent("Callee.cat"), atomically: true, encoding: .utf8)

        let callerText = """
        mlxflow 0.8; offdevice
        1. Read Text   notes.txt
        2. Callee      (1)
        3. Save Text   out.md

        uses:
          Callee = ./Callee.cat
        """
        let doc = try CatParser.parse(callerText)
        let status = FlowFlagInventory.status(of: .offdevice, in: doc, workspace: workspace, flowID: flowID)
        if case .inherited = status {
            Issue.record("a self-declared flag must not read as inherited, got \(status)")
        }
    }

    // MARK: - refusedByChannel

    /// `CatFlowCapabilityGateTests` already pins that the MLXUI test host builds
    /// `APPSTORE_BUILD` — `code`/`improvise` are refused outright there regardless of need.
    @Test func codeOnTheAppStoreBuildIsRefusedByChannelRegardlessOfNeed() throws {
        #expect(CapabilityGate.isAppStoreBuild)
        let doc = try CatParser.parse("""
            mlxflow 0.8; code
            1. Tidy
            transforms:
              Tidy  text -> text
                run: script.sh
            """)
        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        #expect(FlowFlagInventory.status(of: .code, in: doc, workspace: workspace, flowID: "flow-1") == .refusedByChannel)
    }

    // MARK: - The other flags' `requiringRow` predicates

    // No `improvise`-required test: `improvise` is always `.refusedByChannel` on this (App
    // Store) test host regardless of whether a row needs it (proven by
    // `codeOnTheAppStoreBuildIsRefusedByChannelRegardlessOfNeed` above, same mechanism) — its
    // `requiringRow` predicate (an `Improvise` row) can only be observed on the Direct build,
    // which has no test host in this suite (the same standing gap `CatFlowHeaderRepairTests
    // .underAppStoreBuildNoActionForCodeOrImproviseButOneForOffdeviceAndEvents` already
    // documents for `headerRepair`).

    @Test func eventsTriggerRowRequiresTheEventsFlag() throws {
        let doc = try CatParser.parse("""
            mlxflow 0.8; events
            1. On File      inbox/
            2. Save Text    out.md
            """)
        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        let status = FlowFlagInventory.status(of: .events, in: doc, workspace: workspace, flowID: "flow-1")
        guard case .requiredAndDeclared(let rowNumber, _) = status else {
            Issue.record("expected .requiredAndDeclared, got \(status)")
            return
        }
        #expect(rowNumber == "1")
    }

    @Test func aRemoteModelRowRequiresTheOffdeviceFlag() throws {
        let doc = try CatParser.parse("""
            mlxflow 0.8; offdevice
            1. Read Text    memo.txt
            2. Answer       (1)  claude-sonnet @ anthropic
            3. Save Text    out.md

            models:
              claude-sonnet @ anthropic = anthropic/claude-sonnet-4
            """)
        let (workspace, cleanup) = tempWorkspace()
        defer { cleanup() }
        let status = FlowFlagInventory.status(of: .offdevice, in: doc, workspace: workspace, flowID: "flow-1")
        guard case .requiredAndDeclared(let rowNumber, let task) = status else {
            Issue.record("expected .requiredAndDeclared, got \(status)")
            return
        }
        #expect(rowNumber == "2")
        #expect(task == "Answer")
    }
}
