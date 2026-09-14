import Testing
@testable import MLXUI

/// SET-3's G2 seam and G4 regression guard (`RSI/DelegateSettingsBacklog.md`) — a newly
/// ported tool that isn't placed in an `AgentToolGroup` family, or isn't named in the §8 copy
/// table (`AgentToolCopy`), fails here instead of shipping a bare snake_case row nobody named.
///
/// `MLXUITests` is wired only to the `MLXUI` (App Store) scheme (`CLAUDE.md`, "Build &
/// validate") and its `fileSystemSynchronizedGroups` doesn't include `Apps/` — so it cannot
/// compile `Apps/Direct/ShellTool.swift` and cannot construct a real `DirectTools.all()` to
/// call `ModelRunner.defaultTools()` under `DIRECT_BUILD`. `directOnlyToolNames` is that
/// file's `DirectTools.all()` tool-name list, kept in sync by hand — the closest this target
/// can get to asserting "both editions" without importing Direct-only source.
struct AgentToolGroupingTests {

    /// `Apps/Direct/ShellTool.swift`'s `DirectTools.all()`, minus names shared with the App
    /// Store edition's `ModelRunner.defaultTools()` (already covered by that call).
    private static let directOnlyToolNames: Set<String> = ["run_shell"]

    @Test func everyShippingToolLandsInExactlyOneGroup() {
        for tool in ModelRunner.defaultTools() {
            #expect(AgentToolGroup.group(for: tool.name) != nil)
        }
        for name in Self.directOnlyToolNames {
            #expect(AgentToolGroup.group(for: name) != nil)
        }
    }

    @Test func noGroupIsEmpty() {
        let grouped = AgentToolGroup.grouping(ModelRunner.defaultTools())
        for entry in grouped {
            #expect(!entry.tools.isEmpty)
        }
    }

    @Test func groupingOrderMatchesFamilyOrder() {
        let grouped = AgentToolGroup.grouping(ModelRunner.defaultTools())
        let indices = grouped.map { AgentToolGroup.allCases.firstIndex(of: $0.group)! }
        #expect(indices == indices.sorted())
    }

    @Test func everyShippingToolHasACopyTableEntry() {
        for tool in ModelRunner.defaultTools() {
            #expect(AgentToolCopy.knownToolNames.contains(tool.name))
        }
        for name in Self.directOnlyToolNames {
            #expect(AgentToolCopy.knownToolNames.contains(name))
        }
    }

    @Test func titleFallsBackToRawIdForAnUnknownTool() {
        #expect(AgentToolCopy.title(for: "some_future_tool") == "some_future_tool")
    }
}
