import Foundation

/// SET-3's G2 seam and G4 regression guard (`RSI/DelegateSettingsBacklog.md`) — the §0
/// families, in order. A newly ported tool that isn't added to `membership` fails
/// `AgentToolGroupingTests.everyShippingToolLandsInExactlyOneGroup` instead of shipping an
/// unlabeled row nobody placed anywhere.
nonisolated enum AgentToolGroup: String, CaseIterable, Hashable, Sendable {
    case web
    case computeAndClipboard
    case inAppModels
    case files
    case shell
    case debug

    var title: String {
        switch self {
        case .web: return "Web"
        case .computeAndClipboard: return "Compute & clipboard"
        case .inAppModels: return "In-app models"
        case .files: return "Files"
        case .shell: return "Shell"
        case .debug: return "Debug"
        }
    }

    /// §8's group note for the family that has an off-by-default tool (`write_file` in
    /// Files, `run_shell` in Shell) — rendered once per group, not repeated per tool.
    var offByDefaultNote: String? {
        switch self {
        case .files, .shell:
            return "Off until you turn it on — a model should not be able to propose this on first launch. Each use still asks."
        case .web, .computeAndClipboard, .inAppModels, .debug:
            return nil
        }
    }

    /// Every shipping tool's family, keyed by tool id.
    private static let membership: [String: AgentToolGroup] = [
        "fetch_url": .web,
        "run_javascript": .computeAndClipboard,
        "calculator": .computeAndClipboard,
        "datetime": .computeAndClipboard,
        "read_clipboard": .computeAndClipboard,
        "write_clipboard": .computeAndClipboard,
        "embed_text": .inAppModels,
        "summarize": .inAppModels,
        "semantic_search": .inAppModels,
        "transcribe_audio": .inAppModels,
        "ocr_image": .inAppModels,
        "read_file": .files,
        "write_file": .files,
        "list_directory": .files,
        "search_files": .files,
        "run_shell": .shell,
        "echo": .debug,
    ]

    /// The family a tool id belongs to, or `nil` if `membership` has no entry for it.
    static func group(for toolName: String) -> AgentToolGroup? {
        membership[toolName]
    }

    /// Groups `tools` into families, in family order (§0), dropping any family with no
    /// members among `tools` — the App Store edition naturally has no `.shell` row, and a
    /// Release build naturally has no `.debug` row.
    static func grouping(_ tools: [any AgentTool]) -> [(group: AgentToolGroup, tools: [any AgentTool])] {
        var byGroup: [AgentToolGroup: [any AgentTool]] = [:]
        for tool in tools {
            guard let group = group(for: tool.name) else { continue }
            byGroup[group, default: []].append(tool)
        }
        return allCases.compactMap { group in
            guard let tools = byGroup[group], !tools.isEmpty else { return nil }
            return (group, tools)
        }
    }
}
