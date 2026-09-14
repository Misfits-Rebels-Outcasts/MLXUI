import Foundation

/// SET-3's §8 copy table (`RSI/DelegateSettingsBacklog.md`) — a verbatim transcription, not a
/// rewrite. `title(for:)` is what `ToolsSettingsView` renders instead of a raw snake_case tool
/// id, and it's the other half of the G4 regression guard in `AgentToolGroupingTests`: a tool
/// shipped without an entry here still renders (falls back to its raw id) rather than
/// crashing, but the test fails first so that never reaches a build.
nonisolated enum AgentToolCopy {
    private static let titles: [String: String] = [
        "fetch_url": "Fetch a web page",
        "run_javascript": "Run JavaScript",
        "calculator": "Calculate",
        "datetime": "Date and time",
        "read_clipboard": "Read the clipboard",
        "write_clipboard": "Write to the clipboard",
        "embed_text": "Embed text",
        "summarize": "Summarize",
        "semantic_search": "Search by meaning",
        "transcribe_audio": "Transcribe audio",
        "ocr_image": "Read text in an image",
        "read_file": "Read a file",
        "write_file": "Write a file",
        "list_directory": "List a folder",
        "search_files": "Search files",
        "run_shell": "Run a shell command",
        "echo": "Echo (debug build only)",
    ]

    /// The tool's human title from §8, or its raw id when this table has no entry yet.
    static func title(for toolName: String) -> String {
        titles[toolName] ?? toolName
    }

    static var knownToolNames: Set<String> { Set(titles.keys) }
}
