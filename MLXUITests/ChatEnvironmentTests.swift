import Foundation
import Testing
@testable import MLXUI

/// CW-1 — the environment block (C3). Pure and deterministic: same inputs, same bytes, no dates.
/// The full text is pinned as a golden so the model-facing wording is reviewable at ▣ CW-A, and
/// the length cap keeps an 8B model's context from being spent on its own instructions.
struct ChatEnvironmentTests {

    private let readOnlyTools: Set<String> = ["read_file", "list_directory", "search_files"]
    private let writeTools: Set<String> = ["read_file", "write_file", "edit_file"]

    private let threeFolders = FileScope(
        folders: [
            "/Users/me/Desktop/workfolder",
            "/Users/me/Documents/Invoices 2026",
            "/Users/me/Projects/Some Long Named Project Folder",
        ],
        workingFolder: "/Users/me/Desktop/workfolder")

    // MARK: nil cases

    @Test func nilWhenNoFolderAndNoFileTool() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .appStore,
            fileToolsEnabled: [], writeEnabled: false)
        #expect(block == nil)
    }

    @Test func nilWhenFoldersExistButNoFileToolCanUseThem() {
        // The folder is in scope but every file tool is off, so there is nothing actionable to say.
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/workfolder"], workingFolder: "/Users/me/workfolder"),
            edition: .appStore, fileToolsEnabled: [], writeEnabled: false)
        #expect(block == nil)
    }

    // MARK: goldens

    @Test func goldenOneFolderWorkingReadWrite() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/Desktop/workfolder"],
                             workingFolder: "/Users/me/Desktop/workfolder"),
            edition: .appStore, fileToolsEnabled: writeTools, writeEnabled: true)
        #expect(block == """
        Folders you can use in this chat:
        - workfolder — /Users/me/Desktop/workfolder (working folder; you may write in it)
        Relative paths resolve against the working folder (/Users/me/Desktop/workfolder).
        """)
    }

    @Test func goldenThreeFoldersWorkingFolderFirst() {
        let block = ChatEnvironment.instructions(
            scope: threeFolders, edition: .appStore,
            fileToolsEnabled: writeTools, writeEnabled: true)
        #expect(block == """
        Folders you can use in this chat:
        - workfolder — /Users/me/Desktop/workfolder (working folder; you may write in it)
        - Invoices 2026 — /Users/me/Documents/Invoices 2026 (you may write in it)
        - Some Long Named Project Folder — /Users/me/Projects/Some Long Named Project Folder (you may write in it)
        Relative paths resolve against the working folder (/Users/me/Desktop/workfolder).
        """)
    }

    @Test func readOnlyFoldersAreMarkedAndSayWhatToDoNext() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/Desktop/workfolder"],
                             workingFolder: "/Users/me/Desktop/workfolder"),
            edition: .appStore, fileToolsEnabled: readOnlyTools, writeEnabled: false)
        #expect(block?.contains("read only") == true)
        #expect(block?.contains("Next: read the folders above; ask the user before writing.") == true)
        #expect(block?.contains("you may write in it") == false)
    }

    /// The write marker follows the write tools, not the file tools: with only `write_file` on,
    /// folders still read "you may write in it" even though no read/list tool is enabled. (The
    /// "Next:" nudge is for the read-only case; see the test above.)
    @Test func writeMarkerFollowsTheWriteToolsNotTheFileTools() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/workfolder"], workingFolder: "/Users/me/workfolder"),
            edition: .appStore, fileToolsEnabled: ["write_file"], writeEnabled: true)
        #expect(block == """
        Folders you can use in this chat:
        - workfolder — /Users/me/workfolder (working folder; you may write in it)
        Relative paths resolve against the working folder (/Users/me/workfolder).
        """)
    }

    @Test func noFoldersWithFileToolsAsksForOne() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .appStore,
            fileToolsEnabled: readOnlyTools, writeEnabled: false)
        #expect(block == FileScope.noFolderMessage)
    }

    /// Two folders and no working folder: nothing can resolve a relative path, so say so.
    @Test func twoFoldersWithoutAWorkingFolderWarnAboutRelativePaths() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/a", "/Users/me/b"]), edition: .appStore,
            fileToolsEnabled: readOnlyTools, writeEnabled: false)
        #expect(block == """
        Folders you can use in this chat:
        - a — /Users/me/a (read only (ask the user to switch on write_file or edit_file before writing))
        - b — /Users/me/b (read only (ask the user to switch on write_file or edit_file before writing))
        Relative paths do not work until the user picks a working folder; use the full path above.
        Next: read the folders above; ask the user before writing.
        """)
    }

    // MARK: Direct

    @Test func directWithNoFoldersNamesTheHomeFolderAndWhatToDo() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .direct,
            fileToolsEnabled: readOnlyTools, writeEnabled: false, homePath: "/Users/me")
        #expect(block == "You have no folder attached. Your home folder is /Users/me — ask the user for one, or use an absolute path under it.")
    }

    @Test func directWithoutAHomePathSaysNothing() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .direct,
            fileToolsEnabled: readOnlyTools, writeEnabled: false)
        #expect(block == nil)
    }

    @Test func directWithAFolderListstheFolder() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/proj"], workingFolder: "/Users/me/proj"),
            edition: .direct, fileToolsEnabled: readOnlyTools, writeEnabled: false,
            homePath: "/Users/me")
        #expect(block?.contains("- proj — /Users/me/proj (working folder; read only") == true)
        #expect(block?.contains("home folder") == false)
    }

    // MARK: determinism + size

    @Test func isDeterministicAcrossCalls() {
        let first = ChatEnvironment.instructions(
            scope: threeFolders, edition: .appStore, fileToolsEnabled: writeTools, writeEnabled: true)
        let second = ChatEnvironment.instructions(
            scope: threeFolders, edition: .appStore, fileToolsEnabled: writeTools, writeEnabled: true)
        #expect(first == second)
        #expect(first?.contains("2026") == true)  // …but only because a folder is *named* 2026
    }

    @Test func staysUnderAKilobyteForThreeLongPaths() {
        let block = ChatEnvironment.instructions(
            scope: threeFolders, edition: .appStore, fileToolsEnabled: writeTools, writeEnabled: true)
        let bytes = block?.utf8.count ?? 0
        #expect(bytes > 0)
        #expect(bytes < 1024)
    }

    /// The attachments note is CW-3's; CW-1 only proves the seam exists and is appended last.
    @Test func attachmentsNoteIsAppendedWhenGiven() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/workfolder"], workingFolder: "/Users/me/workfolder"),
            edition: .appStore, fileToolsEnabled: readOnlyTools, writeEnabled: false,
            attachmentsNote: "Attached file: rates.csv (read it with read_file).")
        #expect(block?.hasSuffix("Attached file: rates.csv (read it with read_file).") == true)
    }
}
