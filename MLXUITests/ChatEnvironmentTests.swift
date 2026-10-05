import Foundation
import Testing
@testable import MLXUI

/// CW-1 — the environment block (C3, wording W1/W2 from the CW-A review). Pure and deterministic:
/// same inputs, same bytes, no dates. The full text is pinned as a golden so the model-facing
/// wording is reviewable, and the length cap keeps an 8B model's context from being spent on its
/// own instructions.
struct ChatEnvironmentTests {

    private let readOnlyTools: Set<String> = ["read_file", "list_directory", "search_files"]
    private let writeTools: Set<String> = ["read_file", "write_file", "edit_file"]

    private let readLine =
        "You can read these folders. Writing is off; if asked to change a file, tell the user"
        + " to turn on write_file or edit_file."
    private let writeLine =
        "You can read and change files in these folders. The user approves each change."

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

    @Test func nilWhenFoldersExistButNoToolCanUseThem() {
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
        - workfolder: /Users/me/Desktop/workfolder (working folder)
        Relative paths start in the working folder, e.g. notes/todo.md.
        \(writeLine)
        """)
    }

    @Test func goldenThreeFoldersWorkingFolderFirst() {
        let block = ChatEnvironment.instructions(
            scope: threeFolders, edition: .appStore,
            fileToolsEnabled: writeTools, writeEnabled: true)
        #expect(block == """
        Folders you can use in this chat:
        - workfolder: /Users/me/Desktop/workfolder (working folder)
        - Invoices 2026: /Users/me/Documents/Invoices 2026
        - Some Long Named Project Folder: /Users/me/Projects/Some Long Named Project Folder
        Relative paths start in the working folder, e.g. notes/todo.md.
        \(writeLine)
        """)
    }

    @Test func readOnlyFoldersGetTheOneWritingLine() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/Desktop/workfolder"],
                             workingFolder: "/Users/me/Desktop/workfolder"),
            edition: .appStore, fileToolsEnabled: readOnlyTools, writeEnabled: false)
        #expect(block?.hasSuffix(readLine) == true)
        #expect(block?.contains("(working folder)") == true)
        #expect(block?.contains("read only") == false)
        #expect(block?.contains("Next:") == false)
    }

    @Test func noFoldersWithFileToolsAsksForOne() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .appStore,
            fileToolsEnabled: readOnlyTools, writeEnabled: false)
        #expect(block == FileScope.noFolderMessage)
    }

    /// Two folders and no working folder: nothing can resolve a relative path, so say so.
    @Test func twoFoldersWithoutAWorkingFolderSayToUseFullPaths() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/a", "/Users/me/b"]), edition: .appStore,
            fileToolsEnabled: readOnlyTools, writeEnabled: false)
        #expect(block == """
        Folders you can use in this chat:
        - a: /Users/me/a
        - b: /Users/me/b
        There is no working folder, so use full paths.
        \(readLine)
        """)
    }

    // MARK: Direct (W2, F3)

    @Test func directWithNoFoldersNamesTheHomeFolderAndWhatToDo() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .direct,
            fileToolsEnabled: readOnlyTools, writeEnabled: false, homePath: "/Users/me")
        #expect(block == "No folder is attached to this chat. The user's home folder is /Users/me."
            + " Use full paths under it, or paths starting with ~/.")
    }

    /// F3: Direct used to inject the home line whatever the tool state.
    @Test func directWithNoToolAtAllSaysNothing() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .direct,
            fileToolsEnabled: [], writeEnabled: false, homePath: "/Users/me")
        #expect(block == nil)
    }

    @Test func directWithOnlyRunShellStillNamesTheHomeFolder() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .direct,
            fileToolsEnabled: ["run_shell"], writeEnabled: false, homePath: "/Users/me")
        #expect(block?.contains("home folder is /Users/me") == true)
    }

    @Test func directWithOnlyAWriteToolStillNamesTheHomeFolder() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .direct,
            fileToolsEnabled: ["write_file"], writeEnabled: true, homePath: "/Users/me")
        #expect(block?.contains("home folder is /Users/me") == true)
    }

    @Test func directWithoutAHomePathSaysNothing() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: []), edition: .direct,
            fileToolsEnabled: readOnlyTools, writeEnabled: false)
        #expect(block == nil)
    }

    @Test func directWithAFolderListsTheFolder() {
        let block = ChatEnvironment.instructions(
            scope: FileScope(folders: ["/Users/me/proj"], workingFolder: "/Users/me/proj"),
            edition: .direct, fileToolsEnabled: readOnlyTools, writeEnabled: false,
            homePath: "/Users/me")
        #expect(block?.contains("- proj: /Users/me/proj (working folder)") == true)
        #expect(block?.contains("home folder") == false)
    }

    // MARK: determinism + size

    @Test func isDeterministicAcrossCalls() {
        let first = ChatEnvironment.instructions(
            scope: threeFolders, edition: .appStore, fileToolsEnabled: writeTools, writeEnabled: true)
        let second = ChatEnvironment.instructions(
            scope: threeFolders, edition: .appStore, fileToolsEnabled: writeTools, writeEnabled: true)
        #expect(first == second)
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
