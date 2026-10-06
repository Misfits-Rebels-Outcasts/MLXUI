import Foundation
import Testing
@testable import MLXUI

/// CW-UX-1 — `FolderSuggestion`: the decision behind the composer's hint row and the buttons on a
/// refused tool card. Pure, so none of this needs a runner, a grant or a view.
///
/// The report it comes from: a folder granted in Settings, visible in the "+" menu and **unticked**,
/// then "What files are in workfolder?" answered with "No folder is attached to this chat". Correct
/// by design (G1 ∩ G2), but it left no way to find the missing click.
struct FolderSuggestionTests {

    private let workfolder = "/Users/me/Desktop/workfolder"
    private let invoices = "/Users/me/Documents/Invoices 2026"

    // MARK: - candidates: the four empty cases

    @Test func emptyWhenAFolderIsAlreadyAttached() {
        let candidates = FolderSuggestion.candidates(
            attached: [workfolder], available: [invoices], fileToolsOn: true)
        #expect(candidates.isEmpty)
    }

    @Test func emptyWhenNoFileToolIsOn() {
        let candidates = FolderSuggestion.candidates(
            attached: [], available: [invoices], fileToolsOn: false)
        #expect(candidates.isEmpty)
    }

    @Test func emptyWhenNothingIsAvailable() {
        let candidates = FolderSuggestion.candidates(
            attached: [], available: [], fileToolsOn: true)
        #expect(candidates.isEmpty)
    }

    /// Direct with an empty recent-folders list: nothing to offer, so the view shows Add folder….
    @Test func emptyInDirectWithNoRecentFolders() {
        let candidates = FolderSuggestion.candidates(
            attached: [], available: [], fileToolsOn: true)
        #expect(candidates.isEmpty)
        #expect(!FolderSuggestion.row(candidates: candidates).hasMore)
    }

    // MARK: - candidates: the positive case

    @Test func offersEveryAvailableFolderInItsExistingOrder() {
        let candidates = FolderSuggestion.candidates(
            attached: [], available: [workfolder, invoices], fileToolsOn: true)
        #expect(candidates == [workfolder, invoices])
    }

    /// The order is the caller's — this function does not sort, deduplicate or re-derive.
    @Test func preservesTheCallersOrder() {
        let available = ["/z/last", "/a/first", "/m/middle"]
        #expect(FolderSuggestion.candidates(
            attached: [], available: available, fileToolsOn: true) == available)
    }

    // MARK: - "a file tool is on"

    /// Each file tool on its own is enough — the definition is
    /// `ChatEnvironment.fileToolNames ∪ writeToolNames`, not a list kept here.
    @Test func fileToolsOnIsTrueForEachFileToolAlone() {
        for name in ChatEnvironment.fileToolNames {
            #expect(FolderSuggestion.fileToolsOn(enabledToolNames: [name]),
                    "\(name) alone should count as a file tool")
        }
        for name in ChatEnvironment.writeToolNames {
            #expect(FolderSuggestion.fileToolsOn(enabledToolNames: [name]),
                    "\(name) alone should count as a file tool")
        }
    }

    /// `run_shell` is not a file tool: attaching a folder would not change what the model can do.
    @Test func fileToolsOnIsFalseForRunShellAlone() {
        #expect(!FolderSuggestion.fileToolsOn(enabledToolNames: ["run_shell"]))
        #expect(!FolderSuggestion.fileToolsOn(enabledToolNames: []))
    }

    /// The five names the contract lists are exactly the two sets' union.
    @Test func theFiveFileToolsAreTheUnions() {
        let union = ChatEnvironment.fileToolNames.union(ChatEnvironment.writeToolNames)
        #expect(union == ["read_file", "list_directory", "search_files", "write_file", "edit_file"])
    }

    // MARK: - the row's three-and-More shape

    @Test func rowShowsUpToThreeAndFlagsMore() {
        #expect(FolderSuggestion.maxButtons == 3)
        let three = FolderSuggestion.row(candidates: ["/a", "/b", "/c"])
        #expect(three.shown == ["/a", "/b", "/c"])
        #expect(!three.hasMore)

        let four = FolderSuggestion.row(candidates: ["/a", "/b", "/c", "/d"])
        #expect(four.shown == ["/a", "/b", "/c"])
        #expect(four.hasMore)
    }

    // MARK: - the card predicate (G3 below)

    @Test func showsOnCardForTheNoFolderRefusal() {
        #expect(FolderSuggestion.showsOnCard(
            result: FileScope.noFolderMessage, attached: []))
    }

    @Test func doesNotShowOnceAnythingIsAttached() {
        #expect(!FolderSuggestion.showsOnCard(
            result: FileScope.noFolderMessage, attached: [workfolder]))
    }

    @Test func doesNotShowForTheC5Refusal() {
        let c5 = "Error: access to /Users/me/Desktop/other/file.txt is not permitted."
            + " This chat can use: workfolder (/Users/me/Desktop/workfolder)."
            + " The user can add a folder to this chat with the + button next to the message box."
        #expect(!FolderSuggestion.showsOnCard(result: c5, attached: []))
    }

    @Test func doesNotShowForTheNoPathMessage() {
        #expect(!FolderSuggestion.showsOnCard(result: FileScope.noPathMessage, attached: []))
    }

    @Test func doesNotShowForOtherResultsOrNone() {
        #expect(!FolderSuggestion.showsOnCard(result: "contents of the file", attached: []))
        #expect(!FolderSuggestion.showsOnCard(result: "", attached: []))
        #expect(!FolderSuggestion.showsOnCard(result: nil, attached: []))
        // A near miss: the refusal with a trailing newline is not the constant.
        #expect(!FolderSuggestion.showsOnCard(
            result: FileScope.noFolderMessage + "\n", attached: []))
    }
}
