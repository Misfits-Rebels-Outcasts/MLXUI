import Testing
import Foundation
@testable import MLXUI

/// MT4-7-FIX-4 (A1, owner option a, 2026-11-12): the file tools and the shell are cue-gated for Apple Intelligence.
struct AppleFileToolGatingTests {
    private let all = ["run_shell", "read_file", "list_directory", "search_files", "write_file", "edit_file"]
    private let fileTools = ["read_file", "list_directory", "search_files", "write_file", "edit_file"]

    private func offered(_ prompt: String, router: ToolExposurePolicy.Router = .appleIntelligence, folder: Bool = false,
                         turn: Set<Kind> = []) -> [String] {
        ToolExposurePolicy.select(ToolExposurePolicy.Input(
            router: router, modelToolsEnabled: true, conversationAssetKinds: turn, turnAssetKinds: turn, prompt: prompt,
            enabledTools: all, gateAmbiguousTools: router == .appleIntelligence, foldersAttached: folder)).tools
    }

    @Test func step31aFolderAttachedAndReadFileOnAPathOffersReadFile() {
        let tools = offered("Use read_file to read /Users/x/Documents/notes.txt", folder: true)
        #expect(tools.contains("read_file"))
    }

    @Test func step27AnAudioFileAttachedNoFolderAndAWritingPromptOffersNeitherShellNorFileTools() {
        let tools = offered("write 1500 words about the history of the sea and sailing", turn: [.audio])
        #expect(!tools.contains("run_shell"))
        for tool in fileTools { #expect(!tools.contains(tool), "\(tool)") }
    }

    @Test func noCueAndNoFolderOffersNoneOfTheFive() {
        for prompt in ["What is the capital of France?", "tell me a joke", "summarize the French revolution"] {
            let tools = offered(prompt)
            for tool in fileTools { #expect(!tools.contains(tool), "\(prompt): \(tool)") }
        }
    }

    @Test func aFolderAttachedAndAPlainQuestionOffersReadListSearchButNotWriteOrEdit() {
        let tools = offered("What is the capital of France?", folder: true)
        #expect(tools.contains("read_file") && tools.contains("list_directory") && tools.contains("search_files"))
        #expect(!tools.contains("write_file") && !tools.contains("edit_file") && !tools.contains("run_shell"))
    }

    @Test func writeAndEditNeedTheirOwnCues() {
        #expect(offered("save this to a file").contains("write_file"))
        #expect(offered("write the summary to ~/Desktop/out.md").contains("write_file"))
        #expect(offered("edit notes.txt and fix the typo").contains("edit_file"))
        #expect(!offered("save this to a file").contains("search_files"))
        #expect(!offered("read notes.txt").contains("write_file"))
        #expect(!offered("read notes.txt").contains("edit_file"))
    }

    @Test func readListAndSearchCues() {
        let path = offered("what is in ./src")
        #expect(path.contains("read_file") && path.contains("list_directory") && !path.contains("search_files"))
        let word = offered("show me what is in my folder")
        #expect(word.contains("list_directory") && !word.contains("search_files"))
        #expect(offered("search my files for invoices").contains("search_files"))
        #expect(offered("find the word budget in this folder").contains("search_files"))
        #expect(!offered("find me a good restaurant").contains("search_files"))
    }

    @Test func mlxIsUnchangedForTheFileTools() {
        let tools = offered("What is the capital of France?", router: .mlx(hasVision: false))
        for tool in fileTools { #expect(tools.contains(tool), "\(tool)") }
        #expect(!tools.contains("run_shell"))
    }

    @Test func gatedToolsAreReportedAsNotCalledFor() {
        let out = ToolExposurePolicy.select(ToolExposurePolicy.Input(
            router: .appleIntelligence, modelToolsEnabled: true, prompt: "hello", enabledTools: all, gateAmbiguousTools: true))
        #expect(Set(out.dropped.filter { $0.reason == .notCalledFor }.map(\.name)) == Set(all))
    }
}
