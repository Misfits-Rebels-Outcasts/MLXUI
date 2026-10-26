import Foundation
import Testing
@testable import MLXUI

/// CW-FIX (owner, 2026-10-26): "Models whose template ignores tools get no environment block."
/// The gate is `ModelRunner.environmentBlock`, keyed on the installed `config.json`'s `model_type`
/// the way the Local Server reads it.
@MainActor
@Suite(.serialized)
struct ChatEnvironmentModelGateTests {

    private func runner(modelType: String) throws -> (ModelRunner, ModelEntry) {
        let entry = makeEntry(id: "gate-\(modelType)", displayName: "Gate Model")
        let store = FileManager.default.temporaryDirectory
            .appendingPathComponent("cw-fix-store-\(UUID().uuidString)")
        let modelDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cw-fix-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data(#"{"model_type":"\#(modelType)"}"#.utf8)
            .write(to: modelDir.appendingPathComponent("config.json"))

        let runner = ModelRunner()
        runner.prepare(for: entry, in: store)
        runner.modelDirectoryForTesting = { _ in modelDir }
        for tool in ["read_file", "list_directory", "search_files"] {
            runner.setToolEnabled(true, for: tool)
        }
        runner.attachFolder(path: NSTemporaryDirectory())
        return (runner, entry)
    }

    @Test func aTemplateThatIgnoresToolsGetsNoBlockEvenWithFoldersAttached() throws {
        let (runner, entry) = try runner(modelType: "gemma3_text")
        #expect(ToolCallFormatTable.toolSupport(forModelType: "gemma3_text") == .templateIgnoresTools)
        #expect(runner.currentEnvironmentBlock(for: entry) == nil)
    }

    #if DIRECT_BUILD
    /// Control: the same setup on a tool-capable family keeps its block (Direct attaches without
    /// a grant, so the folder list is non-nil there).
    @Test func aToolCapableTemplateKeepsItsBlock() throws {
        let (runner, entry) = try runner(modelType: "llama")
        #expect(runner.currentEnvironmentBlock(for: entry) != nil)
    }
    #endif
}
