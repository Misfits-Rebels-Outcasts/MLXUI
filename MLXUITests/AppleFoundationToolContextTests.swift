import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// MT4-2 — a tool the Apple Intelligence router runs sees the chat's folder scope, model-tool runtime and assets, exactly as a
/// tool run by the MLX router's `AgentSession` does. Without this the model tools answered "isn't available here" and the file
/// tools ignored the chat's attached folders under Apple Intelligence. Fixture-driven (no device).
@MainActor
@Suite(.serialized)
struct AppleFoundationToolContextTests {
    typealias Fixtures = AppleFoundationRouterTests

    /// Reports what the task-locals hold at the moment it runs.
    nonisolated struct ProbeTool: AgentTool {
        let name = "probe"
        let toolDescription = "Reports the bound context."
        let parameters: [ToolParameter] = []
        func execute(arguments: [String: JSONValue]) async throws -> String {
            let folders = FileScope.current?.folders.joined(separator: ",") ?? "none"
            let host = ModelToolHost.current.map { $0.assets.conversationID.uuidString } ?? "none"
            let runtime = ModelToolRuntime.current == nil ? "none" : "runtime"
            return "folders=\(folders) host=\(host) runtime=\(runtime)"
        }
    }

    private func run(_ turn: RouterTurn) async throws -> String {
        let chatter = Fixtures.FakeChatter(script: { _, handler in
            [.snapshot(await handler("probe", [:]))]
        })
        let router = AppleFoundationRouter(chatter: chatter)
        var text = ""
        for try await event in router.streamTurn(turn) {
            if case .text(let chunk) = event { text += chunk }
        }
        return text
    }

    @Test func theBoundContextReachesTheToolRunByTheRouter() async throws {
        let id = UUID()
        let store = ChatAssetStore(conversationID: id, in: FileManager.default.temporaryDirectory.appendingPathComponent("mt42ctx-\(id)"))
        let runtime = ModelToolRuntime(catalog: [], installedModelIDs: [], claimableModelIDs: [], totalRAMGB: 16,
                                       makeExecutor: { _, _ in MT13Executor.text("x") })
        let context = RouterToolContext(fileScope: FileScope(folders: ["/a/work"], workingFolder: "/a/work"),
                                        modelToolRuntime: runtime, modelToolHost: ModelToolHost(assets: store))
        let text = try await run(RouterTurn(prompt: "go", tools: [ProbeTool()], toolContext: context))
        #expect(text.contains("folders=/a/work host=\(id.uuidString) runtime=runtime"))
    }

    @Test func withoutAContextNothingIsBound() async throws {
        let text = try await run(RouterTurn(prompt: "go", tools: [ProbeTool()]))
        #expect(text.contains("folders=none host=none runtime=none"))
    }
}
