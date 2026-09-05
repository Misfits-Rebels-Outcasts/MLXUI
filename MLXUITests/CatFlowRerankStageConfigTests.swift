import Testing
import Foundation
@testable import MLXUI

/// MoC-3-2 (`RSI/DelegateMoCBacklog.md`) — `query`/`top_k` reach the stage. `StageConfig` gains
/// `query`/`topK`; `RealExecutor.stageConfig` reads `query=` off an `engines.rerank.*` row's
/// settings, falling back to the first bare token (the same fallback `firstBare()` already gives
/// the diffusion prompt), plus `top_k=`. A row with neither must fail with a named error — never
/// run with an empty query and never silently pass, the `CFM-R16-1` lesson applied here.
struct CatFlowRerankStageConfigTests {

    private func makeExecutor() -> RealExecutor {
        RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "moc3",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in StubTextStage() },
            installedModelIDs: [],
            catalog: [])
    }

    private func rerankDesc() throws -> TaskDescriptor {
        try #require(TaskCatalog.get("Rerank"))
    }

    // MARK: - query= and top_k= reach the config, at the stageConfig boundary

    @Test func queryAndTopKSettingsBecomeStageConfig() throws {
        let desc = try rerankDesc()
        let row = Row(task: "Rerank", model: "Qwen3 Reranker 0.6B", settings: "query=\"what is MLX\"; top_k=3")
        let config = try makeExecutor().stageConfig(for: desc, row: row)
        #expect(config.query == "what is MLX")
        #expect(config.topK == 3)
    }

    @Test func bareTokenFallsBackToQueryWhenNoKeyIsGiven() throws {
        // Same fallback `firstBare()` already gives the diffusion prompt (RealExecutor.swift's
        // `engines.diffusion.*` branch) — a bare token is the query when `query=` is absent.
        // Quoted so the multi-word query survives as one bare token, not three.
        let desc = try rerankDesc()
        let row = Row(task: "Rerank", model: "Qwen3 Reranker 0.6B", settings: "\"what is MLX\"")
        let config = try makeExecutor().stageConfig(for: desc, row: row)
        #expect(config.query == "what is MLX")
        #expect(config.topK == nil)
    }

    @Test func queryKeyBeatsTheBareToken() throws {
        let desc = try rerankDesc()
        let row = Row(task: "Rerank", model: "Qwen3 Reranker 0.6B", settings: "ignored bare text; query=\"real query\"")
        let config = try makeExecutor().stageConfig(for: desc, row: row)
        #expect(config.query == "real query")
    }

    // MARK: - No query, no bare token → a named error, never a silent empty query

    @Test func noQueryFailsWithANamedError() throws {
        let desc = try rerankDesc()
        let row = Row(task: "Rerank", model: "Qwen3 Reranker 0.6B", settings: "top_k=3")
        #expect(throws: FlowError.self) {
            _ = try makeExecutor().stageConfig(for: desc, row: row, path: "?")
        }
        do {
            _ = try makeExecutor().stageConfig(for: desc, row: row, path: "?")
            Issue.record("expected throw")
        } catch let error as FlowError {
            guard case .missingRerankQuery(let path) = error else {
                Issue.record("wrong FlowError case: \(error)")
                return
            }
            #expect(path == "?")
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    @Test func noSettingsAtAllFailsWithANamedError() throws {
        let desc = try rerankDesc()
        let row = Row(task: "Rerank", model: "Qwen3 Reranker 0.6B", settings: nil)
        #expect(throws: FlowError.self) {
            _ = try makeExecutor().stageConfig(for: desc, row: row, path: "5")
        }
    }

    @Test func missingQueryErrorNamesWhatIsWrong() {
        // The reference's own wording (RSI/DelegateMoCBacklog.md MoC-3-2): "Rerank needs a
        // query -- e.g. Rerank BGE Reranker; query=\"...\"."
        let sentence = FlowErrorDisplay.sentence(for: FlowError.missingRerankQuery(row: "3"))
        #expect(sentence.contains("query"))
        #expect(sentence.contains("3"))
    }

    // MARK: - Non-rerank rows do not pick up query/top_k

    @Test func nonRerankRowsDoNotPickUpQueryOrTopK() throws {
        let desc = try #require(TaskCatalog.get("Transcribe"))
        let row = Row(task: "Transcribe", model: "Whisper Small", settings: "query=foo; top_k=3")
        let config = try makeExecutor().stageConfig(for: desc, row: row)
        #expect(config.query == nil)
        #expect(config.topK == nil)
    }

    // MARK: - StageConfig's Hashable conformance keys EngineCache correctly on the new fields

    @Test func stageConfigHashDiffersByQueryAndTopK() {
        let base = StageConfig(query: "a", topK: 3)
        let differentQuery = StageConfig(query: "b", topK: 3)
        let differentTopK = StageConfig(query: "a", topK: 5)
        let same = StageConfig(query: "a", topK: 3)
        #expect(base == same)
        #expect(base != differentQuery)
        #expect(base != differentTopK)
        var seen = Set<StageConfig>()
        seen.insert(base)
        seen.insert(differentQuery)
        seen.insert(differentTopK)
        seen.insert(same)
        #expect(seen.count == 3)
    }
}

/// A local text→text stub for RealExecutor config tests (no model needed).
private struct StubTextStage: PipelineStage {
    let id = "stub.rerank"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}
