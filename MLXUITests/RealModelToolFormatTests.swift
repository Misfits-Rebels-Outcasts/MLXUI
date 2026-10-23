import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

// WW2-2-FIX-1 — the tool-call format each installed catalog chat model really loads with.
//
// Owner-approved 2026-10-22: the `model_type` → format expectations that pinned mlx-swift-lm 3.31's mapping
// (ToolCallFormatTableTests, AgentToolTests, MiMoServeTests, Gemma4ServeTests) moved here. Under 3.32 the
// format is resolved inside the container (model declaration, `tool_parser_type`, then the chat template), so
// the only honest check is a real load through `ModelContainerPool` — the same path the Local Server reads
// (`context.configuration.toolCallFormat ?? .json`).
//
// OWNER-RUN (KICKOFF rule 16). Enabled only when `MLXUI_TEST_MODELS_DIR` names a folder holding installed
// model directories (`mlx-community--Qwen3-8B-4bit`, …); a model whose folder is absent is skipped, not red.
// A skipped test does not count as a pass.
@Suite(.serialized)
struct RealModelToolFormatTests {
    static let modelsDir = ProcessInfo.processInfo.environment["MLXUI_TEST_MODELS_DIR"]

    /// The effective format per catalog chat model: main's (mlx-swift-lm 3.31.4). `nil` on main meant "the
    /// container's `.json` default", so it is `.json` here. `qwen3_5` is `.xmlFunction`: 3.32 declares `.qwen35`
    /// upstream, but that parser refuses a call to an undeclared tool (SV-8, WW2-3-3), so
    /// `ToolCallFormatTable.override` restores main's served format (owner-approved 2026-10-23).
    static let expected: [(repo: String, format: ToolCallFormat)] = [
        ("prism-ml/Ternary-Bonsai-27B-mlx-2bit", .xmlFunction),
        ("mlx-community/Qwen3.5-9B-MLX-4bit", .xmlFunction),
        ("lukaskremla/Qwen3.8-27B-3bit-MLX", .xmlFunction),
        ("mlx-community/Qwen3.8-27B-4bit", .xmlFunction),
        ("mlx-community/gemma-3-1b-it-qat-4bit", .json),
        ("mlx-community/Ministral-3-3B-Instruct-2512-4bit", .mistral),
        ("mlx-community/Qwen3-8B-4bit", .json),
        ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", .llama3),
        ("mlx-community/gemma-2-9b-it-4bit", .json),
        ("mlx-community/Qwen2.5-14B-Instruct-4bit", .json),
        ("mlx-community/gpt-oss-20b-MXFP4-Q8", .json),
        ("mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", .mistral),
        // WW2-3-2 (owner 2026-10-22): the two catalog chat models the table was missing.
        ("mlx-community/gemma-4-e4b-it-4bit", .gemma4),
        ("mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit", .xmlFunction),
    ]

    private static func directory(_ repo: String) -> URL? {
        guard let root = modelsDir else { return nil }
        let url = URL(fileURLWithPath: root).appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--"))
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    @Test(.enabled(if: RealModelToolFormatTests.modelsDir != nil),
          arguments: RealModelToolFormatTests.expected.map(\.repo))
    func containerFormatMatchesTheExpectedTable(repo: String) async throws {
        guard let dir = Self.directory(repo) else {
            print("TOOLFORMAT \(repo): not installed — skipped")
            return
        }
        let expected = try #require(Self.expected.first { $0.repo == repo }).format
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let container = try await pool.container(directory: dir, architecture: nil)
        let raw = await container.configuration.toolCallFormat
        print("TOOLFORMAT \(repo): container.toolCallFormat = \(raw.map { "\($0)" } ?? "nil") (effective \(raw ?? .json))")
        #expect((raw ?? .json) == expected, "\(repo)")
    }
}
