import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG6a — dispatch-level per-tool execution timeout. The per-tool `executionTimeout` values are
/// tested synchronously; the actual timeout race + dispatch integration are async (they await a
/// real delay). If this async suite executes while `AgentToolTests` does not, AG0-b is a
/// suite-specific selection quirk, not a systemic async-dispatch hang.
struct SafetyRailsTests {

    // MARK: per-tool timeout policy (synchronous)

    @Test func runJavaScriptOptsIntoAShortCap() {
        #expect(RunJavaScriptTool().executionTimeout == 15)
    }

    @Test func otherToolsHaveNoCapByDefault() {
        #expect(CalculatorTool().executionTimeout == nil)   // charset-guarded — can't loop
        #expect(FetchURLTool().executionTimeout == nil)      // bounded by its URLSession timeout
        #expect(SummarizeTool().executionTimeout == nil)     // model inference legitimately slow
        #expect(TranscribeAudioTool().executionTimeout == nil)
        #expect(SemanticSearchTool().executionTimeout == nil)
    }

    // MARK: withTimeout race (async)

    @Test func withTimeoutReturnsFastResult() async throws {
        let result = try await AgentSession.withTimeout(seconds: 5) { "done" }
        #expect(result == "done")
    }

    @Test func withTimeoutReturnsNilWhenOperationExceedsCap() async throws {
        let result = try await AgentSession.withTimeout(seconds: 0.05) {
            try await Task.sleep(nanoseconds: 3_000_000_000)  // 3s — far past the 50ms cap
            return "should not arrive"
        }
        #expect(result == nil)
    }

    @Test func withTimeoutPropagatesOperationError() async {
        await #expect(throws: StubError.self) {
            _ = try await AgentSession.withTimeout(seconds: 5) { throw StubError.boom }
        }
    }

    // MARK: dispatch enforces the cap (async)

    @Test func dispatchReportsTimeoutForASlowTool() async {
        let registry = ToolRegistry([SlowTool()])
        var events: [String] = []
        let result = await AgentSession.dispatch(
            ToolCall(function: .init(name: "slow", arguments: [:])),
            registry: registry,
            budget: ToolBudget(limit: 8),
            approve: { _, _ in true },
            onEvent: { event in if case .failed(_, let m) = event { events.append(m) } })

        guard case .failed(let message) = result else {
            Issue.record("expected .failed, got \(result)")
            return
        }
        #expect(message.contains("timed out"))
        #expect(events.contains { $0.contains("timed out") })
    }

    // MARK: fixtures

    enum StubError: Error { case boom }

    /// A tool that sleeps well past its own tiny timeout, so dispatch must abandon it.
    nonisolated struct SlowTool: AgentTool {
        let name = "slow"
        let toolDescription = "test-only: sleeps past its timeout"
        let parameters: [ToolParameter] = []
        var executionTimeout: TimeInterval? { 0.05 }
        func execute(arguments: [String: JSONValue]) async throws -> String {
            try await Task.sleep(nanoseconds: 3_000_000_000)
            return "should not arrive"
        }
    }
}
