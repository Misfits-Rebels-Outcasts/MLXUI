import Testing
@testable import MLXUI

/// MT3-FIX-2 — the context meter counts what the router sends.
@MainActor
struct ChatMeterToolResultTests {
    private let big = String(repeating: "x", count: 80_000)

    @Test func appleIntelligenceCountsAToolResultOnlyUpToTheRoutersCap() {
        let seen = ChatInspectorModel.modelVisibleToolResult(big, isApple: true)
        #expect(seen.count == AppleFoundationRouter.maxToolOutputCharacters)
        let meter = ChatInspectorModel.meter(estimatedTokens: ChatInspectorModel.estimatedTokens(of: seen), window: 8_192)
        #expect(meter.fraction < 1)
    }

    @Test func anMLXModelCountsTheWholeResult() {
        #expect(ChatInspectorModel.modelVisibleToolResult(big, isApple: false) == big)
    }

    @Test func shortAndMissingResultsPassThrough() {
        #expect(ChatInspectorModel.modelVisibleToolResult("ok", isApple: true) == "ok")
        #expect(ChatInspectorModel.modelVisibleToolResult(nil, isApple: true) == "")
        #expect(ChatInspectorModel.modelVisibleToolResult(nil, isApple: false) == "")
    }
}
