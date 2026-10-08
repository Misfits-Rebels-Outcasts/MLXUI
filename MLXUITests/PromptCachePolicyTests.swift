import Testing
import Foundation
@testable import MLXUI

/// PC-2 — the pure parts: when the cache may run, and where it checkpoints. (The real-model
/// equivalence tests are in `PromptCacheRealModelTests`.)
struct PromptCachePolicyTests {

    // MARK: kill switch and allowlist

    @Test func theKillSwitchIsShippedOffAndPromptCacheFollowsItAndTheServerGate() {
        #expect(AppState.disablePromptCache == false)               // shipped at Review PC-A, 2026-10-08 (99fda41)
        PromptCache.overrideForTesting = nil
        #expect(PromptCache.isEnabled == (!AppState.disablePromptCache && LocalServerGate.isAvailable))
        PromptCache.overrideForTesting = false
        #expect(PromptCache.isEnabled == false)                     // the test override still wins either way
        PromptCache.overrideForTesting = true
        #expect(PromptCache.isEnabled == true)
        PromptCache.overrideForTesting = nil
    }

    @Test func theAllowlistIsExactlyTheFamiliesWithAPassingRealModelTest() {
        // Pinned (ruling D1). Adding a family means a journal line showing its real-model test.
        #expect(PromptCacheAllowlist.tested == ["qwen3", "qwen3_5", "gemma3_text", "llama", "gemma2", "mistral3"])
        // By model_type (owner, 2026-10-08): `mistral3` admits Devstral too, as `qwen3_5` admits Qwen3.8/Bonsai.
        for untested in ["gpt_oss", "qwen2", "ministral3", "qwen3_5_text", ""] {
            #expect(PromptCacheAllowlist.allows(untested) == false, "\(untested)")
        }
    }

    @Test func anUntestedModelTypeBypassesBeforeAnythingElse() {
        #expect(PromptCachePolicy.bypass(enabled: true, modelType: "gpt_oss", seed: nil, kvBits: nil, maxKVSize: nil) == .modelNotTested)
        #expect(PromptCachePolicy.bypass(enabled: false, modelType: "qwen3", seed: nil, kvBits: nil, maxKVSize: nil) == .disabled)
        #expect(PromptCachePolicy.bypass(enabled: true, modelType: "qwen3", seed: 7, kvBits: nil, maxKVSize: nil) == .seeded)
        #expect(PromptCachePolicy.bypass(enabled: true, modelType: "qwen3", seed: nil, kvBits: 4, maxKVSize: nil) == .customKVCache)
        #expect(PromptCachePolicy.bypass(enabled: true, modelType: "qwen3", seed: nil, kvBits: nil, maxKVSize: 1024) == .customKVCache)
        #expect(PromptCachePolicy.bypass(enabled: true, modelType: "gemma3_text", seed: nil, kvBits: nil, maxKVSize: nil) == nil)
    }

    // MARK: probe conversations

    @Test func theProbeAddsAnAssistantAndAUserTurnAndTheSystemProbeKeepsOnlyTheLeadingSystemMessages() {
        let messages = [ServeMessage(role: .system, text: "S1"), ServeMessage(role: .system, text: "S2"),
                        ServeMessage(role: .user, text: "u1"), ServeMessage(role: .assistant, text: "a1"),
                        ServeMessage(role: .user, text: "u2")]
        let probes = PromptCheckpoints.probeMessages(for: messages)
        #expect(probes.turn.count == messages.count + 2)
        #expect(Array(probes.turn.prefix(messages.count)) == messages)
        #expect(probes.turn.suffix(2).map(\.role) == [.assistant, .user])
        #expect(probes.system?.map(\.text) == ["S1", "S2", PromptCheckpoints.probeUser])
        #expect(probes.system?.map(\.role) == [.system, .system, .user])
        #expect(PromptCheckpoints.probeMessages(for: [ServeMessage(role: .user, text: "hi")]).system == nil)
    }

    // MARK: positions

    private func ids(_ n: Int) -> [Int] { Array(0..<n) }

    @Test func aChatMLStyleTemplatePutsTBeforeTheGenerationPromptAndSBeforeTheUserWords() {
        // actual = 1000 tokens, the last 4 being the generation prompt; probes diverge at 996 / 700.
        let actual = ids(1000)
        let turn = Array(actual[0..<996]) + [-1, -2, -3]
        let system = Array(actual[0..<700]) + [-9]
        #expect(PromptCheckpoints.positions(actual: actual, turnProbe: turn, systemProbe: system, minimumLength: 256) == [700, 996])
    }

    @Test func aTemplateWhoseWholePromptIsTheStableBoundaryIsClampedToLeaveATailToken() {
        // Gemma 3: nothing follows `model⏎`, so the probe agrees with the whole prompt (T == n).
        let actual = ids(500)
        let turn = actual + [-1, -2]
        #expect(PromptCheckpoints.positions(actual: actual, turnProbe: turn, systemProbe: nil, minimumLength: 256) == [499])
    }

    @Test func positionsBelowTheMinimumAreDroppedAndEqualOnesAreNotRepeated() {
        let actual = ids(1000)
        #expect(PromptCheckpoints.positions(actual: actual, turnProbe: Array(actual[0..<100]) + [-1], systemProbe: nil, minimumLength: 256).isEmpty)
        let same = Array(actual[0..<800]) + [-1]
        #expect(PromptCheckpoints.positions(actual: actual, turnProbe: same, systemProbe: same, minimumLength: 256) == [800])
        // a failed render (nil probe) just means no checkpoint from it
        #expect(PromptCheckpoints.positions(actual: actual, turnProbe: nil, systemProbe: nil, minimumLength: 256).isEmpty)
    }
}
