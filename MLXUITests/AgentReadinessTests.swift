import Foundation
import Testing
@testable import MLXUI

// S2-4 — Agent-ready. The 12 catalog chat models, their real `config.json` shapes (fetched
// 2026-10-07, `RSI/notes/s2-4-research/kv-keys.md`), and the criteria over them.
//
// The budget used throughout is **this Mac's**: 18 GB RAM → `EngineCache.defaultBudget` =
// 10.8 GiB = 11,596,411,699 B. It is injected rather than read, so the suite doesn't depend on the
// machine it runs on.

struct AgentReadinessCriteriaTests {

    /// The owner's Mac's budget, as `MemoryBudget.shared` computes it.
    private static let budget = EngineCache.defaultBudget(totalRAMGB: 18)

    /// A fixture: the model's id, `model_type`, the config fragment that carries its KV keys, the
    /// catalog's `ramGB`, and the KV bytes a 32k context needs (hand-computed below).
    private struct Fixture {
        var id: String
        var modelType: String
        var ramGB: Double
        var kvBytes: Int
        /// Kept last (with a default) so the call sites read id → type → ram → kv → config.
        var verifiedFirst = false
        var config: String
    }

    /// A config as JSON, built from a dictionary so nesting and escaping can't drift.
    private static func json(_ object: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    /// The real `qwen3_5` shape: keys under `text_config`, `layer_types` = 3 `linear_attention` then
    /// 1 `full_attention` per group, plus the linear-state keys. `layers` is the length of
    /// `layer_types` (16 full + 48 linear for the 27B, 8 + 24 for the 9B).
    private static func hybrid(layers: Int, attnHeads: Int, kvHeads: Int, headDim: Int, hidden: Int,
                               maxPosition: Int, valueHeads: Int, keyHeads: Int = 16,
                               convKernel: Int = 4) -> String {
        let pattern = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        let types = (0 ..< (layers / 4)).flatMap { _ in pattern }
        return json(["model_type": "qwen3_5", "text_config": [
            "model_type": "qwen3_5_text", "num_hidden_layers": layers, "num_attention_heads": attnHeads,
            "num_key_value_heads": kvHeads, "head_dim": headDim, "hidden_size": hidden,
            "max_position_embeddings": maxPosition, "full_attention_interval": 4,
            "linear_conv_kernel_dim": convKernel, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_num_key_heads": keyHeads,
            "linear_num_value_heads": valueHeads, "layer_types": types,
        ]])
    }

    /// Hand-computed KV bytes, from `layers × kv_heads × head_dim × 2 (K and V) × 2 bytes × 32768`,
    /// with the linear-state term for the hybrids:
    ///
    /// * Qwen3-8B: 36 × 8 × 128 × 2 × 2 × 32768 = 4,831,838,208
    /// * Qwen3.8-27B-3bit: 16 attention layers × 4 × 256 × 2 × 2 × 32768 = 2,147,483,648, plus
    ///   48 linear layers × (48 × 128 × 128 × 2 recurrent + 3 × 10,240 × 2 conv) = 78,446,592
    private static let catalog: [Fixture] = [
        // 4 × qwen3_5 hybrids (`text_config`; the 27B class has 48 linear value heads, the 9B 32).
        Fixture(id: "prism-ml/Ternary-Bonsai-27B-mlx-2bit", modelType: "qwen3_5", ramGB: 12.66,
                kvBytes: 2_225_930_240,
                config: hybrid(layers: 64, attnHeads: 24, kvHeads: 4, headDim: 256, hidden: 5120,
                               maxPosition: 262144, valueHeads: 48)),
        Fixture(id: "mlx-community/Qwen3.5-9B-MLX-4bit", modelType: "qwen3_5", ramGB: 8.96,
                kvBytes: 1_100_087_296,
                config: hybrid(layers: 32, attnHeads: 16, kvHeads: 4, headDim: 256, hidden: 4096,
                               maxPosition: 262144, valueHeads: 32)),
        Fixture(id: "lukaskremla/Qwen3.8-27B-3bit-MLX", modelType: "qwen3_5", ramGB: 19.08,
                kvBytes: 2_225_930_240, verifiedFirst: true,
                config: hybrid(layers: 64, attnHeads: 24, kvHeads: 4, headDim: 256, hidden: 5120,
                               maxPosition: 262144, valueHeads: 48)),
        Fixture(id: "mlx-community/Qwen3.8-27B-4bit", modelType: "qwen3_5", ramGB: 26.0,
                kvBytes: 2_225_930_240,
                config: hybrid(layers: 64, attnHeads: 24, kvHeads: 4, headDim: 256, hidden: 5120,
                               maxPosition: 262144, valueHeads: 48)),
        // Plain decoders, keys at the top level.
        Fixture(id: "mlx-community/gemma-3-1b-it-qat-4bit", modelType: "gemma3_text", ramGB: 0.84,
                kvBytes: 872_415_232,
                config: json(["model_type": "gemma3_text", "num_hidden_layers": 26,
                              "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
                              "hidden_size": 1152, "max_position_embeddings": 32768,
                              "sliding_window": 512, "sliding_window_pattern": 6])),
        Fixture(id: "mlx-community/Ministral-3-3B-Instruct-2512-4bit", modelType: "mistral3", ramGB: 2.53,
                kvBytes: 3_489_660_928, verifiedFirst: true,
                config: json(["model_type": "mistral3", "text_config": [
                    "model_type": "ministral3", "num_hidden_layers": 26, "num_attention_heads": 32,
                    "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 3072,
                    "max_position_embeddings": 262144, "sliding_window": NSNull(),
                ]])),
        Fixture(id: "mlx-community/Qwen3-8B-4bit", modelType: "qwen3", ramGB: 6.75,
                kvBytes: 4_831_838_208, verifiedFirst: true,
                config: json(["model_type": "qwen3", "num_hidden_layers": 36, "num_attention_heads": 32,
                              "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096,
                              "max_position_embeddings": 40960, "sliding_window": NSNull(),
                              "use_sliding_window": false])),
        // `head_dim` absent → hidden_size / num_attention_heads (4096/32, 5120/40).
        Fixture(id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", modelType: "llama", ramGB: 6.75,
                kvBytes: 4_294_967_296, verifiedFirst: true,
                config: json(["model_type": "llama", "num_hidden_layers": 32, "num_attention_heads": 32,
                              "num_key_value_heads": 8, "hidden_size": 4096,
                              "max_position_embeddings": 131072])),
        Fixture(id: "mlx-community/gemma-2-9b-it-4bit", modelType: "gemma2", ramGB: 7.59,
                kvBytes: 11_274_289_152, verifiedFirst: true,
                config: json(["model_type": "gemma2", "num_hidden_layers": 42, "num_attention_heads": 16,
                              "num_key_value_heads": 8, "head_dim": 256, "hidden_size": 3584,
                              "max_position_embeddings": 8192, "sliding_window": 4096,
                              "sliding_window_size": 4096])),
        Fixture(id: "mlx-community/Qwen2.5-14B-Instruct-4bit", modelType: "qwen2", ramGB: 11.81,
                kvBytes: 6_442_450_944, verifiedFirst: true,
                config: json(["model_type": "qwen2", "num_hidden_layers": 48, "num_attention_heads": 40,
                              "num_key_value_heads": 8, "hidden_size": 5120,
                              "max_position_embeddings": 32768, "sliding_window": 131072,
                              "use_sliding_window": false])),
        // MoE + alternating attention: the 12 `sliding_attention` layers are counted at the full
        // window (an upper bound — their window is 128 tokens), so this is 24 layers' worth.
        Fixture(id: "mlx-community/gpt-oss-20b-MXFP4-Q8", modelType: "gpt_oss", ramGB: 17.65,
                kvBytes: 1_610_612_736, verifiedFirst: true,
                config: json(["model_type": "gpt_oss", "num_hidden_layers": 24, "num_attention_heads": 64,
                              "num_key_value_heads": 8, "head_dim": 64, "hidden_size": 2880,
                              "max_position_embeddings": 131072, "sliding_window": 128,
                              "num_local_experts": 32, "num_experts_per_tok": 4,
                              "layer_types": (0 ..< 12).flatMap { _ in ["sliding_attention", "full_attention"] }])),
        Fixture(id: "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", modelType: "mistral3", ramGB: 20.25,
                kvBytes: 5_368_709_120,
                config: json(["model_type": "mistral3", "text_config": [
                    "model_type": "ministral3", "num_hidden_layers": 40, "num_attention_heads": 32,
                    "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 5120,
                    "max_position_embeddings": 393216, "sliding_window": NSNull(),
                ]])),
    ]

    private func data(_ json: String) -> Data { Data(json.utf8) }

    /// The KV estimate, one dense and one hybrid, against the hand-computed numbers above — with the
    /// hybrid's two terms checked separately, because that is where the arithmetic could be wrong.
    @Test func theMemoryEstimateMatchesHandComputedNumbers() throws {
        let dense = try #require(Self.catalog.first { $0.id == "mlx-community/Qwen3-8B-4bit" })
        // 36 × 8 × 128 × 2 (K and V) × 2 bytes × 32768 = 4.50 GiB.
        #expect(AgentReadiness.kvBytes(configData: data(dense.config)) == 4_831_838_208)
        #expect(AgentReadiness.kvBytes(configData: data(dense.config)) == 36 * 8 * 128 * 2 * 2 * 32_768)

        let hybrid = try #require(Self.catalog.first { $0.id == "lukaskremla/Qwen3.8-27B-3bit-MLX" })
        let kv = try #require(AgentReadiness.kvBytes(configData: data(hybrid.config)))
        // 16 full-attention layers of 4 kv heads × 256 head_dim (2.00 GiB)…
        let attention = 16 * 4 * 256 * 2 * 2 * 32_768
        #expect(attention == 2_147_483_648)
        // …plus 48 linear layers holding a fixed state: recurrent 48×128×128×2 and conv
        // (4−1) × (2×16×128 + 48×128) × 2.
        let recurrent = 48 * 128 * 128 * 2
        let convDim = 2 * 16 * 128 + 48 * 128
        let linear = 48 * (recurrent + 3 * convDim * 2)
        #expect(recurrent == 1_572_864 && convDim == 10_240)
        #expect(linear == 78_446_592)
        #expect(kv == attention + linear)
        // The hybrid's cache is dominated by its 16 full-attention layers — the point of counting
        // only those.
        #expect(attention > 27 * linear)
    }

    /// Every catalog config parses to its hand-computed KV number.
    @Test func everyCatalogConfigEstimates() throws {
        for fixture in Self.catalog {
            let kv = try #require(AgentReadiness.kvBytes(configData: data(fixture.config)), "\(fixture.id)")
            #expect(kv == fixture.kvBytes, "\(fixture.id): \(kv) != \(fixture.kvBytes)")
        }
    }

    /// A config that doesn't declare enough is `nil`, and never a made-up number.
    @Test func anUnestimatableConfigIsNil() {
        #expect(AgentReadiness.kvBytes(configData: nil) == nil)
        #expect(AgentReadiness.kvBytes(configData: data("not json")) == nil)
        #expect(AgentReadiness.kvBytes(configData: data(#"{"model_type":"qwen3"}"#)) == nil)
        // Layers but no heads: no head_dim, and hidden_size can't be divided by a missing head count.
        #expect(AgentReadiness.kvBytes(configData: data(#"{"num_hidden_layers":36,"hidden_size":4096}"#)) == nil)
    }

    /// **The catalog, with no SV-6 rows** — the state the app ships in. Everything with tools and a
    /// 32k window is `eligibleNotVerified`; the two Gemmas and gpt-oss are refused for their
    /// templates. Qwen2.5-14B sits exactly on the 32k boundary and passes.
    @Test func theShippedTableLeavesEverythingUnverified() {
        let expected: [(String, AgentReadiness.State)] = [
            ("prism-ml/Ternary-Bonsai-27B-mlx-2bit", .eligibleNotVerified),
            ("mlx-community/Qwen3.5-9B-MLX-4bit", .eligibleNotVerified),
            ("lukaskremla/Qwen3.8-27B-3bit-MLX", .eligibleNotVerified),
            ("mlx-community/Qwen3.8-27B-4bit", .eligibleNotVerified),
            ("mlx-community/gemma-3-1b-it-qat-4bit",
             .notAgentReady(reason: "This model's template has no tool support.")),
            ("mlx-community/Ministral-3-3B-Instruct-2512-4bit", .eligibleNotVerified),
            ("mlx-community/Qwen3-8B-4bit", .eligibleNotVerified),
            ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", .eligibleNotVerified),
            ("mlx-community/gemma-2-9b-it-4bit",
             .notAgentReady(reason: "This model's template has no tool support.")),
            ("mlx-community/Qwen2.5-14B-Instruct-4bit", .eligibleNotVerified),
            ("mlx-community/gpt-oss-20b-MXFP4-Q8",
             .notAgentReady(reason: "This model's template has no tool support.")),
            ("mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", .eligibleNotVerified),
        ]
        #expect(AgentReadiness.verified.isEmpty)          // the table ships empty: never inferred
        #expect(Self.catalog.count == expected.count)
        for (id, want) in expected {
            let fixture = try? #require(Self.catalog.first { $0.id == id })
            let state = AgentReadiness.state(modelType: fixture?.modelType ?? "", contextLength: nil,
                                             catalogRAMGB: fixture?.ramGB ?? 0,
                                             configData: fixture.map { data($0.config) },
                                             budgetBytes: Self.budget)
            #expect(state == want, "\(id): \(state)")
        }
    }

    /// **With SV-6 rows** — criterion (d) does the deciding, and it is tight on this Mac: 8B models
    /// land within 1% of the budget in both directions.
    @Test func aVerifiedModelIsJudgedOnMemory() throws {
        func state(_ id: String, verified: Bool) throws -> AgentReadiness.State {
            let fixture = try #require(Self.catalog.first { $0.id == id })
            return AgentReadiness.state(modelType: fixture.modelType, contextLength: nil,
                                        catalogRAMGB: fixture.ramGB, configData: data(fixture.config),
                                        budgetBytes: Self.budget,
                                        verification: verified ? AgentReadiness.Verification(
                                            hfModelID: id, date: "2026-10-07", journal: "test") : nil)
        }
        // 8.96 + 1.02 = 9.98 GiB ≤ 10.80 → ready.
        #expect(try state("mlx-community/Qwen3.5-9B-MLX-4bit", verified: true) == .agentReady)
        // 6.75 + 4.00 = 10.75 GiB ≤ 10.80 → ready, by 53 MB.
        #expect(try state("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", verified: true) == .agentReady)
        // 6.75 + 4.50 = 11.25 GiB > 10.80 → may not fit.
        #expect(try state("mlx-community/Qwen3-8B-4bit", verified: true) == .mayNotFit)
        // 11.81 + 6.00 = 17.81 GiB → may not fit.
        #expect(try state("mlx-community/Qwen2.5-14B-Instruct-4bit", verified: true) == .mayNotFit)
        // A row cannot rescue a model whose template ignores tools: (a) is checked first.
        #expect(try state("mlx-community/gemma-2-9b-it-4bit", verified: true)
                == .notAgentReady(reason: "This model's template has no tool support."))
        // And the same model without a row is still refused for the same reason.
        #expect(try state("mlx-community/gemma-2-9b-it-4bit", verified: false)
                == .notAgentReady(reason: "This model's template has no tool support."))
    }

    /// (b) on its own: a 32k window passes, 8,192 doesn't, and an unknown window is refused with the
    /// word "unknown" rather than assumed large.
    @Test func theContextCriterionIsInclusiveAt32k() {
        func state(context: Int?) -> AgentReadiness.State {
            AgentReadiness.state(modelType: "qwen3", contextLength: context, catalogRAMGB: 1,
                                 configData: nil, budgetBytes: Self.budget)
        }
        #expect(state(context: 32_768) == .eligibleNotVerified)
        #expect(state(context: 32_767) == .notAgentReady(reason: "Its context window is 32767 tokens; agent use needs 32768."))
        #expect(state(context: 8_192) == .notAgentReady(reason: "Its context window is 8192 tokens; agent use needs 32768."))
        #expect(state(context: nil) == .notAgentReady(reason: "Its context window is unknown; agent use needs 32768."))
    }

    /// The config's own window is used when the caller doesn't pass one — one parse, S2-6's.
    @Test func theWindowComesFromTheConfigWhenNotGiven() throws {
        let qwen = try #require(Self.catalog.first { $0.id == "mlx-community/Qwen3-8B-4bit" })
        #expect(AgentReadiness.state(modelType: "qwen3", contextLength: nil, catalogRAMGB: 6.75,
                                     configData: data(qwen.config), budgetBytes: Self.budget)
                == .eligibleNotVerified)
        // gemma-2's 8,192 would refuse it, but its template refuses it first — the reason says so.
        let gemma = try #require(Self.catalog.first { $0.id == "mlx-community/gemma-2-9b-it-4bit" })
        #expect(AgentReadiness.state(modelType: "gemma2", contextLength: nil, catalogRAMGB: 7.59,
                                     configData: data(gemma.config), budgetBytes: Self.budget)
                == .notAgentReady(reason: "This model's template has no tool support."))
    }

    /// An unusable estimate must not *refuse* a model — fail open, like the pre-count.
    @Test func anUnknownBudgetOrConfigDoesNotRefuse() {
        let verification = AgentReadiness.Verification(hfModelID: "x", date: "2026-10-07", journal: "j")
        #expect(AgentReadiness.state(modelType: "qwen3", contextLength: 40_960, catalogRAMGB: 999,
                                     configData: nil, budgetBytes: 0, verification: verification) == .agentReady)
        #expect(AgentReadiness.state(modelType: "qwen3", contextLength: 40_960, catalogRAMGB: 999,
                                     configData: nil, budgetBytes: Self.budget,
                                     verification: verification) == .agentReady)
    }

    /// A family the table doesn't know is not Agent-ready — the conservative answer for a badge.
    @Test func anUnknownFamilyIsRefused() {
        #expect(AgentReadiness.state(modelType: "some_new_arch", contextLength: 131_072, catalogRAMGB: 1,
                                     configData: nil, budgetBytes: Self.budget)
                == .notAgentReady(reason: "This model's template has no tool support."))
    }

    /// The copy: one short line per state, and for `notAgentReady` the line **is** the reason.
    @Test func everyStateHasOneLineAndABadge() {
        let states: [AgentReadiness.State] = [.agentReady, .eligibleNotVerified, .mayNotFit,
                                              .notAgentReady(reason: "Because.")]
        for state in states {
            let line = AgentReadiness.line(for: state)
            #expect(!line.isEmpty && !line.contains("\n"), "\(state)")
            #expect(line.count < 130, "\(state): \(line.count) chars")
            #expect(!AgentReadiness.badge(for: state).isEmpty)
        }
        #expect(AgentReadiness.line(for: .notAgentReady(reason: "Because.")) == "Because.")
        // The shipped copy must not still claim tools are unsupported (S1's line).
        #expect(!AgentReadiness.line(for: .agentReady).contains("aren't supported"))
    }

    /// Rule 14/R27: gate closed ⇒ no badge, no line. Only installed MLX chat models get one.
    @Test func theBadgeObeysTheGateAndTheRunnerKind() {
        #expect(AgentReadiness.isVisible(gateAvailable: true, runnerKind: .llm, isInstalled: true))
        #expect(!AgentReadiness.isVisible(gateAvailable: false, runnerKind: .llm, isInstalled: true))
        #expect(!AgentReadiness.isVisible(gateAvailable: true, runnerKind: .llm, isInstalled: false))
        for kind in [RunnerKind.asr, .tts, .vision, .embedding, .ocr, .image, .unsupported] {
            #expect(!AgentReadiness.isVisible(gateAvailable: true, runnerKind: kind, isInstalled: true), "\(kind)")
        }
    }
}

// MARK: - The Connect sheet's line (S1-5's fitness line, S2-4's replacement)

struct ConnectFitnessLineTests {
    @Test func theLineCarriesTheContextAndTheState() {
        let line = ConnectSnippets.fitnessLine(contextLength: 40_960, agent: .eligibleNotVerified)
        #expect(line.contains("40,960"))
        #expect(line.contains(AgentReadiness.line(for: .eligibleNotVerified)))
        // S1's promise that tools never work is gone from the sheet.
        #expect(!line.contains("tool calls aren't supported"))
    }

    @Test func anUndeclaredContextStillSaysSoAndStillGivesTheState() {
        let line = ConnectSnippets.fitnessLine(contextLength: nil, agent: .agentReady)
        #expect(line.contains("not stated"))
        #expect(line.contains(AgentReadiness.line(for: .agentReady)))
    }
}
