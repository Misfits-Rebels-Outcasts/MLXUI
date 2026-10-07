import Foundation

/// The copy-paste text on the "Connect your app" sheet (design §5). **Every snippet is checked
/// against the harness's current docs and recorded in `RSI/DelegateServeReadme.md`'s copy table**
/// (C-rows, with URL + date) — no config key here is invented. S3 adds Claude Code and Codex.
///
/// **S2-5** — the app snippets (OpenCode, Continue) are per **model** now: what they advertise
/// depends on `AgentReadiness` and on whether the family can think, so a model that isn't Agent-ready
/// gets S1's plain-chat block rather than a tools or limits claim that would be false.
nonisolated enum ConnectSnippets {
    enum Harness: String, CaseIterable, Identifiable, Sendable {
        case curl = "curl"
        case openCode = "OpenCode"
        case xcode = "Xcode"
        case python = "Python"
        /// S2-5. `continue` is a Swift keyword, so the case is spelt out; the raw value is what the
        /// sheet shows.
        case continueDev = "Continue"
        var id: String { rawValue }
    }

    /// What a snippet may claim about the selected model (S2-5). Assembled in the app by
    /// `AgentReadinessFacts.connectFacts(hfModelID:)`, so the sheet, the tests and the readme's
    /// C-rows all describe the same facts.
    nonisolated struct ModelFacts: Equatable, Sendable {
        var modelID: String
        /// S2-4's state. `notAgentReady` ⇒ S1's plain-chat block, with **no** tools or limits claims.
        var agent: AgentReadiness.State
        /// The context window — the number S2-6's pre-count enforces, and what `/v1/models` reports.
        var contextLength: Int?
        /// Does S2-3's thinking control reach this family's template? (`ReasoningTemplateTable`.)
        var thinkingAvailable: Bool

        /// The S1 shape: the model's name and nothing it can't do.
        static func plain(_ modelID: String, contextLength: Int? = nil, thinkingAvailable: Bool = false) -> Self {
            ModelFacts(modelID: modelID, agent: .notAgentReady(reason: ""), contextLength: contextLength,
                       thinkingAvailable: thinkingAvailable)
        }

        /// Whether tools and limits may be advertised at all (S2-5's rule).
        var isAgentCapable: Bool {
            if case .notAgentReady = agent { return false }
            return true
        }
    }

    /// `http://127.0.0.1:<port>/v1` — the Connect sheet always shows the IPv4 form (S1-A ruling 1).
    static func baseURL(port: Int) -> String { "http://127.0.0.1:\(port)/v1" }

    /// The S1 convenience: a model with no facts gets the plain-chat block.
    static func text(for harness: Harness, port: Int, modelID: String) -> String {
        text(for: harness, port: port, model: .plain(modelID))
    }

    static func text(for harness: Harness, port: Int, model: ModelFacts) -> String {
        let base = baseURL(port: port)
        let modelID = model.modelID
        switch harness {
        case .curl:
            // `-N` turns off curl's output buffering so a streamed reply shows token by token.
            return """
            curl -N \(base)/chat/completions \\
              -H 'Content-Type: application/json' \\
              -d '{"model": "\(jsonEscaped(modelID))", "stream": true, "messages": [{"role": "user", "content": "Hello"}]}'
            """
        case .openCode:
            return openCode(base: base, model: model)
        case .xcode:
            return """
            1. Open Xcode ▸ Settings ▸ Intelligence.
            2. Click Add Chat Provider… and choose Locally Hosted.
            3. Set Port to \(port).
            Model: \(modelID)
            """
        case .python:
            return """
            from openai import OpenAI

            client = OpenAI(base_url="\(base)", api_key="unused")
            stream = client.chat.completions.create(
                model="\(pythonEscaped(modelID))",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            for event in stream:
                print(event.choices[0].delta.content or "", end="")
            """
        case .continueDev:
            return continueConfig(base: base, model: model)
        }
    }

    // MARK: - OpenCode (C2)

    /// opencode.json — https://opencode.ai/docs/providers/ ("custom provider"), verified 2026-10-07.
    ///
    /// The per-model keys are what makes tools and compaction work:
    ///
    /// * `tool_call` — the model-level flag (snake_case; the config schema's `models.<id>.tool_call`,
    ///   default `true`, only an explicit `false` turns it off).
    /// * `limit` — `context` + `output`, both **required** by the schema when `limit` is present.
    ///   OpenCode compacts against `context` and puts `output` in `max_tokens`, so `context` is the
    ///   window S2-6's pre-count enforces: OpenCode compacts *before* our `400`.
    /// * `reasoning` + `options.reasoningEffort` — only for a family whose template reads
    ///   `enable_thinking` (S2-3). `reasoning: true` is what makes OpenCode offer the effort variants;
    ///   `options.reasoningEffort` is what reaches the wire, via the SDK's `reasoning_effort`.
    private static func openCode(base: String, model: ModelFacts) -> String {
        let id = jsonEscaped(model.modelID)
        var members = ["\"name\": \"\(id)\""]
        if model.isAgentCapable {
            members.append("\"tool_call\": true")
            if let context = model.contextLength {
                members.append("\"limit\": { \"context\": \(context), \"output\": \(serverDefaultMaxTokens) }")
            }
            if model.thinkingAvailable {
                members.append("\"reasoning\": true")
                members.append("\"options\": { \"reasoningEffort\": \"high\" }")
            }
        }
        // 10 spaces: inside `"models"` (6) → the model key (8) → its object.
        let body = members.map { "          " + $0 }.joined(separator: ",\n")
        return """
        {
          "$schema": "https://opencode.ai/config.json",
          "provider": {
            "mlxui": {
              "npm": "@ai-sdk/openai-compatible",
              "name": "AI Browser (local)",
              "options": {
                "baseURL": "\(base)"
              },
              "models": {
                "\(id)": {
        \(body)
                }
              }
            }
          }
        }
        """
    }

    // MARK: - Continue (C7)

    /// ~/.continue/config.yaml — https://docs.continue.dev/reference ("models"), verified 2026-10-07.
    ///
    /// * `provider: openai` + `apiBase` is the documented way to point Continue at an
    ///   OpenAI-compatible server (there is no `openai-compatible` provider).
    /// * `capabilities: [tool_use]` is the one key that makes Continue send tools — it overrides its
    ///   own autodetection ("required for Agent mode").
    /// * `contextLength` lives under `defaultCompletionOptions` in the reference.
    /// * Thinking goes through `requestOptions.extraBodyProperties`, which merges arbitrary keys into
    ///   the HTTP body — `reasoning_effort` is our S2-3 wire name.
    private static func continueConfig(base: String, model: ModelFacts) -> String {
        var lines = [
            "# ~/.continue/config.yaml",
            "name: AI Browser (local)",
            "version: 0.0.1",
            "schema: v1",
            "",
            "models:",
            "  - name: \(model.modelID)",
            "    provider: openai",
            "    model: \(model.modelID)",
            "    apiBase: \(base)",
            "    apiKey: unused    # the local server doesn't check it",
        ]
        if model.isAgentCapable {
            lines.append("    capabilities:")
            lines.append("      - tool_use")
            if let context = model.contextLength {
                lines.append("    defaultCompletionOptions:")
                lines.append("      contextLength: \(context)")
            }
            if model.thinkingAvailable {
                lines.append("    requestOptions:")
                lines.append("      extraBodyProperties:")
                lines.append(#"        reasoning_effort: "high""#)
            }
        }
        return lines.joined(separator: "\n")
    }

    /// Where each snippet is documented, for the sheet's footnote.
    static func docsURL(for harness: Harness) -> URL? {
        switch harness {
        case .curl: return nil
        case .openCode: return URL(string: "https://opencode.ai/docs/providers/")
        case .xcode: return URL(string: "https://developer.apple.com/videos/play/wwdc2026/232/")
        case .python: return URL(string: "https://github.com/openai/openai-python")
        case .continueDev: return URL(string: "https://docs.continue.dev/reference")
        }
    }

    /// What a request that doesn't name `max_tokens` gets (R16). Advertised as OpenCode's
    /// `limit.output` so the client asks for exactly what the server would do by default.
    static let serverDefaultMaxTokens = ServeRequest.defaultMaxTokens

    /// One honest line about a model's fitness (design §5 item 2): its context length when the
    /// checkpoint's `config.json` says, plus its **Agent-ready state** (S2-4).
    ///
    /// S1's wording ended "It answers in plain text — tool calls aren't supported yet." S2-2 made
    /// that false, so the second sentence is now `AgentReadiness.line(for:)` — one short line, and
    /// for a model that isn't Agent-ready it is the plain-English reason.
    static func fitnessLine(contextLength: Int?, agent: AgentReadiness.State) -> String {
        let context = contextLength.map { "Context length: \(formatted($0)) tokens." }
            ?? "Context length: not stated by this model's config."
        return context + " " + AgentReadiness.line(for: agent)
    }

    /// `max_position_embeddings` from a checkpoint's `config.json` (top level, or inside
    /// `text_config` for multimodal checkpoints). `nil` when absent or unreadable.
    /// The checkpoint's declared window. The parse lives in `ModelContextWindow` (S2-6) because the
    /// Local Server's pre-count must use the same number this line shows; this stays as the Connect
    /// sheet's name for it.
    static func contextLength(configJSON: Data?) -> Int? {
        ModelContextWindow.limit(configData: configJSON)
    }

    /// Apple Foundation Models' window (measured on macOS 27.0: `SystemLanguageModel.contextSize`).
    static let appleFoundationContextLength = 8192

    private static func formatted(_ number: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = Locale(identifier: "en_US")
        return formatter.string(from: NSNumber(value: number)) ?? String(number)
    }

    private static func jsonEscaped(_ text: String) -> String {
        text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    }
    private static func pythonEscaped(_ text: String) -> String { jsonEscaped(text) }
}

/// Rule 14/R27 for the one surface that hands out snippets: **no gate, no sheet, no snippet.**
/// Extracted from `MLXUIApp`'s sheet binding so "gate closed ⇒ nothing" is a pure, testable rule
/// (S2-5) rather than a clause inside a view.
nonisolated enum ConnectSheetPresentation {
    static func isPresented(gateAvailable: Bool, requested: Bool) -> Bool {
        gateAvailable && requested
    }
}
