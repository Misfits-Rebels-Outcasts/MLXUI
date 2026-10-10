import Foundation

/// The copy-paste text on the "Connect your app" sheet (design §5). **Every snippet is checked
/// against the harness's current docs and recorded in `RSI/DelegateServeReadme.md`'s copy table**
/// (C-rows, with URL + date) — no config key here is invented. S3 adds Claude Code and Codex.
///
/// **S2-5** — the app snippets (OpenCode, Continue) are per **model** now: what they advertise
/// depends on the model's **tool capability** (`ToolCallFormatTable.toolSupport`, S2-1) and on whether
/// the family can think, so a model that can't take tools gets S1's plain-chat block rather than a tools
/// or limits claim that would be false. (R51, 2026-10-10: there is no "Agent-ready" state any more.)
nonisolated enum ConnectSnippets {
    enum Harness: String, CaseIterable, Identifiable, Sendable {
        case curl = "curl"
        case openCode = "OpenCode"
        /// S3-4 (C5, C6).
        case claudeCode = "Claude Code"
        case codex = "Codex"
        case xcode = "Xcode"
        case python = "Python"
        /// S2-5. `continue` is a Swift keyword, so the case is spelt out; the raw value is what the
        /// sheet shows.
        case continueDev = "Continue"
        var id: String { rawValue }
    }

    /// What a snippet may claim about the selected model (S2-5). Assembled in the app by
    /// `ServeModelFacts.connectFacts(hfModelID:)`, so the sheet, the tests and the readme's
    /// C-rows all describe the same facts.
    nonisolated struct ModelFacts: Equatable, Sendable {
        var modelID: String
        /// Can this family's template take tools (`ToolCallFormatTable.toolSupport(...).acceptsTools`)?
        /// `false` ⇒ S1's plain-chat block, with **no** tools or limits claims.
        var supportsTools: Bool
        /// The context window — the number S2-6's pre-count enforces, and what `/v1/models` reports.
        var contextLength: Int?
        /// Does S2-3's thinking control reach this family's template? (`ReasoningTemplateTable`.)
        var thinkingAvailable: Bool
        /// R53 — `min(context_length, the largest prompt memory admission accepts on this Mac)`: the number the
        /// Claude Code and Codex blocks tell the client, so it compacts before the server would refuse. `nil` means
        /// "not computed" and falls back to `contextLength` (see `usable`).
        var usableTokens: Int? = nil

        /// What a client app may send: the memory-aware figure, else the model's own window, else unknown.
        var usable: Int? { usableTokens ?? contextLength }

        /// The S1 shape: the model's name and nothing it can't do.
        static func plain(_ modelID: String, contextLength: Int? = nil, thinkingAvailable: Bool = false) -> Self {
            ModelFacts(modelID: modelID, supportsTools: false, contextLength: contextLength,
                       thinkingAvailable: thinkingAvailable)
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
        case .claudeCode:
            return claudeCode(port: port, model: model)
        case .codex:
            return codex(base: base, model: model)
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
    /// * `reasoning: true` — only for a family whose template reads `enable_thinking` (S2-3), and it
    ///   is the **capability** flag alone (S2-5b): it makes OpenCode offer the effort variants; the
    ///   user picks one when they want it. Nothing here asks for thinking — R30, and with
    ///   `limit.output` covering reasoning *and* answer an unattended default can end `length`
    ///   before the tool call.
    private static func openCode(base: String, model: ModelFacts) -> String {
        let id = jsonEscaped(model.modelID)
        var members = ["\"name\": \"\(id)\""]
        if model.supportsTools {
            members.append("\"tool_call\": true")
            if let context = model.contextLength {
                members.append("\"limit\": { \"context\": \(context), \"output\": \(serverDefaultMaxTokens) }")
            }
            if model.thinkingAvailable {
                // S2-5b: the **capability** flag only. R30's "thinking is off unless asked" means no
                // `options.reasoningEffort` here — a default effort would ask on every turn, and
                // with `limit.output` covering reasoning *and* answer a long think can end `length`
                // before the tool call. OpenCode shows the effort variants because of this flag; the
                // user picks one when they want it (the readme's C2 row says how).
                members.append("\"reasoning\": true")
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

    // MARK: - Claude Code (C5) and Codex (C6)

    /// **Every served model gets both blocks (owner ruling R56, 2026-10-10)**: like LM Studio and Ollama, the
    /// sheet shows setup for any model and recommends, it never gates. Two plain lines say what the model can't do:
    /// this one for a model without tool support (replaces S3-4's "can't be used with Claude Code/Codex" lines)…
    static let noToolCallsLine = "This model doesn't support tool calls, so agent tools won't work."

    /// …and this one when Claude Code's room is small. Claude Code's own prompt and tools are ~20k tokens and it
    /// reserves a ~33k compaction buffer (SV-7, journal 2026-446), so under 64k it thrashes.
    static let claudeCodeComfortableTokens = 65_536
    static func claudeCodeContextLine(usable: Int) -> String {
        "Claude Code works best with 64k+ tokens of context; this model holds about \(formatted(usable)) on this Mac."
    }

    /// Claude Code — verified 2026-10-10 against https://code.claude.com/docs/en/env-vars and
    /// https://code.claude.com/docs/en/model-config (the readme's C5 row has the quotes):
    ///
    /// * `ANTHROPIC_BASE_URL` (no `/v1`: Claude Code appends its own path) and `ANTHROPIC_AUTH_TOKEN`
    ///   ("custom value for the `Authorization` header").
    /// * `ANTHROPIC_MODEL`, and `ANTHROPIC_DEFAULT_HAIKU_MODEL` for the small/fast background model
    ///   (`ANTHROPIC_SMALL_FAST_MODEL` is documented as deprecated).
    /// * `CLAUDE_CODE_MAX_CONTEXT_TOKENS` — for an id that doesn't start with `claude-` (ours never does) "the
    ///   variable applies directly and proactive compaction continues at the declared window". `usable`, R53.
    ///   (`CLAUDE_CODE_AUTO_COMPACT_WINDOW` is not used: its documented range starts at 100,000.)
    /// * `--strict-mcp-config --mcp-config '{"mcpServers":{}}'` — "only use MCP servers from --mcp-config",
    ///   checked against `claude --help` of 2.1.296.
    ///
    /// Thinking: nothing to set (the server ignores adaptive thinking, R44).
    private static func claudeCode(port: Int, model: ModelFacts) -> String {
        var lines: [String] = []
        if !model.supportsTools { lines.append("# " + noToolCallsLine) }
        if let usable = model.usable, usable < claudeCodeComfortableTokens {
            lines.append("# " + claudeCodeContextLine(usable: usable))
        }
        if !lines.isEmpty { lines.append("") }
        lines += [
            "export ANTHROPIC_BASE_URL=http://127.0.0.1:\(port)",
            "export ANTHROPIC_AUTH_TOKEN=unused    # any non-empty value; the local server doesn't check it",
            "export ANTHROPIC_MODEL=\(shellQuoted(model.modelID))",
            "export ANTHROPIC_DEFAULT_HAIKU_MODEL=\(shellQuoted(model.modelID))    # Claude Code's small background model",
        ]
        let leanNote: String
        if let usable = model.usable {
            lines.append("export CLAUDE_CODE_MAX_CONTEXT_TOKENS=\(usable)    # so it compacts before the server would refuse")
            leanNote = "# MCP servers add to every request, and this model holds \(formatted(usable)) tokens."
        } else {
            leanNote = "# MCP servers add to every request, and this model's memory is limited."
        }
        lines.append("")
        lines.append(leanNote)
        lines.append("# For a lean session, start Claude Code without MCP servers:")
        // No `--settings modelOverrides` here (R58): it makes Claude Code treat the id as a known model, and then
        // `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is ignored (/context showed 1M, journal 2026-448). The warning
        // "[claude-code:unrecognized_model]" it would silence is cosmetic.
        lines.append(#"claude --strict-mcp-config --mcp-config '{"mcpServers":{}}'"#)
        return lines.joined(separator: "\n")
    }

    /// Codex — verified 2026-10-10 against https://learn.chatgpt.com/docs/config-file/config-reference and
    /// .../config-advanced (the redirect target of developers.openai.com/codex/config-*): `model`,
    /// `model_provider`, `model_context_window`, `model_auto_compact_token_limit`, `web_search` (`disabled`),
    /// `[model_providers.<id>]` with `name`, `base_url`, `env_key`, `wire_api` (only `responses` is accepted)
    /// and `stream_idle_timeout_ms` (default 300000; a local model's long prefill needs more).
    ///
    /// `model_auto_compact_token_limit` is **90 % of `usable`**, not `usable`: Codex's compaction request carries
    /// the whole history, so compacting only at the refusal line would be refused itself.
    /// No `model_reasoning_effort` (R40: thinking is off unless asked).
    private static func codex(base: String, model: ModelFacts) -> String {
        var top = [
            "model = \"\(tomlEscaped(model.modelID))\"",
            "model_provider = \"mlxui\"",
            "web_search = \"disabled\"",
        ]
        if let usable = model.usable {
            top.append("model_context_window = \(usable)")
            top.append("model_auto_compact_token_limit = \(codexCompactLimit(usable: usable))")
        }
        var lines: [String] = []
        if !model.supportsTools { lines += ["# " + noToolCallsLine, ""] }
        lines += [
            "# In your shell (any non-empty value; the local server doesn't check it):",
            "export MLXUI_API_KEY=unused",
            "",
            "# ~/.codex/config.toml — run Codex without MCP servers for now.",
        ]
        lines += top
        lines += [
            "",
            "[model_providers.mlxui]",
            "name = \"AI Browser (local)\"",
            "base_url = \"\(base)\"",
            "wire_api = \"responses\"",
            "env_key = \"MLXUI_API_KEY\"",
            "stream_idle_timeout_ms = 1800000",
        ]
        return lines.joined(separator: "\n")
    }

    /// Where Codex should compact: 90 % of the usable context (see `codex(base:model:)`).
    static func codexCompactLimit(usable: Int) -> Int { usable * 9 / 10 }

    // MARK: - Continue (C7)

    /// ~/.continue/config.yaml — https://docs.continue.dev/reference ("models"), verified 2026-10-07.
    ///
    /// * `provider: openai` + `apiBase` is the documented way to point Continue at an
    ///   OpenAI-compatible server (there is no `openai-compatible` provider).
    /// * `capabilities: [tool_use]` is the one key that makes Continue send tools — it overrides its
    ///   own autodetection ("required for Agent mode").
    /// * `contextLength` lives under `defaultCompletionOptions` in the reference.
    /// * Thinking goes through `requestOptions.extraBodyProperties`, which merges arbitrary keys into
    ///   the HTTP body — `reasoning_effort` is our S2-3 wire name. S2-5b ships it **commented out**
    ///   (R30: off unless asked), with the one-line hint above it.
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
        if model.supportsTools {
            lines.append("    capabilities:")
            lines.append("      - tool_use")
            if let context = model.contextLength {
                lines.append("    defaultCompletionOptions:")
                lines.append("      contextLength: \(context)")
            }
            if model.thinkingAvailable {
                // S2-5b: commented out, not active — same reason as OpenCode's (R30). Uncommenting
                // it turns thinking on for this model; until then the block never asks.
                lines.append("    # Uncommenting this turns thinking on for this model:")
                lines.append("    # requestOptions:")
                lines.append("    #   extraBodyProperties:")
                lines.append(#"    #     reasoning_effort: "high""#)
            }
        }
        return lines.joined(separator: "\n")
    }

    /// Where each snippet is documented, for the sheet's footnote.
    static func docsURL(for harness: Harness) -> URL? {
        switch harness {
        case .curl: return nil
        case .openCode: return URL(string: "https://opencode.ai/docs/providers/")
        case .claudeCode: return URL(string: "https://code.claude.com/docs/en/env-vars")
        case .codex: return URL(string: "https://developers.openai.com/codex/config-reference")
        case .xcode: return URL(string: "https://developer.apple.com/videos/play/wwdc2026/232/")
        case .python: return URL(string: "https://github.com/openai/openai-python")
        case .continueDev: return URL(string: "https://docs.continue.dev/reference")
        }
    }

    /// What a request that doesn't name `max_tokens` gets (R16). Advertised as OpenCode's
    /// `limit.output` so the client asks for exactly what the server would do by default.
    static let serverDefaultMaxTokens = ServeRequest.defaultMaxTokens

    /// The Connect sheet's one plain fact about a model: its context length, when the checkpoint's `config.json`
    /// says. Kept because it is true, useful (it is the window the server enforces and `/v1/models` reports) and
    /// not a verdict; the readiness line that used to follow it is gone (R51).
    static func contextLine(contextLength: Int?, usable: Int? = nil) -> String {
        guard let contextLength else { return "Context length: not stated by this model's config." }
        var line = "Context length: \(formatted(contextLength)) tokens"
        if let usable, usable < contextLength { line += " (this Mac holds about \(formatted(usable)))" }   // R56
        return line + "."
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
    private static func tomlEscaped(_ text: String) -> String { jsonEscaped(text) }
    private static func shellQuoted(_ text: String) -> String {
        text.allSatisfy { $0.isLetter || $0.isNumber || "-_./:@".contains($0) }
            ? text : "'" + text.replacingOccurrences(of: "'", with: "'\\''") + "'"
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

/// The Connect sheet's layout decisions that a test can pin (R59, owner 2026-10-10): the client choice is a
/// **menu**, so the row stays one line however many clients there are, and the sheet has one size for every model.
nonisolated enum ConnectSheetLayout {
    enum ClientPickerStyle: Equatable, Sendable { case menu, segmented }
    static let clientPickerStyle: ClientPickerStyle = .menu
    static let size = CGSize(width: 600, height: 520)
}
