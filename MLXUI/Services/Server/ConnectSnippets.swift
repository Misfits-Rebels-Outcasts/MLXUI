import Foundation

/// The copy-paste text on the "Connect your app" sheet (design §5). **Every snippet is checked
/// against the harness's current docs and recorded in `RSI/DelegateServeReadme.md`'s copy table**
/// (C-rows, with URL + date) — no config key here is invented. S3 adds Claude Code and Codex.
nonisolated enum ConnectSnippets {
    enum Harness: String, CaseIterable, Identifiable, Sendable {
        case curl = "curl"
        case openCode = "OpenCode"
        case xcode = "Xcode"
        case python = "Python"
        var id: String { rawValue }
    }

    /// `http://127.0.0.1:<port>/v1` — the Connect sheet always shows the IPv4 form (S1-A ruling 1).
    static func baseURL(port: Int) -> String { "http://127.0.0.1:\(port)/v1" }

    static func text(for harness: Harness, port: Int, modelID: String) -> String {
        let base = baseURL(port: port)
        switch harness {
        case .curl:
            // `-N` turns off curl's output buffering so a streamed reply shows token by token.
            return """
            curl -N \(base)/chat/completions \\
              -H 'Content-Type: application/json' \\
              -d '{"model": "\(jsonEscaped(modelID))", "stream": true, "messages": [{"role": "user", "content": "Hello"}]}'
            """
        case .openCode:
            // opencode.json — https://opencode.ai/docs/providers/ ("custom provider").
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
                    "\(jsonEscaped(modelID))": {
                      "name": "\(jsonEscaped(modelID))"
                    }
                  }
                }
              }
            }
            """
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
        }
    }

    /// Where each snippet is documented, for the sheet's footnote.
    static func docsURL(for harness: Harness) -> URL? {
        switch harness {
        case .curl: return nil
        case .openCode: return URL(string: "https://opencode.ai/docs/providers/")
        case .xcode: return URL(string: "https://developer.apple.com/videos/play/wwdc2026/232/")
        case .python: return URL(string: "https://github.com/openai/openai-python")
        }
    }

    /// One honest line about a model's fitness (design §5 item 2): its context length when the
    /// checkpoint's `config.json` says, and that S1 answers in plain text only.
    static func fitnessLine(contextLength: Int?) -> String {
        let context = contextLength.map { "Context length: \(formatted($0)) tokens." }
            ?? "Context length: not stated by this model's config."
        return context + " It answers in plain text — tool calls aren't supported yet."
    }

    /// `max_position_embeddings` from a checkpoint's `config.json` (top level, or inside
    /// `text_config` for multimodal checkpoints). `nil` when absent or unreadable.
    static func contextLength(configJSON: Data?) -> Int? {
        guard let configJSON,
              let object = (try? JSONSerialization.jsonObject(with: configJSON)) as? [String: Any] else { return nil }
        func positions(_ dictionary: [String: Any]) -> Int? {
            (dictionary["max_position_embeddings"] as? NSNumber).flatMap { $0.intValue > 0 ? $0.intValue : nil }
        }
        return positions(object) ?? (object["text_config"] as? [String: Any]).flatMap(positions)
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
