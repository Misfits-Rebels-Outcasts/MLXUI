import Foundation

/// How a decoded `[ServeMessage]` becomes the message list a model's chat template is handed.
///
/// Three rules, all applied to the neutral core so every dialect (S3's Anthropic and Responses
/// translators included) gets them:
///
/// 1. **Same-role merge** — owner ruling R21, 2026-10-02. Flattening the tool traffic turns each
///    tool result into a `user` turn, and strict-alternation templates (the Mistral family)
///    reject two `user` turns in a row; a client may also send consecutive same-role messages
///    itself. Applied to every role, except the two structured S2-2 cases below.
/// 2. **Tool traffic** — a model whose template can't take tools gets S1's flattening (owner
///    ruling R20, 2026-10-02); a model that can gets the client's structured turns untouched.
/// 3. **`system` placement** — owner ruling R22, 2026-10-02: "hoisting system messages to the
///    front that requires it is deferred to S2-1's per-family template work". This is that work.
///    It runs *after* the merge, and re-applies the merge because moving a `system` message out
///    of the middle can make two same-role messages adjacent that were not before.
///
/// The per-family answer lives in `ToolCallFormatTable.systemPlacement(forModelType:)`; the
/// server's MLX backend asks for it because only the backend knows the checkpoint's `model_type`.
nonisolated enum ServeMessageAssembly {

    /// The whole assembly, in the order the server applies it: tool traffic settled first, then the
    /// family's one-call-per-turn rule (S2-8), then its `system` placement.
    ///
    /// The split runs **before** the hoisting, so the hoisting's own re-merge sees the final shape —
    /// and the split's pieces can't be glued back together, because `canMerge` refuses both a turn
    /// with `toolCalls` and a `tool` turn (which is exactly why the interleaving order matters).
    static func assembling(_ messages: [ServeMessage],
                           acceptsTools: Bool,
                           oneCallPerTurn: Bool = false,
                           placement: ToolCallFormatTable.SystemPlacement) -> [ServeMessage] {
        let settled = acceptsTools ? messages : flatteningToolTraffic(messages)
        let history = oneCallPerTurn ? splittingMultiCallTurns(settled) : settled
        return hoistingSystemMessages(history, placement: placement)
    }

    /// **S2-8 — a family whose template takes one tool call per assistant turn.** Llama 3.1's
    /// template *raises* on a turn with two (`ToolCallHistoryTable`), so a model that answered with
    /// several calls could not be asked again: the histories the client replays were unrenderable
    /// (SV-6's Llama run, journal `2026-415`).
    ///
    /// An assistant turn with N > 1 calls becomes N single-call turns, each followed by the `tool`
    /// results that belong to it — `assistant(A), tool(A), assistant(B), tool(B) …` — matched on
    /// `tool_call_id`. The turn's text rides on the **first** piece, and every piece keeps the
    /// original role ordering around it.
    ///
    /// **Edge cases (implementer's call, pending owner confirmation):**
    ///
    /// * a call with **no id**, or a result whose id matches no call in the turn — the calls are
    ///   still split (otherwise the template raises and the request dies), but the results are left
    ///   **in the client's order** after the last piece, rather than guessed at;
    /// * results that arrive *before* their assistant turn (a replayed conversation may do this) are
    ///   matched the same way and moved to their call;
    /// * a turn with a single call is untouched — text, id and result position stay as the client
    ///   sent them, which is the shape every family's template already handles.
    static func splittingMultiCallTurns(_ messages: [ServeMessage]) -> [ServeMessage] {
        guard messages.contains(where: { $0.role == .assistant && $0.toolCalls.count > 1 }) else {
            return messages                                             // nothing to do: byte-identical
        }
        // Which results **move** — only those naming a call of a multi-call turn. Everything else
        // stays exactly where the client put it, which is what "keeps the client's order" means.
        var owners: [Int: (turn: Int, id: String)] = [:]                // result index → its call
        for (turn, message) in messages.enumerated()
        where message.role == .assistant && message.toolCalls.count > 1 {
            let ids = Set(message.toolCalls.compactMap(\.id))
            for (index, candidate) in messages.enumerated()
            where candidate.role == .tool && candidate.toolCallID.map(ids.contains) == true {
                owners[index] = (turn, candidate.toolCallID ?? "")
            }
        }

        var out: [ServeMessage] = []
        for (index, message) in messages.enumerated() {
            if message.role == .tool, owners[index] != nil {
                continue                                                // placed with its call below
            }
            guard message.role == .assistant, message.toolCalls.count > 1 else {
                out.append(message)
                continue
            }
            for (position, call) in message.toolCalls.enumerated() {
                // The turn's text rides on the first piece; the rest are calls only.
                out.append(ServeMessage(role: .assistant, text: position == 0 ? message.text : "",
                                        toolCalls: [call]))
                for (resultIndex, result) in messages.enumerated()
                where owners[resultIndex]?.turn == index && owners[resultIndex]?.id == call.id {
                    out.append(result)
                }
            }
        }
        return out
    }

    /// Adjacent messages with the same role become one, joined by a blank line. An empty side adds
    /// no stray blank line.
    ///
    /// **Two exceptions (S2-2)**, because a message now carries more than text:
    ///
    /// - a `tool` turn is one tool result — gluing two together would hand the template a single
    ///   result where the client sent two (the Qwen templates group consecutive tool results
    ///   themselves, and Mistral wraps each in `[TOOL_RESULTS]`);
    /// - an assistant turn carrying `tool_calls` keeps them: merging it into a neighbouring
    ///   assistant turn would drop the calls on the floor — and merging a plain turn *into* it
    ///   would silently add text to a structured turn. Both sides of the merge are checked.
    static func mergingAdjacentRoles(_ messages: [ServeMessage]) -> [ServeMessage] {
        var merged: [ServeMessage] = []
        for message in messages {
            if let last = merged.last, last.role == message.role,
               canMerge(last), canMerge(message) {
                merged[merged.count - 1].text = [last.text, message.text].filter { !$0.isEmpty }
                    .joined(separator: "\n\n")
            } else {
                merged.append(message)
            }
        }
        return merged
    }

    private static func canMerge(_ message: ServeMessage) -> Bool {
        message.role != .tool && message.toolCalls.isEmpty
    }

    /// S1's flattening of tool traffic into plain text (owner ruling R20, 2026-10-02: don't 400 —
    /// S1 ignores `tools`, but a harness replays its whole conversation, tool turns included):
    ///
    /// - an assistant turn's `tool_calls` become lines `[called <name> with <arguments>]` after any
    ///   text it had;
    /// - a `tool` message becomes a **user** turn `[tool result for <name>: <content>]`, where
    ///   `<name>` is the message's `name`, else the name of the call with that `tool_call_id`,
    ///   else `tool`.
    ///
    /// Wording is S1's, unchanged — a model that can't take tools must see exactly what it saw
    /// before S2-2. Re-merged afterwards, because flattening can put two user turns side by side.
    static func flatteningToolTraffic(_ messages: [ServeMessage]) -> [ServeMessage] {
        var names: [String: String] = [:]               // tool_call id → function name
        var flattened: [ServeMessage] = []
        for message in messages {
            switch message.role {
            case .assistant where !message.toolCalls.isEmpty:
                var lines: [String] = message.text.isEmpty ? [] : [message.text]
                for call in message.toolCalls {
                    if let id = call.id { names[id] = call.name }
                    lines.append("[called \(call.name) with \(call.argumentsJSON)]")
                }
                flattened.append(ServeMessage(role: .assistant, text: lines.joined(separator: "\n")))
            case .tool:
                let name = message.toolName
                    ?? message.toolCallID.flatMap { names[$0] } ?? "tool"
                flattened.append(ServeMessage(role: .user, text: "[tool result for \(name): \(message.text)]"))
            default:
                flattened.append(message)
            }
        }
        return mergingAdjacentRoles(flattened)
    }

    /// Move or fold `system` messages for families whose template can't take them where the client
    /// put them.
    ///
    /// - `.anywhere` — returned unchanged. This is also the answer for an unknown family, so S1's
    ///   behaviour is preserved wherever this table has no evidence.
    /// - `.firstOnly` — every `system` message's text is joined (blank line) into **one** leading
    ///   `system` message, and the rest keep their order; the result is re-merged. A list that
    ///   already has exactly one leading system message is returned untouched.
    /// - `.unsupported` — the `system` text is prepended to the **first `user`** turn, since the
    ///   template has no system role to put it in; if there is no user turn, a leading one is
    ///   created. Re-merged for the same reason.
    ///
    /// A list that needs no moving is returned exactly as it came — a single leading system
    /// message is fine for a `.firstOnly` family even when its text is empty. Where the list *is*
    /// rebuilt, an empty system message leaves no leading turn behind (it carries nothing for the
    /// template to render).
    static func hoistingSystemMessages(
        _ messages: [ServeMessage],
        placement: ToolCallFormatTable.SystemPlacement
    ) -> [ServeMessage] {
        switch placement {
        case .anywhere:
            return messages

        case .firstOnly:
            let systems = messages.filter { $0.role == .system }
            guard !systems.isEmpty else { return messages }
            if systems.count == 1, messages.first?.role == .system { return messages }
            let text = joined(systems)
            var reordered = messages.filter { $0.role != .system }
            if !text.isEmpty { reordered.insert(ServeMessage(role: .system, text: text), at: 0) }
            return mergingAdjacentRoles(reordered)

        case .unsupported:
            let systems = messages.filter { $0.role == .system }
            guard !systems.isEmpty else { return messages }
            let text = joined(systems)
            var rest = messages.filter { $0.role != .system }
            if !text.isEmpty {
                if let index = rest.firstIndex(where: { $0.role == .user }) {
                    rest[index].text = rest[index].text.isEmpty ? text : text + "\n\n" + rest[index].text
                } else {
                    rest.insert(ServeMessage(role: .user, text: text), at: 0)
                }
            }
            return mergingAdjacentRoles(rest)
        }
    }

    private static func joined(_ systems: [ServeMessage]) -> String {
        systems.map(\.text).filter { !$0.isEmpty }.joined(separator: "\n\n")
    }
}
