import Foundation

/// How a decoded `[ServeMessage]` becomes the message list a model's chat template is handed.
///
/// Two rules, both applied to the neutral core so every dialect (S3's Anthropic and Responses
/// translators included) gets them:
///
/// 1. **Same-role merge** — owner ruling R21, 2026-10-02. Flattening the tool traffic turns each
///    tool result into a `user` turn, and strict-alternation templates (the Mistral family)
///    reject two `user` turns in a row; a client may also send consecutive same-role messages
///    itself. Applied to every role.
/// 2. **`system` placement** — owner ruling R22, 2026-10-02: "hoisting system messages to the
///    front that requires it is deferred to S2-1's per-family template work". This is that work.
///    It runs *after* the merge, and re-applies the merge because moving a `system` message out
///    of the middle can make two same-role messages adjacent that were not before.
///
/// The per-family answer lives in `ToolCallFormatTable.systemPlacement(forModelType:)`; the
/// server's MLX backend asks for it because only the backend knows the checkpoint's `model_type`.
nonisolated enum ServeMessageAssembly {

    /// Adjacent messages with the same role become one, joined by a blank line. An empty side adds
    /// no stray blank line.
    static func mergingAdjacentRoles(_ messages: [ServeMessage]) -> [ServeMessage] {
        var merged: [ServeMessage] = []
        for message in messages {
            if let last = merged.last, last.role == message.role {
                merged[merged.count - 1].text = [last.text, message.text].filter { !$0.isEmpty }
                    .joined(separator: "\n\n")
            } else {
                merged.append(message)
            }
        }
        return merged
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
