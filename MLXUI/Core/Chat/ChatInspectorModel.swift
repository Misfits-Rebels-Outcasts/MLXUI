//
//  ChatInspectorModel.swift
//  MLXUI — chat workspace, CW-4 (`RSI/DelegateChatWorkspaceBacklog.md`).
//
//  The inspector's arithmetic, kept out of the view so it is unit-testable: the C15 context meter,
//  the per-reply tool-call count, and the read/read+write badge for an attached folder. The view
//  renders what this says and decides nothing.
//
//  Pure `nonisolated` value type: no SwiftUI, no runner, no I/O.
//

import Foundation

nonisolated struct ChatInspectorModel: Sendable, Equatable {

    // MARK: - Context meter (C15)

    /// How full the model's window is, as the app's estimate convention (`~`, from `utf8/4`).
    struct Meter: Sendable, Equatable {
        /// `ceil(utf8(environment + history + draft) / 4)`.
        let estimatedTokens: Int
        /// The model's window: `ModelEntry.contextWindow`, else its `config.json`, else 8 192.
        let window: Int

        var fraction: Double {
            guard window > 0 else { return 0 }
            return Double(estimatedTokens) / Double(window)
        }

        /// Amber at 80% — small models hit this first once files are inlined.
        var isAmber: Bool { fraction >= 0.8 }

        /// `~12,300 / 40,960 (30%)`.
        var label: String {
            let percent = Int((fraction * 100).rounded())
            return "~\(Self.decimal(estimatedTokens)) / \(Self.decimal(window)) (\(percent)%)"
        }

        private static func decimal(_ value: Int) -> String {
            let formatter = NumberFormatter()
            formatter.numberStyle = .decimal
            formatter.groupingSeparator = ","
            return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
        }
    }

    /// The meter for one turn's inputs. `estimatedTokens` is passed in already counted, so the
    /// view can feed it the same text the model is about to receive.
    static func meter(estimatedTokens: Int, window: Int) -> Meter {
        Meter(estimatedTokens: max(0, estimatedTokens), window: max(0, window))
    }

    /// Tokens for a block of text, by the app's `ceil(utf8/4)` estimate.
    static func estimatedTokens(of text: String) -> Int {
        ChatAttachment.estimatedTokens(forUTF8Count: text.utf8.count)
    }

    // MARK: - Session

    /// Tool calls in the current or last reply: the `.tool` items after the last user message.
    static func toolCallsSinceLastUser(_ items: [TranscriptItem]) -> Int {
        var count = 0
        for item in items.reversed() {
            switch item {
            case .message(let message):
                if message.role == .user { return count }
            case .tool:
                count += 1
            }
        }
        return count
    }

    /// `3 / 8` for the Session section.
    static func toolCallLabel(used: Int, limit: Int) -> String {
        "\(used) / \(limit)"
    }

    // MARK: - Attached folders (Context)

    /// Whether an attached folder is writable by the current build's file tools — C3's rule, from
    /// the enabled set rather than a second opinion.
    static func canWrite(enabledToolNames: Set<String>) -> Bool {
        !enabledToolNames.isDisjoint(with: ChatEnvironment.writeToolNames)
    }

    /// The Context badge: `read + write` or `read only`.
    static func accessBadge(canWrite: Bool) -> String {
        canWrite ? "read + write" : "read only"
    }

    /// `4 of 12 tools on`.
    static func toolsOnLabel(enabled: Int, available: Int) -> String {
        "\(enabled) of \(available) tools on"
    }
}
