import Foundation

/// CP-2: the words of the "Delete All Conversations" confirmation, as a pure function so the
/// singular/plural and bench/page wording is pinned by tests.
nonisolated enum ChatDeleteAllText {
    /// `modelName` is the bench's model (the sheet only lists that model's chats); `nil` on the page.
    static func title(count: Int, modelName: String?) -> String {
        switch (count == 1, modelName) {
        case (true, nil): return "Delete the conversation?"
        case (false, nil): return "Delete all \(count) conversations?"
        case (true, let name?): return "Delete the conversation with \(name)?"
        case (false, let name?): return "Delete all \(count) conversations with \(name)?"
        }
    }

    static func message(modelName: String?) -> String {
        if let modelName {
            return "This permanently removes only the conversations with \(modelName) from this Mac. Chats with other models are kept."
        }
        return "This permanently removes every conversation from this Mac."
    }
}
