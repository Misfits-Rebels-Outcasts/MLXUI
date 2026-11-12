import SwiftUI

/// CP-5: Settings → Chat. Text size for messages and for the conversation list.
struct ChatSettingsView: View {
    @AppStorage(ChatTextSize.messageKey) private var messageStored = ChatTextSize.standard.rawValue
    @AppStorage(ChatTextSize.listKey) private var listStored = ChatTextSize.standard.rawValue

    private var messageSize: Binding<ChatTextSize> {
        Binding(get: { ChatTextSize(stored: messageStored) }, set: { messageStored = $0.rawValue })
    }
    private var listSize: Binding<ChatTextSize> {
        Binding(get: { ChatTextSize(stored: listStored) }, set: { listStored = $0.rawValue })
    }

    var body: some View {
        Form {
            Section("Messages") {
                Picker("Text size", selection: messageSize) {
                    ForEach(ChatTextSize.allCases, id: \.self) { Text($0.title).tag($0) }
                }
                Text("The quick brown fox jumps over the lazy dog.")
                    .chatTextSize(.message, ChatTextSize(stored: messageStored))
                Text("⌘= larger, ⌘- smaller, ⌘0 standard, in a chat.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Conversation list") {
                Picker("Text size", selection: listSize) {
                    ForEach(ChatTextSize.allCases, id: \.self) { Text($0.title).tag($0) }
                }
                Text("Meeting notes")
                    .font(ChatTextSize(stored: listStored).font(.listTitle, standard: .callout))
                Text("Qwen3 8B · 2 hours ago")
                    .font(ChatTextSize(stored: listStored).font(.listDetail, standard: .caption2))
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}
