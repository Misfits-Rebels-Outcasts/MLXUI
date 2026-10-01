import SwiftUI

/// KC-5 — the one place a secret is typed. Used by the HuggingFace token row and by the
/// Providers pane's Add / Change flows. It **never pre-fills** and never reads the existing
/// value: `Change…` starts from an empty field, and the key only ever goes from the field into
/// `onSave`.
struct CredentialKeySheet: View {
    /// What the sheet is for: a single fixed target (`Change…`, `Set Up…`) or a choice among
    /// the not-yet-configured providers of one kind (`Add …`).
    let title: String
    let choices: [String]
    let caption: String?
    let placeholder: (String) -> String
    let isValid: (String) -> Bool
    let signupLink: (String) -> (label: String, url: URL)?
    let onSave: (_ choice: String, _ key: String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var selected: String
    @State private var keyInput: String = ""

    init(title: String, choices: [String], caption: String? = nil,
         placeholder: @escaping (String) -> String,
         isValid: @escaping (String) -> Bool = { !$0.isEmpty },
         signupLink: @escaping (String) -> (label: String, url: URL)?,
         onSave: @escaping (_ choice: String, _ key: String) -> Void) {
        self.title = title
        self.choices = choices
        self.caption = caption
        self.placeholder = placeholder
        self.isValid = isValid
        self.signupLink = signupLink
        self.onSave = onSave
        _selected = State(initialValue: choices.first ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.headline)

            if let caption {
                Text(caption).font(.caption).foregroundStyle(.secondary)
            }
            if choices.count > 1 {
                Picker("Provider", selection: $selected) {
                    ForEach(choices, id: \.self) { Text($0.capitalized).tag($0) }
                }
            }

            SecureField(placeholder(selected), text: $keyInput)
                .textFieldStyle(.roundedBorder)

            if let link = signupLink(selected) {
                Link(link.label, destination: link.url).font(.caption)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") {
                    onSave(selected, keyInput)
                    keyInput = ""
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!isValid(keyInput) || selected.isEmpty)
            }
        }
        .padding(20)
        .frame(width: 420)
    }
}
