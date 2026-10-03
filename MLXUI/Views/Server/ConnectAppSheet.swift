import SwiftUI
import AppKit

/// "Connect your app" (design §5): base URL, model, and copy-paste setup for curl, OpenCode,
/// Xcode and Python. Presented the first time a model is served, and from Settings → Local Server.
/// Only ever shown through `LocalServerGate` (rule 14) — its presenters check the gate.
struct ConnectAppSheet: View {
    @Environment(\.dismiss) private var dismiss
    let controller: LocalServerController
    let initialModelID: String?

    @State private var harness: ConnectSnippets.Harness = .curl
    @State private var modelID: String = ""
    @State private var copied: String?

    private var port: Int {
        if case .running(let port) = controller.server.status { return Int(port) }
        return controller.server.settings.port
    }
    private var models: [String] { controller.reachableIDs.sorted() }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Connect your app").font(.title2.bold())
            Text(controller.statusSentence)
                .font(.callout)
                .foregroundStyle(controller.isRunning ? Color.secondary : Color.orange)
            Text(LocalServerPolicy.chatOnlyNote)
                .font(.caption).foregroundStyle(.secondary)

            row("Base URL", ConnectSnippets.baseURL(port: port))
            if models.isEmpty {
                Text("No model is being served yet — turn on Serve for a model first.")
                    .foregroundStyle(.secondary)
            } else {
                HStack {
                    Text("Model").frame(width: 70, alignment: .leading).foregroundStyle(.secondary)
                    Picker("Model", selection: $modelID) {
                        ForEach(models, id: \.self) { Text($0).tag($0) }
                    }
                    .labelsHidden()
                    Button("Copy") { copy(modelID, label: "model") }
                }
                Text(fitness).font(.caption).foregroundStyle(.secondary)
            }

            Picker("App", selection: $harness) {
                ForEach(ConnectSnippets.Harness.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            ScrollView {
                Text(snippet)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .frame(minHeight: 150)
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))

            HStack {
                if let url = ConnectSnippets.docsURL(for: harness) {
                    Link("Docs", destination: url).font(.caption)
                }
                Text("Nothing here leaves this Mac; the server only answers requests from it.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let copied { Text("Copied \(copied)").font(.caption).foregroundStyle(.secondary) }
                Button("Copy") { copy(snippet, label: harness.rawValue) }
                    .disabled(models.isEmpty)
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 600, height: 520)
        .onAppear { modelID = initialModelID.flatMap { models.contains($0) ? $0 : nil } ?? models.first ?? "" }
    }

    private var snippet: String {
        ConnectSnippets.text(for: harness, port: port, modelID: modelID.isEmpty ? "<model id>" : modelID)
    }

    /// One honest line about the selected model.
    private var fitness: String {
        let length: Int?
        if modelID == ServedModels.appleFoundationID {
            length = ConnectSnippets.appleFoundationContextLength
        } else {
            let config = ModelStore.shared.directory(forHFModelID: modelID).appendingPathComponent("config.json")
            length = ConnectSnippets.contextLength(configJSON: try? Data(contentsOf: config))
        }
        return ConnectSnippets.fitnessLine(contextLength: length)
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).frame(width: 70, alignment: .leading).foregroundStyle(.secondary)
            Text(value).font(.system(.body, design: .monospaced)).textSelection(.enabled)
            Spacer()
            Button("Copy") { copy(value, label: title) }
        }
    }

    private func copy(_ text: String, label: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copied = label
    }
}
