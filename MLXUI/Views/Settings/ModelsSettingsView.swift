import SwiftUI
import AppKit

/// SET-6 (`RSI/DelegateSettingsBacklog.md`) — the Models pane: the HuggingFace token (reworded
/// per §8 — what it's for, what happens without it), where models live on disk and their total
/// size (+ Reveal in Finder), and the app-wide flow cache size (+ Clear Cache, per OG-6 —
/// `FlowMaintenanceMenu`'s per-flow clearing stays exactly as it is).
struct ModelsSettingsView: View {
    @State private var tokenInput: String = ""
    @State private var hasToken: Bool = KeychainHelper.getToken() != nil
    @State private var status: String = ""
    @State private var storageSizeBytes: Int?
    @State private var cacheSizeBytes: Int64?

    var body: some View {
        Form {
            Section("HuggingFace Access Token") {
                Text("Only needed for models marked gated or private on HuggingFace. Everything else downloads without it.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Stored securely in your macOS Keychain — never shared.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                SecureField("Paste your HuggingFace token", text: $tokenInput)
                    .textFieldStyle(.roundedBorder)

                HStack {
                    Button("Save") { save() }
                        .disabled(!HFTokenValidator.isPlausible(tokenInput))
                    Button("Clear", role: .destructive) { clear() }
                        .disabled(!hasToken)
                    Spacer()
                    if !status.isEmpty {
                        Text(status)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Label(hasToken ? "A token is saved." : "No token saved.",
                      systemImage: hasToken ? "checkmark.seal.fill" : "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(hasToken ? Color.green : Color.secondary)

                Link("Get a token on huggingface.co",
                     destination: URL(string: "https://huggingface.co/settings/tokens")!)
                    .font(.caption)
            }

            Section("Storage") {
                Text(ModelStore.shared.modelsDirectory.path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                HStack {
                    if let storageSizeBytes {
                        Text(Self.byteFormatter.string(fromByteCount: Int64(storageSizeBytes)))
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    Spacer()
                    Button("Reveal in Finder") { revealModelsInFinder() }
                }
            }

            Section("Flow Cache") {
                HStack {
                    if let cacheSizeBytes {
                        Text(Self.byteFormatter.string(fromByteCount: cacheSizeBytes))
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    Spacer()
                    Button("Clear Cache") { clearFlowCache() }
                        .disabled(cacheSizeBytes == 0)
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 520)
        .onAppear { hasToken = KeychainHelper.getToken() != nil }
        .task { await loadStorageSize() }
        .task { await loadCacheSize() }
    }

    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter
    }()

    // MARK: - HuggingFace token

    private func save() {
        KeychainHelper.saveToken(HFTokenValidator.normalized(tokenInput))
        hasToken = true
        tokenInput = ""
        status = "Saved."
    }

    private func clear() {
        KeychainHelper.deleteToken()
        hasToken = false
        tokenInput = ""
        status = "Cleared."
    }

    // MARK: - Storage

    /// Off the main actor, once per appearance — a `models/` tree with many files shouldn't
    /// stall the window opening.
    private func loadStorageSize() async {
        let bytes = await Task.detached(priority: .utility) {
            ModelStore.directorySize(at: ModelStore.shared.modelsDirectory)
        }.value
        storageSizeBytes = bytes
    }

    private func revealModelsInFinder() {
        let dir = ModelStore.shared.modelsDirectory
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        NSWorkspace.shared.activateFileViewerSelecting([dir])
    }

    // MARK: - Flow cache (OG-6 — app-wide only; per-flow stays in FlowMaintenanceMenu)

    private func loadCacheSize() async {
        let bytes = await Task.detached(priority: .utility) {
            FlowCacheStore.shared.totalBytes
        }.value
        cacheSizeBytes = bytes
    }

    private func clearFlowCache() {
        try? FlowCacheStore.shared.clear()
        cacheSizeBytes = 0
    }
}
