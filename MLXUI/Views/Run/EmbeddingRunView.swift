import SwiftUI

/// Run surface for MLX text embedding. Enter one or two texts → Embed → see the vector
/// dimensionality and (for two texts) their cosine similarity. Vectors are L2-normalized, so
/// cosine similarity is the dot product. Calls `EmbeddingEngine` directly (one model load for
/// both texts), mirroring how `ASRRunView` reads its input directly. `plan-nonchat-aiui.md` EM2.
struct EmbeddingRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    let family: String

    @State private var model = EmbeddingRunModel()
    @State private var textA = "A cat sits on the mat."
    @State private var textB = "A feline rests on the rug."
    /// Prepend the model's query/document task prefixes (asymmetric models only).
    @State private var asQueryDocument = true

    /// The `(query, document)` prefixes for this model, or nil if it's symmetric.
    private var prefixes: (query: String, document: String)? {
        EmbeddingPrefixes.queryDocument(family: family)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            inputs
            Text(license ?? "License: see model page")
                .font(.caption).foregroundStyle(.secondary)
            if model.isRunning { runningRow }
            if model.didEmbed { resultView }
            if let error = model.errorText { errorBar(error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 560, height: 560)
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "point.3.connected.trianglepath.dotted")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Text → embedding vector").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Text A").font(.subheadline.bold())
                TextEditor(text: $textA)
                    .font(.body).frame(minHeight: 60)
                    .padding(6)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Text B").font(.subheadline.bold())
                    + Text("  (optional — for cosine similarity)").font(.caption).foregroundColor(.secondary)
                TextEditor(text: $textB)
                    .font(.body).frame(minHeight: 60)
                    .padding(6)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
            // Asymmetric models (nomic) want query/document task prefixes; hidden otherwise.
            if let prefixes {
                Toggle(isOn: $asQueryDocument) {
                    Text("Retrieval mode — prefix A as query, B as document")
                        + Text("  (\(prefixes.query.trimmingCharacters(in: .whitespaces)) / \(prefixes.document.trimmingCharacters(in: .whitespaces)))")
                        .font(.caption).foregroundColor(.secondary)
                }
                .toggleStyle(.checkbox)
                .font(.callout)
            }
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            Text("Embedding…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultView: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Embedded", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green).font(.subheadline.bold())
            if let dims = model.dimensions {
                Text("Dimensions: \(dims)").font(.callout.monospacedDigit())
            }
            if let similarity = model.similarity {
                Text("Cosine similarity: " + String(format: "%.4f", similarity))
                    .font(.callout.monospacedDigit())
                Text(similarityHint(similarity)).font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func similarityHint(_ s: Double) -> String {
        switch s {
        case 0.8...:      return "Very similar"
        case 0.5..<0.8:   return "Related"
        case 0.2..<0.5:   return "Loosely related"
        default:          return "Unrelated"
        }
    }

    private func errorBar(_ message: String) -> some View {
        RunErrorBar(message: message)
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Embedding…" : "Embed") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(textA.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.isRunning)
        }
    }

    private func run() {
        var a = textA.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !a.isEmpty else { return }
        var b = textB.trimmingCharacters(in: .whitespacesAndNewlines)
        // Apply the model's query/document task prefixes when retrieval mode is on.
        if asQueryDocument, let prefixes {
            a = prefixes.query + a
            if !b.isEmpty { b = prefixes.document + b }
        }
        model.embed(textA: a, textB: b.isEmpty ? nil : b, modelID: modelID)
    }
}

/// View-model driving one embedding run: embeds one or two texts off-actor and folds the
/// dimensionality + cosine similarity back onto the main actor. Mirrors `ASRRunModel`.
@MainActor
@Observable
final class EmbeddingRunModel {
    var isRunning = false
    var didEmbed = false
    var dimensions: Int?
    var similarity: Double?
    var errorText: String?

    func embed(textA: String, textB: String?, modelID: String) {
        isRunning = true
        didEmbed = false
        dimensions = nil
        similarity = nil
        errorText = nil

        let directory = EmbeddingEngine.installedModelDirectory(id: modelID)
        let texts = textB.map { [textA, $0] } ?? [textA]
        Task {
            do {
                let vectors = try await EmbeddingEngine.embed(texts, modelDirectory: directory)
                self.dimensions = vectors.first?.count
                if vectors.count == 2 {
                    // Vectors are L2-normalized, so cosine similarity == dot product.
                    let dot = zip(vectors[0], vectors[1]).map(*).reduce(0, +)
                    self.similarity = Double(dot)
                }
                self.didEmbed = true
                self.isRunning = false
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }
}
