import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// `ModelUI` for OpenAI Privacy Filter. Presents the **Redact** screen (owner decision 6): paste text or open a
/// `.txt`/`.md` file (or drop one), Redact with a working Stop, the original with every detected span highlighted next
/// to the redacted text, eight category checkboxes that re-render without re-running the model, Copy redacted,
/// Save .txt and Export spans .json. No chat history, no prompt, no attachments.
struct PrivacyFilterUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        PrivacyFilterSDK().claim(model)
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        let footprint = Int64(model.ramGB * 1_073_741_824)
        return AnyView(PrivacyFilterRunView(
            modelDisplayName: model.displayName,
            license: model.license,
            licenseURL: model.licenseUrl,
            modelID: model.id,
            handedStage: stage,
            model: RedactRunModel(
                modelID: model.id,
                displayName: model.displayName,
                modelDirectory: PrivacyFilterEngine.modelDirectory(for: model.id),
                footprintBytes: footprint)))
    }
}

/// The Redact screen. The view binds to ``RedactRunModel`` and does nothing else; everything the model can decide
/// (what is refused, what the checkboxes change, what is exported) lives there and is unit-tested without weights.
struct PrivacyFilterRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let licenseURL: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run goes through it, so the shared-budget load
    /// and the "may I push out a served model?" question are the sheet's, not this view's.
    let handedStage: any PipelineStage

    @State private var model: RedactRunModel
    @State private var showImporter = false
    @State private var isDropTargeted = false

    /// The app's path. The model is a render-test seam (`TranslateGemmaRunView`'s pattern).
    @MainActor init(modelDisplayName: String, license: String?, licenseURL: String?, modelID: String,
                    handedStage: any PipelineStage, model: RedactRunModel) {
        self.modelDisplayName = modelDisplayName
        self.license = license
        self.licenseURL = licenseURL
        self.modelID = modelID
        self.handedStage = handedStage
        _model = State(initialValue: model)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            inputRow
            if model.isRunning { runningRow }
            categoryRow
            if let error = model.errorText { RunErrorBar(message: error) }
            if model.hasResult { resultView } else { placeholderView }
            Spacer(minLength: 0)
            caption
            footer
        }
        .padding(18)
        .frame(width: 980, height: 760)
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: [.plainText, .init(filenameExtension: "md") ?? .plainText],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result, let url = urls.first { model.open(url) }
        }
        .dropDestination(for: URL.self) { urls, _ in
            model.drop(urls)
            return true
        } isTargeted: { isDropTargeted = $0 }
        .overlay {
            if isDropTargeted {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .padding(6)
                    .allowsHitTesting(false)
            }
        }
        .onDisappear { model.teardown() }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: ModelType.redact.sfSymbol).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Find personal data in text and replace each span with a placeholder.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Text").font(.callout.weight(.medium))
                Button("Open…") { showImporter = true }.disabled(model.isRunning)
                if let source = model.sourceName {
                    Text(source).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
                Spacer()
                if let tokens = model.tokenCount {
                    Text("\(tokens) tokens").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Button("Clear") { model.clear() }.disabled(model.isRunning)
            }
            TextEditor(text: $model.inputText)
                .font(.body)
                .frame(minHeight: 110)
                .overlay {
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(.secondary.opacity(0.35))
                        .allowsHitTesting(false)
                }
                .disabled(model.isRunning)
            Text("Paste text, or open a .txt / .md file (or drop one here). Up to 128,000 tokens.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text("Redacting…").font(.callout).foregroundStyle(.secondary)
            Spacer()
            Button("Stop") { model.stop() }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var categoryRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Replace").font(.callout.weight(.medium))
                Spacer()
                Text("\(model.selectedCategoryCount) of 8")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            // Two rows of four, all on by default (decision 6). Toggling one re-renders from the cached spans; it never
            // re-runs the model.
            let columns = [GridItem(.adaptive(minimum: 180), alignment: .leading)]
            LazyVGrid(columns: columns, alignment: .leading, spacing: 4) {
                ForEach(PrivacyFilterCategory.allCases, id: \.self) { category in
                    Toggle(isOn: Binding(
                        get: { model.selectedCategories.contains(category) },
                        set: { isOn in
                            if isOn { model.selectedCategories.insert(category) }
                            else { model.selectedCategories.remove(category) }
                            model.categoriesChanged()
                        })) {
                        Label(category.displayName, systemImage: "circle.fill")
                            .labelStyle(.titleAndIcon)
                            .foregroundStyle(category.highlightColor)
                    }
                    .toggleStyle(.checkbox)
                }
            }
        }
    }

    private var resultView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Original").font(.callout.weight(.medium))
                Spacer()
                Text("Redacted").font(.callout.weight(.medium))
                Spacer()
                Button("Copy redacted") { model.copyRedacted() }.disabled(!model.canCopy)
                Button("Save .txt…") { model.saveRedacted() }.disabled(!model.canCopy)
                Button("Export spans .json…") { model.exportSpans() }.disabled(!model.canExportSpans)
            }
            .labelsHidden()
            HStack(alignment: .top, spacing: 12) {
                scrollPane {
                    Text(PrivacyFilterHighlight.attributed(model.resultSourceText, spans: model.spans))
                        .font(.body).textSelection(.enabled)
                }
                scrollPane {
                    Text(model.redactedText ?? "")
                        .font(.body).textSelection(.enabled)
                }
            }
        }
    }

    private var placeholderView: some View {
        HStack(spacing: 8) {
            Image(systemName: "text.magnifyingglass").foregroundStyle(.secondary)
            Text(model.didStop
                 ? "Stopped. The redacted text appears here when a run finishes."
                 : "The original and the redacted text appear here, side by side.")
                .font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity, minHeight: 160, alignment: .leading)
    }

    private func scrollPane<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        ScrollView {
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
        }
        .frame(minHeight: 220)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 6))
    }

    private var caption: some View {
        Text(RedactRunModel.caption)
            .font(.caption).foregroundStyle(.secondary)
    }

    private var footer: some View {
        HStack {
            if let license {
                Text("License: \(DataNormalizer.normalizeLicense(license))")
                    .font(.caption).foregroundStyle(.secondary)
                if let licenseURL, let url = URL(string: licenseURL) {
                    Link("Read the license", destination: url).font(.caption)
                }
            }
            Spacer()
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Button(model.isRunning ? "Redacting…" : "Redact") {
                model.start(handedStage: handedStage)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(!model.canRedact)
        }
    }
}

/// PF-7 — highlight colours and UI names for the eight categories. Implementer's call, pending owner confirmation:
/// distinguishable hues rather than one accent, checked in light and dark appearances; a span is tinted (background)
/// so the text stays readable.
nonisolated extension PrivacyFilterCategory {
    var displayName: String {
        switch self {
        case .person: return "People"
        case .email: return "Email"
        case .phone: return "Phone"
        case .url: return "URL"
        case .address: return "Address"
        case .date: return "Date"
        case .account: return "Account number"
        case .secret: return "Secret"
        }
    }

    var highlightColor: Color {
        switch self {
        case .person: return .blue
        case .email: return .green
        case .phone: return .orange
        case .url: return .purple
        case .address: return .brown
        case .date: return .teal
        case .account: return .pink
        case .secret: return .red
        }
    }
}

/// Builds the highlighted original: every detected span tinted by its category, whether or not its box is checked
/// (the boxes change only which spans are *replaced* — decision 6). Pure, so the UI shows exactly what the model had.
nonisolated enum PrivacyFilterHighlight {
    static func attributed(_ text: String, spans: [PrivacyFilterSpan]) -> AttributedString {
        var attributed = AttributedString(text)
        let characters = attributed.characters
        let count = characters.count
        for span in spans {
            let lower = min(max(0, span.characterRange.lowerBound), count)
            let upper = min(max(lower, span.characterRange.upperBound), count)
            guard upper > lower else { continue }
            let start = characters.index(characters.startIndex, offsetBy: lower)
            let end = characters.index(characters.startIndex, offsetBy: upper)
            attributed[start..<end].backgroundColor = span.category.highlightColor.opacity(0.26)
        }
        return attributed
    }
}
