import SwiftUI

/// Run surface for Laya's decision encoder. Enter text + a typed question (choice/score/
/// yes-no) → Decide → see the winner, a probability bar per option, the confidence and the
/// latency. Calls `LayaEngine` directly (via the shared `LayaEngineCache`, `Modules/Laya/
/// LayaSDK.swift`), mirroring how `EmbeddingRunView`/`ImageQARunView` read their input
/// directly rather than going through the generic `LayaStage` JSON boundary — the same reason
/// those two don't route through their own stage either. `RSI/DelegateLayaBacklog.md` LY-5.
struct LayaRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String

    @State private var model = LayaRunModel()
    @State private var stateText = "I was billed twice. Please refund the duplicate."
    @State private var questionType: LayaQuestionType = .choice
    @State private var instructions = "Who should handle this?"
    @State private var choiceOptions: [LayaRunOption] = [
        LayaRunOption(label: "billing"), LayaRunOption(label: "technical"), LayaRunOption(label: "sales"),
    ]
    @State private var scoreOptions: [LayaRunOption] = [
        LayaRunOption(label: "not urgent"), LayaRunOption(label: "soon"), LayaRunOption(label: "critical"),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                Divider()
                inputs
                Text(license ?? "License: see model page")
                    .font(.caption).foregroundStyle(.secondary)
                if model.isRunning { runningRow }
                if let answer = model.answer { resultView(answer) }
                if let error = model.errorText { errorBar(error) }
            }
            .padding(18)
        }
        .frame(width: 560, height: 680)
        .safeAreaInset(edge: .bottom) { footer.padding(18).background(.regularMaterial) }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.triangle.branch")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Text + typed question → label, probabilities, confidence")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Text").font(.subheadline.bold())
                TextEditor(text: $stateText)
                    .font(.body).frame(minHeight: 70)
                    .padding(6)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
            Picker("Question type", selection: $questionType) {
                Text("Choice").tag(LayaQuestionType.choice)
                Text("Score").tag(LayaQuestionType.score)
                Text("Yes / No").tag(LayaQuestionType.noul)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            VStack(alignment: .leading, spacing: 4) {
                Text("Instructions").font(.subheadline.bold())
                TextField("What should Laya decide?", text: $instructions)
                    .textFieldStyle(.roundedBorder)
            }
            switch questionType {
            case .choice: optionsList(title: "Options", options: $choiceOptions, placeholder: "label (e.g. billing)")
            case .score: optionsList(title: "Levels, low to high", options: $scoreOptions, placeholder: "level description")
            case .noul: EmptyView()
            }
        }
    }

    private func optionsList(title: String, options: Binding<[LayaRunOption]>, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.subheadline.bold())
                Spacer()
                Button {
                    options.wrappedValue.append(LayaRunOption())
                } label: {
                    Image(systemName: "plus.circle")
                }
                .buttonStyle(.borderless)
            }
            ForEach(options.wrappedValue.indices, id: \.self) { index in
                HStack {
                    Text("\(index + 1).").foregroundStyle(.secondary).frame(width: 18, alignment: .trailing)
                    TextField(placeholder, text: options[index].label)
                        .textFieldStyle(.roundedBorder)
                    if options.wrappedValue.count > 2 {
                        Button {
                            options.wrappedValue.remove(at: index)
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
        }
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            Text("Deciding…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func resultView(_ answer: LayaAnswer) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(winnerText(answer), systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Spacer()
                if let latency = model.latencyMS {
                    Text(String(format: "%.0f ms", latency)).font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("Confidence: " + String(format: "%.1f%%", answer.confidence * 100))
                .font(.callout.monospacedDigit())
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(zip(answer.optionLabels, answer.probabilities)), id: \.0) { label, probability in
                    probabilityBar(label: label, probability: probability)
                }
            }
            if answer.stateTruncated {
                Label("Input was truncated to fit 512 tokens", systemImage: "scissors")
                    .font(.caption).foregroundStyle(.orange)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func probabilityBar(label: String, probability: Double) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption).lineLimit(1).truncationMode(.tail)
                Spacer()
                Text(String(format: "%.1f%%", probability * 100)).font(.caption.monospacedDigit())
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3).fill(.quaternary)
                    RoundedRectangle(cornerRadius: 3).fill(Color.accentColor)
                        .frame(width: geometry.size.width * max(0, min(1, probability)))
                }
            }
            .frame(height: 6)
        }
    }

    private func winnerText(_ answer: LayaAnswer) -> String {
        switch answer.type {
        case .choice:
            return answer.choiceLabel ?? "—"
        case .score:
            let level = answer.probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            let levelText = answer.scoreLevels.indices.contains(level) ? answer.scoreLevels[level] : "level \(level)"
            return "\(levelText) (expected \(String(format: "%.2f", answer.scoreValue ?? 0)))"
        case .noul:
            let isTrue = (answer.noulProbability ?? 0) >= 0.5
            return isTrue ? "Yes" : "No"
        }
    }

    private func errorBar(_ message: String) -> some View {
        RunErrorBar(message: message)
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Deciding…" : "Decide") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(stateText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || model.isRunning)
        }
    }

    private func run() {
        let text = stateText.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedInstructions = instructions.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !trimmedInstructions.isEmpty else { return }

        let criteria: LayaJSON?
        switch questionType {
        case .choice:
            // A plain array of labels — `LayaQuestionDefinition.resolve()`'s `dict.fromkeys`
            // path, giving every label "no description" (a bare label, not "label: null").
            let labels = choiceOptions.map { $0.label.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            criteria = .array(labels.map { .string($0) })
        case .score:
            let levels = scoreOptions.map { $0.label.trimmingCharacters(in: .whitespacesAndNewlines) }
            criteria = .array(levels.map { .string($0.isEmpty ? "—" : $0) })
        case .noul:
            criteria = nil
        }

        let definition = LayaQuestionDefinition(type: questionType, instructions: .string(trimmedInstructions), criteria: criteria)
        model.run(modelID: modelID, state: text, definition: definition)
    }
}

/// One editable option row (a choice label or a score level's description).
private struct LayaRunOption: Identifiable {
    let id = UUID()
    var label = ""
}

/// View-model driving one Laya decision: resolves the question, runs it off-actor through the
/// shared `LayaEngineCache`, and folds the answer + latency back onto the main actor. Mirrors
/// `EmbeddingRunModel`/`ImageQARunModel`.
@MainActor
@Observable
final class LayaRunModel {
    var isRunning = false
    var answer: LayaAnswer?
    var errorText: String?
    var latencyMS: Double?

    func run(modelID: String, state: String, definition: LayaQuestionDefinition) {
        isRunning = true
        answer = nil
        errorText = nil
        latencyMS = nil

        let directory = ModelStore.shared.directory(forModelID: modelID)
        Task {
            let start = Date()
            do {
                let question = try definition.resolve()
                let engine = try await LayaEngineCache.shared.engine(for: directory)
                let answers = try engine.predict(state: state, questions: [question])
                self.latencyMS = Date().timeIntervalSince(start) * 1000
                self.answer = answers.first
                self.isRunning = false
            } catch {
                self.errorText = "\(error)"
                self.isRunning = false
            }
        }
    }
}
