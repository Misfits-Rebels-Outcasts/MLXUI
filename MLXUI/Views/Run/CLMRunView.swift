import SwiftUI

/// Run surface for CLM's decision encoder. **A copy of `Views/Run/LayaRunView.swift`** (278
/// lines at commit `22e0314`, per `RSI/DelegateCLMBacklog.md` CL-0 gate G, "overridden: a
/// duplicate Run screen" — the owner's own words: "G. Make it a duplicate Run screen.") — a
/// later fix to one should be carried to the other by diffing against that commit, not by
/// re-deriving. `LayaRunView.swift` itself is not touched by this file.
///
/// Differences from the copy: drives `CLMEngineCache`/`CLMEngine` instead of
/// `LayaEngineCache`/`LayaEngine`; the truncation note reads "Input was truncated to 2048
/// tokens" (`CLMEncoder.maxTokens`, not Laya's 512); the first Decide after launch shows
/// "Loading CLM 8B (about 8 GB)…" instead of a bare spinner (`CLMRunModel.hasLoadedOnce`); the
/// placeholder reads "What should CLM decide?"; the option-row type is `CLMRunOption`, private
/// to this file. `CLMEngine.answer` takes `CLMJSON` state/instructions/criteria directly (no
/// `LayaQuestionDefinition.resolve()` step — `CLMSchema.candidates` does its own validation).
struct CLMRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String

    @State private var model = CLMRunModel()
    @State private var stateText = "I was billed twice. Please refund the duplicate."
    @State private var questionType: LayaQuestionType = .choice
    @State private var instructions = "Who should handle this?"
    @State private var choiceOptions: [CLMRunOption] = [
        CLMRunOption(label: "billing"), CLMRunOption(label: "technical"), CLMRunOption(label: "sales"),
    ]
    @State private var scoreOptions: [CLMRunOption] = [
        CLMRunOption(label: "not urgent"), CLMRunOption(label: "soon"), CLMRunOption(label: "critical"),
    ]
    /// A validation error caught before the engine is ever called (CL-5-FIX-1) — distinct
    /// from `model.errorText`, which is the engine's own runtime errors.
    @State private var inlineError: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                Divider()
                inputs
                Text(license ?? "License: see model page")
                    .font(.caption).foregroundStyle(.secondary)
                if let inlineError { errorBar(inlineError) }
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
                TextField("What should CLM decide?", text: $instructions)
                    .textFieldStyle(.roundedBorder)
            }
            switch questionType {
            case .choice: optionsList(title: "Options", options: $choiceOptions, placeholder: "label (e.g. billing)")
            case .score: optionsList(title: "Levels, low to high", options: $scoreOptions, placeholder: "level description")
            case .noul: EmptyView()
            }
        }
    }

    private func optionsList(title: String, options: Binding<[CLMRunOption]>, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.subheadline.bold())
                Spacer()
                Button {
                    options.wrappedValue.append(CLMRunOption())
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
            Text(model.hasLoadedOnce ? "Deciding…" : "Loading CLM 8B (about 8 GB)…")
                .font(.callout).foregroundStyle(.secondary)
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
                Label("Input was truncated to 2048 tokens", systemImage: "scissors")
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

        inlineError = nil
        let criteria: CLMJSON?
        do {
            criteria = try clmRunCriteria(
                type: questionType,
                choiceLabels: choiceOptions.map(\.label),
                scoreLevels: scoreOptions.map(\.label))
        } catch {
            inlineError = "Each option needs a different name"
            return
        }

        model.run(
            modelID: modelID, state: text, type: questionType,
            instructions: .string(trimmedInstructions), criteria: criteria)
    }
}

/// One editable option row (a choice label or a score level's description).
private struct CLMRunOption: Identifiable {
    let id = UUID()
    var label = ""
}

/// CL-5-FIX-1: screen state → `CLMJSON` criteria, pure and UI-independent (so it's directly
/// testable without driving `CLMRunView` itself). Mirrors `CLMSchema.candidates`' shape
/// requirements exactly: **choice** needs an **object** (`dict.fromkeys`-shaped — a key per
/// option, "" description, so `CLMSchema.candidates`'s `isEmptyValue` fallback uses the key
/// itself as the candidate text — the comment this replaced already said this, the code
/// didn't); **score** stays an **array** (`schema.py` requires an ordered list there, not a
/// dict). Blank option rows are dropped before building. Duplicate choice labels are rejected
/// here, before the engine ever sees them — a Python `dict.fromkeys`/object literal would
/// silently collapse them onto one key, changing option count and answer shape underneath
/// the caller.
enum CLMRunQuestionError: Error, Equatable {
    case duplicateChoiceLabels
}

nonisolated func clmRunCriteria(
    type: LayaQuestionType, choiceLabels: [String], scoreLevels: [String]
) throws -> CLMJSON? {
    switch type {
    case .choice:
        let labels = choiceLabels
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard Set(labels).count == labels.count else {
            throw CLMRunQuestionError.duplicateChoiceLabels
        }
        return .object(labels.map { CLMJSONField(key: $0, value: .string("")) })
    case .score:
        let levels = scoreLevels.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        return .array(levels.map { .string($0.isEmpty ? "—" : $0) })
    case .noul:
        return nil
    }
}

/// View-model driving one CLM decision: runs off-actor through the shared `CLMEngineCache`,
/// and folds the answer + latency back onto the main actor. Mirrors `LayaRunModel`.
@MainActor
@Observable
final class CLMRunModel {
    var isRunning = false
    var answer: LayaAnswer?
    var errorText: String?
    var latencyMS: Double?
    /// Flips to `true` after the first Decide completes (success or failure) — drives
    /// `CLMRunView.runningRow`'s "Loading CLM 8B (about 8 GB)…" vs. plain "Deciding…" text.
    var hasLoadedOnce = false

    func run(modelID: String, state: String, type: LayaQuestionType, instructions: CLMJSON, criteria: CLMJSON?) {
        isRunning = true
        answer = nil
        errorText = nil
        latencyMS = nil

        let directory = ModelStore.shared.directory(forModelID: modelID)
        // `.detached`, not a plain `Task { }`: see `LayaRunModel.run`'s own comment — keeps
        // every bit of MLX work off the main thread; only the final state writes hop back on.
        Task.detached(priority: .userInitiated) {
            let start = Date()
            do {
                let engine = try await CLMEngineCache.shared.engine(for: directory)
                let results = try await engine.answer(
                    state: .string(state),
                    questions: [(id: "run", type: type, instructions: instructions, criteria: criteria)])
                let elapsedMS = Date().timeIntervalSince(start) * 1000
                await MainActor.run {
                    self.latencyMS = elapsedMS
                    self.answer = results.first?.answer
                    self.isRunning = false
                    self.hasLoadedOnce = true
                }
            } catch {
                await MainActor.run {
                    self.errorText = "\(error)"
                    self.isRunning = false
                    self.hasLoadedOnce = true
                }
            }
        }
    }
}
