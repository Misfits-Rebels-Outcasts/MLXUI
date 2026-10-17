import SwiftUI

/// Run surface for Clef Flash's decision runner. **A copy of `Views/Run/CLMRunView.swift`** (341
/// lines at commit `456cc67`, per `RSI/DelegateClefBacklog.md` CF-6 under gate **E1** — "a separate
/// `ClefRunView` copied from `CLMRunView`") — a later fix to one should be carried to the other by
/// diffing against that commit, not by re-deriving. `CLMRunView.swift` and `LayaRunView.swift` are not
/// touched by this file.
///
/// Differences from the copy:
/// - drives `ClefEngineCache`/`ClefEngine` instead of `CLMEngineCache`/`CLMEngine`;
/// - the truncation note reads **"Input was truncated to 8,192 tokens"** — gate **F2**'s cap
///   (`ClefPrompt.defaultMaxLength`), not F1's 16,384;
/// - the first Decide after launch shows **"Loading Clef Flash (about 6 GB)…"** instead of a bare
///   spinner (`ClefRunModel.hasLoadedOnce`);
/// - the instructions placeholder reads **"What should Clef decide?"**;
/// - the option-row type is `ClefRunOption`, private to this file.
///
/// **Choice criteria go as a JSON object `{label: ""}`** — the CL-5-FIX-1 lesson, carried over
/// deliberately: `ClefPrompt.questionOptions` sorts a choice's criteria by id and keeps an empty
/// description, and a bare array would not encode the option ids the answer comes back keyed by.
/// `ClefEngine.answer` takes `ClefJSON` state/instructions/criteria directly.
struct ClefRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String

    @Environment(\.servedLoadConfirmer) private var confirmer      // SC-5a (gate D)
    @State private var model = ClefRunModel()
    @State private var stateText = "I was billed twice. Please refund the duplicate."
    @State private var questionType: ClefQuestionType = .choice
    @State private var instructions = "Who should handle this?"
    @State private var choiceOptions: [ClefRunOption] = [
        ClefRunOption(label: "billing"), ClefRunOption(label: "technical"), ClefRunOption(label: "sales"),
    ]
    @State private var scoreOptions: [ClefRunOption] = [
        ClefRunOption(label: "not urgent"), ClefRunOption(label: "soon"), ClefRunOption(label: "critical"),
    ]
    /// A validation error caught before the engine is ever called (CL-5-FIX-1's shape, kept) —
    /// distinct from `model.errorText`, which is the engine's own runtime errors.
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
                Text("Choice").tag(ClefQuestionType.choice)
                Text("Score").tag(ClefQuestionType.score)
                Text("Yes / No").tag(ClefQuestionType.noul)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            VStack(alignment: .leading, spacing: 4) {
                Text("Instructions").font(.subheadline.bold())
                TextField("What should Clef decide?", text: $instructions)
                    .textFieldStyle(.roundedBorder)
            }
            switch questionType {
            case .choice: optionsList(title: "Options", options: $choiceOptions, placeholder: "label (e.g. billing)")
            case .score: optionsList(title: "Levels, low to high", options: $scoreOptions, placeholder: "level description")
            case .noul: EmptyView()
            }
        }
    }

    private func optionsList(title: String, options: Binding<[ClefRunOption]>, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.subheadline.bold())
                Spacer()
                Button {
                    options.wrappedValue.append(ClefRunOption())
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
            Text(model.hasLoadedOnce ? "Deciding…" : "Loading Clef Flash (about 6 GB)…")
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
                // Gate F2: the request budget is 8,192 tokens (`ClefPrompt.defaultMaxLength`).
                Label("Input was truncated to 8,192 tokens", systemImage: "scissors")
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
        let criteria: ClefJSON?
        do {
            criteria = try clefRunCriteria(
                type: questionType,
                choiceLabels: choiceOptions.map(\.label),
                scoreLevels: scoreOptions.map(\.label))
        } catch {
            inlineError = "Each option needs a different name"
            return
        }

        model.run(
            modelID: modelID, state: text, type: questionType,
            instructions: .string(trimmedInstructions), criteria: criteria,
            confirmer: confirmer, displayName: modelDisplayName)
    }
}

/// One editable option row (a choice label or a score level's description).
private struct ClefRunOption: Identifiable {
    let id = UUID()
    var label = ""
}

enum ClefRunQuestionError: Error, Equatable {
    case duplicateChoiceLabels
}

/// Screen state → `ClefJSON` criteria, pure and UI-independent (so it is directly testable). The
/// **choice** shape is an **object** `{label: ""}`: `ClefPrompt.questionOptions` sorts a choice's
/// criteria by id, keeps the option ids as keys, and keeps an empty description as a description
/// (`""` is not `null`), so a bare array would lose the ids the answer is keyed by — the CL-5-FIX-1
/// bug, carried over deliberately. **Score** stays an ordered **array** (levels low → high).
/// Blank option rows are dropped; duplicate choice labels are rejected here, before the engine ever
/// sees them — an object literal would silently collapse them onto one key.
nonisolated func clefRunCriteria(
    type: ClefQuestionType, choiceLabels: [String], scoreLevels: [String]
) throws -> ClefJSON? {
    switch type {
    case .choice:
        let labels = choiceLabels
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard Set(labels).count == labels.count else {
            throw ClefRunQuestionError.duplicateChoiceLabels
        }
        return .object(labels.map { ClefJSONField(key: $0, value: .string("")) })
    case .score:
        let levels = scoreLevels.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        return .array(levels.map { .string($0.isEmpty ? "—" : $0) })
    case .noul:
        return nil
    }
}

/// View-model driving one Clef decision: runs off-actor through the shared `ClefEngineCache`, and
/// folds the answer + latency back onto the main actor. Mirrors `CLMRunModel`.
@MainActor
@Observable
final class ClefRunModel {
    var isRunning = false
    var answer: LayaAnswer?
    var errorText: String?
    var latencyMS: Double?
    /// Flips to `true` after the first Decide completes (success or failure) — drives
    /// `ClefRunView.runningRow`'s "Loading Clef Flash (about 6 GB)…" vs. plain "Deciding…" text.
    var hasLoadedOnce = false

    func run(modelID: String, state: String, type: ClefQuestionType, instructions: ClefJSON, criteria: ClefJSON?,
             confirmer: ServedLoadConfirmer? = nil, displayName: String = "this model") {
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
                // SC-5a (gate D): ask before pushing out a served model. Cancel: the pass doesn't start, no error.
                if let confirmer {
                    let needed = await ClefEngineCache.shared.bytesNeeded(directory: directory)
                    guard await confirmer.confirm(modelName: displayName, loadBytes: needed, requester: .runWindow) else {
                        await MainActor.run { self.isRunning = false }
                        return
                    }
                }
                // MLX-ERR-2: inside the detached task (see `LayaRunModel.run`'s note) —
                // mlx-swift's handler is `@TaskLocal`, so the guard must not sit outside it.
                let results = try await MLXErrorGuard.guardedRun {
                    // SC-2b: leased for the pass (shared budget, real size).
                    let lease = try await ClefEngineCache.shared.lease(for: directory, holder: .runWindow)
                    defer { lease.release() }
                    return try await lease.value.answer(
                        state: .string(state),
                        questions: [(id: "run", type: type, instructions: instructions, criteria: criteria)])
                }
                let elapsedMS = Date().timeIntervalSince(start) * 1000
                await MainActor.run {
                    self.latencyMS = elapsedMS
                    self.answer = results.first?.answer
                    self.isRunning = false
                    self.hasLoadedOnce = true
                }
            } catch {
                await MainActor.run {
                    self.errorText = MLXErrorGuard.sentence(for: error)
                    self.isRunning = false
                    self.hasLoadedOnce = true
                }
            }
        }
    }
}
