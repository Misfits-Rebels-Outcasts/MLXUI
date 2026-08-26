import SwiftUI

/// CFM-R10-Human — the parked human-row prompt (answer `g3`): friendly wording
/// ("Ask me before sending (waits for you)" / "Wait for me to type (waits for you)"), the
/// row's own criterion, the waiting policy as one plain sentence (never `timeout=4h`), and
/// the answer controls — tag buttons for `Ask Human`, a text field for `Human Input`. Answering
/// re-runs the flow with the answer; the interpreter resolves the parked row and continues.
struct FlowHumanPromptView: View {
    let parked: FlowRunSession.ParkedInfo
    let row: Row
    let doc: FlowDocument
    let session: FlowRunSession
    let runner: FlowRunner
    let context: FlowRunner.RunContext

    @State private var typedText = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(friendlyTitle, systemImage: "person.circle")
                .font(.headline)
            Text(parked.prompt)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let policy = policySentence {
                Text(policy)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            // A `timeout=` row shows a live countdown; the run falls back to the row's
            // default when it reaches zero unanswered.
            if parked.deadline != nil {
                countdown
            }
            if row.task == "Ask Human" {
                HStack {
                    ForEach(tags, id: \.self) { tag in
                        Button(tag) {
                            session.answer(tag: tag, for: parked, doc: doc,
                                           runner: runner, context: context)
                        }
                        .buttonStyle(.bordered)
                    }
                }
            } else {
                TextField("Your reply…", text: $typedText)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Spacer()
                    Button("Send") {
                        session.answer(text: typedText, for: parked, doc: doc,
                                       runner: runner, context: context)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(typedText.isEmpty)
                    .keyboardShortcut(.defaultAction)
                }
            }
            HStack {
                Spacer()
                Button("Stop the run") {
                    session.cancel()
                }
                .keyboardShortcut(.cancelAction)
            }
        }
        .padding(20)
        .frame(width: 460)
        // If the user never answers, wait out the deadline and fall back to the row's
        // default — answering dismisses the sheet and cancels this task first.
        .task {
            await autoFallbackIfNeeded()
        }
    }

    /// The live countdown to a `timeout=` row's fallback deadline.
    private var countdown: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let remaining = remainingSeconds(now: context.date)
            let urgent = remaining > 0 && remaining <= 30
            let style: Color = remaining <= 0 ? .orange : (urgent ? .orange : .secondary)
            HStack(spacing: 6) {
                Image(systemName: "timer")
                Text(remainingText(now: context.date))
                    .font(.callout.monospacedDigit().weight(.semibold))
            }
            .foregroundStyle(style)
        }
    }

    private func remainingSeconds(now: Date) -> TimeInterval {
        guard let deadline = parked.deadline else { return 0 }
        return deadline.timeIntervalSince(now)
    }

    private func remainingText(now: Date) -> String {
        let remaining = remainingSeconds(now: now)
        if remaining <= 0 { return "Time's up — proceeding with the default…" }
        let total = Int(remaining.rounded())
        return "Answer in \(total / 60):\(String(format: "%02d", total % 60))"
    }

    /// Wait out the deadline; if the run is still parked unanswered, re-run with the row's
    /// fallback (the sheet's dismissal cancels this task the moment an answer lands).
    private func autoFallbackIfNeeded() async {
        guard let deadline = parked.deadline else { return }
        while Date() < deadline {
            try? await Task.sleep(for: .milliseconds(250))
            if Task.isCancelled { return }
        }
        guard !Task.isCancelled else { return }
        session.fallbackTimeout(for: parked, doc: doc, runner: runner, context: context)
    }

    /// The g3 friendly title with the "(waits for you)" suffix.
    private var friendlyTitle: String {
        row.task == "Ask Human" ? "Ask me before sending (waits for you)"
                                : "Wait for me to type (waits for you)"
    }

    /// The `Ask Human` row's answerable tags — declared `tags:`, else the decide clause edges.
    private var tags: [String] {
        if let t = row.tags, !t.isEmpty { return t }
        if let edges = row.clause?.edges { return edges.map(\.tag) }
        return []
    }

    /// The waiting policy as one plain sentence, never `timeout=4h` (answer `g3`).
    private var policySentence: String? {
        let s = FlowSettings(row.settings)
        guard let timeout = s.value(for: "timeout") else { return nil }
        let fallback: String
        if let dflt = s.value(for: "default") {
            fallback = "keep my \(dflt)"
        } else {
            fallback = row.task == "Ask Human" ? "keep my edit" : "leave it unchanged"
        }
        return "If I don't answer in \(timeout), \(fallback)."
    }
}
