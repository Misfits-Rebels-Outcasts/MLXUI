import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// `ModelUI` for Sortformer speaker diarization. Presents `SortformerRunView` (pick or drop an audio file, set the threshold,
/// Run → a speaker timeline, a segment table, Copy as text / CSV). Same claim predicate as `SortformerSDK`.
struct SortformerUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .diarization,
              model.source == .mlx,
              model.family.lowercased().contains("sortformer") else {
            return .no
        }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(SortformerRunView(modelDisplayName: model.displayName,
                                  license: model.license,
                                  modelID: model.id,
                                  handedStage: stage))
    }
}

/// The run surface: choose or drop an audio file, set the threshold, Run, then read the timeline and copy the result.
struct SortformerRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run goes through it.
    let handedStage: any PipelineStage

    @State private var model = SortformerRunModel()
    @State private var audioURL: URL?
    @State private var showImporter = false
    @State private var isDropTargeted = false
    @State private var threshold = Double(SortformerEngine.defaultThreshold)

    private var defaultThreshold: Double { Double(SortformerEngine.defaultThreshold) }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            inputs
            if model.isRunning { runningRow }
            if let error = model.errorText { RunErrorBar(message: error) }
            if let result = model.result { resultView(result) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 640, height: 720)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio],
                      allowsMultipleSelection: false) { result in
            if case let .success(urls) = result { audioURL = urls.first }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            audioURL = url
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
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: ModelType.diarization.sfSymbol).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Who spoke when: audio → speaker timeline (up to 4 speakers)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("Choose Audio…") { showImporter = true }
                Text(audioURL?.lastPathComponent ?? "No file selected (or drop one here)")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            HStack {
                Text("Threshold").frame(width: 70, alignment: .leading)
                Slider(value: $threshold,
                       in: Double(SortformerEngine.thresholdRange.lowerBound)...Double(SortformerEngine.thresholdRange.upperBound),
                       step: 0.05)
                    .frame(maxWidth: 220)
                Text(String(format: "%.2f", threshold)).monospacedDigit().frame(width: 40, alignment: .trailing)
                Button("Reset") { threshold = defaultThreshold }
                    .buttonStyle(.borderless).disabled(threshold == defaultThreshold)
                Spacer()
            }
            Text("A frame counts as a speaker talking when the model's probability is above the threshold. Lower finds quieter or shorter turns and may add noise; higher is stricter.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text(Self.lengthNote).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text(license ?? "License: see model page").font(.caption).foregroundStyle(.secondary)
        }
    }

    /// What the length limit means for the person (the engine's two paths and its refusal; measured numbers are in its comment).
    static var lengthNote: String {
        let limit = Int(SortformerEngine.maxSeconds / 60)
        let single = Int(SortformerEngine.offlineMaxSeconds)
        return "Handles recordings up to \(limit) minutes. Up to \(single) seconds is analysed in one pass; a longer recording is "
            + "analysed in 5-second steps, which keeps memory use the same for any length but can place turn boundaries a "
            + "fraction of a second differently. Longer than \(limit) minutes is refused: split it into parts."
    }

    private var runningRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            Text("Finding speakers…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func resultView(_ result: SortformerRunModel.Result) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(summaryLine(result),
                      systemImage: result.segments.isEmpty ? "exclamationmark.circle" : "checkmark.circle.fill")
                    .foregroundStyle(result.segments.isEmpty ? Color.secondary : Color.green)
                    .font(.subheadline.bold())
                Spacer()
                Button("Copy as Text") { copy(SpeakerSegments.text(result.segments)) }.disabled(result.segments.isEmpty)
                Button("Copy as CSV") { copy(SpeakerSegments.csv(result.segments)) }.disabled(result.segments.isEmpty)
            }
            if !result.segments.isEmpty {
                SpeakerTimelineView(segments: result.segments, duration: result.duration)
                segmentTable(result.segments)
            }
        }
    }

    private func summaryLine(_ result: SortformerRunModel.Result) -> String {
        guard !result.segments.isEmpty else { return "No speech found" }
        let speakers = result.speakerCount
        let count = result.segments.count
        return "\(speakers) speaker\(speakers == 1 ? "" : "s"), \(count) segment\(count == 1 ? "" : "s")"
    }

    private func segmentTable(_ segments: [SpeakerSegment]) -> some View {
        Table(segments) {
            TableColumn("Speaker") { s in
                HStack(spacing: 6) {
                    Circle().fill(SpeakerTimelineView.color(for: s.speaker)).frame(width: 8, height: 8)
                    Text(s.label)
                }
            }
            TableColumn("Start") { s in Text(SpeakerSegments.clock(s.start)).monospacedDigit() }
            TableColumn("End") { s in Text(SpeakerSegments.clock(s.end)).monospacedDigit() }
            TableColumn("Length") { s in Text(SpeakerSegments.seconds(s.duration) + " s").monospacedDigit() }
        }
        .frame(minHeight: 160)
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Working…" : "Run") { run() }
                .keyboardShortcut(.defaultAction)
                .disabled(audioURL == nil || model.isRunning)
        }
    }

    // MARK: - Actions

    private func run() {
        guard let url = audioURL else { return }
        let modelID = self.modelID
        let threshold = Float(self.threshold)
        let stage = LeasingStage.perRun(handedStage, config: .default) { _ in
            SortformerStage(modelID: modelID, threshold: threshold)
        }
        model.start(audioURL: url, stage: stage)
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

extension SpeakerSegment: Identifiable {
    var id: String { "\(speaker)-\(start)-\(end)" }
}

/// One lane per speaker that spoke, a bar per segment, and a time axis underneath.
struct SpeakerTimelineView: View {
    let segments: [SpeakerSegment]
    let duration: Double

    private static let palette: [Color] = [.blue, .orange, .green, .purple]
    static func color(for speaker: Int) -> Color { palette[speaker % palette.count] }

    private let labelWidth: CGFloat = 78
    private let laneHeight: CGFloat = 22

    var body: some View {
        let lanes = SpeakerSegments.lanes(segments)
        let span = max(duration, segments.map(\.end).max() ?? 0, 0.001)
        VStack(alignment: .leading, spacing: 4) {
            ForEach(lanes, id: \.speaker) { lane in
                HStack(spacing: 6) {
                    Text("Speaker \(lane.speaker + 1)").font(.caption).frame(width: labelWidth, alignment: .leading)
                    GeometryReader { proxy in
                        ZStack(alignment: .topLeading) {
                            RoundedRectangle(cornerRadius: 4).fill(.quaternary.opacity(0.4))
                            ForEach(lane.segments) { s in
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Self.color(for: s.speaker))
                                    .frame(width: max(2, proxy.size.width * s.duration / span), height: laneHeight)
                                    .offset(x: proxy.size.width * s.start / span)
                                    .help("\(s.label): \(SpeakerSegments.clock(s.start)) – \(SpeakerSegments.clock(s.end))")
                            }
                        }
                    }
                    .frame(height: laneHeight)
                }
            }
            HStack(spacing: 6) {
                Color.clear.frame(width: labelWidth, height: 1)
                GeometryReader { proxy in
                    ZStack(alignment: .topLeading) {
                        ForEach(SpeakerSegments.axisTicks(duration: span), id: \.self) { tick in
                            Text(Self.tickLabel(tick))
                                .font(.system(size: 9)).foregroundStyle(.secondary).monospacedDigit()
                                .fixedSize()
                                .offset(x: min(proxy.size.width * tick / span, max(0, proxy.size.width - 24)))
                        }
                    }
                }
                .frame(height: 14)
            }
        }
        .padding(8)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    /// `m:ss` for an axis tick (ticks are whole seconds).
    static func tickLabel(_ seconds: Double) -> String {
        let clock = SpeakerSegments.clock(seconds)
        return clock.hasSuffix(".000") ? String(clock.dropLast(4)) : clock
    }
}

/// View-model driving one diarization run: reads the chosen file, runs the stage, and folds the segments back onto the
/// main actor. Reads the file directly via `AudioFileReader`. No Combine.
@MainActor
@Observable
final class SortformerRunModel {
    struct Result: Equatable {
        let segments: [SpeakerSegment]
        let duration: Double
        var speakerCount: Int { SpeakerSegments.speakers(in: segments).count }
    }

    var isRunning = false
    var result: Result?
    var errorText: String?

    func start(audioURL: URL, stage: any PipelineStage) {
        isRunning = true
        result = nil
        errorText = nil

        let accessed = audioURL.startAccessingSecurityScopedResource()
        Task {
            defer { if accessed { audioURL.stopAccessingSecurityScopedResource() } }
            do {
                // Decode off the main actor; a long recording is many megabytes of samples.
                let buffer = try await Task.detached { try AudioFileReader.read(audioURL) }.value
                let duration = buffer.sampleRate > 0 ? Double(buffer.samples.count) / Double(buffer.sampleRate) : 0
                // Refuse an over-long recording before anything is loaded (the stage and the engine check again).
                try SortformerEngine.requireWithinLimit(seconds: duration)
                // MLX-ERR-2 (see MLXErrorGuard.guardedRun). The library's `generate` hops to its own detached task, where the
                // scoped handler does not reach; the process-wide fallback records an error there and guardedRun rethrows it.
                let output = try await MLXErrorGuard.guardedRun {
                    try await stage.run(.audio(buffer)) { _ in }
                }
                guard case let .text(csv) = output else { throw StageError.kindMismatch(expected: .text, got: output.kind) }
                self.result = Result(segments: try SpeakerSegments.parseCSV(csv), duration: duration)
            } catch is CancellationError {
                // The user declined the served-load question: not a failure.
            } catch {
                self.errorText = MLXErrorGuard.sentence(for: error)
            }
            self.isRunning = false
        }
    }
}
