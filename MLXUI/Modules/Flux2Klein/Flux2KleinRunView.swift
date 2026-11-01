import AppKit
import SwiftUI
import UniformTypeIdentifiers

// FLUX.2 Klein 4B run surface (MP7-4): prompt, Size (width x height, multiples of 16 up to this Mac's cap), steps, seed, the image model's quantize
// (fixed at 8: MP7-4-FIX-1 dropped the choice; flows still accept 4 and 8), progress per stage and step, cancel, Save PNG. It reuses the shape of `FluxRunView` / `FluxRunModel` but NOT FluxUI's Size picker,
// whose selection is never read (MP7-1 spike 7.4): every value shown here reaches both the `StageConfig` and the leased stage.

/// Settings parsing and run state for one Klein run window. `@MainActor @Observable`, no Combine.
@MainActor
@Observable
final class Flux2KleinRunModel {
    // Fields hold text so a half-typed value is not rewritten under the user's cursor.
    var prompt = "a red fox in a snowy forest at sunrise"
    var widthText: String
    var heightText: String
    var stepsText = "\(Flux2KleinRunSettings.defaultSteps)"
    var seedText = ""
    /// The Run UI offers no quantize choice (owner, 2026-11-01): every run is 8-bit.
    let quantize = Flux2KleinRunSettings.defaultQuantize

    var isRunning = false
    var progress = 0.0
    var phase: Flux2KleinPhase?
    var image: CGImage?
    var errorText: String?
    /// The run was cancelled by the user (not a failure).
    var wasStopped = false
    /// The seed the last run used (the typed one, or the random one it picked), so a good picture can be reproduced.
    var usedSeed: UInt64?
    private(set) var usedSettings: Flux2KleinRunSettings?

    /// Klein's own cap budget; injectable so a test can stand in for a small Mac.
    var budgetBytes: @Sendable () -> Int64 = { Flux2KleinMemoryPlan.defaultBudgetBytes }

    private var task: Task<Void, Never>?
    private var generation = 0

    static let sizePresets = [512, 768, 1024, 1280]

    /// `budgetBytes` is the cap budget this window plans against (a test stands in for a small Mac). The size fields start at `defaultSide(...)`.
    init(budgetBytes: (@Sendable () -> Int64)? = nil) {
        let budget = budgetBytes ?? { Flux2KleinMemoryPlan.defaultBudgetBytes }
        self.budgetBytes = budget
        let side = Self.defaultSide(quantize: Flux2KleinRunSettings.defaultQuantize, budgetBytes: budget())
        widthText = "\(side)"
        heightText = "\(side)"
    }

    /// The preset sizes this Mac can make at `quantize`, smallest first. Empty when not even the smallest preset fits.
    nonisolated static func offeredSizes(quantize: Int, budgetBytes: Int64) -> [Int] {
        let largest = Flux2KleinMemoryPlan.largestSquareSide(quantize: quantize, budgetBytes: budgetBytes)
        return sizePresets.filter { $0 <= largest }
    }

    /// The largest offered size at or below the manifest's 1024; the smallest preset when none is offered (so the cap sentence names a real size).
    nonisolated static func defaultSide(quantize: Int, budgetBytes: Int64) -> Int {
        offeredSizes(quantize: quantize, budgetBytes: budgetBytes).last { $0 <= Flux2KleinRunSettings.defaultSize } ?? sizePresets[0]
    }

    /// The sizes the Presets menu lists (this Mac's, not every preset).
    var offeredSizes: [Int] { Self.offeredSizes(quantize: quantize, budgetBytes: budgetBytes()) }

    // MARK: Settings

    /// The settings the fields describe, or the sentence saying why they can't run. Everything the engine would refuse is refused here first, in the engine's own words
    /// (size not a multiple of 16: SPEC-Q242; over the cap: names the size asked and the largest this Mac can make).
    func resolve() -> Result<Flux2KleinRunSettings, Flux2KleinSettingsRefusal> {
        Self.resolve(width: widthText, height: heightText, steps: stepsText, seed: seedText, quantize: quantize, budgetBytes: budgetBytes())
    }

    nonisolated static func resolve(width: String, height: String, steps: String, seed: String, quantize: Int,
                                    budgetBytes: Int64) -> Result<Flux2KleinRunSettings, Flux2KleinSettingsRefusal> {
        func whole(_ text: String) -> Int? { Int(text.trimmingCharacters(in: .whitespacesAndNewlines)) }
        guard let w = whole(width) else { return .failure(.init(message: "Width must be a whole number of pixels.")) }
        guard let h = whole(height) else { return .failure(.init(message: "Height must be a whole number of pixels.")) }
        guard let s = whole(steps), s >= 1 else { return .failure(.init(message: "Steps must be a whole number, 1 or more.")) }
        let seedTrimmed = seed.trimmingCharacters(in: .whitespacesAndNewlines)
        var seedValue: UInt64?
        if !seedTrimmed.isEmpty {
            guard let parsed = UInt64(seedTrimmed) else { return .failure(.init(message: "Seed must be a whole number, or blank for a random one.")) }
            seedValue = parsed
        }
        let settings = Flux2KleinRunSettings(width: w, height: h, steps: s, seed: seedValue, quantize: quantize)
        do {
            try Flux2KleinEngine.validate(settings.request(prompt: "", seed: 0, row: Flux2KleinStage.runWindowLabel), budgetBytes: budgetBytes)
        } catch {
            return .failure(.init(message: (error as? LocalizedError)?.errorDescription ?? String(describing: error)))
        }
        return .success(settings)
    }

    var settingsRefusal: String? {
        if case let .failure(refusal) = resolve() { return refusal.message }
        return nil
    }

    var promptIsEmpty: Bool { prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    var canRun: Bool { !isRunning && !promptIsEmpty && settingsRefusal == nil }

    /// The largest square this Mac can make at the chosen quantize, for the caption under the size fields.
    var largestSquare: Int { Flux2KleinMemoryPlan.largestSquareSide(quantize: quantize, budgetBytes: budgetBytes()) }

    func applySizePreset(_ side: Int) {
        widthText = "\(side)"
        heightText = "\(side)"
    }

    // MARK: Run

    typealias StageBuilder = (_ settings: Flux2KleinRunSettings, _ onPhase: @escaping @Sendable (Flux2KleinPhase) -> Void) -> any PipelineStage

    /// Starts one run through `handedStage` (the Run sheet's `LeasingStage`, so the run leases its memory from the shared budget). `makeStage` is the production stage unless a test
    /// stands in. Does nothing when the fields are invalid, the prompt is empty, or a run is already going.
    func start(modelID: String, handedStage: any PipelineStage, makeStage: StageBuilder? = nil) {
        guard !isRunning, !promptIsEmpty, case let .success(parsed) = resolve() else { return }
        var settings = parsed
        let seed = settings.seed ?? Flux2KleinRunSettings.randomSeed()
        settings.seed = seed
        generation += 1
        let mine = generation
        isRunning = true
        progress = 0
        phase = nil
        image = nil
        errorText = nil
        wasStopped = false
        usedSeed = seed
        usedSettings = settings
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        // Strong capture is deliberate: the run's task holds the stage, the stage holds this closure, and all of it is released when the run ends.
        let onPhase: @Sendable (Flux2KleinPhase) -> Void = { phase in
            Task { @MainActor in
                guard self.generation == mine, self.isRunning else { return }
                self.phase = phase
                self.progress = max(self.progress, phase.fraction)
            }
        }
        let built = makeStage?(settings, onPhase) ?? Flux2KleinStage(modelID: modelID, settings: settings, onPhase: onPhase)
        // The per-run settings ride on `config` for the record; the engine cache is keyed without them (they don't change which weights load).
        let stage = LeasingStage.perRun(handedStage, config: StageConfig(seed: seed, width: settings.width, height: settings.height, steps: settings.steps)) { _ in built }
        task = Task {
            do {
                // MLX-ERR-2 (see MLXErrorGuard.guardedRun): the main actor starts it, the engine runs off it (`@concurrent`).
                let result = try await MLXErrorGuard.guardedRun {
                    try await stage.run(.text(text)) { _ in }
                }
                guard self.generation == mine else { return }
                if case let .image(media) = result { self.image = media.cgImage }
                self.progress = 1
                self.isRunning = false
            } catch is CancellationError {
                guard self.generation == mine else { return }
                self.wasStopped = true
                self.isRunning = false
            } catch {
                guard self.generation == mine else { return }
                self.errorText = MLXErrorGuard.sentence(for: error)
                self.isRunning = false
            }
        }
    }

    /// Stops the run; the engine checks between steps and blocks, releases its component and puts the cache limit back, then the lease is returned.
    func cancel() { task?.cancel() }

    /// Waits for the current run to finish (tests).
    func waitForRun() async { await task?.value }

    // MARK: Save

    nonisolated static func pngData(for image: CGImage) -> Data? {
        NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
    }

    nonisolated static func suggestedFileName(seed: UInt64?) -> String { "klein-\(seed.map(String.init) ?? "image").png" }
}

nonisolated struct Flux2KleinSettingsRefusal: Error, Equatable {
    let message: String
}

struct Flux2KleinRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run goes through it.
    let handedStage: any PipelineStage

    @State private var model: Flux2KleinRunModel

    init(modelDisplayName: String, license: String?, modelID: String, handedStage: any PipelineStage, model: Flux2KleinRunModel? = nil) {
        self.modelDisplayName = modelDisplayName
        self.license = license
        self.modelID = modelID
        self.handedStage = handedStage
        _model = State(initialValue: model ?? Flux2KleinRunModel())
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            promptField
            options
            if model.isRunning { progressRow }
            if let image = model.image { resultView(image) }
            if model.wasStopped { Text("Stopped.").font(.callout).foregroundStyle(.secondary) }
            if let error = model.errorText { RunErrorBar(message: error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 640, height: 780)
    }

    // MARK: Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "photo.artframe").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Text → image").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var promptField: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Prompt").font(.subheadline.bold())
            TextEditor(text: $model.prompt)
                .font(.body)
                .frame(minHeight: 90)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
                .disabled(model.isRunning)
        }
    }

    private var options: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Size").frame(width: 70, alignment: .leading)
                TextField("Width", text: $model.widthText).textFieldStyle(.roundedBorder).frame(width: 80)
                Text("×")
                TextField("Height", text: $model.heightText).textFieldStyle(.roundedBorder).frame(width: 80)
                Menu("Presets") {
                    ForEach(model.offeredSizes, id: \.self) { side in
                        Button("\(side) × \(side)") { model.applySizePreset(side) }
                    }
                }
                .fixedSize()
                .disabled(model.offeredSizes.isEmpty)
                Spacer()
            }
            Text(sizeCaption).font(.caption).foregroundStyle(.secondary).padding(.leading, 70)
            HStack {
                Text("Steps").frame(width: 70, alignment: .leading)
                TextField("Steps", text: $model.stepsText).textFieldStyle(.roundedBorder).frame(width: 80)
                Text("4 is what the model is made for").font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            HStack {
                Text("Seed").frame(width: 70, alignment: .leading)
                TextField("Seed (blank = random)", text: $model.seedText).textFieldStyle(.roundedBorder).frame(maxWidth: 200)
                Spacer()
            }
            if let refusal = model.settingsRefusal {
                Text(refusal).font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            }
            Text(license ?? "License: see model page").font(.caption).foregroundStyle(.secondary)
        }
        .disabled(model.isRunning)
    }

    private var sizeCaption: String {
        let largest = model.largestSquare
        let rule = "Multiples of 16."
        return largest >= Flux2KleinLatents.sizeMultiple
            ? "\(rule) The largest this Mac can make is \(largest) × \(largest) (or any shape with no more pixels)."
            : "\(rule) This Mac doesn't have the memory for even a small image with this model."
    }

    private var progressRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: model.progress).progressViewStyle(.linear)
                Text("\(Int(model.progress * 100))%").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(model.phase?.label ?? "Starting…").font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func resultView(_ image: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(nsImage: NSImage(cgImage: image, size: .zero))
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 320)
                .background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
            HStack(spacing: 12) {
                Label("Ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.subheadline.bold())
                if let seed = model.usedSeed { Text("Seed \(seed)").font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                Button { save(image) } label: { Label("Save PNG…", systemImage: "square.and.arrow.down") }
                Spacer()
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            if model.isRunning {
                Button("Cancel") { model.cancel() }
            }
            Button(model.isRunning ? "Generating…" : "Run") {
                model.start(modelID: modelID, handedStage: handedStage)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(!model.canRun)
        }
    }

    // MARK: Save

    private func save(_ image: CGImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = Flux2KleinRunModel.suggestedFileName(seed: model.usedSeed)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let data = Flux2KleinRunModel.pngData(for: image) else {
            model.errorText = "Could not encode the image as PNG."
            return
        }
        do { try data.write(to: url) } catch { model.errorText = "Save failed: \(error.localizedDescription)" }
    }
}
