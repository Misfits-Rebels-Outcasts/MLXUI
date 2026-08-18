import SwiftUI
import AVKit
import AppKit
import UniformTypeIdentifiers

// MARK: - Resolution presets

enum WanResolution: String, CaseIterable {
    case full   = "832×480"
    case square = "480×480"
    case medium = "512×288"
    case small  = "320×192"

    var latH: Int { switch self { case .full: return 60; case .square: return 60; case .medium: return 36; case .small: return 24 } }
    var latW: Int { switch self { case .full: return 104; case .square: return 60; case .medium: return 64; case .small: return 40 } }
}

// MARK: - Run view

struct WanVideoRunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let modelID: String

    @State private var model          = WanRunModel()
    @State private var prompt         = "A cat walks on the grass, realistic style."
    @State private var showNeg        = false
    @State private var negativePrompt = ""
    @State private var numFrames      = 3
    @State private var numSteps       = 20
    @State private var resolution     = WanResolution.square
    @State private var player: AVPlayer?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header.padding([.horizontal, .top], 18).padding(.bottom, 12)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    promptSection
                    if showNeg { negPromptField }
                    configRow
                    if model.isRunning { progressRow }
                    if let url = model.videoURL { videoSection(url) }
                    else if let img = model.thumbnail { thumbnailSection(img) }
                    if let err = model.errorText { RunErrorBar(message: err) }
                    Spacer(minLength: 0)
                }
                .padding(18)
            }
            Divider()
            footer.padding([.horizontal, .bottom], 18).padding(.top, 12)
        }
        .frame(width: 700, height: 800)
        .onChange(of: model.videoURL) { _, url in
            guard let url else { return }
            let p = AVPlayer(url: url)
            player = p
            p.play()
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "film.stack").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Text → video · \(resolution.rawValue) · 16 fps").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Prompt").font(.subheadline.bold())
                Spacer()
                Toggle("Negative prompt", isOn: $showNeg.animation())
                    .toggleStyle(.checkbox)
                    .font(.caption).foregroundStyle(.secondary)
            }
            TextEditor(text: $prompt)
                .font(.body)
                .frame(minHeight: 80)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var negPromptField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Negative prompt").font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $negativePrompt)
                .font(.body)
                .frame(minHeight: 50)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var configRow: some View {
        HStack(spacing: 20) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Frames").font(.caption).foregroundStyle(.secondary)
                Picker("", selection: $numFrames) {
                    Text("3 (0.25s)").tag(3)
                    Text("5 (0.5s)").tag(5)
                    Text("17 (1.25s)").tag(17)
                    Text("81 (5s)").tag(81)
                }
                .pickerStyle(.segmented)
                .frame(width: 200)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("Steps: \(numSteps)").font(.caption).foregroundStyle(.secondary)
                Slider(
                    value: Binding(get: { Double(numSteps) },
                                   set: { numSteps = Int($0.rounded()) }),
                    in: 10...50, step: 5
                )
                .frame(width: 160)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("Resolution").font(.caption).foregroundStyle(.secondary)
                Picker("", selection: $resolution) {
                    ForEach(WanResolution.allCases, id: \.self) { r in
                        Text(r.rawValue).tag(r)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 280)
            }
            Spacer()
        }
    }

    private var progressRow: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                ProgressView(value: model.progress).progressViewStyle(.linear)
                Text("\(Int(model.progress * 100))%")
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(model.progressLabel).font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func videoSection(_ url: URL) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let p = player {
                VideoPlayer(player: p)
                    .frame(height: 330)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .onAppear { p.play() }
            }
            HStack(spacing: 12) {
                Label("Video ready", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Button { saveVideo(url) } label: {
                    Label("Save Video…", systemImage: "square.and.arrow.down")
                }
                Spacer()
            }
        }
    }

    private func thumbnailSection(_ image: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Image(nsImage: NSImage(cgImage: image, size: .zero))
                .resizable().scaledToFit()
                .frame(maxHeight: 280)
                .background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
            Text("First frame decoded — assembling MP4…")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Generating…" : "Generate") { startGeneration() }
                .keyboardShortcut(.defaultAction)
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.isRunning)
        }
    }

    // MARK: - Actions

    private func startGeneration() {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !p.isEmpty else { return }
        player = nil
        model.start(
            prompt:         p,
            negativePrompt: showNeg ? negativePrompt : "",
            modelID:        modelID,
            numSteps:       numSteps,
            numFrames:      numFrames,
            latH:           resolution.latH,
            latW:           resolution.latW
        )
    }

    private func saveVideo(_ url: URL) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType.mpeg4Movie]
        panel.nameFieldStringValue = "wan-video.mp4"
        guard panel.runModal() == .OK, let dest = panel.url else { return }
        try? FileManager.default.removeItem(at: dest)
        try? FileManager.default.copyItem(at: url, to: dest)
    }
}

// MARK: - View model

@MainActor @Observable
final class WanRunModel {
    var isRunning     = false
    var progress      = 0.0
    var progressLabel = ""
    var thumbnail:  CGImage?
    var videoURL:   URL?
    var errorText:  String?

    func start(
        prompt:         String,
        negativePrompt: String,
        modelID:        String,
        numSteps:       Int,
        numFrames:      Int,
        latH:           Int = WanVideoEngine.latentH,
        latW:           Int = WanVideoEngine.latentW
    ) {
        isRunning = true; progress = 0; progressLabel = "Starting…"
        thumbnail = nil; videoURL = nil; errorText = nil

        Task {
            do {
                let frames = try await WanVideoEngine.generateAll(
                    prompt:         prompt,
                    negativePrompt: negativePrompt,
                    modelID:        modelID,
                    numSteps:       numSteps,
                    numFrames:      numFrames,
                    latH:           latH,
                    latW:           latW,
                    progress: { fraction, label in
                        Task { @MainActor in
                            self.progress = fraction
                            self.progressLabel = label
                        }
                    }
                )
                if let first = frames.first { thumbnail = first }
                progressLabel = "Assembling MP4…"

                // Use app-container Caches dir — AVAssetWriter fails on the global temp dir in sandbox
                let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
                    ?? FileManager.default.temporaryDirectory
                let tmpURL = cacheDir
                    .appendingPathComponent("wan-\(UInt64(Date().timeIntervalSince1970 * 1000)).mp4")
                try await WanVideoEngine.assembleMp4(frames: frames, fps: WanVideoEngine.targetFPS, to: tmpURL)

                videoURL = tmpURL
                isRunning = false
                progress = 1.0
                progressLabel = "Done — \(frames.count) frames"
            } catch {
                errorText = "\(error)"; isRunning = false
            }
        }
    }
}
