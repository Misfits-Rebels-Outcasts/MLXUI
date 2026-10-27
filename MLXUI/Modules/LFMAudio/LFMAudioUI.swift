import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// `ModelUI` for LFM2.5-Audio. Presents `LFMAudioRunView`: one window, three modes (Voice chat, Speech to text, Text to speech).
struct LFMAudioUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .voice,
              model.source == .mlx,
              model.family.lowercased().contains("lfm2.5-audio") else {
            return .no
        }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(LFMAudioRunView(modelDisplayName: model.displayName, license: model.license, modelID: model.id, handedStage: stage))
    }
}

/// The run surface. Each tab owns its own view-model, so switching tabs keeps what each one has. Leaving the window stops the
/// microphone and the sound and drops the conversation (RAM only).
struct LFMAudioRunView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case voice = "Voice chat", transcribe = "Speech to text", speak = "Text to speech"
        var id: String { rawValue }
    }

    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let license: String?
    let modelID: String
    /// SC-5c: the stage the Run sheet handed (a `LeasingStage`); every run goes through it.
    let handedStage: any PipelineStage

    @State private var mode: Mode = .voice
    @State private var voice: LFMVoiceRunModel?
    @State private var transcribe = LFMTranscribeRunModel()
    @State private var speak = LFMSpeakRunModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            Divider()
            switch mode {
            case .voice:
                if let voice { LFMVoiceChatView(model: voice) }
            case .transcribe:
                LFMTranscribeView(model: transcribe, modelID: modelID, handedStage: handedStage)
            case .speak:
                LFMSpeakView(model: speak, modelID: modelID, handedStage: handedStage)
            }
            Divider()
            footer
        }
        .padding(18)
        .frame(width: 640, height: 720)
        .onAppear {
            if voice == nil {
                let dir = LFMAudioEngine.installedModelDirectory(id: modelID)
                let stage = handedStage
                voice = LFMVoiceRunModel(makeStage: { conversation, onEvent in
                    LeasingStage.perRun(stage, config: .default) { _ in
                        LFMVoiceTurnStage(modelDirectory: dir, conversation: conversation, onEvent: onEvent)
                    }
                })
            }
        }
        .onDisappear {
            voice?.shutdown()
            transcribe.cancel()
            speak.cancel()
            speak.stopPlayback()
        }
        .onChange(of: mode) { _, _ in
            // Hold-to-talk is a Voice chat control: leaving the tab ends any capture and any reply sound.
            voice?.pressCancelled()
            voice?.stop()
            speak.stopPlayback()
        }
        .task {
            // The app losing focus mid-hold would swallow the mouse-up; treat it as a cancelled press.
            for await _ in NotificationCenter.default.notifications(named: NSApplication.didResignActiveNotification) {
                voice?.pressCancelled()
            }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: ModelType.voice.sfSymbol).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Voice chat, speech to text and text to speech. English only.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Text(license ?? "License: see model page").font(.caption).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Voice chat

struct LFMVoiceChatView: View {
    @Bindable var model: LFMVoiceRunModel
    @State private var pressing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            transcript
            if model.permissionDenied { deniedNotice }
            if let hint = model.hint { Text(hint).font(.caption).foregroundStyle(.secondary) }
            if let error = model.errorText { RunErrorBar(message: error) }
            status
            controls
            Text("Hold the button while you speak, then let go. Your voice is processed on this Mac. The conversation is kept only while this window is open.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    if model.messages.isEmpty {
                        Text("Nothing said yet. Hold the button and ask something.")
                            .foregroundStyle(.secondary).padding(.top, 8)
                    }
                    ForEach(model.messages) { message in
                        bubble(message).id(message.id)
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            .onChange(of: model.messages.count) { _, _ in
                if let last = model.messages.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
    }

    private func bubble(_ message: LFMVoiceRunModel.Message) -> some View {
        let isUser = message.role == .user
        return HStack(alignment: .top, spacing: 6) {
            if isUser { Spacer(minLength: 60) }
            VStack(alignment: .leading, spacing: 4) {
                Label(message.text, systemImage: isUser ? "waveform" : "text.bubble")
                    .labelStyle(BubbleLabelStyle(showIcon: isUser))
                    .textSelection(.enabled)
                if message.cutShort {
                    Text(LFMVoiceRunModel.cutShortLine).font(.caption).foregroundStyle(.secondary)
                }
                if !message.audio.isEmpty {
                    Button { model.replay(message) } label: { Label("Replay", systemImage: "play.fill") }
                        .buttonStyle(.borderless).font(.caption)
                        .disabled(model.phase != .idle)
                }
            }
            .padding(8)
            .background(isUser ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
            if !isUser { Spacer(minLength: 60) }
        }
    }

    private var deniedNotice: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "mic.slash").foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 4) {
                Text("AI Browser doesn't have permission to use the microphone, so voice chat can't hear you.")
                    .font(.callout).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Button("Open System Settings") { NSWorkspace.shared.open(LFMMicSettings.url) }
                    Button("Check again") { model.recorder.refreshPermission() }
                }
                Text("Turn on AI Browser under Privacy & Security, Microphone. Speech to text and text to speech work without it.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder private var status: some View {
        switch model.phase {
        case .recording:
            HStack(spacing: 8) {
                Image(systemName: "record.circle.fill").foregroundStyle(.red)
                Text("Listening \(Self.clock(model.recorder.elapsed)) of \(Self.clock(model.recorder.maxSeconds))")
                    .monospacedDigit()
                Spacer()
            }
            .font(.callout)
        case .thinking:
            HStack(spacing: 8) {
                ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
                Text("Thinking…").font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Stop") { model.stop() }
            }
        case .speaking:
            HStack(spacing: 8) {
                Image(systemName: "speaker.wave.2.fill").foregroundStyle(.secondary)
                Text("Speaking…").font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Stop") { model.stop() }
            }
        case .idle:
            EmptyView()
        }
    }

    private var controls: some View {
        HStack {
            Button { model.newConversation() } label: { Label("New conversation", systemImage: "square.and.pencil") }
                .disabled(model.phase == .thinking || (model.messages.isEmpty && model.phase == .idle))
            Spacer()
            holdButton
            Spacer()
            // balances the layout
            Color.clear.frame(width: 150, height: 1)
        }
    }

    private var holdButton: some View {
        Label(model.phase == .recording ? "Release to send" : "Hold to talk", systemImage: "mic.fill")
            .font(.headline)
            .padding(.horizontal, 22).padding(.vertical, 12)
            .foregroundStyle(.white)
            .background(model.canHold ? (model.phase == .recording ? Color.red : Color.accentColor) : Color.gray, in: Capsule())
            .opacity(model.canHold ? 1 : 0.5)
            .contentShape(Capsule())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !pressing else { return }
                    pressing = true
                    model.pressBegan()
                }
                .onEnded { _ in
                    pressing = false
                    model.pressEnded()
                })
            .accessibilityLabel("Hold to talk")
            .accessibilityAddTraits(.isButton)
    }

    static func clock(_ seconds: Double) -> String {
        let total = max(Int(seconds.rounded(.down)), 0)
        return "\(total / 60):" + String(format: "%02d", total % 60)
    }
}

/// A label whose icon can be hidden (the assistant's bubble is just text).
private struct BubbleLabelStyle: LabelStyle {
    let showIcon: Bool
    func makeBody(configuration: Configuration) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            if showIcon { configuration.icon.foregroundStyle(.secondary) }
            configuration.title
        }
    }
}

// MARK: - Speech to text

struct LFMTranscribeView: View {
    @Bindable var model: LFMTranscribeRunModel
    let modelID: String
    let handedStage: any PipelineStage

    @State private var audioURL: URL?
    @State private var showImporter = false
    @State private var isDropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Button("Choose Audio…") { showImporter = true }
                Text(audioURL?.lastPathComponent ?? "No file selected (or drop one here)")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            Text("English speech. Any sample rate works; long recordings are read in 30-second parts.")
                .font(.caption).foregroundStyle(.secondary)
            if model.isRunning {
                HStack(spacing: 8) {
                    ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
                    Text("Transcribing…").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Stop") { model.cancel() }
                }
                .padding(10)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
            if let error = model.errorText { RunErrorBar(message: error) }
            if let transcript = model.transcript {
                HStack {
                    Label("Transcript", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.subheadline.bold())
                    Spacer()
                    Button("Copy") { model.copy() }
                    Button("Save…") { model.save(suggestedName: Self.transcriptName(for: audioURL)) }
                }
                ScrollView {
                    Text(transcript).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(8)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button(model.isRunning ? "Transcribing…" : "Transcribe") { run() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(audioURL == nil || model.isRunning)
            }
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio], allowsMultipleSelection: false) { result in
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
                    .allowsHitTesting(false)
            }
        }
    }

    private func run() {
        guard let url = audioURL else { return }
        let modelID = self.modelID
        let stage = LeasingStage.perRun(handedStage, config: .default) { _ in LFMAudioStages.asr(modelID: modelID) }
        model.start(audioURL: url, stage: stage)
    }

    static func transcriptName(for url: URL?) -> String {
        let base = url?.deletingPathExtension().lastPathComponent ?? ""
        return (base.isEmpty ? "transcript" : base) + ".txt"
    }
}

// MARK: - Text to speech

struct LFMSpeakView: View {
    @Bindable var model: LFMSpeakRunModel
    let modelID: String
    let handedStage: any PipelineStage

    @State private var text = "Hello! This is LFM2.5-Audio speaking, running on this Mac."
    @State private var style = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Text to speak").font(.subheadline.bold())
            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 150)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 4) {
                Text("Voice style (optional)").font(.subheadline.bold())
                TextField("e.g. Use a UK male voice.", text: $style).textFieldStyle(.roundedBorder)
                Text("Free text added to the instruction. The model decides how to follow it; there is no list of voices. English only. Long text is spoken in parts.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if model.isRunning {
                HStack(spacing: 8) {
                    ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
                    Text("Synthesizing…").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Stop") { model.cancel() }
                }
                .padding(10)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
            if let error = model.errorText { RunErrorBar(message: error) }
            if model.didSynthesize {
                HStack(spacing: 12) {
                    Label("Ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.subheadline.bold())
                    Button { model.play() } label: { Label("Play", systemImage: "play.fill") }
                    Button { model.stopPlayback() } label: { Label("Stop", systemImage: "stop.fill") }.disabled(!model.isPlaying)
                    Button { model.save() } label: { Label("Save WAV…", systemImage: "square.and.arrow.down") }
                    Spacer()
                }
            }
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button(model.isRunning ? "Synthesizing…" : "Speak") { run() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.isRunning)
            }
        }
    }

    private func run() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let (modelID, style) = (self.modelID, self.style)
        let stage = LeasingStage.perRun(handedStage, config: .default) { _ in LFMAudioStages.tts(modelID: modelID, style: style) }
        model.synthesize(text: trimmed, stage: stage)
    }
}
