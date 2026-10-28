import Foundation
import Testing
@testable import MLXUI

/// MP6-3b: the microphone configuration, the hold-to-talk recorder's rules, the Voice chat view-model and the "nothing is
/// persisted" and "never asked at launch" structure. No microphone, no speaker, no model: fakes for the permission, the capture
/// device, the reply player and the stage.

private func repoRoot() -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
}

private func sourceFiles(under path: String) -> [URL] {
    let base = repoRoot().appendingPathComponent(path)
    guard let walker = FileManager.default.enumerator(at: base, includingPropertiesForKeys: nil) else { return [] }
    return walker.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
}

// MARK: - Configuration (owner ruling, Boo, 2026-10-27)

struct LFMMicConfigurationTests {
    private func text(_ path: String) throws -> String {
        try String(contentsOf: repoRoot().appendingPathComponent(path), encoding: .utf8)
    }

    private func entitlements(_ path: String) throws -> [String: Any] {
        let data = try Data(contentsOf: repoRoot().appendingPathComponent(path))
        return try #require(try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any])
    }

    @Test func bothEditionsDeclareTheAudioInputEntitlementAndNothingElseChanged() throws {
        let appStore = try entitlements("MLXUI/MLXUI-AppStore.entitlements")
        let direct = try entitlements("MLXUI/MLXUI-Direct.entitlements")
        #expect(appStore["com.apple.security.device.audio-input"] as? Bool == true)
        #expect(direct["com.apple.security.device.audio-input"] as? Bool == true)
        // the rest is exactly what it was before MP6-3b
        #expect(Set(appStore.keys) == ["com.apple.security.app-sandbox", "com.apple.security.files.user-selected.read-write",
                                       "com.apple.security.network.client", "com.apple.security.network.server",
                                       "keychain-access-groups", "com.apple.security.device.audio-input"])
        #expect(Set(direct.keys) == ["keychain-access-groups", "com.apple.security.device.audio-input"],
                "the Direct edition is Hardened Runtime only: the microphone key is the only addition")
    }

    @Test func theUsageDescriptionIsTheOwnersWordsInTheSharedConfig() throws {
        let shared = try text("Config/Shared.xcconfig")
        let line = "INFOPLIST_KEY_NSMicrophoneUsageDescription = AI Browser uses the microphone only while you hold to talk, to send your voice to a model running on this Mac. Audio never leaves your Mac."
        #expect(shared.contains(line))
        // one definition, in the shared file: neither edition file re-defines (or blanks) it
        #expect(!(try text("Config/AppStore.xcconfig")).contains("NSMicrophoneUsageDescription"))
        #expect(!(try text("Config/Direct.xcconfig")).contains("NSMicrophoneUsageDescription"))
    }

    /// Nothing may ask for the microphone but the recorder, and nothing may call the recorder but a press: the system prompt
    /// can only appear on the first hold-to-talk, never at launch.
    @Test func onlyTheRecorderAsksForPermissionAndNothingAtLaunchTouchesIt() throws {
        let askers = ["requestAccess(for", "requestRecordPermission", "AVAudioApplication.requestRecordPermission"]
        var found: [String] = []
        for file in sourceFiles(under: "MLXUI") {
            let source = try String(contentsOf: file, encoding: .utf8)
            if askers.contains(where: { source.contains($0) }) { found.append(file.lastPathComponent) }
        }
        #expect(found == ["LFMAudioMic.swift"], "permission requests: \(found)")
        let launch = try text("MLXUI/MLXUIApp.swift") + (try text("MLXUI/State/AppState.swift"))
        #expect(!launch.contains("LFMMic") && !launch.contains("AVCaptureDevice"))
        // within the module: `request()` is called from one place, `begin`
        let mic = try text("MLXUI/Modules/LFMAudio/LFMAudioMic.swift")
        #expect(mic.components(separatedBy: "permission.request()").count == 2)
    }

    @Test func theVoiceCodeTouchesNoFileOrDefaultsApi() throws {
        let forbidden = ["FileManager", "write(to", "createFile", "FileHandle", "UserDefaults", "AudioWriter", "NSSavePanel",
                         "ModelStore", "@AppStorage", "SceneStorage", "Keychain"]
        for name in ["LFMAudioVoiceModel.swift", "LFMAudioMic.swift", "LFMAudioPlayback.swift", "LFMAudioTypes.swift"] {
            let source = try text("MLXUI/Modules/LFMAudio/" + name)
            for token in forbidden { #expect(!source.contains(token), "\(name) mentions \(token)") }
        }
    }
}

// MARK: - Fakes

@MainActor
final class FakeReplyPlayer: LFMReplyPlaying {
    var onDrained: (() -> Void)?
    private(set) var appended: [[Float]] = []
    private(set) var finished = 0
    private(set) var stopped = 0
    func append(_ samples: [Float]) { appended.append(samples) }
    func finish() { finished += 1; onDrained?() }
    func stop() { stopped += 1 }
}

nonisolated final class FakePermission: LFMMicPermission, @unchecked Sendable {
    private let lock = NSLock()
    private var current: LFMMicAuthorization
    private let grants: Bool
    private var _requests = 0
    var gate: (@Sendable () async -> Void)?
    init(_ status: LFMMicAuthorization, grants: Bool = true) { current = status; self.grants = grants }
    var requests: Int { lock.withLock { _requests } }
    func status() -> LFMMicAuthorization { lock.withLock { current } }
    func request() async -> Bool {
        lock.withLock { _requests += 1 }
        await gate?()
        lock.withLock { current = grants ? .authorized : .denied }
        return grants
    }
}

nonisolated final class FakeCapture: LFMCaptureDevice, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var started = 0
    private(set) var stopped = 0
    var samples: [Float]
    var failure: Error?
    private var cap: (@Sendable () -> Void)?
    private(set) var maxSamples = 0
    init(samples: [Float] = []) { self.samples = samples }
    func start(maxSamples: Int, onCap: @escaping @Sendable () -> Void) throws {
        if let failure { throw failure }
        lock.withLock { started += 1; self.maxSamples = maxSamples; cap = onCap }
    }
    func stop() -> [Float] { lock.withLock { stopped += 1; return samples } }
    func hitCap() { cap?() }
}

/// A stage whose result and events are scripted.
nonisolated struct ScriptedVoiceStage: PipelineStage {
    var id = "scripted.voice"
    var name = "Scripted"
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .text }
    var events: [LFMVoiceEvent]
    var reply = "the reply"
    var failure: (any Error)?
    var delay: Duration?
    var onEvent: @Sendable (LFMVoiceEvent) -> Void

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        if let delay { try await Task.sleep(for: delay) }
        if let failure { throw failure }
        for event in events { onEvent(event) }
        return .text(reply)
    }
}

// MARK: - Recorder

@MainActor
struct LFMMicRecorderTests {
    private func recorder(_ permission: FakePermission, device: FakeCapture = FakeCapture(samples: [0.1, 0.2]),
                          maxSeconds: Double = LFMAudioEngine.maxUserTurnSeconds) -> (LFMMicRecorder, FakeCapture) {
        (LFMMicRecorder(maxSeconds: maxSeconds, permission: permission, makeDevice: { device }), device)
    }

    @Test func creatingTheRecorderAsksForNothing() {
        let permission = FakePermission(.notDetermined)
        _ = recorder(permission)
        #expect(permission.requests == 0)
    }

    @Test func theFirstPressAsksOnceAndLaterPressesDoNot() async {
        let permission = FakePermission(.notDetermined)
        let (rec, device) = recorder(permission)
        #expect(await rec.begin() == .started)
        #expect(permission.requests == 1)
        #expect(rec.isRecording)
        #expect(device.maxSamples == 60 * 16_000, "the 60 s cap, in 16 kHz samples")
        #expect(rec.end() == [0.1, 0.2])
        #expect(!rec.isRecording)
        #expect(await rec.begin() == .started)
        #expect(permission.requests == 1, "asked on the first hold only")
        _ = rec.end()
    }

    @Test func aDeniedPermissionIsAMessageNotAnErrorAndStartsNothing() async {
        let denied = FakePermission(.denied)
        let (rec, device) = recorder(denied)
        #expect(await rec.begin() == .denied)
        #expect(rec.permissionDenied)
        #expect(device.started == 0)
        #expect(denied.requests == 0, "never re-prompts: the system won't show it again")
        // the user declines the first-time prompt
        let declines = FakePermission(.notDetermined, grants: false)
        let (rec2, device2) = recorder(declines)
        #expect(await rec2.begin() == .denied)
        #expect(rec2.permissionDenied && device2.started == 0)
        // back from System Settings with access on
        denied.gate = nil
        let (rec3, _) = recorder(FakePermission(.authorized))
        #expect(await rec3.begin() == .started)
        #expect(!rec3.permissionDenied)
        _ = rec3.end()
    }

    @Test func aPressThatEndedWhileThePromptWasUpStartsNothing() async {
        let permission = FakePermission(.notDetermined)
        let (rec, device) = recorder(permission)
        #expect(await rec.begin(stillHeld: { false }) == .pressEnded)
        #expect(device.started == 0)
        #expect(!rec.isRecording)
    }

    @Test func aDeviceFailureIsASentenceAndLeavesNoRecording() async {
        struct Broken: LocalizedError { var errorDescription: String? { "No microphone is available." } }
        let device = FakeCapture()
        device.failure = Broken()
        let (rec, _) = recorder(FakePermission(.authorized), device: device)
        #expect(await rec.begin() == .failed("No microphone is available."))
        #expect(!rec.isRecording)
    }

    @Test func cancelStopsTheMicrophoneAndKeepsNothing() async {
        let (rec, device) = recorder(FakePermission(.authorized))
        _ = await rec.begin()
        rec.cancel()
        #expect(device.stopped == 1)
        #expect(!rec.isRecording)
        #expect(rec.end() == nil)
    }

    @Test func reachingTheCapStopsCleanlyAndHandsOverTheSamples() async throws {
        let (rec, device) = recorder(FakePermission(.authorized), device: FakeCapture(samples: [0.5, 0.6, 0.7]))
        var delivered: [Float]?
        rec.onAutoStop = { delivered = $0 }
        _ = await rec.begin()
        device.hitCap()
        try await Task.sleep(for: .milliseconds(100))   // the cap callback hops to the main actor
        #expect(delivered == [0.5, 0.6, 0.7])
        #expect(device.stopped == 1)
        #expect(!rec.isRecording)
    }

    @Test func sampleSinkStopsAtTheCapAndSaysSoOnce() {
        let hits = LockedCounter()
        let sink = LFMSampleSink(maxSamples: 5) { hits.add() }
        sink.append([1, 2, 3])
        sink.append([4, 5, 6, 7])
        sink.append([8])
        #expect(sink.take() == [1, 2, 3, 4, 5])
        #expect(hits.value == 1)
    }

    @Test func theSystemSettingsLinkOpensThePrivacyMicrophonePane() {
        #expect(LFMMicSettings.url.absoluteString == "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
    }
}

nonisolated final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func add() { lock.withLock { n += 1 } }
    var value: Int { lock.withLock { n } }
}

// MARK: - Voice chat view-model

@MainActor
struct LFMVoiceRunModelTests {
    private let speech = [Float](repeating: 0.1, count: 16_000 * 2)

    private func model(events: [LFMVoiceEvent] = [.text("the reply"), .audio([0.1, 0.2]), .audio([0.3])],
                       failure: (any Error)? = nil, delay: Duration? = nil, permission: LFMMicAuthorization = .authorized,
                       device: FakeCapture = FakeCapture(), player fake: FakeReplyPlayer? = nil
    ) -> (LFMVoiceRunModel, FakeCapture, FakeReplyPlayer) {
        let player = fake ?? FakeReplyPlayer()
        let recorder = LFMMicRecorder(permission: FakePermission(permission), makeDevice: { device })
        let model = LFMVoiceRunModel(recorder: recorder, player: player, makeStage: { _, onEvent in
            ScriptedVoiceStage(events: events, failure: failure, delay: delay, onEvent: onEvent)
        })
        return (model, device, player)
    }

    private func settle(_ model: LFMVoiceRunModel, until phase: LFMVoiceRunModel.Phase = .idle) async throws {
        for _ in 0..<200 {
            if model.phase == phase { return }
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    @Test func aReplyThatHitTheBudgetCarriesAPlainCutShortLineAndNoError() async throws {
        let (model, _, _) = model(events: [.text("a long reply"), .cutShort, .audio([0.1])])
        model.send(speech)
        try await settle(model)
        #expect(model.messages.last?.cutShort == true)
        #expect(LFMVoiceRunModel.cutShortLine == "(reply cut short)")
        #expect(model.errorText == nil)
        let (plain, _, _) = self.model()
        plain.send(speech)
        try await settle(plain)
        #expect(plain.messages.last?.cutShort == false)
    }

    @Test func aTurnShowsTheLabelThenTheReplyAndRemembersTheExchange() async throws {
        let (model, _, player) = model()
        model.send(speech)
        #expect(model.phase == .thinking)
        #expect(model.messages.map(\.text) == ["Voice message 0:02"])
        try await settle(model)
        #expect(model.messages.map(\.text) == ["Voice message 0:02", "the reply"])
        #expect(model.messages.last?.audio == [0.1, 0.2, 0.3])
        #expect(player.appended == [[0.1, 0.2], [0.3]])
        #expect(player.finished == 1)
        #expect(model.conversation.turns.count == 1)
        #expect(model.conversation.turns[0].assistantText == "the reply")
        #expect(model.errorText == nil)
    }

    /// MP6-3d-prep (owner ruling, Boo, 2026-10-27): the hold that reaches 60 s is sent, with a plain line, never an error.
    @Test func reachingTheLimitSendsTheTurnAndSaysItWasCut() async throws {
        let device = FakeCapture(samples: [Float](repeating: 0.2, count: 60 * 16_000))
        let (model, device2, _) = model(device: device)
        model.pressBegan()
        try await settle(model, until: .recording)
        device2.hitCap()
        try await settle(model, until: .thinking)
        try await settle(model)
        #expect(model.errorText == nil)
        #expect(model.hint == "That message was cut at 1 minute.")
        #expect(model.messages.first?.text == "Voice message 1:00")
        #expect(model.conversation.turns.count == 1, "the cut turn was sent and remembered")
        #expect(model.conversation.turns[0].userSamples.count == 60 * 16_000)
    }

    @Test func aLongerRecordingIsCutToTheLimitNotRefused() async throws {
        let (model, _, _) = model()
        model.send([Float](repeating: 0.1, count: 60 * 16_000 + 4_096))
        try await settle(model)
        #expect(model.errorText == nil)
        #expect(model.conversation.turns.count == 1)
        #expect(model.conversation.turns[0].userSamples.count == LFMVoiceRunModel.maximumSamples)
    }

    @Test func holdToTalkRecordsThenSendsOnRelease() async throws {
        let (model, device, _) = model(device: FakeCapture(samples: [Float](repeating: 0.2, count: 16_000)))
        model.pressBegan()
        try await settle(model, until: .recording)
        #expect(model.phase == .recording)
        model.pressEnded()
        #expect(device.stopped == 1)
        try await settle(model)
        #expect(model.messages.count == 2)
    }

    @Test func aTapIsAHintNotATurn() async throws {
        let (model, _, _) = model(device: FakeCapture(samples: [0.1, 0.1]))
        model.pressBegan()
        try await settle(model, until: .recording)
        model.pressEnded()
        #expect(model.phase == .idle)
        #expect(model.hint != nil)
        #expect(model.messages.isEmpty)
        #expect(model.errorText == nil)
    }

    @Test func aDeniedMicrophoneShowsTheNoticeAndNoError() async throws {
        let (model, device, _) = model(permission: .denied)
        model.pressBegan()
        try await Task.sleep(for: .milliseconds(100))
        #expect(model.permissionDenied)
        #expect(model.errorText == nil)
        #expect(model.phase == .idle)
        #expect(device.started == 0)
    }

    @Test func aNewHoldSilencesThePreviousReplyBeforeTheMicrophoneOpens() async throws {
        let (model, _, player) = model()
        model.send(speech)
        try await settle(model)
        let stopsBefore = player.stopped
        model.pressBegan()
        #expect(player.stopped > stopsBefore, "the reply's sound is stopped as the new hold begins")
        model.pressCancelled()
    }

    @Test func stopWhileThinkingIsQuietAndRecordsNothing() async throws {
        let (model, _, player) = model(delay: .milliseconds(300))
        model.send(speech)
        #expect(model.phase == .thinking)
        model.stop()
        #expect(model.phase == .idle)
        #expect(model.messages.isEmpty, "the pending message is dropped")
        try await Task.sleep(for: .milliseconds(450))   // the abandoned run finishes; its result is discarded
        #expect(model.messages.isEmpty)
        #expect(model.conversation.turns.isEmpty)
        #expect(model.errorText == nil, "a cancelled turn is never an error")
        #expect(player.appended.isEmpty)
    }

    @Test func aCancellationErrorFromTheStageIsNotShownEither() async throws {
        let (model, _, _) = model(failure: CancellationError())
        model.send(speech)
        try await settle(model)
        #expect(model.errorText == nil)
        #expect(model.messages.isEmpty)
        #expect(model.conversation.turns.isEmpty)
    }

    @Test func aRealFailureIsAnErrorSentenceAndRecordsNothing() async throws {
        struct Boom: LocalizedError { var errorDescription: String? { "the model failed" } }
        let (model, _, _) = model(failure: Boom())
        model.send(speech)
        try await settle(model)
        #expect(model.errorText == "the model failed")
        #expect(model.conversation.turns.isEmpty)
        #expect(model.phase == .idle)
    }

    @Test func newConversationAndShutdownClearTheMemoryAndStopEverything() async throws {
        let device = FakeCapture()
        let (model, _, player) = model(device: device)
        model.send(speech)
        try await settle(model)
        #expect(!model.conversation.turns.isEmpty)
        model.newConversation()
        #expect(model.conversation.turns.isEmpty && model.messages.isEmpty)
        model.send(speech)
        try await settle(model)
        model.pressBegan()
        try await settle(model, until: .recording)
        model.shutdown()
        #expect(model.conversation.turns.isEmpty && model.messages.isEmpty)
        #expect(!model.recorder.isRecording)
        #expect(device.stopped >= 1)
        #expect(player.stopped >= 1)
    }

    @Test func aWholeTurnWritesNothingUnderTheModelStore() async throws {
        let before = Self.listing(ModelStore.shared.baseDirectory)
        let (model, _, _) = model()
        model.send(speech)
        try await settle(model)
        model.newConversation()
        #expect(Self.listing(ModelStore.shared.baseDirectory) == before)
    }

    private static func listing(_ url: URL) -> [String] {
        guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return [] }
        return walker.compactMap { ($0 as? URL).map { "\($0.path) \((try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)" } }.sorted()
    }
}

// MARK: - The voice stage

struct LFMVoiceTurnStageTests {
    @Test func theStageResamplesToSixteenKHzRunsTheTurnAndReturnsTheReplyText() async throws {
        let backend = MockLFMBackend { _ in LFMGenerated(text: " spoken ", audioFrames: [[1], [2]]) }
        let events = LFMEventBox()
        var conversation = LFMConversation()
        conversation.record(userSamples: [0.1], assistantText: "before")
        let stage = LFMVoiceTurnStage(modelDirectory: URL(fileURLWithPath: "/tmp/lfm-model"), conversation: conversation,
                                      factory: { _ in backend }) { events.add($0) }
        #expect(stage.accepts == .audio && stage.produces == .text)
        let input = AudioBuffer(samples: [Float](repeating: 0.05, count: 44_100), sampleRate: 44_100)
        let out = try await stage.run(.audio(input)) { _ in }
        guard case let .text(text) = out else { Issue.record("not text"); return }
        #expect(text == "spoken")
        guard case let .audio(heard) = backend.calls[0].prompt.turns.last?.parts.first else { Issue.record("no audio"); return }
        #expect(abs(heard.count - 16_000) < 64)
        #expect(backend.calls[0].prompt.turns.map(\.role) == ["system", "user", "assistant", "user"])
        #expect(events.values.first == .text("spoken"))
    }
}
