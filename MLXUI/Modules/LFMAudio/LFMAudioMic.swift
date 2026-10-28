import Foundation
import AVFoundation
import AppKit

// MP6-3b: hold-to-talk capture for the LFM2.5-Audio voice chat. Owner ruling (Boo, 2026-10-27): "Ask the mic permission only on
// the first hold-to-talk, never at launch. A denied permission shows a plain message linking to System Settings, not an error."
//
// Three seams keep the logic testable without a microphone: `LFMMicPermission` (the system prompt), `LFMCaptureDevice` (the
// audio engine) and the recorder itself, which owns the rules: ask on the first press only, never start after the press ended,
// stop at the cap, stop on cancel or when the view goes away. Nothing here is written to disk.

nonisolated enum LFMMicAuthorization: Sendable, Equatable {
    case notDetermined
    case authorized
    /// Denied by the user, or restricted by policy (parental controls, MDM). Both show the same plain message.
    case denied
}

/// The system's microphone permission. The ONLY place in the app that asks for it (a structural test enforces that).
nonisolated protocol LFMMicPermission: Sendable {
    func status() -> LFMMicAuthorization
    /// Shows the system prompt (first time only); returns whether access was granted.
    func request() async -> Bool
}

nonisolated struct LFMSystemMicPermission: LFMMicPermission {
    func status() -> LFMMicAuthorization {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return .authorized
        case .notDetermined: return .notDetermined
        case .denied, .restricted: return .denied
        @unknown default: return .denied
        }
    }

    func request() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }
}

/// Where a denied permission is fixed: System Settings > Privacy & Security > Microphone.
nonisolated enum LFMMicSettings {
    static let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!
}

/// A running capture. `stop()` returns the 16 kHz mono samples gathered so far and releases the microphone.
nonisolated protocol LFMCaptureDevice: AnyObject {
    /// Starts capturing. `onCap` is called (from any thread) once when `maxSamples` have been gathered.
    func start(maxSamples: Int, onCap: @escaping @Sendable () -> Void) throws
    func stop() -> [Float]
}

/// Thread-safe sample sink the audio thread appends to.
nonisolated final class LFMSampleSink: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    private let maxSamples: Int
    private var capped = false
    private let onCap: @Sendable () -> Void

    init(maxSamples: Int, onCap: @escaping @Sendable () -> Void) {
        self.maxSamples = maxSamples
        self.onCap = onCap
        samples.reserveCapacity(min(maxSamples, 16_000 * 10))
    }

    /// Appends up to the cap; calls `onCap` once when the cap is reached.
    func append(_ chunk: [Float]) {
        var reached = false
        lock.withLock {
            guard !capped else { return }
            let room = maxSamples - samples.count
            samples.append(contentsOf: chunk.prefix(room))
            if samples.count >= maxSamples { capped = true; reached = true }
        }
        if reached { onCap() }
    }

    var count: Int { lock.withLock { samples.count } }
    func take() -> [Float] { lock.withLock { let out = samples; samples = []; return out } }
}

/// The real device: `AVAudioEngine`'s input node, converted to 16 kHz mono Float32. Everything the tap touches is nonisolated
/// (the tap runs on the audio thread).
nonisolated final class LFMEngineCapture: LFMCaptureDevice, @unchecked Sendable {
    struct NoInputDevice: Error, LocalizedError {
        var errorDescription: String? { "No microphone is available. Connect or enable one and try again." }
    }

    private var engine: AVAudioEngine?
    private var sink: LFMSampleSink?

    func start(maxSamples: Int, onCap: @escaping @Sendable () -> Void) throws {
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else { throw NoInputDevice() }
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(LFMAudioEngine.inputSampleRate),
                                            channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat) else { throw NoInputDevice() }
        let sink = LFMSampleSink(maxSamples: maxSamples, onCap: onCap)
        let ratio = outFormat.sampleRate / inFormat.sampleRate
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
            guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
            var supplied = false
            var failure: NSError?
            converter.convert(to: out, error: &failure) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return buffer
            }
            guard failure == nil, let channel = out.floatChannelData?[0], out.frameLength > 0 else { return }
            sink.append(Array(UnsafeBufferPointer(start: channel, count: Int(out.frameLength))))
        }
        engine.prepare()
        do { try engine.start() } catch {
            input.removeTap(onBus: 0)
            throw error
        }
        self.engine = engine
        self.sink = sink
    }

    func stop() -> [Float] {
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        engine = nil
        let samples = sink?.take() ?? []
        sink = nil
        return samples
    }
}

/// A weak reference that may cross to another thread (the audio thread's cap callback hops back to the main actor with it).
nonisolated final class LFMWeakBox<T: AnyObject>: @unchecked Sendable {
    weak var value: T?
    init(_ value: T) { self.value = value }
}

/// Hold-to-talk recorder: the rules around a capture device and the permission.
@MainActor
@Observable
final class LFMMicRecorder {
    enum Phase: Equatable { case idle, recording }
    enum Begin: Equatable {
        /// Capturing now.
        case started
        /// The press ended while the system prompt was up (first use); nothing was started. The next press records.
        case pressEnded
        /// Permission denied or restricted: show the plain message with the System Settings button. Not an error.
        case denied
        /// The device couldn't start (no microphone, engine failure); the sentence to show.
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    /// Seconds captured so far, for the visible countdown (updated about 10 times a second while recording).
    private(set) var elapsed: Double = 0
    /// True once a press found the permission denied; the view shows the plain message while it is true.
    private(set) var permissionDenied = false

    /// The cap on one recording: the engine's refusal limit, so a hold that reaches it is sent, not refused.
    let maxSeconds: Double
    private let permission: any LFMMicPermission
    private let makeDevice: () -> any LFMCaptureDevice
    private var device: (any LFMCaptureDevice)?
    private var ticker: Task<Void, Never>?
    private var startedAt: Date?
    /// Called when the cap is reached and the capture was stopped for the user: the samples gathered. The view-model sends them.
    var onAutoStop: (([Float]) -> Void)?

    init(maxSeconds: Double = LFMAudioEngine.maxUserTurnSeconds,
         permission: any LFMMicPermission = LFMSystemMicPermission(),
         makeDevice: @escaping () -> any LFMCaptureDevice = { LFMEngineCapture() }) {
        self.maxSeconds = maxSeconds
        self.permission = permission
        self.makeDevice = makeDevice
    }

    var isRecording: Bool { phase == .recording }
    var secondsLeft: Double { max(maxSeconds - elapsed, 0) }

    /// A press began. `stillHeld` is asked after the permission prompt returns, so a user who let go while the prompt was up
    /// does not start a recording nobody is holding.
    func begin(stillHeld: () -> Bool = { true }) async -> Begin {
        guard phase == .idle else { return .started }
        switch permission.status() {
        case .denied:
            permissionDenied = true
            return .denied
        case .notDetermined:
            // The only place the system prompt can appear: the first hold-to-talk press.
            let granted = await permission.request()
            guard granted else { permissionDenied = true; return .denied }
            guard stillHeld() else { return .pressEnded }
        case .authorized:
            break
        }
        permissionDenied = false
        let device = makeDevice()
        do {
            let target = LFMWeakBox(self)
            try device.start(maxSamples: Int(maxSeconds * Double(LFMAudioEngine.inputSampleRate))) {
                Task { @MainActor in target.value?.capReached() }
            }
        } catch {
            return .failed(MLXErrorGuard.sentence(for: error))
        }
        self.device = device
        phase = .recording
        elapsed = 0
        startedAt = Date()
        ticker = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, let startedAt = self.startedAt else { return }
                self.elapsed = min(Date().timeIntervalSince(startedAt), self.maxSeconds)
            }
        }
        return .started
    }

    /// The press ended: stop and return what was captured (nil if nothing was recording).
    func end() -> [Float]? {
        guard let device else { return nil }
        let samples = device.stop()
        reset()
        return samples
    }

    /// The press was cancelled or the view went away: stop and drop everything.
    func cancel() { _ = end() }

    /// Re-read after the user came back from System Settings.
    func refreshPermission() {
        if permission.status() != .denied { permissionDenied = false }
    }

    private func capReached() {
        guard phase == .recording, let device else { return }
        let samples = device.stop()
        reset()
        onAutoStop?(samples)
    }

    private func reset() {
        ticker?.cancel()
        ticker = nil
        device = nil
        startedAt = nil
        phase = .idle
    }
}
