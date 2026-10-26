import Foundation
import AVFoundation

/// What the voice view-model needs from a reply player (a fake stands in under test, so no test makes a sound).
@MainActor
protocol LFMReplyPlaying: AnyObject {
    var onDrained: (() -> Void)? { get set }
    func append(_ samples: [Float])
    func finish()
    func stop()
}

/// Plays the voice reply as it arrives: 24 kHz mono slices scheduled back to back on one `AVAudioPlayerNode`, so the first slice
/// starts playing while later ones are still being delivered. All work is a few buffer copies on the main actor; nothing blocks.
/// Stopping tears the engine down, which also drops slices not yet played (a new hold-to-talk, Stop, a closed view).
@MainActor
final class LFMReplyPlayer: LFMReplyPlaying {
    private var engine: AVAudioEngine?
    private var node: AVAudioPlayerNode?
    private let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(LFMAudioEngine.outputSampleRate),
                                       channels: 1, interleaved: false)
    private var scheduled = 0
    private var finished = 0
    /// Called on the main actor when every scheduled slice has played (and `finish()` was called).
    var onDrained: (() -> Void)?
    private var closing = false
    /// Bumped by `stop()`, so a completion callback from a torn-down engine can't count against the next one.
    private var generation = 0

    var isPlaying: Bool { engine != nil }

    func append(_ samples: [Float]) {
        guard !samples.isEmpty, let format,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { return }
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if engine == nil && !start(format) { return }
        scheduled += 1
        let generation = self.generation
        let target = LFMWeakBox(self)
        node?.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
            Task { @MainActor in target.value?.sliceFinished(generation) }
        }
    }

    /// No more slices are coming; `onDrained` fires once the queue has played out.
    func finish() {
        closing = true
        if scheduled == finished { drained() }
    }

    func stop() {
        generation += 1
        node?.stop()
        engine?.stop()
        node = nil
        engine = nil
        scheduled = 0
        finished = 0
        closing = false
    }

    private func start(_ format: AVAudioFormat) -> Bool {
        let engine = AVAudioEngine()
        let node = AVAudioPlayerNode()
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        do { try engine.start() } catch { return false }
        node.play()
        self.engine = engine
        self.node = node
        return true
    }

    private func sliceFinished(_ generation: Int) {
        guard engine != nil, generation == self.generation else { return }
        finished += 1
        if closing && finished == scheduled { drained() }
    }

    private func drained() {
        stop()
        onDrained?()
    }
}
