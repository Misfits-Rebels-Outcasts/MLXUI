import Foundation
import Testing
@testable import MLXUI

/// MC1-R — writes the listening sets for the owner's `review.html`. Gated: the R5 opt-in, the weights (see
/// `SAMAudioRealLoad.directory`) **and** `MLXUI_TEST_SAMAUDIO_REVIEW_OUT=<dir>`; it writes nothing into the repo. Needs a host that
/// can write the output folder (the unsandboxed smoke build; the default sandboxed host cannot write `/private/tmp`).
///
/// Set 2 is the 65 s deterministic voice-over-music clip (the windowed test's and the MC1-9b table's), run through the shipped path
/// (`SAMAudioEngine.separate`, "speech", seed 1, the shipped 30 s windows with 3 s overlap). Set 3 is optional:
/// `MLXUI_TEST_SAMAUDIO_REVIEW_IN=<audio file>` (the owner's recording, decoded the way the Run path does:
/// `SAMAudioInput.read`, then resampled to the model's rate); it is never copied into the repo and has no reference to score.
/// Lines: `SAMREVIEW set=… file=… bytes=… seconds=…` and, for the synthetic set, `SAMREVIEW-CORR …`.
enum SAMAudioReviewGate {
    static func setting(_ name: String) -> String? {
        SAMAudioRealLoad.env[name] ?? SAMAudioRealLoad.env["TEST_RUNNER_" + name]
    }
    static var enabled: Bool { SAMAudioRealLoad.canRun && setting("MLXUI_TEST_SAMAUDIO_REVIEW_OUT") != nil }
}

@Suite(.enabled(if: SAMAudioReviewGate.enabled), .serialized)
struct SAMAudioReviewArtifactsTests {
    static func setting(_ name: String) -> String? { SAMAudioReviewGate.setting(name) }

    private func write(_ samples: [Float], rate: Int, set: String, name: String, out: URL) throws {
        let url = out.appendingPathComponent(name)
        try AudioWriter.writeWAV(AudioBuffer(samples: samples, sampleRate: rate), to: url)
        let bytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        print(String(format: "SAMREVIEW set=%@ file=%@ bytes=%d seconds=%.1f", set, name, bytes, Double(samples.count) / Double(rate)))
    }

    @Test func writesTheSyntheticClipSet() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        let out = URL(fileURLWithPath: try #require(Self.setting("MLXUI_TEST_SAMAUDIO_REVIEW_OUT")), isDirectory: true)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try await SAMAudioRealLoad.withClip(seconds: 65) { clip in
            let result = try await SAMAudioEngine.separate(clip.mix, description: "speech", modelDirectory: dir, seed: 1)
            try write(clip.mix, rate: 48_000, set: "synthetic-65s", name: "voice-over-music-input.wav", out: out)
            try write(result.isolated, rate: result.sampleRate, set: "synthetic-65s", name: "voice-over-music-isolated.wav", out: out)
            try write(result.other, rate: result.sampleRate, set: "synthetic-65s", name: "voice-over-music-everything-else.wav", out: out)
            print(String(format: "SAMREVIEW-CORR set=synthetic-65s seed=1 isolatedVsSpeech=%.3f otherVsSpeech=%.3f windows=3 chunk=%.0f overlap=%.0f",
                         SAMAudioRealLoad.correlation(result.isolated, clip.speech),
                         SAMAudioRealLoad.correlation(result.other, clip.speech),
                         SAMAudioEngine.chunkSeconds, SAMAudioEngine.overlapSeconds))
        }
    }

    @Test(.enabled(if: SAMAudioReviewGate.setting("MLXUI_TEST_SAMAUDIO_REVIEW_IN") != nil))
    func writesTheOwnersRecordingSet() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        let out = URL(fileURLWithPath: try #require(Self.setting("MLXUI_TEST_SAMAUDIO_REVIEW_OUT")), isDirectory: true)
        let input = URL(fileURLWithPath: try #require(Self.setting("MLXUI_TEST_SAMAUDIO_REVIEW_IN")))
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let rate = SAMAudioEngine.sampleRate(in: dir)
        let decoded = try SAMAudioInput.read(input)
        let mono = try AudioResampler.resample(decoded, to: rate).samples
        let seconds = Double(mono.count) / Double(rate)
        print(String(format: "SAMREVIEW-IN seconds=%.1f", seconds))
        guard seconds <= SAMAudioEngine.maxDurationSeconds else {
            Issue.record("the recording is \(seconds) s, over the 30-minute limit: not run")
            return
        }
        let result = try await SAMAudioEngine.separate(mono, description: "speech", modelDirectory: dir, seed: 1)
        try write(mono, rate: rate, set: "owner-recording", name: "owner-recording-input.wav", out: out)
        try write(result.isolated, rate: result.sampleRate, set: "owner-recording", name: "owner-recording-isolated.wav", out: out)
        try write(result.other, rate: result.sampleRate, set: "owner-recording", name: "owner-recording-everything-else.wav", out: out)
    }
}
