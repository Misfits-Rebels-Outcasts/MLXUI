import Foundation
import Testing
import MLX
@testable import MLXUI

/// MC1-9 — the window loop (`SAMAudioEngine.runWindows`) tested through its seam, with a fake in place of
/// `SAMAudio.separate`: no weights, runs in the normal suite. Covers Stop between windows, the per-window seeded noise
/// and the crossfade invariant. (That the loop never touches the global `MLXRandom` state is by construction —
/// `windowNoise` passes an explicit key — and is not asserted here, because other suites share the process's global
/// state and a parallel draw would make such a test flaky.)
struct SAMAudioWindowLoopTests {
    private static let hop = 4
    private static let channels = 3

    private final class Calls: @unchecked Sendable {
        private let lock = NSLock()
        private var indices: [Int] = []
        private var noises: [[Float]] = []
        private var shapes: [[Int]] = []
        func add(index: Int, noise: [Float], shape: [Int]) {
            lock.lock(); indices.append(index); noises.append(noise); shapes.append(shape); lock.unlock()
        }
        var count: Int { lock.lock(); defer { lock.unlock() }; return indices.count }
        var allNoises: [[Float]] { lock.lock(); defer { lock.unlock() }; return noises }
        var allShapes: [[Int]] { lock.lock(); defer { lock.unlock() }; return shapes }
    }

    private final class TaskBox: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<Void, Error>?
        private var pending = false
        func set(_ t: Task<Void, Error>) { lock.lock(); task = t; let p = pending; lock.unlock(); if p { t.cancel() } }
        func cancel() { lock.lock(); pending = true; let t = task; lock.unlock(); t?.cancel() }
    }

    private static func signal(_ count: Int) -> [Float] {
        (0..<count).map { 0.5 + 0.4 * Float(sin(Double($0) * 0.05)) + Float($0 % 7) * 0.01 }
    }

    /// Runs the loop over `samples` with `fake` and returns the two tracks.
    private static func run(_ samples: [Float], chunk: Int = 1_000, overlap: Int = 300, seed: UInt64 = 7,
                            fake: @escaping SAMAudioEngine.WindowSeparator,
                            progress: @Sendable @escaping (Double) -> Void = { _ in }) async throws -> (isolated: [Float], other: [Float]) {
        let plan = SAMAudioEngine.chunkPlan(sampleCount: samples.count, chunkSamples: chunk, overlapSamples: overlap)
        return try await SAMAudioEngine.runWindows(
            sampleCount: samples.count, plan: plan, samples: samples, seed: seed, hopLength: hop, noiseChannels: channels,
            separateWindow: fake, progress: progress)
    }

    private static func recording(_ calls: Calls) -> SAMAudioEngine.WindowSeparator {
        { window, index, noise in
            let values = try MLXErrorGuard.read(noise, as: Float.self)
            calls.add(index: index, noise: values, shape: noise.shape)
            return .init(target: window, residual: window)
        }
    }

    @Test func stopAfterTheFirstWindowEndsTheRunWithoutCallingTheSecond() async throws {
        let calls = Calls()
        let box = TaskBox()
        let samples = Self.signal(5_000)
        let task = Task {
            _ = try await Self.run(samples, fake: Self.recording(calls)) { _ in box.cancel() }
        }
        box.set(task)
        do {
            try await task.value
            Issue.record("a cancelled run must not return audio")
        } catch is CancellationError {
            #expect(calls.count == 1, "Stop is checked before every window: only window 1 may have run")
        }
    }

    @Test func progressComesAfterEachWindowAndRisesToOne() async throws {
        final class Log: @unchecked Sendable { let lock = NSLock(); var v: [Double] = []
            func add(_ x: Double) { lock.lock(); v.append(x); lock.unlock() } }
        let log = Log(), calls = Calls()
        let samples = Self.signal(3_000)
        _ = try await Self.run(samples, fake: Self.recording(calls)) { log.add($0) }
        #expect(log.v.count == calls.count && calls.count > 1)
        #expect(log.v == log.v.sorted() && log.v.last == 1.0)
    }

    @Test func eachWindowGetsItsOwnNoiseAndTheSameSeedReproducesIt() async throws {
        let samples = Self.signal(3_000)
        let a = Calls(), b = Calls(), c = Calls()
        _ = try await Self.run(samples, seed: 11, fake: Self.recording(a))
        _ = try await Self.run(samples, seed: 11, fake: Self.recording(b))
        _ = try await Self.run(samples, seed: 12, fake: Self.recording(c))
        #expect(a.count > 2)
        #expect(a.allNoises == b.allNoises, "the same seed gives the same noise for every window")
        let first = a.allNoises
        for i in 1..<first.count { #expect(first[i] != first[0], "window \(i) must not reuse window 0's noise") }
        #expect(a.allNoises != c.allNoises, "a different seed gives different noise")
    }

    @Test func noiseHasTheShapeAndDTypeSeparateExpects() async throws {
        // 3,000 samples, 1,000-sample windows every 700: the last window is shorter than the others.
        let samples = Self.signal(3_000)
        let calls = Calls()
        let plan = SAMAudioEngine.chunkPlan(sampleCount: 3_000, chunkSamples: 1_000, overlapSamples: 300)
        _ = try await Self.run(samples, fake: Self.recording(calls))
        let shapes = calls.allShapes
        #expect(shapes.count == plan.windows.count)
        for (shape, window) in zip(shapes, plan.windows) {
            #expect(shape == [1, (window.count + Self.hop - 1) / Self.hop, Self.channels])
        }
        let noise = try SAMAudioEngine.windowNoise(seed: 1, index: 0, frames: 5, channels: 6)
        #expect(noise.dtype == .float32, "the library's own default is float32, cast to the projection's dtype inside separate")
    }

    @Test func aPassThroughFakeReproducesTheInputAcrossTheSeams() async throws {
        // 4,321 samples is not a multiple of the 700-sample hop.
        let samples = Self.signal(4_321)
        let out = try await Self.run(samples, fake: { window, _, _ in .init(target: window, residual: window) })
        #expect(out.isolated.count == samples.count && out.other.count == samples.count)
        let worst = zip(out.isolated, samples).map { abs($0 - $1) }.max() ?? 0
        let worstOther = zip(out.other, samples).map { abs($0 - $1) }.max() ?? 0
        #expect(worst < 1e-5 && worstOther < 1e-5, "the crossfade must sum to one at every seam (\(worst), \(worstOther))")
    }

    @Test func tracksAreTrimmedOrPaddedToTheInputLength() async throws {
        let samples = Self.signal(2_500)
        let longer = try await Self.run(samples, fake: { window, _, _ in .init(target: window + [9, 9, 9], residual: window + [9]) })
        #expect(longer.isolated.count == samples.count && longer.other.count == samples.count)
        let shorter = try await Self.run(samples, fake: { window, _, _ in .init(target: Array(window.dropLast(3)), residual: Array(window.dropLast(3))) })
        #expect(shorter.isolated.count == samples.count && shorter.other.count == samples.count)
        let single = try await Self.run(Self.signal(500), fake: { window, _, _ in .init(target: Array(window.dropLast(2)), residual: window + [1, 1]) })
        #expect(single.isolated.count == 500 && single.other.count == 500, "the one-window path pads and trims too")
    }
}
