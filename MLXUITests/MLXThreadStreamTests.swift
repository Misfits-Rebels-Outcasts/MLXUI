import Foundation
import Testing
import Cmlx
import MLXAudioTTS
@testable import MLX
@testable import MLXUI

// WW2-3-5 — pin mlx-swift 0.32's thread-bound streams (owner ruling 2026-10-23; journal 2026-557).
//
// MLX 0.32's C++ default stream is **per thread** (`default_stream`, `stream.cpp`): the first `eval` on a thread
// creates one, with a command encoder that exists only on that thread (`get_command_encoder`, `device.cpp:927`).
// An op recorded on such a stream cannot be evaluated by another thread: `Fence::wait` / `gpu::eval` raises
// "There is no Stream(gpu, N) in current thread". 0.31 had no such error.
//
// mlx-swift's own default (`Stream.defaultStream`) is a thread-UNSAFE global stream, which is why ordinary Swift ops
// cross threads fine (control below). The hazard is a graph holding an op on a thread-bound stream — as the Kokoro
// path does (found in Part B, B10: `generate` runs off the main thread, the Run UI's `asArray` reads on the main
// actor) — and nothing in the Swift API says which results those are.
//
// What these tests pin, and what they deliberately do not do:
//   - the raw behaviour, so an mlx-swift update that changes it (upstream's `MaterializedArray`, #418) fails here;
//   - that **our** entry points (`MLXErrorGuard.guarded` / `guardedRun`) turn it into a thrown `MLXError`;
//   - nothing here READS the array after a failed `eval`: `asArray` on it dereferences a buffer-less array and
//     segfaults the test host (that is the Kokoro crash). A read site must check for the error first — a proposal
//     in the journal, not code, so not asserted.

/// MLX arrays are not `Sendable`; this carries one across the task boundary on purpose — the hazard under test.
private final class CrossThreadArray: @unchecked Sendable {
    let array: MLXArray
    init(_ array: MLXArray) { self.array = array }
}

/// A result whose matmul is recorded on the building thread's own (thread-bound) default stream, unevaluated.
private func buildOnThreadBoundStream(evaluateHere: Bool = false) async -> CrossThreadArray {
    await Task.detached(priority: .userInitiated) {
        var device = mlx_device_new()
        mlx_get_default_device(&device)
        var raw = mlx_stream_new()
        mlx_get_default_stream(&raw, device)              // the C++ default stream of THIS thread
        let a = MLXArray(0 ..< 4096).reshaped([64, 64]).asType(.float32)
        let result = matmul(a, a, stream: .stream(Stream(raw))) + 1
        if evaluateHere { eval(result) }
        return CrossThreadArray(result)
    }.value
}

@MainActor
struct MLXThreadStreamTests {

    /// The behaviour: an array whose op sits on another thread's stream cannot be evaluated here.
    @Test func evaluatingAThreadBoundStreamOpOnAnotherThreadRaises() async {
        let box = await buildOnThreadBoundStream()
        #expect(Thread.isMainThread)
        #expect(box.array.shape == [64, 64], "shape is known without evaluating")
        do {
            try withError { eval(box.array) }
            Issue.record("expected an MLX error: on mlx-swift 0.32 a thread-bound stream's op can only be evaluated on its own thread")
        } catch let error as MLXError {
            guard case .caught(let message) = error else { Issue.record("unexpected \(error)"); return }
            #expect(message.contains("no Stream"), "\(message)")
            #expect(message.contains("in current thread"), "\(message)")
        } catch {
            Issue.record("expected an MLXError, got \(error)")
        }
    }

    /// Our path: the scoped guard and the Run UI's `guardedRun` both surface it as a thrown error.
    @Test func ourGuardsTurnItIntoAThrownError() async {
        let first = await buildOnThreadBoundStream()
        do {
            _ = try await MLXErrorGuard.guarded { eval(first.array) }
            Issue.record("`guarded` should have thrown")
        } catch {
            #expect(MLXErrorGuard.isMLXError(error))
            #expect(MLXErrorGuard.sentence(for: error).contains("no Stream"))
        }
        let second = await buildOnThreadBoundStream()
        do {
            _ = try await MLXErrorGuard.guardedRun { eval(second.array) }
            Issue.record("`guardedRun` should have thrown")
        } catch {
            #expect(MLXErrorGuard.isMLXError(error))
        }
    }

    /// Control 1: ordinary Swift ops (mlx-swift's global default stream) cross threads without a problem — the
    /// hazard is specific to thread-bound streams, which is why most engines are unaffected.
    @Test func ordinaryOpsCrossThreadsFine() async {
        let box = await Task.detached(priority: .userInitiated) {
            let a = MLXArray(0 ..< 4096).reshaped([64, 64]).asType(.float32)
            return CrossThreadArray(matmul(a, a) + 1)
        }.value
        #expect(Thread.isMainThread)
        let samples = box.array.asArray(Float.self)
        #expect(samples.count == 64 * 64)
    }

    /// The fix pattern's core: an array evaluated at its producer, before it crosses, can then be read anywhere —
    /// including one built on a thread-bound stream, which raised above when left lazy.
    @Test func anArrayEvaluatedAtTheProducerIsSafeToReadOnTheMainActor() async {
        let box = await buildOnThreadBoundStream(evaluateHere: true)
        #expect(Thread.isMainThread)
        do {
            try withError { eval(box.array) }
            #expect(box.array.asArray(Float.self).count == 64 * 64)
        } catch {
            Issue.record("an evaluated array should be safe to hand over: \(error)")
        }
    }
}

// MARK: - Real-model, app-context probes (gated on an installed Kokoro: TEST_RUNNER_MLXUI_TEST_MODELS_DIR)

/// One dedicated OS thread that runs every job enqueued on it — a `TaskExecutor`. The candidate fix for library
/// calls that return lazy thread-bound results (journal 2026-557 §3): run load, `generate`, `eval` and the read
/// on this one thread, started from anywhere.
final class PinnedThreadExecutor: TaskExecutor, @unchecked Sendable {
    private let condition = NSCondition()
    private var jobs: [UnownedJob] = []
    private var thread: Thread?

    init(name: String = "pinned-mlx") {
        let thread = Thread { [unowned self] in
            while true {
                condition.lock()
                while jobs.isEmpty { condition.wait() }
                let job = jobs.removeFirst()
                condition.unlock()
                job.runSynchronously(on: asUnownedTaskExecutor())
            }
        }
        thread.name = name
        thread.start()
        self.thread = thread
    }

    func enqueue(_ job: consuming ExecutorJob) {
        let unowned = UnownedJob(job)
        condition.lock()
        jobs.append(unowned)
        condition.signal()
        condition.unlock()
    }
}

@MainActor
struct KokoroThreadingProbeTests {
    nonisolated private static let modelsDir = ProcessInfo.processInfo.environment["MLXUI_TEST_MODELS_DIR"]
    nonisolated private static let kokoroSlug = "mlx-community--Kokoro-82M-bf16"
    nonisolated static var kokoroDir: URL? {
        guard let root = modelsDir else { return nil }
        let url = URL(fileURLWithPath: root).appendingPathComponent(kokoroSlug)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// The Part B (B10) crash, as the app hits it: the Run UI awaits the engine from the main actor; `generate` builds
    /// its audio on another thread; the main thread's `eval` raises. Stops at the error — never reads the array.
    @Test(.enabled(if: KokoroThreadingProbeTests.kokoroDir != nil))
    func kokoroGenerateFromTheMainActorRaisesTheStreamError() async throws {
        let dir = try #require(Self.kokoroDir)
        let model = try await KokoroModel.fromModelDirectory(dir, textProcessor: KokoroMultilingualProcessor())
        let audio = try await model.generate(text: "Local models run entirely on this Mac.", voice: "af_heart",
                                             refAudio: nil, refText: nil, language: nil)
        #expect(audio.shape == [76200] || audio.shape.count == 1, "generate returns a lazy 1-D float vector")
        do {
            try withError { eval(audio) }
            Issue.record("expected the stream error on mlx-swift 0.32 (if it no longer raises, the audit in 2026-557 changes)")
        } catch let error as MLXError {
            guard case .caught(let message) = error else { return }
            #expect(message.contains("no Stream"), "\(message)")
        }
    }

    /// The candidate fix, verified: the real `KokoroEngine.synthesize`, awaited from the main actor, runs through a
    /// `@concurrent` wrapper under a pinned-thread executor and returns real audio (twice, same thread).
    @Test(.enabled(if: KokoroThreadingProbeTests.kokoroDir != nil))
    func theRealEngineOnAPinnedThreadReturnsAudioFromTheMainActor() async throws {
        let dir = try #require(Self.kokoroDir)
        let executor = PinnedThreadExecutor()
        let first = try await withTaskExecutorPreference(executor) {
            try await Self.synthesizeOffMain(text: "Local models run entirely on this Mac.", dir: dir)
        }
        #expect(first.samples > 0)
        #expect(first.thread == "pinned-mlx")
        let second = try await withTaskExecutorPreference(executor) {
            try await Self.synthesizeOffMain(text: "Hello", dir: dir)
        }
        #expect(second.samples > 0)
        #expect(second.thread == "pinned-mlx")
    }

    @concurrent nonisolated static func synthesizeOffMain(text: String, dir: URL) async throws -> (samples: Int, thread: String) {
        let audio = try await KokoroEngine.synthesize(text, voice: KokoroEngine.defaultVoice, speed: 1.0, modelDirectory: dir)
        return (audio.samples.count, Thread.current.name ?? "?")
    }
}
