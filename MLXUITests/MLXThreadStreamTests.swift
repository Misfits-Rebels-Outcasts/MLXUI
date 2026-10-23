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

// MARK: - Real-model, app-context probes (gated on installed weights: TEST_RUNNER_MLXUI_TEST_MODELS_DIR)

/// The shape the Run UI gives the engines: awaited from a main-actor `Task`, inside `MLXErrorGuard.guardedRun`
/// (`TTSRunModel.synthesize`). Both audio engines now run their body on `MLXAudioThread`; these fail if that stops
/// working (or if the engine reads a failed array again — that would segfault the host, which is the old crash).
@MainActor
struct AudioEngineMainActorProbeTests {
    nonisolated private static let modelsDir = ProcessInfo.processInfo.environment["MLXUI_TEST_MODELS_DIR"]

    nonisolated static func installed(_ slug: String) -> URL? {
        guard let root = modelsDir else { return nil }
        let url = URL(fileURLWithPath: root).appendingPathComponent(slug)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// The raw behaviour the engines work around: `generate` awaited from the main actor returns a lazy array that the
    /// main thread cannot evaluate on mlx-swift 0.32. Stops at the error — never reads the array.
    @Test(.enabled(if: AudioEngineMainActorProbeTests.installed("mlx-community--Kokoro-82M-bf16") != nil))
    func kokoroGenerateFromTheMainActorRaisesTheStreamError() async throws {
        let dir = try #require(Self.installed("mlx-community--Kokoro-82M-bf16"))
        let model = try await KokoroModel.fromModelDirectory(dir, textProcessor: KokoroMultilingualProcessor())
        let audio = try await model.generate(text: "Local models run entirely on this Mac.", voice: "af_heart",
                                             refAudio: nil, refText: nil, language: nil)
        #expect(audio.ndim == 1, "generate returns a lazy 1-D float vector")
        do {
            try withError { eval(audio) }
            Issue.record("expected the stream error on mlx-swift 0.32 (if it no longer raises, the audit in 2026-557 changes: `MLXAudioThread` can go)")
        } catch let error as MLXError {
            guard case .caught(let message) = error else { return }
            #expect(message.contains("no Stream"), "\(message)")
        }
    }

    /// `KokoroEngine.synthesize` from the main actor: real audio, no MLX error recorded anywhere, and the same call twice.
    @Test(.enabled(if: AudioEngineMainActorProbeTests.installed("mlx-community--Kokoro-82M-bf16") != nil))
    func kokoroSynthesizeFromTheMainActorReturnsAudio() async throws {
        let dir = try #require(Self.installed("mlx-community--Kokoro-82M-bf16"))
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        for text in ["Local models run entirely on this Mac.", "Hello"] {
            let audio = try await MLXErrorGuard.guardedRun {
                try await KokoroEngine.synthesize(text, voice: KokoroEngine.defaultVoice, speed: 1.0, modelDirectory: dir)
            }
            #expect(audio.samples.count > 1000, "\(text)")
            #expect(audio.sampleRate == 24_000)
            #expect(audio.samples.contains { $0 != 0 }, "not silence")
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }

    /// `MLXAudioTTSEngine.synthesize` from the main actor, for each installed non-Kokoro family.
    @Test(arguments: ["mlx-community--Qwen3-TTS-12Hz-0.6B-Base-bf16", "mlx-community--chatterbox-fp16"])
    func mlxAudioTTSSynthesizeFromTheMainActorReturnsAudio(slug: String) async throws {
        guard let dir = Self.installed(slug) else {
            print("AUDIOPROBE \(slug): not installed — skipped")
            return
        }
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        // chatterbox clones a voice and refuses to run without a reference clip; a Kokoro-made sentence stands in.
        var reference: AudioBuffer?
        if slug.contains("chatterbox") {
            guard let kokoro = Self.installed("mlx-community--Kokoro-82M-bf16") else {
                print("AUDIOPROBE \(slug): needs Kokoro for a reference clip — skipped")
                return
            }
            reference = try await KokoroEngine.synthesize("This is a short reference voice sample.", voice: KokoroEngine.defaultVoice,
                                                          speed: 1.0, modelDirectory: kokoro)
        }
        let audio = try await MLXErrorGuard.guardedRun { [reference] in
            try await MLXAudioTTSEngine.synthesize("Local models run entirely on this Mac.", voice: nil,
                                                   referenceAudio: reference, modelDirectory: dir)
        }
        print("AUDIOPROBE \(slug): \(audio.samples.count) samples @ \(audio.sampleRate)")
        #expect(audio.samples.count > 1000)
        #expect(audio.samples.contains { $0 != 0 }, "not silence")
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }
}
