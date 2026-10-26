import Foundation
import MLX
import Testing
@testable import MLXUI

/// MP6-3a: the LFM2.5-Audio engine's logic with a mock backend (no weights), the stages, the SDK claim and the rate proofs.
/// The real-weights tests are in `LFMAudioRealLoadTests.swift`.

/// Records every call and answers from a script. Reference type so the engine's calls are visible to the test.
final class MockLFMBackend: LFMAudioBackend, @unchecked Sendable {
    struct Call { var prompt: LFMPrompt; var mode: LFMGenerationMode; var sampling: LFMSampling }
    private let lock = NSLock()
    private var _calls: [Call] = []
    private var _decoded: [[[Int32]]] = []
    var onGenerate: (@Sendable (Int) throws -> Void)?
    var reply: @Sendable (Int) -> LFMGenerated
    var samplesPerFrame = 4
    init(reply: @escaping @Sendable (Int) -> LFMGenerated = { _ in LFMGenerated(text: "hello", audioFrames: [[1, 2]]) }) { self.reply = reply }
    var calls: [Call] { lock.withLock { _calls } }
    var decoded: [[[Int32]]] { lock.withLock { _decoded } }

    func generate(_ prompt: LFMPrompt, mode: LFMGenerationMode, sampling: LFMSampling) async throws -> LFMGenerated {
        let n = lock.withLock { () -> Int in _calls.append(Call(prompt: prompt, mode: mode, sampling: sampling)); return _calls.count }
        try onGenerate?(n)
        return reply(n)
    }

    func decode(frames: [[Int32]]) async throws -> [Float] {
        lock.withLock { _decoded.append(frames) }
        return [Float](repeating: 0.1, count: frames.count * samplesPerFrame)
    }
}

enum LFMFixture {
    /// The pinned repo's config.json (RSI/notes/modelpack/hf/mlx-community_LFM2.5-Audio-1.5B-6bit @ c875bcf), verbatim.
    static let pinnedConfig = #"""
{
    "architectures": [
        "Lfm2AudioForConditionalGeneration"
    ],
    "codebook_weight": "log",
    "codebooks": 8,
    "depthformer": {
        "layers": 6,
        "dim": 1024,
        "tie": true
    },
    "encoder": {
        "feat_in": 128,
        "feat_out": -1,
        "n_layers": 17,
        "d_model": 512,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "causal_downsampling": false,
        "reduction": null,
        "reduction_position": null,
        "reduction_factor": 1,
        "ff_expansion_factor": 4,
        "self_attention_model": "rel_pos",
        "n_heads": 8,
        "att_context_size": [
            -1,
            -1
        ],
        "xscaling": false,
        "untie_biases": true,
        "pos_emb_max_len": 5000,
        "conv_kernel_size": 9,
        "conv_norm_type": "batch_norm",
        "conv_context_size": null,
        "dropout": 0.1,
        "dropout_pre_encoder": 0.1,
        "dropout_emb": 0,
        "dropout_att": 0.1
    },
    "interleaved_n_audio": 12,
    "interleaved_n_text": 6,
    "lfm": {
        "_name_or_path": "LiquidAI/LFM2-1.2B",
        "architectures": [
            "Lfm2ForCausalLM"
        ],
        "block_auto_adjust_ff_dim": true,
        "block_dim": 2048,
        "block_ff_dim": 12288,
        "block_ffn_dim_multiplier": 1,
        "block_mlp_init_scale": 1,
        "block_multiple_of": 256,
        "block_norm_eps": 1e-05,
        "block_out_init_scale": 1,
        "block_use_swiglu": true,
        "block_use_xavier_init": true,
        "conv_L_cache": 3,
        "conv_bias": false,
        "conv_dim": 2048,
        "conv_dim_out": 2048,
        "conv_use_xavier_init": true,
        "eos_token_id": 7,
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "layer_types": [
            "conv",
            "conv",
            "full_attention",
            "conv",
            "conv",
            "full_attention",
            "conv",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv"
        ],
        "max_position_embeddings": 128000,
        "model_type": "lfm2",
        "norm_eps": 1e-05,
        "num_attention_heads": 32,
        "num_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "rope_theta": 1000000,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "use_pos_enc": true,
        "vocab_size": 65536
    },
    "model_type": "lfm_audio",
    "preprocessor": {
        "sample_rate": 16000,
        "normalize": "per_feature",
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "features": 128,
        "n_fft": 512,
        "log": true,
        "frame_splicing": 1,
        "dither": 1e-05,
        "pad_to": 0,
        "pad_value": 0.0
    },
    "quantization": {
        "group_size": 64,
        "bits": 6,
        "mode": "affine"
    },
    "quantization_config": {
        "group_size": 64,
        "bits": 6,
        "mode": "affine"
    },
    "semantic_codebook_factor": 100,
    "tie_audio_embeddings": false
}
"""#
}

struct LFMAudioEngineTests {

    private let silence16k = { (seconds: Double) in [Float](repeating: 0.01, count: Int(seconds * 16_000)) }

    // MARK: Rates (settled from the pinned source)

    @Test func inputRateIsSixteenKHzAndOutputIsTwentyFourFromThePinnedConfig() throws {
        let rates = try LFMAudioRealBackend.rates(inConfig: Data(LFMFixture.pinnedConfig.utf8))
        #expect(rates.input == 16_000)
        #expect(rates.output == 24_000)
        #expect(rates.input == LFMAudioEngine.inputSampleRate)
        #expect(rates.output == LFMAudioEngine.outputSampleRate)
    }

    /// The library's preprocessor ignores the rate it is told, so a 24 kHz claim changes nothing: the engine must resample to 16 kHz.
    @Test func theLibraryPreprocessorIgnoresTheSampleRateItIsTold() throws {
        let data = Data(LFMFixture.pinnedConfig.utf8)
        let at16 = try LFMAudioRealBackend.melFrameCount(inConfig: data, sampleCount: 16_000, claimedRate: 16_000)
        let at24 = try LFMAudioRealBackend.melFrameCount(inConfig: data, sampleCount: 16_000, claimedRate: 24_000)
        #expect(at16 == at24)
        #expect(at16 == 16_000 / 160 + 1, "10 ms hop at 16 kHz, centred")
    }

    // MARK: Pure helpers

    @Test func windowsCoverTheClipWithTheLastShorter() {
        #expect(LFMAudioEngine.windows(sampleCount: 0, windowSamples: 10).isEmpty)
        #expect(LFMAudioEngine.windows(sampleCount: 25, windowSamples: 10) == [0..<10, 10..<20, 20..<25])
        #expect(LFMAudioEngine.windows(sampleCount: 10, windowSamples: 10) == [0..<10])
    }

    @Test func transcriptsJoinWithASingleSpaceAndSkipEmpties() {
        #expect(LFMAudioEngine.joinTranscripts([" one ", "", "two\n", "  "]) == "one two")
    }

    @Test func textChunksRespectParagraphsAndTheCharacterCap() {
        let long = String(repeating: "A sentence of moderate length. ", count: 30)
        let chunks = LFMAudioEngine.textChunks("First paragraph.\n\n" + long)
        #expect(chunks.first == "First paragraph.")
        #expect(chunks.count > 2)
        #expect(chunks.allSatisfy { $0.unicodeScalars.count <= LFMAudioEngine.ttsChunkCharacters })
        #expect(LFMAudioEngine.textChunks("  \n ").isEmpty)
    }

    // MARK: Speech to text

    @Test func sttRunsOneCallPerWindowEachWithItsOwnBudget() async throws {
        let backend = MockLFMBackend { LFMGenerated(text: " piece\($0) ", audioFrames: []) }
        let progress = LFMProgressBox()
        // 70 s -> windows of 30 s, 30 s, 10 s
        let text = try await LFMAudioEngine.transcribe(silence16k(70), backend: backend) { progress.add($0) }
        #expect(text == "piece1 piece2 piece3")
        let calls = backend.calls
        #expect(calls.count == 3)
        #expect(calls.allSatisfy { $0.mode == .interleaved })
        #expect(calls.allSatisfy { $0.sampling.maxNewTokens == LFMAudioEngine.sttMaxNewTokens }, "the budget is per window, not per session")
        #expect(calls.allSatisfy { $0.sampling.temperature == 0 }, "transcription is greedy")
        #expect(progress.values == [1.0 / 3, 2.0 / 3, 1.0])
        // each prompt is one user turn: the window's audio, then the instruction
        guard case let .audio(first) = calls[0].prompt.turns[0].parts[0], case let .audio(last) = calls[2].prompt.turns[0].parts[0] else {
            Issue.record("prompt shape"); return
        }
        #expect(first.count == 30 * 16_000)
        #expect(last.count == 10 * 16_000)
        #expect(calls[0].prompt.turns[0].parts[1] == .text("Transcribe the audio."))
        #expect(calls[0].prompt.assistantStartsAudio == false)
    }

    @Test func sttRefusesEmptyAudio() async {
        await #expect(throws: LFMAudioEngine.EngineError.emptyAudio) {
            _ = try await LFMAudioEngine.transcribe([], backend: MockLFMBackend())
        }
    }

    // MARK: Text to speech

    @Test func ttsSpeaksEveryChunkWithItsOwnBudgetAndJoinsTheAudio() async throws {
        let backend = MockLFMBackend { _ in LFMGenerated(text: "", audioFrames: [[1, 2, 3], [4, 5, 6]]) }
        let text = String(repeating: "A sentence of moderate length. ", count: 30)
        let chunkCount = LFMAudioEngine.textChunks(text).count
        #expect(chunkCount > 1)
        let samples = try await LFMAudioEngine.synthesize(text, style: "Use a UK male voice.", backend: backend)
        let calls = backend.calls
        #expect(calls.count == chunkCount)
        #expect(calls.allSatisfy { $0.mode == .sequential && $0.prompt.assistantStartsAudio })
        #expect(calls.allSatisfy { $0.sampling.maxNewTokens == LFMAudioEngine.ttsMaxNewTokens })
        #expect(calls[0].prompt.turns[0] == .system("Perform TTS. Use a UK male voice."))
        #expect(samples.count == chunkCount * 2 * backend.samplesPerFrame)
    }

    @Test func ttsWithoutStyleUsesTheBareInstruction() async throws {
        let backend = MockLFMBackend { _ in LFMGenerated(text: "", audioFrames: [[1]]) }
        _ = try await LFMAudioEngine.synthesize("Hello there.", backend: backend)
        #expect(backend.calls[0].prompt.turns[0] == .system("Perform TTS."))
        #expect(backend.calls[0].prompt.turns[1] == .user(.text("Hello there.")))
    }

    @Test func ttsRefusesEmptyTextAndSilentOutput() async {
        await #expect(throws: LFMAudioEngine.EngineError.emptyText) {
            _ = try await LFMAudioEngine.synthesize("  \n ", backend: MockLFMBackend())
        }
        await #expect(throws: LFMAudioEngine.EngineError.noAudio) {
            _ = try await LFMAudioEngine.synthesize("Hello.", backend: MockLFMBackend { _ in LFMGenerated(text: "x", audioFrames: []) })
        }
    }

    // MARK: Voice turn and memory

    @Test func aVoiceTurnReplaysHistoryAndStreamsTextThenAudioSlices() async throws {
        var conversation = LFMConversation()
        conversation.record(userSamples: [0.1, 0.2], assistantText: "first answer")
        let frames = (0..<60).map { [Int32($0), 1] }
        let backend = MockLFMBackend { _ in LFMGenerated(text: " the reply ", audioFrames: frames) }
        let events = LFMEventBox()
        let reply = try await LFMAudioEngine.voiceTurn(userSamples: [0.3, 0.4, 0.5], conversation: conversation, backend: backend) { events.add($0) }
        #expect(reply.text == "the reply")
        // 60 frames in slices of 25: 25, 25, 10
        #expect(backend.decoded.map(\.count) == [25, 25, 10])
        #expect(reply.audioSampleCount == 60 * backend.samplesPerFrame)
        let seen = events.values
        #expect(seen.first == .text("the reply"))
        #expect(seen.dropFirst().allSatisfy { if case .audio = $0 { return true } else { return false } })
        #expect(seen.count == 4)
        let call = try #require(backend.calls.first)
        #expect(call.mode == .interleaved)
        #expect(call.sampling.maxNewTokens == LFMAudioEngine.voiceMaxNewTokens)
        // system, past user audio, past assistant text, new user audio
        #expect(call.prompt.turns.map(\.role) == ["system", "user", "assistant", "user"])
        #expect(call.prompt.turns[1].parts == [.audio([0.1, 0.2])])
        #expect(call.prompt.turns[2].parts == [.text("first answer")])
        #expect(call.prompt.turns[3].parts == [.audio([0.3, 0.4, 0.5])])
        #expect(call.prompt.turns[0] == .system("Respond with interleaved text and speech audio."))
    }

    @Test func aFailedOrRefusedTurnLeavesTheMemoryAlone() async {
        let conversation = LFMConversation()
        let backend = MockLFMBackend()
        await #expect(throws: LFMAudioEngine.LimitError.self) {
            _ = try await LFMAudioEngine.voiceTurn(userSamples: [Float](repeating: 0, count: 121 * 16_000), conversation: conversation,
                                                   backend: backend) { _ in }
        }
        #expect(backend.calls.isEmpty, "refused before the model is touched")
        #expect(conversation.turns.isEmpty)
    }

    @Test func conversationMemoryIsCappedResettableAndNeverEncodable() {
        var conversation = LFMConversation()
        for i in 0..<(LFMConversation.maxTurns + 3) { conversation.record(userSamples: [Float(i)], assistantText: "a\(i)") }
        #expect(conversation.turns.count == LFMConversation.maxTurns)
        #expect(conversation.turns.first?.assistantText == "a3", "the oldest exchanges are dropped")
        conversation.reset()
        #expect(conversation.turns.isEmpty)
        // RAM only: persisting it would need a new type and a decision.
        #expect((LFMConversation.self as Any.Type) is Encodable.Type == false)
        #expect((LFMConversation.Turn.self as Any.Type) is Encodable.Type == false)
    }

    // MARK: Cancellation and errors

    @Test func aCancelledRunStopsBetweenWindowsAndIsNotAnEngineError() async throws {
        let backend = MockLFMBackend()
        let task = Task {
            try await LFMAudioEngine.transcribe([Float](repeating: 0, count: 16_000 * 100), backend: backend)
        }
        backend.onGenerate = { n in if n == 1 { task.cancel() } }
        // the first call may already be running when the cancel lands; either way no window after it runs
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(backend.calls.count <= 1)
    }

    @Test func withBackendMapsErrorsLikeTheOtherEngines() async throws {
        struct Boom: Error {}
        let dir = URL(fileURLWithPath: "/tmp/lfm-model")
        // a library error becomes a stage engine failure
        do {
            _ = try await LFMAudioEngine.withBackend(dir, factory: { _ in throw Boom() }) { _ in 1 }
            Issue.record("no throw")
        } catch let error as StageError {
            guard case .engineFailure = error else { Issue.record("\(error)"); return }
        }
        // an engine error passes through unchanged
        await #expect(throws: LFMAudioEngine.EngineError.noAudio) {
            _ = try await LFMAudioEngine.withBackend(dir, factory: { _ in MockLFMBackend() }) { _ in throw LFMAudioEngine.EngineError.noAudio }
        }
        // cancellation is cancellation
        await #expect(throws: CancellationError.self) {
            _ = try await LFMAudioEngine.withBackend(dir, factory: { _ in MockLFMBackend() }) { _ in throw CancellationError() }
        }
    }

    @Test func theBackendIsBuiltAndUsedOffTheMainThread() async throws {
        let flags = try await LFMAudioEngine.withBackend(URL(fileURLWithPath: "/tmp/lfm-model"), factory: { _ in
            ThreadProbeBackend(created: onMainThread())
        }) { backend in
            let probe = try #require(backend as? ThreadProbeBackend)
            _ = try await probe.generate(LFMPrompt(turns: []), mode: .interleaved, sampling: LFMAudioEngine.sttSampling)
            return [probe.createdOnMain, probe.usedOnMain]
        }
        #expect(flags == [false, false])
    }

    // MARK: Real backend, no weights

    @Test func theLoaderNamesMissingFilesAndNeverRequiresTheRootTokenizerCheckpoint() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("lfm-files-\(UUID().uuidString)")
        let model = root.appendingPathComponent("mlx-community--LFM2.5-Audio-1.5B-6bit")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: model.appendingPathComponent("audio_detokenizer"), withIntermediateDirectories: true)
        #expect(Set(LFMAudioRealBackend.missingFiles(in: model)) == Set(LFMAudioRealBackend.requiredFiles))
        await #expect(throws: LFMAudioEngine.EngineError.noWeights(model.lastPathComponent)) {
            _ = try await LFMAudioRealBackend.load(model)
        }
        for name in LFMAudioRealBackend.requiredFiles {
            FileManager.default.createFile(atPath: model.appendingPathComponent(name).path, contents: Data("x".utf8))
        }
        // exactly the installed layout the selector produces: no tokenizer-*.safetensors anywhere
        #expect(LFMAudioRealBackend.missingFiles(in: model).isEmpty)
        #expect(!LFMAudioRealBackend.requiredFiles.contains { $0.contains("tokenizer-") })
    }

    // MARK: Stages

    @Test func theAsrStageResamplesToSixteenKHzBeforeTheEngine() async throws {
        let backend = MockLFMBackend { _ in LFMGenerated(text: "spoken words", audioFrames: []) }
        let stage = LFMAudioStages.asr(modelDirectory: URL(fileURLWithPath: "/tmp/lfm-model"), factory: { _ in backend })
        #expect(stage.accepts == .audio && stage.produces == .text)
        let input = AudioBuffer(samples: [Float](repeating: 0.05, count: 44_100 * 2), sampleRate: 44_100)
        let out = try await stage.run(.audio(input)) { _ in }
        guard case let .text(text) = out else { Issue.record("not text"); return }
        #expect(text == "spoken words")
        guard case let .audio(heard) = backend.calls[0].prompt.turns[0].parts[0] else { Issue.record("no audio"); return }
        #expect(abs(heard.count - 2 * 16_000) < 64, "two seconds at 16 kHz, not 44.1")
    }

    @Test func theTtsStageReturnsTwentyFourKHzAudio() async throws {
        let backend = MockLFMBackend { _ in LFMGenerated(text: "", audioFrames: [[1], [2]]) }
        let stage = LFMAudioStages.tts(modelDirectory: URL(fileURLWithPath: "/tmp/lfm-model"), factory: { _ in backend })
        #expect(stage.accepts == .text && stage.produces == .audio)
        let out = try await stage.run(.text("Hello.")) { _ in }
        guard case let .audio(buffer) = out else { Issue.record("not audio"); return }
        #expect(buffer.sampleRate == 24_000)
        #expect(buffer.samples.count == 2 * backend.samplesPerFrame)
    }

    // MARK: SDK

    @Test func theSdkClaimsOnlyLfmAudioVoiceEntries() throws {
        let sdk = LFMAudioSDK()
        let voice = makeEntry(family: "LFM2.5-Audio", displayName: "LFM2.5-Audio 1.5B", modelType: .voice)
        #expect(sdk.claim(voice) == .exact)
        #expect(sdk.claim(makeEntry(family: "LFM2.5-Audio", modelType: .asr)) == .exact, "the family override routes it to .voice")
        #expect(sdk.claim(makeEntry(family: "Kokoro", modelType: .tts)) == .no)
        #expect(sdk.claim(makeEntry(family: "Whisper", modelType: .asr)) == .no)
        #expect(sdk.claim(makeEntry(family: "sam-audio", modelType: .asr)) == .no)
        #expect(sdk.claim(makeEntry(family: "LFM2-VL", modelType: .vision)) == .no)
        let stage = try sdk.makeStage(for: voice, config: StageConfig())
        #expect(stage.accepts == .audio && stage.produces == .text)
        #expect(LFMAudioModule.descriptor.modelTypes == [.voice])
    }
}

/// `Thread.isMainThread` is unavailable from async code; the C call is the same question.
func onMainThread() -> Bool { pthread_main_np() != 0 }

final class ThreadProbeBackend: LFMAudioBackend, @unchecked Sendable {
    let createdOnMain: Bool
    private(set) var usedOnMain = true
    init(created: Bool) { createdOnMain = created }
    func generate(_ prompt: LFMPrompt, mode: LFMGenerationMode, sampling: LFMSampling) async throws -> LFMGenerated {
        usedOnMain = onMainThread()
        return LFMGenerated(text: "", audioFrames: [])
    }
    func decode(frames: [[Int32]]) async throws -> [Float] { [] }
}

final class LFMProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _values: [Double] = []
    func add(_ v: Double) { lock.withLock { _values.append(v) } }
    var values: [Double] { lock.withLock { _values } }
}

final class LFMEventBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _values: [LFMVoiceEvent] = []
    func add(_ v: LFMVoiceEvent) { lock.withLock { _values.append(v) } }
    var values: [LFMVoiceEvent] { lock.withLock { _values } }
}
