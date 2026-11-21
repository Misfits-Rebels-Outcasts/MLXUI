import Foundation
import MLX
import MLXNN

// MC5-3i: the ACE-Step 1.5 turbo engine on the no-LM text2music path: text -> condition encoder -> 8-step Euler/DCW sampler over the DiT -> Oobleck VAE
// -> peak-normalised 48 kHz stereo. It only composes 3c (weights), 3d (text), 3e (conditioning), 3f (DiT), 3g (sampler), 3h (VAE).
//
// STAGED RESIDENCY (one component at a time, each released, with Memory.clearCache(), before the next loads):
//   text (Qwen3-Embedding 0.88 GB + lyric table 0.31 GB) -> condition encoder (1.2 GB) -> DiT (3.15 GB) -> VAE (0.34 GB read as f32, f16 resident).
// What survives a stage is small: the text inputs, then `encoder_hidden_states` [1, L, 2048] and `context_latents` [1, T, 128], then the latents [1, T, 64].
// The DiT is released before the VAE loads (nothing of it is needed after the last step), so the busiest moment is the DiT stage.
//
// Memory.cacheLimit is process-global: it is set on entry and put back on EVERY exit (success, throw, cancel) by `Flux2KleinCacheLimitGuard`, the same
// guard the Klein engine uses (it is generic; only its name says Klein).
//
// No download path: every file is checked first and a missing one throws `AceStepWeightsError.missingFile(<relative path>)`.

nonisolated struct AceStepRequest: Sendable, Equatable {
    var caption: String
    /// Empty (after trimming) = instrumental (upstream's `[Instrumental]` / `unknown`, T18).
    var lyrics: String = ""
    /// The lyric language code upstream's `_format_lyrics` takes when there are lyrics ("en", "zh", ...).
    var language: String = "en"
    /// Seconds; the engine accepts `AceStepEngine.Configuration.minDuration ... maxDuration` (10 ... 240). Latent frames = `Int(duration * 25)`.
    var duration: Double = 60
    var seed: UInt64 = 0

    var isInstrumental: Bool { lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}

nonisolated enum AceStepPhase: Sendable, Equatable {
    case text
    case condition
    case denoise(step: Int, of: Int)
    case decode(chunk: Int, of: Int)

    /// 0...1 overall progress: text 0-5%, condition 5-10%, denoise 10-85%, decode 85-98%.
    var fraction: Double {
        switch self {
        case .text: return 0.0
        case .condition: return 0.05
        case .denoise(let s, let n): return 0.10 + 0.75 * Double(s) / Double(max(n, 1))
        case .decode(let c, let n): return 0.85 + 0.13 * Double(c) / Double(max(n, 1))
        }
    }
}

nonisolated enum AceStepComponent: String, Sendable { case text, condition, dit, vae }

nonisolated enum AceStepEvent: Sendable, Equatable {
    case cacheLimitApplied(bytes: Int)
    case loaded(AceStepComponent)
    case released(AceStepComponent)
    case phase(AceStepPhase)
}

nonisolated enum AceStepEngineError: Error, Equatable, CustomStringConvertible, LocalizedError {
    case durationOutOfRange(Double, min: Double, max: Double)
    case emptyPrompt

    var description: String {
        switch self {
        case .durationOutOfRange(let d, let lo, let hi):
            return "ACE-Step makes songs between \(Int(lo)) and \(Int(hi)) seconds long; \(d) seconds is outside that."
        case .emptyPrompt: return "ACE-Step needs a description of the music to make."
        }
    }
    var errorDescription: String? { description }
}

/// Interleaved 48 kHz stereo (L R L R ...), peak-normalised.
nonisolated struct AceStepAudio: Sendable, Equatable {
    static let sampleRate = 48_000
    static let channels = 2
    var samples: [Float]

    var frameCount: Int { samples.count / Self.channels }
    var seconds: Double { Double(frameCount) / Double(Self.sampleRate) }

    /// 44-byte header + 16-bit little-endian PCM, 2 channels, 48 kHz (the repo's `AudioWriter` is mono).
    func wavData() -> Data {
        let dataBytes = UInt32(samples.count * 2)
        var data = Data(capacity: 44 + Int(dataBytes))
        func put<T: FixedWidthInteger>(_ v: T) { var le = v.littleEndian; withUnsafeBytes(of: &le) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8)); put(UInt32(36) + dataBytes)
        data.append(contentsOf: Array("WAVE".utf8)); data.append(contentsOf: Array("fmt ".utf8))
        put(UInt32(16)); put(UInt16(1)); put(UInt16(Self.channels)); put(UInt32(Self.sampleRate))
        put(UInt32(Self.sampleRate * Self.channels * 2)); put(UInt16(Self.channels * 2)); put(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); put(dataBytes)
        for s in samples { put(Int16((max(-1, min(1, s)) * Float(Int16.max)).rounded())) }
        return data
    }

    /// The flow-facing mono mixdown (`AudioBuffer` is mono): the mean of the two channels.
    var monoMixdown: AudioBuffer {
        var mono = [Float](repeating: 0, count: frameCount)
        for i in 0 ..< frameCount { mono[i] = 0.5 * (samples[2 * i] + samples[2 * i + 1]) }
        return AudioBuffer(samples: mono, sampleRate: Self.sampleRate)
    }
}

nonisolated struct AceStepResult: @unchecked Sendable {
    var audio: AceStepAudio
    /// The sampler's final latents `[1, T, 64]` (f32), before the VAE.
    var latents: MLXArray
}

/// Where the text inputs come from. Production loads the installed Qwen3-Embedding folder per call and lets it go; a test injects tensors.
nonisolated protocol AceStepTextSource: Sendable {
    func inputs(for request: AceStepRequest) async throws -> AceStepTextInputs
}

/// The installed `Qwen3-Embedding-0.6B/` folder (tokenizer + trunk + embedding table). Everything it loads dies when `inputs` returns.
nonisolated struct AceStepInstalledTextSource: AceStepTextSource {
    let directory: URL

    func inputs(for request: AceStepRequest) async throws -> AceStepTextInputs {
        let tokenizer = try await AceStepTokenizer(directory: directory)
        let textIds = tokenizer.encodeText(caption: request.caption, duration: request.duration)
        let lyricIds = tokenizer.encodeLyrics(lyrics: request.lyrics, language: request.language, instrumental: request.isInstrumental)
        let encoder = try await AceStepTextEncoder.load(directory: directory)
        return try encoder.inputs(textIds: textIds, lyricIds: lyricIds)
    }
}

nonisolated struct AceStepEngine: Sendable {
    nonisolated struct Configuration: Sendable {
        var minDuration = 10.0
        var maxDuration = 240.0
        /// `Memory.cacheLimit` cap while a run holds it (never raises a lower limit someone set).
        var cacheLimitBytes = 1 << 30
        var vaeDType: DType = .float16
        var vaeChunk = AceStepVAE.defaultChunk
        static let standard = Configuration()
    }

    /// The model folder: `acestep-v15-turbo/`, `Qwen3-Embedding-0.6B/`, `vae/` (the `InstallManager` layout).
    let directory: URL
    var configuration = Configuration.standard
    /// nil = the installed Qwen3 folder.
    var textSource: (any AceStepTextSource)?
    var silenceLatent: @Sendable () throws -> MLXArray = { try AceStepSilenceLatent.load() }
    var readCacheLimit: @Sendable () -> Int = { Memory.cacheLimit }
    var writeCacheLimit: @Sendable (Int) -> Void = { Memory.cacheLimit = $0 }
    var observer: (@Sendable (AceStepEvent) -> Void)?

    static let turboConfig = "acestep-v15-turbo/config.json"
    static let vaeConfig = "vae/config.json"

    /// Files a run needs, relative to the model folder (the Qwen3 ones only when the engine loads its own text side).
    func requiredFiles() -> [String] {
        var files = [Self.turboConfig, AceStepWeights.turboPath, Self.vaeConfig, AceStepWeights.vaePath]
        if textSource == nil {
            files += ["\(AceStepTokenizer.folderName)/tokenizer.json", "\(AceStepTokenizer.folderName)/config.json", AceStepWeights.qwenPath]
        }
        return files
    }

    /// Everything the engine would refuse, before a byte of weights is read.
    func validate(_ request: AceStepRequest) throws {
        if request.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { throw AceStepEngineError.emptyPrompt }
        guard request.duration >= configuration.minDuration, request.duration <= configuration.maxDuration else {
            throw AceStepEngineError.durationOutOfRange(request.duration, min: configuration.minDuration, max: configuration.maxDuration)
        }
        for file in requiredFiles() where !FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path) {
            throw AceStepWeightsError.missingFile(file)
        }
    }

    /// `noise` (`[1, T, 64]`) is the injection point for tests and goldens; production draws `normal([1, T, 64], key(seed))`.
    /// Runs off the main actor (`@concurrent`): the MLX work would otherwise beachball the UI.
    @concurrent
    nonisolated func generate(_ request: AceStepRequest, noise injectedNoise: MLXArray? = nil) async throws -> AceStepResult {
        try validate(request)
        try Task.checkCancellation()
        let limits = Flux2KleinCacheLimitGuard(read: readCacheLimit, write: writeCacheLimit)
        defer { limits.restore() }
        let applied = limits.apply(cap: configuration.cacheLimitBytes)
        emit(.cacheLimitApplied(bytes: applied))

        let frames = Int(request.duration * 25)
        let turboConfigData = try Data(contentsOf: directory.appendingPathComponent(Self.turboConfig))

        // 1. Text.
        emit(.phase(.text))
        var text: AceStepTextInputs? = try await (textSource ?? AceStepInstalledTextSource(directory: directory)).inputs(for: request)
        emit(.loaded(.text))
        try Task.checkCancellation()
        Memory.clearCache()

        // 2. Condition: encoder_hidden_states and context_latents are all that leave this stage.
        emit(.phase(.condition))
        var encoderHidden: MLXArray
        var context: MLXArray
        do {
            let silence = try silenceLatent()
            let weights = try AceStepWeights.loadConditionEncoder(directory: directory)
            let encoder = try AceStepConditionEncoder(config: AceStepConditionConfig(json: turboConfigData), weights: weights, silenceLatent: silence)
            emit(.loaded(.condition))
            guard let inputs = text else { throw AceStepWeightsError.missingTensor("text inputs") }
            let packed = try encoder.encode(inputs)
            // The DiT runs on f32 activations (the oracle's stage-b dtype note); the weights stay bf16.
            encoderHidden = packed.hidden.asType(.float32)
            context = try encoder.contextLatents(frames: frames).asType(.float32)
            eval(encoderHidden, context)
        }
        text = nil
        emit(.released(.text)); emit(.released(.condition))
        Memory.clearCache()
        try Task.checkCancellation()

        // 3. Denoise.
        let noise = try injectedNoise ?? Self.noise(frames: frames, seed: request.seed)
        let latents: MLXArray
        do {
            let ditConfig = try AceStepDiTConfig(json: turboConfigData)
            let net = try AceStepDiT.load(config: ditConfig, weights: AceStepWeights.loadDecoder(directory: directory))
            emit(.loaded(.dit))
            let total = AceStepSamplerConfig().steps
            latents = try AceStepSampler.run(noise: noise.asType(.float32), config: AceStepSamplerConfig(), velocity: { x, t in
                let ts = MLXArray([t])
                return net(hidden: x, context: context, encoder: encoderHidden, timestep: ts, timestepR: ts)
            }, progress: { done, _ in emit(.phase(.denoise(step: done, of: total))) })
            eval(latents)
        }
        emit(.released(.dit))
        encoderHidden = MLXArray(0); context = MLXArray(0)
        Memory.clearCache()
        try Task.checkCancellation()

        // 4. Decode.
        let audio: MLXArray
        do {
            let vaeConfig = try AceStepVAEConfig(json: Data(contentsOf: directory.appendingPathComponent(Self.vaeConfig)))
            let vae = try AceStepVAE(config: vaeConfig, weights: AceStepWeights.loadVAEDecoder(directory: directory), dtype: configuration.vaeDType)
            emit(.loaded(.vae))
            audio = try vae.decodeChannelsFirst(latents.transposed(0, 2, 1), chunk: configuration.vaeChunk,
                                                progress: { c, n in emit(.phase(.decode(chunk: c, of: n))) })   // [1, 2, S] f32
        }
        emit(.released(.vae))
        Memory.clearCache()
        try Task.checkCancellation()

        return AceStepResult(audio: AceStepAudio(samples: try Self.finish(audio)), latents: latents)
    }

    // MARK: Pieces

    static func noise(frames: Int, seed: UInt64) throws -> MLXArray {
        MLXRandom.normal([1, frames, 64], key: MLXRandom.key(seed))
    }

    /// Upstream's `normalize_audio(target_db = -1)` (`inference.py`, enable_normalization on by default): `gain = 10^(-1/20) / peak`, applied up or down;
    /// a near-silent signal (peak < 1e-6) is left alone. Then interleave `[1, 2, S]` -> L R L R.
    static func finish(_ audio: MLXArray) throws -> [Float] {
        let peak = try MLXErrorGuard.item(abs(audio).max(), as: Float.self)
        var out = audio
        if peak >= 1e-6 { out = audio * Float(pow(10.0, -1.0 / 20.0) / Double(peak)) }
        return try MLXErrorGuard.read(out[0].transposed().reshaped(-1), as: Float.self)
    }

    private func emit(_ event: AceStepEvent) { observer?(event) }
}
