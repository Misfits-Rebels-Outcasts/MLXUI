import Foundation
import MLX
import MLXNN

/// Audio-codec decode seam (MG-ENG2) so the engine's run loop is testable with a stub.
protocol MusicGenAudioDecoding: Sendable {
    func decode(_ codes: MLXArray) -> MLXArray
}

extension MusicGenEncodec: MusicGenAudioDecoding {}

/// MusicGen generation engine (MG-ENG4) — text → 32 kHz mono WAV. Assembles the T5 encoder,
/// causal decoder, and EnCodec decoder (all from MG-ENG1…ENG3) and runs the autoregressive
/// generate loop from `ml-explore/mlx-examples/musicgen/musicgen.py`.
///
/// Pipeline:
/// 1. Load `decoder.safetensors` → `MusicGenDecoder`, `t5.safetensors` → `MusicGenT5Encoder`,
///    `encodec/` → `MusicGenEncodec` (progress 0 → 0.15);
/// 2. Tokenize the prompt (fast `tokenizer.json`, 512 cap) → T5 encode (0.15 → 0.25);
/// 3. Autoregressive loop (`maxSteps`, default 250 ≈ 5 s at 50 frames/s) with classifier-free
///    guidance (conditional + unconditional batch) and top-k sampling per codebook, applying
///    the MusicGen **delay pattern** (0.25 → 0.85);
/// 4. Undo the delay → EnCodec decode → mono waveform (0.85 → 0.95);
/// 5. Build the WAV via `AudioWriter` (0.95 → 1.0).
nonisolated enum MusicGenEngine {

    // MARK: - Weight loading (A2 pattern, mirroring FluxEngine)

    private nonisolated static func loadWeights(
        _ model: Module, _ dir: URL, file: String
    ) async throws {
        let url = dir.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StageError.engineFailure(stage: "MusicGen", underlying: MusicGenEngineError.missingWeights(file))
        }
        let raw = try MLX.loadArrays(url: url)
        try model.update(parameters: ModuleParameters.unflattened(raw), verify: .none)
        eval(model)
    }

    // MARK: - Public entry point

    /// Load the installed model and generate a 32 kHz mono `AudioBuffer` from `prompt`.
    /// `maxSteps` controls duration (250 ≈ 5 s).
    nonisolated static func generate(
        prompt: String,
        modelID: String,
        maxSteps: Int = 250,
        topK: Int = 250,
        temperature: Float = 1.0,
        guidanceScale: Float = 3.0,
        progress: @Sendable (Double) -> Void
    ) async throws -> AudioBuffer {
        let dir = ModelStore.shared.directory(forModelID: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw StageError.modelNotInstalled(id: modelID)
        }

        let encoder = MusicGenT5Encoder()
        let decoder = MusicGenDecoder()
        let encodec = try MusicGenEncodec(modelDirectory: dir)
        try await loadWeights(encoder, dir, file: "t5.safetensors")
        try await loadWeights(decoder, dir, file: "decoder.safetensors")
        progress(0.15)

        let tokenizer = try await MusicGenTokenizer(url: dir)
        let tokenIDs = tokenizer.encode(prompt)
        progress(0.18)

        return try await generate(
            tokenIDs: tokenIDs, encoder: encoder, decoder: decoder, encodec: encodec,
            maxSteps: maxSteps, topK: topK, temperature: temperature,
            guidanceScale: guidanceScale, progress: progress)
    }

    // MARK: - Core run loop (testable seam)

    /// Run the autoregressive loop over already-tokenized ids with loaded components. Exposed
    /// (internal) so the unit test can drive it with synthetic weights + a stub codec.
    nonisolated static func generate(
        tokenIDs: [Int32],
        encoder: MusicGenT5Encoder,
        decoder: MusicGenDecoder,
        encodec: any MusicGenAudioDecoding,
        maxSteps: Int,
        topK: Int,
        temperature: Float,
        guidanceScale: Float,
        progress: @Sendable (Double) -> Void
    ) async throws -> AudioBuffer {
        // 2. T5 encode → project to decoder dim.
        let ids = MLXArray(tokenIDs).expandedDimensions(axis: 0)   // [1, seq]
        let t5Hidden = encoder(ids)                                 // [1, seq, 768]
        let conditioning = decoder.encToDecProj(t5Hidden)           // [1, seq, 1024]
        eval(conditioning)
        progress(0.25)

        // 3. Autoregressive loop with CFG. The decoder has no KV cache, so feed the full prefix
        //    each step (causal mask handles history); the current row sits at absolute position
        //    `offset` (learned positional embedding start = 0).
        let codebooks = decoder.config.codebooks
        let bos: Int32 = Int32(decoder.config.bosTokenID)
        let uncond = MLXArray.zeros(like: conditioning)
        // `mx.concatenate([cond, zeros], axis=0)` — batch 2 along the existing dim (NOT
        // `MLX.stacked`, which would insert a NEW axis and give [2,1,seq,hidden]).
        let both = MLX.concatenated([conditioning, uncond], axis: 0)   // [2, seq, 1024]

        var rows: [MLXArray] = [
            MLXArray([Int32](repeating: bos, count: codebooks))
        ]

        for offset in 0 ..< maxSteps {
            let prefix = MLX.stacked(rows, axis: 0).expandedDimensions(axis: 0)   // [1, offset+1, codebooks]
            let batched = MLX.tiled(prefix, repetitions: [2, 1, 1])               // [2, offset+1, codebooks]
            let logits = decoder(batched, conditioning: both)       // [2, T, vocab, codebooks]
            let last = logits[0..., -1, 0...]                       // [2, vocab, codebooks]
            let cond = last[0 ..< 1]
            let uncondLogits = last[1 ..< 2]
            let guided = uncondLogits + (cond - uncondLogits) * guidanceScale   // [1, vocab, codebooks]
            let sample = guided[0]                                  // [vocab, codebooks]
            let tokens = Self.topKSampling(sample, topK: topK, temperature: temperature)   // [codebooks]

            // Delay pattern: codebook k is BOS before step k and after step maxSteps-1+k.
            var next = tokens
            if offset + 1 < codebooks {
                next = MLX.where(Self.codebookIndexMask(codebooks) .> MLXArray(Int32(offset)), MLXArray(bos), next)
            }
            let tailCount = maxSteps - offset
            if tailCount < codebooks && tailCount > 0 {
                let keepFrom = codebooks - tailCount
                next = MLX.where(Self.codebookIndexMask(codebooks) .< MLXArray(Int32(keepFrom)), MLXArray(bos), next)
            }
            rows.append(next)

            if offset % 10 == 0 {
                eval(rows)
                progress(0.25 + 0.60 * Double(offset + 1) / Double(maxSteps))
            }
        }
        eval(rows)
        progress(0.85)

        // 4. Undo the delay (reference epilogue) then decode.
        let codes = Self.undoDelay(rows, codebooks: codebooks)      // [T', codebooks] int32
        let waveform = encodec.decode(codes)
        eval(waveform)
        progress(0.95)

        // 5. WAV output.
        let samples = waveform.asArray(Float.self)
        return AudioBuffer(samples: samples, sampleRate: 32_000)
    }

    /// Top-k sampling over the vocab axis. No `argsort` in MLX Swift, so we threshold the
    /// **sorted** probabilities at the top-k boundary (reference `top_k_sampling`). `logits` is
    /// `[vocab, codebooks]`; returns `[codebooks]` sampled token ids.
    private nonisolated static func topKSampling(_ logits: MLXArray, topK: Int, temperature: Float) -> MLXArray {
        let probs = softmax(logits * (1.0 / temperature), axis: 0)   // [vocab, codebooks]
        let sortedProbs = sorted(probs, axis: 0)                     // ascending
        let k = min(max(topK, 1), logits.dim(0))
        let threshold = sortedProbs[logits.dim(0) - k, 0...].expandedDimensions(axis: 0)   // [1, codebooks]
        let masked = MLX.where(probs .>= threshold, probs, MLXArray(0.0))
        let logProbs = log(masked + 1e-12)
        return MLXRandom.categorical(logProbs, axis: 0)              // [codebooks]
    }

    /// `[codebooks]` int32 vector `[0, 1, 2, 3]` for the delay masks.
    private nonisolated static func codebookIndexMask(_ codebooks: Int) -> MLXArray {
        MLXArray(Array(0 ..< codebooks).map(Int32.init))
    }

    /// Undo the MusicGen delay pattern (reference `generate` epilogue). For each codebook `k`,
    /// shift its rows left by `k` (row `p` takes the value of row `p+k`), then drop the leading
    /// BOS row and trim the tail. Returns `[T', codebooks]` int32.
    nonisolated static func undoDelay(_ rows: [MLXArray], codebooks: Int) -> MLXArray {
        let T = rows.count
        var orig = rows.map { $0.asArray(Int32.self) }               // [T][codebooks]
        var shifted = orig
        for k in 0 ..< codebooks {
            for p in 0 ..< (T - codebooks) {
                shifted[p][k] = orig[p + k][k]
            }
        }
        // `audio_seq[:, 1 : -codebooks + 1]` → rows [1, T-codebooks).
        let finalRows = Array(shifted[1 ..< (T - codebooks + 1)])
        let flat = finalRows.flatMap { $0 }
        return MLXArray(flat, [finalRows.count, codebooks])
    }
}

nonisolated enum MusicGenEngineError: LocalizedError {
    case missingWeights(String)

    var errorDescription: String? {
        switch self {
        case .missingWeights(let file): return "MusicGen weights missing: \(file). Reinstall the model."
        }
    }
}
