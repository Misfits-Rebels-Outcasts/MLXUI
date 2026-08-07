import Foundation
import CoreGraphics
import MLX
import MLXNN

/// Text → image engine for FLUX.1-family diffusion, assembling the sampler, layers, MMDiT,
/// T5+CLIP encoders, and VAE with weights loaded from the installed model directory. AM4h.
///
/// Pipeline (dev-style, matching `Flux-1.lite-8B`'s `guidance_embeds: true` config):
/// 1. load the 4 component shard sets (`transformer/`, `text_encoder/`, `text_encoder_2/`, `vae/`);
/// 2. tokenize the prompt (T5 → pad 512, CLIP → max 77) and run the encoders;
/// 3. sample the initial latent, pack to 2×2 patches, build text/image RoPE ids;
/// 4. **50-step flow-matching denoise loop** (dev time-shift);
/// 5. unpack the latent, VAE decode → NCHW image → `CGImage`.
enum FluxEngine {
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    private static let numTrainSteps: Float = 1000.0
    private static let guidanceStrength: Float = 4.0
    static let numSteps = 50

    // MARK: - Weight loading (A2 pattern)

    private static func safetensorsBytes(_ dir: URL, subdirs: [String]) -> Int64 {
        var total: Int64 = 0
        for sub in subdirs {
            let componentDir = dir.appendingPathComponent(sub, isDirectory: true)
            guard let files = try? FileManager.default.contentsOfDirectory(
                at: componentDir, includingPropertiesForKeys: [.fileSizeKey]) else { continue }
            for f in files where f.pathExtension == "safetensors" {
                total += Int64((try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
        }
        return total
    }

    private static func loadShardSet(
        _ dir: URL, subdir: String, totalBytes: Int64,
        progress: @Sendable (Double) -> Void
    ) async throws -> [String: MLXArray] {
        let componentDir = dir.appendingPathComponent(subdir, isDirectory: true)
        let files = try FileManager.default.contentsOfDirectory(
            at: componentDir, includingPropertiesForKeys: [.fileSizeKey])
            .filter { $0.pathExtension == "safetensors" }
        var raw: [String: MLXArray] = [:]
        var doneBytes: Int64 = 0
        for file in files {
            let fileBytes = Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            if totalBytes > 0 {
                progress(Double(doneBytes + fileBytes / 2) / Double(totalBytes))
            }
            try MLX.loadArrays(url: file).forEach { raw[$0.key] = $0.value }
            doneBytes += fileBytes
            if totalBytes > 0 {
                progress(Double(doneBytes) / Double(totalBytes))
            }
            // Yield after each shard file so the main actor can flush progress updates to the UI.
            await Task.yield()
        }
        guard !raw.isEmpty else {
            throw StageError.engineFailure(stage: "Flux", underlying: FluxEngineError.emptyWeights(subdir))
        }
        return raw
    }

    private static func loadWeights(
        _ model: Module, _ dir: URL, subdir: String, totalBytes: Int64,
        progress: @Sendable (Double) -> Void
    ) async throws {
        let raw = try await loadShardSet(dir, subdir: subdir, totalBytes: totalBytes, progress: progress)
        try model.update(parameters: ModuleParameters.unflattened(raw), verify: .none)
        eval(model)
    }

    // MARK: - Generate

    /// Generate an image from `prompt`. Reports `0...1` progress across load (0–0.5, weighted by
    /// shard bytes), conditioning (0.5–0.55), the denoise loop (0.55–0.95), and VAE decode
    /// (0.95–1.0). `seed` fixes the initial latent for reproducible output (nil = random).
    nonisolated static func generate(
        prompt: String,
        modelID: String,
        seed: UInt64? = nil,
        progress: @Sendable (Double) -> Void
    ) async throws -> CGImage {
        let dir = installedModelDirectory(id: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw StageError.modelNotInstalled(id: modelID)
        }

        // 1. Load components.
        let transformer = FluxTransformer()
        let t5 = FluxT5Encoder()
        let clip = FluxCLIPEncoder()
        let vae = FluxVAE()
        let totalBytes = safetensorsBytes(dir, subdirs: ["transformer", "text_encoder_2", "text_encoder", "vae"])
        func loadProgress(_ fraction: Double) { progress(0.5 * fraction) }
        try await loadWeights(transformer, dir, subdir: "transformer", totalBytes: totalBytes, progress: loadProgress)
        try await loadWeights(t5, dir, subdir: "text_encoder_2", totalBytes: totalBytes, progress: loadProgress)
        try await loadWeights(clip, dir, subdir: "text_encoder", totalBytes: totalBytes, progress: loadProgress)
        try await loadWeights(vae, dir, subdir: "vae", totalBytes: totalBytes, progress: loadProgress)
        progress(0.5)

        // 2. Tokenize + encode conditioning.
        let t5Model = try FluxSentencePiece.parse(from: dir.appendingPathComponent("tokenizer_2/spiece.model"))
        let clipVocab = try loadJSONDict(dir.appendingPathComponent("tokenizer/vocab.json"))
        let clipMerges = try String(contentsOf: dir.appendingPathComponent("tokenizer/merges.txt"), encoding: .utf8)
            .split(separator: "\n")
            .dropFirst()
            .map { String($0) }
        let clipRanks = Dictionary(uniqueKeysWithValues: clipMerges.enumerated().map {
            (FluxCLIPTokenizer.Bigram($0.element.split(separator: " ")[0].description,
                                      $0.element.split(separator: " ")[1].description),
             $0.offset)
        })

        let t5Tokens = FluxT5Tokenizer.tokenize(prompt, model: t5Model)
        let clipTokens = FluxCLIPTokenizer.tokenize(prompt, bpeRanks: clipRanks, vocabulary: clipVocab)

        let t5Ids = MLXArray(t5Tokens.map(Int32.init)).expandedDimensions(axis: 0)
        let clipIds = MLXArray(clipTokens.map(Int32.init)).expandedDimensions(axis: 0)

        let txt = t5(t5Ids)
        let vec = clip(clipIds)
        eval(txt, vec)
        progress(0.55)

        // 3. Initial latent + packing (latent 64×64 for a 512×512 image).
        let latentH = 64, latentW = 64
        let prior = FluxSampler.samplePrior(shape: [1, latentH, latentW, 16], dtype: .bfloat16, seed: seed)
        let xIDs = FluxTransformer.latentImageIDs(h: latentH, w: latentW)
        let txtIDs = FluxTransformer.textIDs(seqLen: txt.dim(1))
        var xT = FluxSampler.packLatents(prior, h: latentH, w: latentW)

        // 4. Denoise loop (dev flow-matching, 50 steps).
        let timesteps = FluxSampler.timesteps(
            numSteps: numSteps, imageSequenceLength: xT.dim(1), schnell: false)
        let guidance = MLXArray([guidanceStrength * numTrainSteps]).asType(.bfloat16)

        for i in 0 ..< numSteps {
            let t = timesteps[i]
            let tPrev = timesteps[i + 1]
            // Build timestep as Float to avoid float64 GPU crash (ops.cpp:456).
            let timeStep = MLXArray([Float(t) * numTrainSteps]).asType(.bfloat16)
            let pred = transformer(
                img: xT, imgIDs: xIDs, txt: txt, txtIDs: txtIDs, vec: vec,
                timestep: timeStep, guidance: guidance)
            xT = FluxSampler.step(pred: pred, xT: xT, t: t, tPrev: tPrev)
            if i % 5 == 0 { eval(xT) }
            progress(0.55 + 0.40 * Double(i + 1) / Double(numSteps))
        }
        eval(xT)

        // 5. Unpack + VAE decode → NCHW image.
        let latent = FluxSampler.unpackLatents(xT, h: latentH, w: latentW)
        let decoded = vae.decode(latent.transposed(0, 3, 1, 2))
        progress(0.98)

        return try cgImage(from: decoded)
    }

    // MARK: - Pixel conversion

    /// mflux `_denormalize`: the VAE decoder emits values roughly in `[-1, 1]`; map to `[0, 1]`
    /// with `x/2 + 0.5` then clamp. A bare `clip(x, 0, 1)` would turn mid-tones into black.
    nonisolated static func denormalize(_ x: MLXArray) -> MLXArray {
        clip(x / 2 + 0.5, min: 0, max: 1)
    }

    private static func cgImage(from nchw: MLXArray) throws -> CGImage {
        let h = nchw.dim(2), w = nchw.dim(3)
        let nhwc = nchw.transposed(0, 2, 3, 1).asType(.float32)
        let clamped = Self.denormalize(nhwc)
        let bytes = (clamped * 255).asType(.uint8).asArray(UInt8.self)

        var rgba = [UInt8](repeating: 255, count: h * w * 4)
        for p in 0 ..< (h * w) {
            rgba[p * 4 + 0] = bytes[p * 3 + 0]
            rgba[p * 4 + 1] = bytes[p * 3 + 1]
            rgba[p * 4 + 2] = bytes[p * 3 + 2]
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: colorSpace, bitmapInfo: info),
              let image = ctx.makeImage() else {
            throw StageError.engineFailure(stage: "Flux", underlying: FluxEngineError.imageConversionFailed)
        }
        return image
    }

    private static func loadJSONDict(_ url: URL) throws -> [String: Int] {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: Int].self, from: data)
    }
}

nonisolated enum FluxEngineError: LocalizedError {
    case emptyWeights(String)
    case imageConversionFailed

    var errorDescription: String? {
        switch self {
        case .emptyWeights(let subdir): return "No safetensors found under \(subdir)/"
        case .imageConversionFailed: return "Could not convert the decoded image to a CGImage."
        }
    }
}
