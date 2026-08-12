import Foundation
import CoreGraphics
import MLX
import MLXNN

/// Text → image engine for SDXL-Turbo, assembling the dual CLIP encoders, SDXL UNet, DDIM
/// scheduler, and VAE decoder with weights loaded from the installed model directory. Mirrors
/// `FluxEngine` (SD-ENG4).
///
/// Pipeline (from `ml-explore/mlx-examples/stable_diffusion` + diffusers v0.24.0, SDXL-Turbo):
/// 1. load the 4 component shard sets (`unet/`, `text_encoder/`, `text_encoder_2/`, `vae/`),
///    transposing the PyTorch conv weights to MLX layout via `SDWeights.sanitize`;
/// 2. tokenize the prompt with both CLIP tokenizers (77 max each) and run the encoders:
///    `[1,77,768]` + `[1,77,1280]` → concat `[1,77,2048]`; plus the **pooled** text_encoder_2
///    output (via `text_projection`) for the UNet's `text_time` conditioning;
/// 3. sample the initial latent `[1,64,64,4]` (fp16, seeded);
/// 4. **4-step DDIM denoise loop** (no CFG → single batch, no negative prompt);
/// 5. VAE decode `latent/0.13025` → NCHW image → `CGImage`.
enum SDXLTurboEngine {
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    nonisolated static let numSteps = 4

    // MARK: - Weight loading (A2 pattern + PyTorch conv transpose)

    private nonisolated static func safetensorsBytes(_ dir: URL, subdirs: [String]) -> Int64 {
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
            await Task.yield()
        }
        guard !raw.isEmpty else {
            throw StageError.engineFailure(stage: "SDXL-Turbo", underlying: SDXLTurboEngineError.emptyWeights(subdir))
        }
        return SDWeights.sanitize(raw)
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

    /// Generate a 512×512 image from `prompt`. Reports `0...1` progress across load (0–0.5,
    /// weighted by shard bytes), conditioning (0.5–0.55), the 4-step DDIM loop (0.55–0.95), and
    /// VAE decode (0.95–1.0). `seed` fixes the initial latent for reproducible output (nil = random).
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
        let unet = SDUNet(config: SDUNetConfig.sdxl)
        let clip = SDCLIPEncoder()
        let openClip = SDOpenCLIPEncoder()
        let vae = SDVAE()
        let totalBytes = safetensorsBytes(dir, subdirs: ["unet", "text_encoder_2", "text_encoder", "vae"])
        let loadProgress: @Sendable (Double) -> Void = { progress(0.5 * $0) }
        try await loadWeights(unet, dir, subdir: "unet", totalBytes: totalBytes, progress: loadProgress)
        try await loadWeights(openClip, dir, subdir: "text_encoder_2", totalBytes: totalBytes, progress: loadProgress)
        try await loadWeights(clip, dir, subdir: "text_encoder", totalBytes: totalBytes, progress: loadProgress)
        try await loadWeights(vae, dir, subdir: "vae", totalBytes: totalBytes, progress: loadProgress)
        progress(0.5)

        // 2. Tokenize + encode conditioning.
        let clipVocab = try loadJSONDict(dir.appendingPathComponent("tokenizer/vocab.json"))
        let clipRanks = try loadMergeRanks(dir.appendingPathComponent("tokenizer/merges.txt"))
        let openVocab = try loadJSONDict(dir.appendingPathComponent("tokenizer_2/vocab.json"))
        let openRanks = try loadMergeRanks(dir.appendingPathComponent("tokenizer_2/merges.txt"))

        let clipIds = MLXArray(SDCLIPTokenizer.tokenize(prompt, bpeRanks: clipRanks, vocabulary: clipVocab).map(Int32.init))
            .expandedDimensions(axis: 0)
        let openIds = MLXArray(SDCLIPTokenizer.tokenize(prompt, bpeRanks: openRanks, vocabulary: openVocab).map(Int32.init))
            .expandedDimensions(axis: 0)

        let clipSeq = clip(clipIds)            // [1, 77, 768]
        let openSeq = openClip(openIds)        // [1, 77, 1280]
        let encoderStates = concatenated([clipSeq, openSeq], axis: 2)   // [1, 77, 2048]
        let pooled = openClip.pooled(openIds)  // [1, 1280] — text_time added conditioning
        eval(clipSeq, openSeq, encoderStates, pooled)
        progress(0.55)

        // 3. Initial latent (NHWC, fp16 to match the weights).
        if let seed { MLXRandom.seed(seed) }
        var latent = MLXRandom.normal([1, 64, 64, 4], dtype: .float16)
        // 6 time ids: [orig_h, orig_w, crop_h, crop_w, target_h, target_w]; 512×512 output.
        let timeIDs = MLXArray([Float(512), 512, 0, 0, 512, 512]).reshaped([1, 6])

        // 4. Denoise loop (SDXL-Turbo: no CFG, single batch, 4 DDIM steps).
        let scheduler = SDScheduler()
        let timesteps = scheduler.timesteps(numSteps: numSteps)
        for i in 0 ..< numSteps {
            let t = timesteps[i]
            let tPrev = timesteps[i + 1]
            let noise = unet(
                sample: latent,
                timestep: MLXArray([t]).asType(latent.dtype),
                encoderHiddenStates: encoderStates,
                textEmbeds: pooled,
                timeIDs: timeIDs)
            eval(noise)
            latent = scheduler.step(noisePred: noise, sample: latent, t: t, tPrev: tPrev)
            eval(latent)
            progress(0.55 + 0.40 * Double(i + 1) / Double(numSteps))
        }

        // 5. VAE decode → NCHW image.
        let decoded = vae.decode(latent.transposed(0, 3, 1, 2))   // [1, 4, 64, 64] → [1, 3, 512, 512]
        progress(0.98)

        return try cgImage(from: decoded)
    }

    // MARK: - Pixel conversion (mirrors FluxEngine)

    /// mflux `_denormalize`: the VAE decoder emits values roughly in `[-1, 1]`; map to `[0, 1]`
    /// with `x/2 + 0.5` then clamp.
    nonisolated static func denormalize(_ x: MLXArray) -> MLXArray {
        clip(x / 2 + 0.5, min: 0, max: 1)
    }

    private nonisolated static func cgImage(from nchw: MLXArray) throws -> CGImage {
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
            throw StageError.engineFailure(stage: "SDXL-Turbo", underlying: SDXLTurboEngineError.imageConversionFailed)
        }
        return image
    }

    private nonisolated static func loadJSONDict(_ url: URL) throws -> [String: Int] {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: Int].self, from: data)
    }

    private nonisolated static func loadMergeRanks(_ url: URL) throws -> [SDCLIPTokenizer.Bigram: Int] {
        let merges = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n")
            .dropFirst()   // "#version: 0.2" header
            .map { String($0) }
        return Dictionary(uniqueKeysWithValues: merges.enumerated().map {
            let parts = $0.element.split(separator: " ")
            return (SDCLIPTokenizer.Bigram(String(parts[0]), String(parts[1])), $0.offset)
        })
    }
}

nonisolated enum SDXLTurboEngineError: LocalizedError {
    case emptyWeights(String)
    case imageConversionFailed

    var errorDescription: String? {
        switch self {
        case .emptyWeights(let subdir): return "No safetensors found under \(subdir)/"
        case .imageConversionFailed: return "Could not convert the decoded image to a CGImage."
        }
    }
}
