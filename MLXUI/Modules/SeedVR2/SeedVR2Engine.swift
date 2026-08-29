import CoreGraphics
import Foundation
import MLX
import MLXNN

// MARK: - Engine errors

enum SeedVR2EngineError: Error, LocalizedError {
    case missingFile(String)
    case emptyWeights(String)
    case imagePrepFailed
    case decodeOutputFailed

    var errorDescription: String? {
        switch self {
        case .missingFile(let f):    return "SeedVR2: required file not found: \(f)"
        case .emptyWeights(let f):   return "SeedVR2: no arrays in '\(f)'"
        case .imagePrepFailed:       return "SeedVR2: failed to prepare input image"
        case .decodeOutputFailed:    return "SeedVR2: VAE decode produced nil CGImage"
        }
    }
}

// MARK: - Latent creator

private enum SeedVR2LatentCreator {
    /// Gaussian noise latent: [1, 16, 1, H, W].
    static func noiseLatents(seed: UInt64, height: Int, width: Int) -> MLXArray {
        let key = MLXRandom.key(seed)
        return MLXRandom.normal([1, 16, 1, height, width], key: key).asType(.bfloat16)
    }

    /// Condition: concat encoded with ones mask → [1, 17, 1, H, W].
    static func condition(_ encoded: MLXArray) -> MLXArray {
        let ones = MLXArray.ones([1, 1, 1, encoded.dim(3), encoded.dim(4)]).asType(encoded.dtype)
        return MLX.concatenated([encoded, ones], axis: 1)
    }
}

// MARK: - Euler scheduler (1-step)

private struct SeedVR2EulerScheduler {
    let timesteps: [Float]  // typically [1000.0]

    init(numSteps: Int = 1) {
        // Single-step: timestep = 1000 (maximum noise level)
        timesteps = (0 ..< numSteps).map { Float(1000 - $0 * (1000 / max(numSteps, 1))) }
    }

    /// Euler step: denoised = latents - (t/1000) * noisePred.
    func step(noisePred: MLXArray, timestepIdx: Int, latents: MLXArray) -> MLXArray {
        let t = timesteps[timestepIdx]
        let tNorm = MLXArray(t / 1000.0).asType(latents.dtype)
        return latents - tNorm * noisePred
    }
}

// MARK: - Engine

/// Orchestrates SeedVR2 3B single-step diffusion super-resolution.
///
/// Pipeline:
///  1. Bicubic-upsample LR image to target HR size
///  2. VAE encode → latent condition  (+ scale by 0.9152)
///  3. Noise latent (same spatial size)
///  4. Single-step Euler denoise via the int8 transformer
///  5. VAE decode → HR image
nonisolated enum SeedVR2Engine {

    // MARK: - Public entry point

    nonisolated static func upscale(
        image:   CGImage,
        scale:   Int = 2,
        seed:    UInt64? = nil,
        modelID: String,
        progress: @Sendable (Double) -> Void
    ) async throws -> CGImage {
        let dir = ModelStore.shared.directory(forModelID: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw StageError.modelNotInstalled(id: modelID)
        }

        // 1. Bicubic-upsample input to target HR size; then pad to multiple of 16
        //    (VAE stride=8, patchSize=2 → combined stride=16; odd latent dims crash patchIn).
        progress(0.02)
        guard let hrImage = resample(image, scale: scale) else {
            throw SeedVR2EngineError.imagePrepFailed
        }
        let trueW = hrImage.width, trueH = hrImage.height
        let padW = roundUp(trueW, to: 16), padH = roundUp(trueH, to: 16)
        let paddedHR = (padW == trueW && padH == trueH) ? hrImage : padImage(hrImage, toWidth: padW, height: padH)
        guard let padded = paddedHR, let input = cgImageToTensor(padded) else {
            throw SeedVR2EngineError.imagePrepFailed
        }
        progress(0.08)

        // 2. Load text embedding (pos_emb.safetensors — fixed [58, 5120] array)
        let textEmb = try loadPosEmb(dir: dir)
        eval(textEmb)
        progress(0.12)

        // 3. VAE encode input image; release VAE weights
        let encoded = try await loadAndEncode(dir: dir, input: input, progress: { p in
            progress(0.12 + p * 0.20)
        })
        Memory.clearCache(); await Task.yield()
        progress(0.32)

        // 4. Transformer denoise; release transformer weights
        let denoised = try await loadAndDenoise(
            dir: dir, encoded: encoded, textEmb: textEmb,
            seed: seed ?? UInt64(Date().timeIntervalSince1970 * 1000),
            progress: { p in progress(0.32 + p * 0.48) })
        Memory.clearCache(); await Task.yield()
        progress(0.82)

        // 5. VAE decode
        let pixels = try await loadAndDecode(dir: dir, latent: denoised, progress: { p in
            progress(0.82 + p * 0.14)
        })
        progress(0.98)

        guard let cgFull = tensorToCGImage(pixels) else {
            throw SeedVR2EngineError.decodeOutputFailed
        }
        // Crop padding back to true target size
        let cgOut = (cgFull.width == trueW && cgFull.height == trueH)
            ? cgFull
            : cropImage(cgFull, toWidth: trueW, height: trueH) ?? cgFull
        progress(1.0)
        return cgOut
    }

    // MARK: - Stage helpers

    private static func loadAndEncode(
        dir:      URL,
        input:    MLXArray,
        progress: @Sendable (Double) -> Void
    ) async throws -> MLXArray {
        let weights = try loadSafetensors(dir: dir, name: "vae.safetensors")
        let vae = SeedVR2VAE()
        try vae.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(vae)
        progress(0.5)

        let enc = vae.encode(input)
        eval(enc)
        progress(1.0)
        return enc
    }

    private static func loadAndDenoise(
        dir:      URL,
        encoded:  MLXArray,
        textEmb:  MLXArray,
        seed:     UInt64,
        progress: @Sendable (Double) -> Void
    ) async throws -> MLXArray {
        let weights = try loadSafetensors(dir: dir, name: "transformer.safetensors")
        let config  = (try? loadConfig(dir: dir)) ?? SeedVR2Config()
        let transformer = SeedVR2Transformer(c: config)

        // Convert Linear → QuantizedLinear for int8 layers (group_size=64)
        quantize(model: transformer) { path, _ in
            weights["\(path).scales"] != nil ? (64, 8, QuantizationMode.affine) : nil
        }
        try transformer.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(transformer)
        progress(0.25)

        // Build condition and noise
        let condition = SeedVR2LatentCreator.condition(encoded)          // [1, 17, 1, h, w]
        let h = encoded.dim(3), w = encoded.dim(4)
        var latents = SeedVR2LatentCreator.noiseLatents(seed: seed, height: h, width: w)

        let scheduler = SeedVR2EulerScheduler(numSteps: 1)
        for (idx, t) in scheduler.timesteps.enumerated() {
            let modelInput = MLX.concatenated([latents, condition], axis: 1)  // [1, 33, 1, h, w]
            let tTensor    = MLXArray(t).asType(.bfloat16)
            let noisePred  = transformer(modelInput, textEmb: textEmb, timestep: tTensor)
            eval(noisePred)
            latents = scheduler.step(noisePred: noisePred, timestepIdx: idx, latents: latents)
            eval(latents)
            progress(0.25 + 0.75 * Double(idx + 1) / Double(scheduler.timesteps.count))
            await Task.yield()
        }

        return latents
    }

    private static func loadAndDecode(
        dir:      URL,
        latent:   MLXArray,
        progress: @Sendable (Double) -> Void
    ) async throws -> MLXArray {
        let weights = try loadSafetensors(dir: dir, name: "vae.safetensors")
        let vae = SeedVR2VAE()
        try vae.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(vae)
        progress(0.5)

        let decoded = vae.decode(latent)
        eval(decoded)
        progress(1.0)
        return decoded
    }

    // MARK: - File loading

    private static func loadSafetensors(dir: URL, name: String) throws -> [String: MLXArray] {
        let url = dir.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SeedVR2EngineError.missingFile(name)
        }
        let d = try MLX.loadArrays(url: url)
        guard !d.isEmpty else { throw SeedVR2EngineError.emptyWeights(name) }
        return d
    }

    /// Loads the precomputed text embedding from pos_emb.safetensors.
    /// The file contains a single [58, 5120] array (key may vary).
    private static func loadPosEmb(dir: URL) throws -> MLXArray {
        let d = try loadSafetensors(dir: dir, name: "pos_emb.safetensors")
        // Try common key names; fall back to first value
        if let v = d["pos_emb.weight"] ?? d["weight"] ?? d["freqs"] ?? d.values.first {
            return v
        }
        throw SeedVR2EngineError.emptyWeights("pos_emb.safetensors")
    }

    private static func loadConfig(dir: URL) throws -> SeedVR2Config {
        let url = dir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(SeedVR2Config.self, from: data)
    }

    // MARK: - Image conversion

    private static func roundUp(_ n: Int, to multiple: Int) -> Int {
        (n + multiple - 1) / multiple * multiple
    }

    /// Pad image to (toWidth × height) by adding zeros to the right and bottom edges.
    private static func padImage(_ src: CGImage, toWidth w: Int, height h: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue |
                        CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.draw(src, in: CGRect(x: 0, y: 0, width: src.width, height: src.height))
        return ctx.makeImage()
    }

    /// Crop image to (toWidth × height) from the top-left corner.
    private static func cropImage(_ src: CGImage, toWidth w: Int, height h: Int) -> CGImage? {
        src.cropping(to: CGRect(x: 0, y: 0, width: w, height: h))
    }

    /// Bicubic 2× / 4× resample using CoreGraphics.
    private static func resample(_ src: CGImage, scale: Int) -> CGImage? {
        let w = src.width * scale, h = src.height * scale
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue |
                        CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(src, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    /// CGImage → [1, 3, 1, H, W] bfloat16 in [-1, 1].
    private static func cgImageToTensor(_ src: CGImage) -> MLXArray? {
        let w = src.width, h = src.height
        var bytes = [UInt8](repeating: 0, count: h * w * 4)
        guard let ctx = CGContext(
            data: &bytes, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue |
                        CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        ctx.draw(src, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Extract RGB, drop alpha: [H, W, 4] → [H, W, 3]
        var rgb = [Float](repeating: 0, count: h * w * 3)
        for i in 0 ..< h * w {
            rgb[i*3 + 0] = (Float(bytes[i*4 + 0]) / 127.5) - 1.0
            rgb[i*3 + 1] = (Float(bytes[i*4 + 1]) / 127.5) - 1.0
            rgb[i*3 + 2] = (Float(bytes[i*4 + 2]) / 127.5) - 1.0
        }
        // [H, W, 3] → [1, 3, 1, H, W]
        let t = MLXArray(rgb, [h, w, 3]).transposed(2, 0, 1)  // [3, H, W]
                  .expandedDimensions(axis: 1)                 // [3, 1, H, W]
                  .expandedDimensions(axis: 0)                 // [1, 3, 1, H, W]
        return t.asType(.bfloat16)
    }

    /// [1, 3, 1, H, W] in [-1, 1] → CGImage.
    private static func tensorToCGImage(_ t: MLXArray) -> CGImage? {
        // t: [1, 3, 1, H, W] → [H, W, 3] float
        let frame = t[0, 0..., 0, 0..., 0...]          // [3, H, W]
                    .transposed(1, 2, 0)                // [H, W, 3]
        let clipped = MLX.clip(frame * 127.5 + 127.5, min: MLXArray(0.0), max: MLXArray(255.0))
        let pixels = clipped.asType(.uint8)
        let H = pixels.dim(0), W = pixels.dim(1)
        let arr: [UInt8] = pixels.asArray(UInt8.self)
        guard let provider = CGDataProvider(data: Data(arr) as CFData) else { return nil }
        return CGImage(
            width: W, height: H,
            bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: W * 3,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent)
    }
}
