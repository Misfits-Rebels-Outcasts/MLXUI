import Foundation
import CoreGraphics
import AVFoundation
import MLX
import MLXNN

// MARK: - Engine errors

enum WanVideoEngineError: Error, LocalizedError {
    case emptyWeights(String)
    case tokenizerNotFound
    case vaeDecodeFailed

    var errorDescription: String? {
        switch self {
        case .emptyWeights(let s):    return "WAN: no safetensors found in '\(s)'"
        case .tokenizerNotFound:      return "WAN: tokenizer/spiece.model not found"
        case .vaeDecodeFailed:        return "WAN: VAE decode produced no valid frames"
        }
    }
}

// MARK: - WanVideoEngine

/// Orchestrates WAN 2.1 T2V-1.3B inference. Sequential load strategy:
///   T5 (text_encoder/) → encode → release → DiT (transformer/) → denoise → VAE (vae/) → decode.
nonisolated enum WanVideoEngine {

    // MARK: Settings

    static let cfgScale   = 5.0 as Float
    static let targetFPS  = 16
    static let latentH    = 60     // ÷8 spatial factor → 480px height
    static let latentW    = 104    // ÷8 spatial factor → 832px width
    static let latentT    = 21     // legacy constant used by WanVideoStage
    static let maxTextLen = 512

    // MARK: Single-frame entry point (used by WanVideoStage)

    nonisolated static func generate(
        prompt:  String,
        modelID: String,
        seed:    UInt64? = nil,
        progress: @Sendable (Double) -> Void
    ) async throws -> CGImage {
        let dir = ModelStore.shared.directory(forModelID: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw StageError.modelNotInstalled(id: modelID)
        }

        // ── 1. Tokenize ──────────────────────────────────────────────────────────
        progress(0.02)
        let spModel = try loadSentencePiece(dir: dir)
        let tokenIDs = tokenize(prompt, model: spModel, maxLen: maxTextLen)
        let tokenTensor = MLXArray(tokenIDs).reshaped([1, tokenIDs.count])

        // ── 2. Load T5 + encode + release ────────────────────────────────────────
        progress(0.04)
        let t5Raw    = try await loadShards(dir: dir, subdir: "text_encoder")
        let t5Params = WanT5Encoder.sanitize(t5Raw)
        let t5       = WanT5Encoder()
        try t5.update(parameters: ModuleParameters.unflattened(t5Params), verify: .none)
        eval(t5)
        progress(0.15)

        let textCond = t5(tokenTensor)
        eval(textCond)
        Memory.clearCache()
        await Task.yield()
        progress(0.20)

        // ── 3. Load DiT + denoise loop ───────────────────────────────────────────
        let ditRaw    = try await loadShards(dir: dir, subdir: "transformer")
        let ditParams = WanDiT.sanitize(ditRaw)
        let dit       = WanDiT()
        try dit.update(parameters: ModuleParameters.unflattened(ditParams), verify: .none)
        eval(dit)
        progress(0.40)

        let rng    = seed.map { MLXRandom.key($0) } ?? MLXRandom.key(UInt64(Date().timeIntervalSince1970 * 1000))
        var latent = MLXRandom.normal([1, latentT, latentH, latentW, 16], key: rng).asType(.bfloat16)
        eval(latent)

        let numSteps  = 30
        let sigmas    = WanSampler.buildSigmas(numSteps: numSteps)
        let loopStart = 0.40, loopEnd = 0.82

        for i in 0 ..< numSteps {
            let sigmaFrom = sigmas[i]
            let sigmaTo   = sigmas[i + 1]
            let t = WanSampler.sigmaToTimestep(sigmaFrom)
            let tBatch = MLXArray([t, t]).asType(.bfloat16)

            let empty    = MLXArray.zeros(like: textCond)
            let combined = MLX.concatenated([empty, textCond], axis: 0)
            let xIn      = MLX.concatenated([latent, latent], axis: 0)

            let pred = dit(xIn, timestep: tBatch, textCond: combined)
            eval(pred)

            let uncond = pred[0, 0..., 0..., 0..., 0...]
            let cond   = pred[1, 0..., 0..., 0..., 0...]
            let noisePred = (uncond + (cond - uncond) * cfgScale).expandedDimensions(axis: 0)

            latent = WanSampler.eulerStep(x: latent, noisePred: noisePred, sigmaFrom: sigmaFrom, sigmaTo: sigmaTo)
            eval(latent)

            let p = loopStart + (loopEnd - loopStart) * Double(i + 1) / Double(numSteps)
            progress(p)
            await Task.yield()
        }

        Memory.clearCache()
        await Task.yield()

        // ── 4. Load VAE + decode ─────────────────────────────────────────────────
        progress(0.85)
        let vaeRaw    = try await loadShards(dir: dir, subdir: "vae")
        let vaeParams = WanVAEDecoder.sanitize(vaeRaw)
        let vae       = WanVAEDecoder()
        try vae.update(parameters: ModuleParameters.unflattened(vaeParams), verify: .none)
        eval(vae)
        progress(0.90)

        let decoded = vae(latent.asType(.float32))
        eval(decoded)
        progress(0.98)

        guard let image = cgImage(from: decoded, frameIndex: 0) else {
            throw WanVideoEngineError.vaeDecodeFailed
        }
        progress(1.0)
        return image
    }

    // MARK: - Full video generation (AM5)

    /// Generates all frames for a full video. latentT is derived from numFrames.
    /// Progress callback receives (fraction 0-1, phase label).
    nonisolated static func generateAll(
        prompt:          String,
        negativePrompt:  String = "",
        modelID:         String,
        numSteps:        Int    = 20,
        numFrames:       Int    = 17,
        seed:            UInt64? = nil,
        progress:        @Sendable (Double, String) -> Void
    ) async throws -> [CGImage] {
        let dir = ModelStore.shared.directory(forModelID: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw StageError.modelNotInstalled(id: modelID)
        }
        // latentT × 4 ≈ numFrames (4× temporal upsampling in the VAE)
        let latT = max(1, (numFrames + 3) / 4)

        // ── 1. Tokenize ──────────────────────────────────────────────────────────
        progress(0.02, "Tokenizing prompt…")
        let spModel     = try loadSentencePiece(dir: dir)
        let tokenIDs    = tokenize(prompt, model: spModel, maxLen: maxTextLen)
        let tokenTensor = MLXArray(tokenIDs).reshaped([1, tokenIDs.count])

        // ── 2. Load T5 + encode pos + neg + release ───────────────────────────────
        progress(0.04, "Loading text encoder (T5-XXL)…")
        let t5Raw    = try await loadShards(dir: dir, subdir: "text_encoder")
        let t5Params = WanT5Encoder.sanitize(t5Raw)
        let t5       = WanT5Encoder()
        try t5.update(parameters: ModuleParameters.unflattened(t5Params), verify: .none)
        eval(t5)
        progress(0.15, "Encoding text…")

        let textCond = t5(tokenTensor)
        eval(textCond)

        // Negative prompt: encode with T5 while it is still loaded; fall back to zeros.
        let negCond: MLXArray
        if negativePrompt.isEmpty {
            negCond = MLXArray.zeros(like: textCond)
        } else {
            let negIDs  = tokenize(negativePrompt, model: spModel, maxLen: maxTextLen)
            let negToks = MLXArray(negIDs).reshaped([1, negIDs.count])
            negCond = t5(negToks)
            eval(negCond)
        }

        Memory.clearCache()
        await Task.yield()
        progress(0.20, "Text encoded. Loading DiT…")

        // ── 3. Load DiT + denoise loop ───────────────────────────────────────────
        let ditRaw    = try await loadShards(dir: dir, subdir: "transformer")
        let ditParams = WanDiT.sanitize(ditRaw)
        let dit       = WanDiT()
        try dit.update(parameters: ModuleParameters.unflattened(ditParams), verify: .none)
        eval(dit)
        progress(0.40, "Denoising — step 0/\(numSteps)…")

        let rng    = seed.map { MLXRandom.key($0) } ?? MLXRandom.key(UInt64(Date().timeIntervalSince1970 * 1000))
        var latent = MLXRandom.normal([1, latT, latentH, latentW, 16], key: rng).asType(.bfloat16)
        eval(latent)

        let sigmas    = WanSampler.buildSigmas(numSteps: numSteps)
        let loopStart = 0.40, loopEnd = 0.82

        for i in 0 ..< numSteps {
            let sigmaFrom = sigmas[i]
            let sigmaTo   = sigmas[i + 1]
            let t         = WanSampler.sigmaToTimestep(sigmaFrom)
            let tBatch    = MLXArray([t, t]).asType(.bfloat16)

            let combined = MLX.concatenated([negCond, textCond], axis: 0)
            let xIn      = MLX.concatenated([latent, latent], axis: 0)

            let pred = dit(xIn, timestep: tBatch, textCond: combined)
            eval(pred)

            let u         = pred[0, 0..., 0..., 0..., 0...]
            let c         = pred[1, 0..., 0..., 0..., 0...]
            let noisePred = (u + (c - u) * cfgScale).expandedDimensions(axis: 0)

            latent = WanSampler.eulerStep(x: latent, noisePred: noisePred, sigmaFrom: sigmaFrom, sigmaTo: sigmaTo)
            eval(latent)

            let step = i + 1
            let p    = loopStart + (loopEnd - loopStart) * Double(step) / Double(numSteps)
            progress(p, "Denoising — step \(step)/\(numSteps)…")
            await Task.yield()
        }

        Memory.clearCache()
        await Task.yield()

        // ── 4. Load VAE + decode all frames ──────────────────────────────────────
        progress(0.85, "Loading VAE decoder…")
        let vaeRaw    = try await loadShards(dir: dir, subdir: "vae")
        let vaeParams = WanVAEDecoder.sanitize(vaeRaw)
        let vae       = WanVAEDecoder()
        try vae.update(parameters: ModuleParameters.unflattened(vaeParams), verify: .none)
        eval(vae)
        progress(0.90, "Decoding frames…")

        let decoded = vae(latent.asType(.float32))   // [1, T', H', W', 3]
        eval(decoded)
        progress(0.97, "Extracting images…")

        let nDecoded = decoded.dim(1)
        var frames: [CGImage] = []
        for f in 0 ..< nDecoded {
            if let img = cgImage(from: decoded, frameIndex: f) { frames.append(img) }
        }
        guard !frames.isEmpty else { throw WanVideoEngineError.vaeDecodeFailed }
        progress(1.0, "Done — \(frames.count) frames")
        return frames
    }

    // MARK: - MP4 assembly

    /// Assembles CGImage frames into an H.264 MP4 at `outputURL`.
    /// Uses a synchronous continuation so AVAssetWriter is never captured across suspensions.
    nonisolated static func assembleMp4(
        frames:    [CGImage],
        fps:       Int = 16,
        to outputURL: URL
    ) async throws {
        guard !frames.isEmpty else { return }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            do {
                let width  = frames[0].width
                let height = frames[0].height
                try? FileManager.default.removeItem(at: outputURL)

                let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
                let settings: [String: Any] = [
                    AVVideoCodecKey:  AVVideoCodecType.h264,
                    AVVideoWidthKey:  width,
                    AVVideoHeightKey: height,
                ]
                let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
                input.expectsMediaDataInRealTime = false
                let pbAttrs: [String: Any] = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height,
                ]
                let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                    assetWriterInput: input,
                    sourcePixelBufferAttributes: pbAttrs
                )
                writer.add(input)
                writer.startWriting()
                writer.startSession(atSourceTime: .zero)

                let scale = CMTimeScale(fps)
                for (i, frame) in frames.enumerated() {
                    while !input.isReadyForMoreMediaData {
                        Thread.sleep(forTimeInterval: 0.01)
                    }
                    let t = CMTime(value: CMTimeValue(i), timescale: scale)
                    if let pb = makePixelBuffer(from: frame, width: width, height: height) {
                        adaptor.append(pb, withPresentationTime: t)
                    }
                }
                input.markAsFinished()
                writer.finishWriting {
                    if writer.status == .failed {
                        cont.resume(throwing: writer.error ?? CocoaError(.fileWriteUnknown))
                    } else {
                        cont.resume()
                    }
                }
            } catch {
                cont.resume(throwing: error)
            }
        }
    }

    // MARK: - Helpers

    private static func makePixelBuffer(from image: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let pixelBuffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue |
                        CGImageAlphaInfo.premultipliedFirst.rawValue
        )
        ctx?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        return pixelBuffer
    }

    private static func loadShards(dir: URL, subdir: String) async throws -> [String: MLXArray] {
        let sub = dir.appendingPathComponent(subdir, isDirectory: true)
        let files = (try? FileManager.default.contentsOfDirectory(at: sub, includingPropertiesForKeys: nil))
            .map { $0.filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent } }
            ?? []
        guard !files.isEmpty else { throw WanVideoEngineError.emptyWeights(subdir) }
        var out: [String: MLXArray] = [:]
        for f in files {
            try MLX.loadArrays(url: f).forEach { out[$0.key] = $0.value }
            await Task.yield()
        }
        return out
    }

    private static func loadSentencePiece(dir: URL) throws -> FluxSentencePiece.Model {
        let url = dir.appendingPathComponent("tokenizer/spiece.model")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WanVideoEngineError.tokenizerNotFound
        }
        return try FluxSentencePiece.parse(from: url)
    }

    private static func tokenize(_ prompt: String, model: FluxSentencePiece.Model, maxLen: Int) -> [Int32] {
        var ids = FluxT5Tokenizer.tokenize(prompt, model: model).prefix(maxLen - 1)
        ids.append(1)    // EOS
        let padded = Array(ids) + Array(repeating: 0, count: max(0, maxLen - ids.count))
        return padded.map(Int32.init)
    }

    private static func cgImage(from decoded: MLXArray, frameIndex: Int) -> CGImage? {
        // decoded: [1, T, H, W, 3] in [-1, 1]  (float32)
        let frame  = decoded[0, frameIndex, 0..., 0..., 0...]   // [H, W, 3]
        let pixels = ((frame + 1.0) * 127.5).asType(.uint8)
        let H = pixels.dim(0), W = pixels.dim(1)
        let byteArray: [UInt8] = pixels.asArray(UInt8.self)
        let data = Data(byteArray)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(
            width: W, height: H,
            bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: W * 3,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}
