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

/// Orchestrates WAN 2.1 T2V-1.3B inference.
///
/// Each stage (T5, DiT, VAE) runs inside its own private helper function so that
/// the stage model + raw weight dictionaries go out of scope — and ARC frees them —
/// before the next stage loads. `Memory.clearCache()` between stages then actually
/// reclaims GPU memory rather than being a no-op while the previous model is alive.
nonisolated enum WanVideoEngine {

    // MARK: Settings

    static let cfgScale   = 5.0 as Float
    static let targetFPS  = 16
    static let latentH    = 60     // ÷8 spatial factor → 480px height
    static let latentW    = 104    // ÷8 spatial factor → 832px width
    static let latentT    = 21     // legacy constant used by WanVideoStage
    static let maxTextLen = 512

    // VAE latent normalisation constants (from vae/config.json latents_mean / latents_std)
    private static let vatMeanF: [Float] = [-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653,
                                            -0.1517,  1.5508,  0.4134, -0.0715,  0.5517, -0.3632,
                                            -0.1922, -0.9497,  0.2503, -0.2921]
    private static let vatStdF:  [Float] = [ 2.8184,  1.4541,  2.3275,  2.6558,  1.2196,  1.7708,
                                              2.6052,  2.0743,  3.2687,  2.1526,  2.8652,  1.5579,
                                              1.6382,  1.1253,  2.8251,  1.9160]

    // MARK: Single-frame entry point (used by WanVideoStage)

    nonisolated static func generate(
        prompt:   String,
        modelID:  String,
        seed:     UInt64? = nil,
        progress: @Sendable (Double) -> Void
    ) async throws -> CGImage {
        let dir = ModelStore.shared.directory(forModelID: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw StageError.modelNotInstalled(id: modelID)
        }
        progress(0.02)
        let spModel = try loadSentencePiece(dir: dir)

        // ── T5: encode, release ──────────────────────────────────────────────────
        progress(0.04)
        let (textCond, negCond) = try await loadAndEncodeText(
            dir: dir, spModel: spModel, prompt: prompt, negPrompt: "",
            progress: { _, _ in }
        )
        Memory.clearCache(); await Task.yield()
        progress(0.20)

        // ── DiT: denoise, release ────────────────────────────────────────────────
        let latent = try await loadAndDenoise(
            dir: dir, textCond: textCond, negCond: negCond,
            latT: latentT, latH: latentH, latW: latentW,
            numSteps: 50, seed: seed,
            progress: { p, _ in progress(0.20 + p * 0.62) }
        )
        Memory.clearCache(); await Task.yield()
        progress(0.85)

        // ── VAE: decode ──────────────────────────────────────────────────────────
        progress(0.90)
        let decoded = try await loadAndDecode(dir: dir, latent: latent, progress: { _, _ in })
        progress(0.98)

        guard let image = cgImage(from: decoded, frameIndex: 0) else {
            throw WanVideoEngineError.vaeDecodeFailed
        }
        progress(1.0)
        return image
    }

    // MARK: - Full video generation

    nonisolated static func generateAll(
        prompt:          String,
        negativePrompt:  String = "",
        modelID:         String,
        numSteps:        Int    = 50,
        numFrames:       Int    = 17,
        latH:            Int    = latentH,
        latW:            Int    = latentW,
        seed:            UInt64? = nil,
        progress:        @Sendable (Double, String) -> Void
    ) async throws -> [CGImage] {
        let dir = ModelStore.shared.directory(forModelID: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw StageError.modelNotInstalled(id: modelID)
        }
        let latT = max(1, (numFrames + 3) / 4)

        // ── 1. Tokenize ──────────────────────────────────────────────────────────
        progress(0.02, "Tokenizing prompt…")
        let spModel = try loadSentencePiece(dir: dir)

        // ── 2. T5: encode, release before DiT loads ──────────────────────────────
        // T5-XXL is ~5.5 GB at 4-bit. By loading it inside loadAndEncodeText,
        // its weight dicts go out of scope when the function returns. The subsequent
        // Memory.clearCache() then actually frees the GPU memory.
        progress(0.04, "Loading text encoder (T5-XXL)…")
        let (textCond, negCond) = try await loadAndEncodeText(
            dir: dir, spModel: spModel,
            prompt: prompt, negPrompt: negativePrompt,
            progress: { _, l in progress(0.15, l) }
        )
        Memory.clearCache(); await Task.yield()
        progress(0.20, "Text encoded. Loading DiT…")

        // ── 3. DiT: denoise, release before VAE loads ────────────────────────────
        let latent = try await loadAndDenoise(
            dir: dir, textCond: textCond, negCond: negCond,
            latT: latT, latH: latH, latW: latW,
            numSteps: numSteps, seed: seed,
            progress: { p, l in progress(0.20 + p * 0.65, l) }
        )
        Memory.clearCache(); await Task.yield()
        progress(0.85, "Loading VAE decoder…")

        // ── 4. VAE: decode all frames ────────────────────────────────────────────
        progress(0.90, "Decoding frames…")
        let decoded = try await loadAndDecode(dir: dir, latent: latent, progress: { _, l in progress(0.90, l) })
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

    // MARK: - Stage helpers

    /// Loads T5-XXL, encodes prompt + negative prompt, returns materialized text embeddings.
    /// All T5 weight arrays (t5Raw, t5Params, t5 model) are released when this returns.
    private static func loadAndEncodeText(
        dir:      URL,
        spModel:  FluxSentencePiece.Model,
        prompt:   String,
        negPrompt: String,
        progress: @Sendable (Double, String) -> Void
    ) async throws -> (textCond: MLXArray, negCond: MLXArray) {
        let t5Raw    = try await loadShards(dir: dir, subdir: "text_encoder")
        let t5Params = WanT5Encoder.sanitize(t5Raw)
        let t5       = WanT5Encoder()
        try t5.update(parameters: ModuleParameters.unflattened(t5Params), verify: .none)
        eval(t5)
        progress(1.0, "Encoding text…")

        let tokenIDs    = tokenize(prompt, model: spModel, maxLen: maxTextLen)
        let tokenTensor = MLXArray(tokenIDs).reshaped([1, tokenIDs.count])
        let textCond    = t5(tokenTensor)
        eval(textCond)

        let negText = negPrompt.isEmpty ? "" : negPrompt
        let negIDs  = tokenize(negText, model: spModel, maxLen: maxTextLen)
        let negToks = MLXArray(negIDs).reshaped([1, negIDs.count])
        let negCond = t5(negToks)
        eval(negCond)

#if DEBUG
        let tcF = textCond.asType(.float32); eval(tcF)
        let ncF = negCond.asType(.float32);  eval(ncF)
        func stdOf(_ a: MLXArray) -> Float { sqrt(pow(a - a.mean(), 2).mean()).item(Float.self) }
        print("[WAN-DBG] textCond: mean=\(String(format:"%.3f",tcF.mean().item(Float.self))) std=\(String(format:"%.3f",stdOf(tcF))) shape=\(textCond.shape)")
        print("[WAN-DBG] negCond:  mean=\(String(format:"%.3f",ncF.mean().item(Float.self))) std=\(String(format:"%.3f",stdOf(ncF))) shape=\(negCond.shape)")
#endif

        // t5Raw, t5Params, t5 released here; textCond/negCond are materialized and safe.
        return (textCond, negCond)
    }

    /// Loads the DiT, runs the denoising loop, returns the final latent (normalized, in DiT space).
    /// All DiT weight arrays are released when this returns.
    private static func loadAndDenoise(
        dir:      URL,
        textCond: MLXArray,
        negCond:  MLXArray,
        latT:     Int,
        latH:     Int,
        latW:     Int,
        numSteps: Int,
        seed:     UInt64?,
        progress: @Sendable (Double, String) -> Void
    ) async throws -> MLXArray {
        let ditRaw    = try await loadShards(dir: dir, subdir: "transformer")
        // [FFN-DBG] P1 — verify FFN: GEGLU ([17920, 1536]) vs plain GELU ([8960, 1536]).
        printFFNShapeDiagnostics(ditRaw)
        let ditParams = WanDiT.sanitize(ditRaw)
        let dit       = WanDiT()
        try dit.update(parameters: ModuleParameters.unflattened(ditParams), verify: .none)
        eval(dit)
        progress(0.0, "Denoising — step 0/\(numSteps)…")

        let rng    = seed.map { MLXRandom.key($0) }
            ?? MLXRandom.key(UInt64(Date().timeIntervalSince1970 * 1000))
        var latent = MLXRandom.normal([1, latT, latH, latW, 16], key: rng).asType(.bfloat16)
        eval(latent)

        let sigmas = WanSampler.buildSigmas(numSteps: numSteps)

#if DEBUG
        let latInit = latent.asType(.float32); eval(latInit)
        func stdOf(_ a: MLXArray) -> Float { sqrt(pow(a - a.mean(), 2).mean()).item(Float.self) }
        print("[WAN-DBG] init latent: mean=\(String(format:"%.3f",latInit.mean().item(Float.self))) std=\(String(format:"%.3f",stdOf(latInit)))")
#endif

        for i in 0 ..< numSteps {
            let sigmaFrom = sigmas[i]
            let sigmaTo   = sigmas[i + 1]
            let t       = WanSampler.sigmaToTimestep(sigmaFrom)
            let tSingle = MLXArray([t]).asType(.bfloat16)

            let u = dit(latent, timestep: tSingle, textCond: negCond)
            let c = dit(latent, timestep: tSingle, textCond: textCond)
            eval(u); eval(c)

            // Dynamic CFG: diff_std grows throughout denoising and can peak anywhere in
            // σ=0.3–0.7 depending on seed. Ramp from full cfgScale at σ≥0.9 to a 1.5 floor
            // at σ≤0.27, cutting guided_std at mid-σ steps from ~3.0 to ~2.0.
            let effectiveCfg = sigmaFrom >= 0.9
                ? cfgScale
                : max(1.5, cfgScale * (sigmaFrom / 0.9))
            let noisePred = u + (c - u) * effectiveCfg

#if DEBUG
            // [VEL-DBG] velocity logging for every step — shows whether diff_std (uncond-vs-cond
            // divergence) grows across the trajectory or stays ~constant (stuck/uninformative).
            let uF = u.asType(.float32); let cF = c.asType(.float32)
            let gF = noisePred.asType(.float32)
            print("[VEL-DBG] step\(String(format:"%02d",i)): σ=\(String(format:"%.4f",sigmaFrom)) cfg=\(String(format:"%.2f",effectiveCfg)) u_std=\(String(format:"%.3f",stdOf(uF))) c_std=\(String(format:"%.3f",stdOf(cF))) diff_std=\(String(format:"%.3f",stdOf(cF-uF))) guided_std=\(String(format:"%.3f",stdOf(gF)))")
#endif

            latent = WanSampler.eulerStep(x: latent, noisePred: noisePred, sigmaFrom: sigmaFrom, sigmaTo: sigmaTo)
            eval(latent)

#if DEBUG
            // Per-step latent divergence tracker (remove with other [WAN-DBG] prints when working)
            let lF2  = latent.asType(.float32); eval(lF2)
            let lstd = stdOf(lF2)
            let lmn  = lF2.mean().item(Float.self)
            print("[STEP-DBG] step\(String(format:"%02d",i)): σ=\(String(format:"%.4f",sigmaFrom))→\(String(format:"%.4f",sigmaTo)) latent_mean=\(String(format:"%.4f",lmn)) std=\(String(format:"%.4f",lstd))")
#endif

            progress(Double(i + 1) / Double(numSteps), "Denoising — step \(i+1)/\(numSteps)…")
            await Task.yield()
        }

#if DEBUG
        let latF = latent.asType(.float32); eval(latF)
        print("[WAN-DBG] final latent: mean=\(String(format:"%.3f",latF.mean().item(Float.self))) std=\(String(format:"%.3f",stdOf(latF))) min=\(String(format:"%.3f",latF.min().item(Float.self))) max=\(String(format:"%.3f",latF.max().item(Float.self)))")
#endif

        // ditRaw, ditParams, dit released here; latent is materialized and safe.
        return latent
    }

    /// Loads the VAE decoder, decodes `latent` into RGB frames, returns [1, T, H, W, 3].
    /// All VAE weight arrays are released when this returns.
    private static func loadAndDecode(
        dir:      URL,
        latent:   MLXArray,
        progress: @Sendable (Double, String) -> Void
    ) async throws -> MLXArray {
        let vaeRaw = try await loadShards(dir: dir, subdir: "vae")
        printVAEChannelDiagnostics(vaeRaw)
        let vaeParams = WanVAEDecoder.sanitize(vaeRaw)
        printDroppedVAEKeys(vaeRaw)
        let vae = WanVAEDecoder()
        try vae.update(parameters: ModuleParameters.unflattened(vaeParams), verify: .none)
        eval(vae)
        progress(1.0, "Decoding frames…")

        var latentVAE = latent.asType(.float32) * MLXArray(vatStdF) + MLXArray(vatMeanF)

        // diffusers AutoencoderKLWan._decode applies post_quant_conv (a trained 1×1×1 causal conv)
        // BEFORE the decoder. It is a top-level VAE key (not under "decoder."), so WanVAEDecoder.sanitize
        // intentionally drops it. Re-apply it here or every decoded frame is garbage.
        if let postQuant = try loadPostQuantConv(vaeRaw) {
            latentVAE = postQuant(latentVAE)
        }
        eval(latentVAE)

#if DEBUG
        let vInMean = latentVAE.mean().item(Float.self)
        let vInStd  = sqrt(pow(latentVAE - latentVAE.mean(), 2).mean()).item(Float.self)
        print("[WAN-DBG] latentVAE: mean=\(String(format:"%.3f",vInMean)) std=\(String(format:"%.3f",vInStd))")
#endif

        let decoded = vae(latentVAE)
        eval(decoded)

#if DEBUG
        let decMean = decoded.mean().item(Float.self)
        let decStd  = sqrt(pow(decoded - decoded.mean(), 2).mean()).item(Float.self)
        print("[WAN-DBG] decoded: mean=\(String(format:"%.3f",decMean)) std=\(String(format:"%.3f",decStd)) min=\(String(format:"%.3f",decoded.min().item(Float.self))) max=\(String(format:"%.3f",decoded.max().item(Float.self)))")
#endif

        // vaeRaw, vaeParams, vae, postQuant released here; decoded is materialized and safe.
        return decoded
    }

    // MARK: - P1/P2 diagnostics (DelegateFixBacklog)

    /// P1 — prints the loaded FFN weight shapes to verify plain GELU ([8960, 1536])
    /// vs GEGLU ([17920, 1536]).
    private static func printFFNShapeDiagnostics(_ ditRaw: [String: MLXArray]) {
#if DEBUG
        if let w = ditRaw["blocks.0.ffn.net.0.proj.weight"] {
            print("[FFN-DBG] blocks.0 ffn.net.0.proj.weight shape: \(w.shape)")
        }
        if let w = ditRaw["blocks.0.ffn.net.2.weight"] {
            print("[FFN-DBG] blocks.0 ffn.net.2.weight shape: \(w.shape)")
        }
#endif
    }

    /// P2 — prints the up-block resnet channel shapes to verify the decoder channel
    /// progression matches the checkpoint.
    private static func printVAEChannelDiagnostics(_ vaeRaw: [String: MLXArray]) {
#if DEBUG
        let keysOfInterest = ["up_blocks.0.resnets.0.conv1.weight",
                              "up_blocks.1.resnets.0.conv1.weight",
                              "up_blocks.2.resnets.0.conv1.weight",
                              "up_blocks.3.resnets.0.conv1.weight"]
        for k in keysOfInterest {
            if let w = vaeRaw["decoder." + k] ?? vaeRaw[k] { print("[VAE-DBG] \(k): \(w.shape)") }
        }
#endif
    }

    /// P2 — flags decoder keys that WanVAEDecoder.sanitize silently drops.
    /// Any dropped key = module left uninitialized = likely noise source.
    private static func printDroppedVAEKeys(_ vaeRaw: [String: MLXArray]) {
#if DEBUG
        var droppedKeys: [String] = []
        for key in vaeRaw.keys {
            guard !key.hasPrefix("encoder.") && !key.hasPrefix("quant_") else { continue }
            let mapped = WanVAEDecoder.sanitize([key: vaeRaw[key]!])
            if mapped.isEmpty { droppedKeys.append(key) }
        }
        print("[VAE-DBG] Dropped \(droppedKeys.count) decoder keys: \(droppedKeys.prefix(10))")
#endif
    }

    /// Builds the VAE `post_quant_conv` (1×1×1 causal conv, zDim→zDim) from the raw
    /// top-level checkpoint keys. Returns nil if the keys are absent.
    private static func loadPostQuantConv(_ vaeRaw: [String: MLXArray]) throws -> WanCausalConv3d? {
        guard let w = vaeRaw["post_quant_conv.weight"], let b = vaeRaw["post_quant_conv.bias"] else {
#if DEBUG
            print("[VAE-DBG] post_quant_conv not found in checkpoint — skipping")
#endif
            return nil
        }
        let conv = WanCausalConv3d(inCh: w.dim(1), outCh: w.dim(0), k: 1)
        try conv.update(
            parameters: ModuleParameters.unflattened([
                "weight": w.transposed(0, 2, 3, 4, 1),   // [O, I, 1, 1, 1] → [O, 1, 1, 1, I]
                "bias": b,
            ]),
            verify: .none
        )
        return conv
    }

    // MARK: - MP4 assembly

    /// Assembles CGImage frames into an H.264 MP4 at `outputURL`.
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
        // pad:false → returns actual tokens + EOS only, no zero-padding to 512.
        // The default pad:true floods T5 with ~500 PAD representations that collapse textCond
        // to a near-zero mean, driving DiT adaLN gate modulation completely out of distribution.
        let ids = FluxT5Tokenizer.tokenize(prompt, model: model, pad: false)
        return Array(ids.prefix(maxLen)).map(Int32.init)
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
