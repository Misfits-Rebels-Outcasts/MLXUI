import Foundation
import MLX
import MLXNN

nonisolated enum RIFEEngineError: Error, LocalizedError {
    case missingFile(String)
    case unsupportedConfig(String)
    case badFrame(String)

    var errorDescription: String? {
        switch self {
        case .missingFile(let f):        return "RIFE: required file not found: \(f). Install the model again."
        case .unsupportedConfig(let s):  return "RIFE: this checkpoint isn't the 4.25 network (\(s))."
        case .badFrame(let s):           return "RIFE: \(s)"
        }
    }
}

/// `config.json` of mlx-community/RIFE-4.25. The architecture is fixed in `RIFEIFNet`; the file is checked against it so a
/// different RIFE version in the same folder is refused rather than loaded with the wrong shapes.
nonisolated struct RIFEConfig: Decodable, Sendable, Equatable {
    let name: String
    let scaleList: [Float]
    let blockChannels: [Int]
    let blockInPlanes: [Int]
    let padTo: Int

    enum CodingKeys: String, CodingKey {
        case name
        case scaleList = "scale_list"
        case blockChannels = "block_channels"
        case blockInPlanes = "block_in_planes"
        case padTo = "pad_to"
    }

    func validate() throws {
        guard blockChannels == RIFEIFNet.blockChannels, blockInPlanes == RIFEIFNet.blockInPlanes else {
            throw RIFEEngineError.unsupportedConfig("block layout \(blockChannels)")
        }
        guard scaleList.count == 5, padTo > 0 else {
            throw RIFEEngineError.unsupportedConfig("scale_list \(scaleList), pad_to \(padTo)")
        }
    }
}

/// Loads the installed RIFE 4.25 weights and interpolates one frame pair. Frames are `[1, H, W, 3]` float32 in `[0, 1]`.
/// Everything here is synchronous MLX work; callers run it off the main actor (`RIFEVideoPipeline`, RIFE-3c).
nonisolated enum RIFEEngine {

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout.
    static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    struct Loaded: @unchecked Sendable {
        let net: RIFEIFNet
        let config: RIFEConfig
    }

    /// Reads `config.json` + `model.safetensors` from `directory`. Strict: every parameter must be present and no extras.
    static func load(directory: URL) throws -> Loaded {
        let configURL = directory.appendingPathComponent("config.json")
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        for url in [configURL, weightsURL] where !FileManager.default.fileExists(atPath: url.path) {
            throw RIFEEngineError.missingFile(url.lastPathComponent)
        }
        let config = try JSONDecoder().decode(RIFEConfig.self, from: Data(contentsOf: configURL))
        try config.validate()
        let weights = try MLX.loadArrays(url: weightsURL)
        let net = RIFEIFNet()
        try net.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        try MLXErrorGuard.evaluate(net.parameters().flattened().map(\.1))
        return Loaded(net: net, config: config)
    }

    /// The frame at `timestep` between `img0` and `img1` (`inference` of the reference): pads H and W up to a multiple of
    /// `max(padTo, padTo / scale)`, runs the net with the scale list divided by `scale`, crops back. `scale` < 1 trades detail
    /// in the flow for memory (0.5 for 4K).
    static func interpolate(_ loaded: Loaded, _ img0: MLXArray, _ img1: MLXArray,
                            timestep: Float = 0.5, scale: Float = 1.0) throws -> MLXArray {
        guard img0.shape == img1.shape, img0.ndim == 4, img0.dim(3) == 3 else {
            throw RIFEEngineError.badFrame("the two frames must be the same size, [1, H, W, 3]")
        }
        let h = img0.dim(1), w = img0.dim(2)
        let pad = max(loaded.config.padTo, Int((Float(loaded.config.padTo) / scale).rounded()))
        let ph = ((h - 1) / pad + 1) * pad
        let pw = ((w - 1) / pad + 1) * pad
        let widths: [IntOrPair] = [IntOrPair((0, 0)), IntOrPair((0, ph - h)), IntOrPair((0, pw - w)), IntOrPair((0, 0))]
        let x = concatenated([padded(img0, widths: widths), padded(img1, widths: widths)], axis: -1)
        let scales = loaded.config.scaleList.map { $0 / scale }
        let merged = loaded.net(x, timestep: timestep, scaleList: scales)
        return merged[0..., 0 ..< h, 0 ..< w, 0...]
    }
}
