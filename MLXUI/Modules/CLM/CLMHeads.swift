import Foundation
import MLX
import MLXNN

/// Ported from `RealityCat/CLM-v0.1-8B-MLX-8bit`'s vendored `clm_mlx/heads.py` — `Head`,
/// `HeadPair`, `l2`. Same maths as upstream's `clm.heads.make_head` (torch):
///
///     x -> Linear(4096, width) -> GELU -> [Linear(width, width) -> LayerNorm(eps 1e-5) ->
///       GELU] * (depth - 2) -> Linear(width, 512) -> L2-normalise
///
/// GELU is MLXNN's exact `gelu(_:)` (the erf form, `compiledGelu`) — **never**
/// `geluApproximate`; `torch.nn.GELU()`'s default is the exact form. Heads stay **float32**
/// throughout (LY-4 hit an fp16 `LayerNorm` overflow porting Laya's own head, journal
/// `2026-334`; these heads never cast down from float32).
///
/// Golden test: `MLXUITests/CLMHeadsTests.swift` against `Fixtures/CLM/heads.json`
/// (`Fixtures/CLM/PROVENANCE.md`) — cosine ≥ 0.99999 per vector.

/// `heads.py::Head`. Named properties (`inp`/`hidden`/`norms`/`out`) match the checkpoint's
/// own key prefixes (`state_head.inp.*`, `state_head.hidden.0.*`, `state_head.norms.0.*`,
/// `state_head.out.*`) one-for-one — no weight-name remapping needed.
nonisolated final class CLMHead: Module {
    @ModuleInfo(key: "inp") var inp: Linear
    @ModuleInfo(key: "hidden") var hidden: [Linear]
    @ModuleInfo(key: "norms") var norms: [LayerNorm]
    @ModuleInfo(key: "out") var out: Linear
    private let residual: Bool

    init(hiddenSize: Int, width: Int, depth: Int, projectionDim: Int, layernorm: Bool, residual: Bool) {
        _inp.wrappedValue = Linear(hiddenSize, width)
        let hiddenCount = max(0, depth - 2)
        _hidden.wrappedValue = (0 ..< hiddenCount).map { _ in Linear(width, width) }
        _norms.wrappedValue = layernorm
            ? (0 ..< hiddenCount).map { _ in LayerNorm(dimensions: width, eps: 1e-5) } : []
        _out.wrappedValue = Linear(width, projectionDim)
        self.residual = residual
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = gelu(inp(x))
        for (index, linear) in hidden.enumerated() {
            var layer = linear(h)
            if index < norms.count { layer = norms[index](layer) }
            layer = gelu(layer)
            h = residual ? h + layer : layer
        }
        return out(h)
    }
}

/// `heads.py::l2`.
nonisolated func clmL2Normalize(_ x: MLXArray) -> MLXArray {
    x / (norm(x, axis: -1, keepDims: true) + 1e-12)
}

nonisolated enum CLMHeadsError: Error, CustomStringConvertible {
    case invalidConfig(String)

    var description: String {
        switch self {
        case .invalidConfig(let message): return "CLM heads config error: \(message)"
        }
    }
}

/// `heads/config.json`'s decoded shape (the `cfg` sub-object plus the two top-level sizes).
nonisolated private struct CLMHeadsConfigFile: Decodable {
    nonisolated struct Inner: Decodable {
        var width: Int
        var depth: Int
        var activation: String = "gelu"
        var layernorm: Bool = false
        var residual: Bool = false
    }
    var cfg: Inner
    var hiddenSize: Int = 4096
    var projectionDim: Int = 512

    enum CodingKeys: String, CodingKey {
        case cfg
        case hiddenSize = "hidden_size"
        case projectionDim = "projection_dim"
    }
}

/// `heads.py::HeadPair`. State head + action head + score scale, loaded from a capture's
/// `heads/` directory (`config.json` + `CLM_v0.1-8B.safetensors`).
nonisolated final class CLMHeadPair: @unchecked Sendable {
    enum Which { case state, action }

    let state: CLMHead
    let action: CLMHead
    /// `weights["logit_scale"]`.
    let logitScale: Float
    /// `min(exp(logit_scale), 100)` — upstream clamps the score scale at 100.
    let scale: Float

    init(headsDirectory: URL) throws {
        let configURL = headsDirectory.appendingPathComponent("config.json")
        let file = try JSONDecoder().decode(CLMHeadsConfigFile.self, from: Data(contentsOf: configURL))
        guard file.cfg.activation == "gelu" else {
            throw CLMHeadsError.invalidConfig("Unsupported head activation: \(file.cfg.activation); expected gelu")
        }

        let state = CLMHead(
            hiddenSize: file.hiddenSize, width: file.cfg.width, depth: file.cfg.depth,
            projectionDim: file.projectionDim, layernorm: file.cfg.layernorm, residual: file.cfg.residual)
        let action = CLMHead(
            hiddenSize: file.hiddenSize, width: file.cfg.width, depth: file.cfg.depth,
            projectionDim: file.projectionDim, layernorm: file.cfg.layernorm, residual: file.cfg.residual)

        let weightsURL = headsDirectory.appendingPathComponent("CLM_v0.1-8B.safetensors")
        let weights = try MLX.loadArrays(url: weightsURL).mapValues { $0.asType(.float32) }
        for (prefix, head) in [("state_head", state), ("action_head", action)] {
            var stripped: [String: MLXArray] = [:]
            for (key, value) in weights where key.hasPrefix(prefix + ".") {
                stripped[String(key.dropFirst(prefix.count + 1))] = value
            }
            try head.update(parameters: ModuleParameters.unflattened(stripped), verify: .all)
        }
        eval(state, action)

        guard let logitScaleArray = weights["logit_scale"] else {
            throw CLMHeadsError.invalidConfig("heads weights are missing logit_scale")
        }
        let logitScale = logitScaleArray.item(Float.self)
        self.state = state
        self.action = action
        self.logitScale = logitScale
        self.scale = min(Foundation.exp(logitScale), 100.0)
    }

    /// `[n, hidden]` encoder embeddings -> `[n, proj]` L2-normalised projections (float32).
    func project(_ embeddings: MLXArray, which: Which) -> MLXArray {
        let head = which == .state ? state : action
        return clmL2Normalize(head(clmL2Normalize(embeddings.asType(.float32))))
    }
}
