import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// The Qwen3.5-9B text backbone, loaded so its **final-norm hidden states** can be read.
///
/// `clef_mlx.py`'s text path calls the inner text model directly — `self._text_model(ids[None])[0]`
/// (`:463`) — and MLXLLM does not expose that inner model: `Qwen35Model.callAsFunction` applies
/// `lm_head` on the way out. Gate **B1** (CF-0, ruled 2026-10-03) reaches the same states by
/// replacing `language_model.lm_head` with an identity `Linear` through the public
/// `Module.update(modules:)`; the public call then returns `[1, T, 4096]`, exactly what
/// `_text_model(...)` returns in Python.
///
/// The swap does **not** free `lm_head`. `_lexical` (`:449–457`) mean-pools dequantized rows of the
/// output-embedding matrix, and those vectors feed `option_lexical_projection` and the `prior` term
/// of every logit — so the original module is kept and handed to ``ClefLexicalRows`` (CF-1 finding 1,
/// ruled at CF-A0: "B1's `lm_head` retention").
nonisolated enum ClefBackboneError: Error, CustomStringConvertible {
    case notAQwen35Model(String)
    case noOutputEmbedding

    var description: String {
        switch self {
        case .notAQwen35Model(let type):
            return "Clef Flash needs a Qwen3.5 text backbone; the container loaded \(type)"
        case .noOutputEmbedding:
            return "Clef Flash's checkpoint has no language_model.lm_head to read the lexical rows from"
        }
    }
}

/// Gate B1's identity head: `Qwen35TextModel.callAsFunction` applies `lm_head` to the final-norm
/// states, so a `Linear` that returns its input makes the public call return the states themselves.
/// The weight is a single placeholder row — `callAsFunction` never reads it — because `Linear`'s
/// `shape` is otherwise meaningless to anything that introspects the model after the swap.
nonisolated final class ClefIdentityHead: Linear, @unchecked Sendable {
    init(hiddenSize: Int) {
        super.init(weight: MLXArray.zeros([1, hiddenSize]), bias: nil)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

/// `_lexical` (`:449–457`): rows of the output-embedding matrix for the given token ids,
/// **dequantized** when the checkpoint quantized `lm_head` (this one is 4-bit, group size 64).
/// The head mean-pools whatever this returns, so it stays a row set, not a mean.
nonisolated struct ClefLexicalRows: @unchecked Sendable {
    private let weight: MLXArray
    private let scales: MLXArray?
    private let biases: MLXArray?
    private let groupSize: Int?
    private let bits: Int?
    private let mode: QuantizationMode

    init(_ module: Linear?) {
        weight = module?.weight ?? MLXArray.zeros([0, 0])
        if let quantized = module as? QuantizedLinear {
            scales = quantized.scales
            biases = quantized.biases
            groupSize = quantized.groupSize
            bits = quantized.bits
            mode = quantized.mode
        } else {
            scales = nil
            biases = nil
            groupSize = nil
            bits = nil
            mode = .affine
        }
    }

    var isUsable: Bool { weight.size > 0 }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        if let scales, let groupSize, let bits {
            return dequantized(
                weight[ids], scales: scales[ids], biases: biases.map { $0[ids] },
                groupSize: groupSize, bits: bits, mode: mode)
        }
        return weight[ids]
    }
}

/// The loaded backbone plus the lexical row source. `@unchecked Sendable` for the same reason
/// `CLMHeadPair` is: a loaded MLX module is immutable in practice, and every call into it is
/// serialized by ``ClefEngine`` being an actor.
nonisolated final class ClefBackbone: @unchecked Sendable {
    let container: ModelContainer
    let lexicalRows: ClefLexicalRows

    private init(container: ModelContainer, lexicalRows: ClefLexicalRows) {
        self.container = container
        self.lexicalRows = lexicalRows
    }

    /// Loads the installed checkpoint through the app's canonical LLM loader
    /// (`ModelContainerPool.loadFromDisk`, the one place tool-call format is decided), then performs
    /// gate B1's swap. The container is private to this backbone — not the shared pool's — because the
    /// swap mutates the model, and doing that to a container the chat sheet or the Local Server also
    /// holds would change what they compute.
    static func load(modelDirectory: URL, hiddenSize: Int) async throws -> ClefBackbone {
        // `MLXLMCommon.loadWeights` enumerates **every** `.safetensors` in the directory, so the
        // install's own `joint_head.safetensors` would arrive as `language_model.evidence_layers.*`
        // weights and be rejected as unhandled keys. The Python reference never sees them: mlx-lm
        // reads the index, which lists only the two LM shards. A temporary view of symlinks gives the
        // loader the language model alone — no 6 GB copy, and it is removed as soon as the load ends.
        let view = try makeLanguageModelView(of: modelDirectory)
        defer { try? FileManager.default.removeItem(at: view) }
        let container = try await ModelContainerPool.loadFromDisk(
            directory: view, architecture: nil)
        let rows: ClefLexicalRows = try await container.perform { context in
            guard let model = context.model as? Qwen35Model else {
                throw ClefBackboneError.notAQwen35Model(String(describing: type(of: context.model)))
            }
            let output = model.leafModules().flattened()
                .first { $0.0 == "language_model.lm_head" || $0.0.hasSuffix(".language_model.lm_head") }?
                .1 as? Linear
            guard let output else { throw ClefBackboneError.noOutputEmbedding }
            let rows = ClefLexicalRows(output)

            model.update(modules: ModuleChildren.unflattened(
                ["language_model.lm_head": ClefIdentityHead(hiddenSize: hiddenSize)]))
            eval(model)
            return rows
        }
        return ClefBackbone(container: container, lexicalRows: rows)
    }

    /// A directory of symlinks to everything in `modelDirectory` except the joint head and its config.
    private static func makeLanguageModelView(of directory: URL) throws -> URL {
        let view = FileManager.default.temporaryDirectory
            .appendingPathComponent("clef-lm-view-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: view, withIntermediateDirectories: true)
        let excluded: Set<String> = ["joint_head.safetensors", "joint_head_config.json"]
        for name in try FileManager.default.contentsOfDirectory(atPath: directory.path)
        where !excluded.contains(name) {
            try FileManager.default.createSymbolicLink(
                at: view.appendingPathComponent(name),
                withDestinationURL: directory.appendingPathComponent(name))
        }
        return view
    }

    /// The final-norm hidden states for `ids`, shape `(T, hiddenSize)` — the reference's
    /// `_hidden` for the text path. One sequence, no padding, no cache (the reference does not batch;
    /// readme F9). The body runs inside the container's serial access, so it can use the returned
    /// arrays (they are not `Sendable`) and must return only `Sendable` values.
    func withHidden<R: Sendable>(
        ids: [Int],
        _ body: @Sendable (MLXArray, ClefLexicalRows) throws -> R
    ) async throws -> R {
        let tokens = MLXArray(ids.map(Int32.init))
        let rows = lexicalRows
        return try await container.perform { context in
            guard let model = context.model as? Qwen35Model else {
                throw ClefBackboneError.notAQwen35Model(String(describing: type(of: context.model)))
            }
            let states = model(tokens.expandedDimensions(axis: 0), cache: nil)[0]
            return try body(states, rows)
        }
    }
}
