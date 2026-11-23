import Foundation
import MLX

// MC5-3c: reading ACE-Step 1.5 weights from an installed model directory, and the conversions that turn the published
// PyTorch layouts into the MLX ones. Upstream is `ace-step/ACE-Step-1.5` @ ca1e85f (`acestep/models/mlx/dit_convert.py`
// and `vae_convert.py`); the logic here is a line-for-line port, checked against upstream's own output in
// `AceStepWeightsConversionTests`. No module code lives here: the DiT, condition encoder and VAE (MC5-3d ... 3h) take
// the dictionaries this returns.

// MARK: - Safetensors header reader

/// One tensor's place in a `.safetensors` file.
nonisolated struct SafetensorsEntry: Sendable, Equatable {
    let name: String
    let dtype: DType
    let shape: [Int]
    /// Absolute byte range in the file.
    let byteRange: Range<Int>
}

/// A safetensors file's header, read without touching the tensor data. `read` pulls individual tensors from their byte
/// offsets, so memory while loading a group is that group's size (plus the one tensor in flight), never the file's.
nonisolated struct SafetensorsIndex: Sendable {
    let url: URL
    let entries: [String: SafetensorsEntry]
    /// The header's `__metadata__` strings (empty when absent).
    let metadata: [String: String]

    var names: [String] { entries.keys.sorted() }

    init(url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AceStepWeightsError.missingFile(url.lastPathComponent)
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileSize = Int((try? handle.seekToEnd()) ?? 0)
        try handle.seek(toOffset: 0)
        guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else {
            throw AceStepWeightsError.badHeader(url.lastPathComponent, "shorter than 8 bytes")
        }
        let headerLength = Int(lenData.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) })
        guard headerLength > 0, 8 + headerLength <= fileSize else {
            throw AceStepWeightsError.badHeader(url.lastPathComponent, "header length \(headerLength) exceeds the file")
        }
        guard let headerData = try handle.read(upToCount: headerLength), headerData.count == headerLength,
              let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw AceStepWeightsError.badHeader(url.lastPathComponent, "header is not a JSON object")
        }
        let dataStart = 8 + headerLength
        var parsed: [String: SafetensorsEntry] = [:]
        for (name, value) in object where name != "__metadata__" {
            guard let info = value as? [String: Any],
                  let dtypeName = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int], offsets.count == 2,
                  offsets[0] <= offsets[1] else {
                throw AceStepWeightsError.badHeader(url.lastPathComponent, "malformed entry '\(name)'")
            }
            guard let dtype = SafetensorsIndex.dtype(named: dtypeName) else {
                throw AceStepWeightsError.unsupportedDType(name, dtypeName)
            }
            let range = (dataStart + offsets[0])..<(dataStart + offsets[1])
            let expected = shape.reduce(1, *) * dtype.size
            guard range.count == expected, range.upperBound <= fileSize else {
                throw AceStepWeightsError.badHeader(url.lastPathComponent, "'\(name)' byte range does not match its shape")
            }
            parsed[name] = SafetensorsEntry(name: name, dtype: dtype, shape: shape, byteRange: range)
        }
        self.url = url
        self.entries = parsed
        self.metadata = (object["__metadata__"] as? [String: String]) ?? [:]
    }

    static func dtype(named name: String) -> DType? {
        switch name {
        case "F32": return .float32
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "I32": return .int32
        case "I64": return .int64
        case "U8": return .uint8
        default: return nil
        }
    }

    /// Reads exactly `names`, each straight from its byte offsets. The dtype is kept (bf16 stays bf16). Arrays are
    /// materialised (the bytes are copied into MLX), so nothing here depends on lazy loading.
    func read(_ names: [String]) throws -> [String: MLXArray] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var out: [String: MLXArray] = [:]
        for name in names {
            guard let entry = entries[name] else { throw AceStepWeightsError.missingTensor(name) }
            try handle.seek(toOffset: UInt64(entry.byteRange.lowerBound))
            let data = try handle.read(upToCount: entry.byteRange.count) ?? Data()
            guard data.count == entry.byteRange.count else {
                throw AceStepWeightsError.badHeader(url.lastPathComponent, "'\(name)' is truncated")
            }
            out[name] = MLXArray(data, entry.shape, dtype: entry.dtype)
        }
        return out
    }

    /// Names with any of `prefixes` or exactly in `exact`, sorted.
    func names(withPrefixes prefixes: [String], exact: [String] = []) -> [String] {
        names.filter { name in prefixes.contains { name.hasPrefix($0) } || exact.contains(name) }
    }
}

// MARK: - Errors

nonisolated enum AceStepWeightsError: Error, Equatable, CustomStringConvertible {
    case missingFile(String)
    case badHeader(String, String)
    case unsupportedDType(String, String)
    case missingTensor(String)
    case unusedTensor(String)
    case forbiddenGroup(String)
    case unexpectedTensor(String)
    case badShape(String, [Int])

    var description: String {
        switch self {
        case .missingFile(let f): return "ACE-Step: missing file '\(f)' in the model folder"
        case .badHeader(let f, let why): return "ACE-Step: '\(f)' is not a readable safetensors file (\(why))"
        case .unsupportedDType(let n, let d): return "ACE-Step: tensor '\(n)' has unsupported dtype \(d)"
        case .missingTensor(let n): return "ACE-Step: missing tensor '\(n)'"
        case .unusedTensor(let n): return "ACE-Step: unused tensor '\(n)' (not part of the expected set)"
        case .forbiddenGroup(let p): return "ACE-Step: '\(p)' is never read (not needed on the no-LM path)"
        case .unexpectedTensor(let n): return "ACE-Step: unexpected tensor '\(n)' for this conversion"
        case .badShape(let n, let s): return "ACE-Step: tensor '\(n)' has unexpected shape \(s)"
        }
    }
}

// MARK: - Weights

nonisolated enum AceStepWeights {

    // MARK: Layout of an installed model folder

    static let turboPath = "acestep-v15-turbo/model.safetensors"
    static let qwenPath = "Qwen3-Embedding-0.6B/model.safetensors"
    static let vaePath = "vae/diffusion_pytorch_model.safetensors"

    /// Turbo checkpoint groups we read: the DiT and the condition encoder, plus `null_condition_emb`.
    static let turboReadPrefixes = ["decoder.", "encoder."]
    static let turboReadNames = ["null_condition_emb"]
    /// The FSQ tokenizer / detokenizer run once on the no-LM path but their output is discarded (T16): never read.
    static let turboNeverReadPrefixes = ["tokenizer.", "detokenizer."]
    /// The VAE file's `encoder.*` tensors are unused by generation: never read.
    static let vaeReadPrefixes = ["decoder."]
    static let qwenEmbedTokensName = "embed_tokens.weight"

    static func url(_ relative: String, in directory: URL) -> URL {
        directory.appendingPathComponent(relative)
    }

    // MARK: Selective reads

    /// Tensors of the turbo file by prefix / exact name. Refuses the tokenizer and detokenizer groups by name, and
    /// an empty prefix (which would read the whole file).
    static func readTurbo(directory: URL, prefixes: [String], names: [String] = []) throws -> [String: MLXArray] {
        let index = try SafetensorsIndex(url: url(turboPath, in: directory))
        return try readTurbo(index: index, prefixes: prefixes, names: names)
    }

    static func readTurbo(index: SafetensorsIndex, prefixes: [String], names: [String] = []) throws -> [String: MLXArray] {
        for p in prefixes + names {
            if p.isEmpty || turboNeverReadPrefixes.contains(where: { p.hasPrefix($0) || $0.hasPrefix(p) }) {
                throw AceStepWeightsError.forbiddenGroup(p)
            }
        }
        let wanted = index.names(withPrefixes: prefixes, exact: names)
        precondition(!wanted.contains { name in turboNeverReadPrefixes.contains { name.hasPrefix($0) } })
        for n in names where index.entries[n] == nil { throw AceStepWeightsError.missingTensor(n) }
        return try index.read(wanted)
    }

    /// Tensors of the Qwen3-Embedding file by prefix / exact name (the text side, MC5-3d, decides what it needs).
    static func readQwen(directory: URL, prefixes: [String], names: [String] = []) throws -> [String: MLXArray] {
        let index = try SafetensorsIndex(url: url(qwenPath, in: directory))
        for n in names where index.entries[n] == nil { throw AceStepWeightsError.missingTensor(n) }
        return try index.read(index.names(withPrefixes: prefixes, exact: names))
    }

    /// `embed_tokens.weight` `[151669, 1024]` on its own (0.311 GB bf16), the lyric table (readme T18).
    static func loadQwenEmbedTokens(directory: URL) throws -> MLXArray {
        let tensors = try readQwen(directory: directory, prefixes: [], names: [qwenEmbedTokensName])
        guard let table = tensors[qwenEmbedTokensName] else { throw AceStepWeightsError.missingTensor(qwenEmbedTokensName) }
        return table
    }

    /// The DiT: `decoder.*` converted to MLX names and layouts, dtype kept (bf16), evaluated.
    static func loadDecoder(directory: URL) throws -> [String: MLXArray] {
        let converted: [String: MLXArray]
        do {   // MC5-T-mem2: the raw dictionary is out of scope as soon as the conversion has been evaluated
            let raw = try readTurbo(directory: directory, prefixes: ["decoder."])
            converted = try convertDecoder(raw)
            eval(Array(converted.values))
        }
        return converted
    }

    /// The condition encoder (`encoder.*`: text projector, lyric encoder, timbre encoder), names and layouts as published,
    /// bf16. No conversion exists upstream for it (it runs in torch there), so MC5-3e takes these as they are.
    static func loadConditionEncoder(directory: URL) throws -> [String: MLXArray] {
        try readTurbo(directory: directory, prefixes: ["encoder."])
    }

    /// `null_condition_emb` `[1, 1, 2048]` bf16.
    static func loadNullConditionEmbedding(directory: URL) throws -> MLXArray {
        let tensors = try readTurbo(directory: directory, prefixes: [], names: ["null_condition_emb"])
        guard let t = tensors["null_condition_emb"] else { throw AceStepWeightsError.missingTensor("null_condition_emb") }
        return t
    }

    /// The VAE decoder only (`decoder.*`; the file's `encoder.*` is never read), weight-norm fused and laid out for MLX.
    /// Returns float32 (see `convertVAE`), evaluated.
    static func loadVAEDecoder(directory: URL) throws -> [String: MLXArray] {
        let index = try SafetensorsIndex(url: url(vaePath, in: directory))
        let converted: [String: MLXArray]
        do {
            let raw = try index.read(index.names(withPrefixes: vaeReadPrefixes))
            converted = try convertVAE(raw)
            eval(Array(converted.values))
        }
        return converted
    }

    // MARK: Conversions (pure functions over arrays)

    /// DiT (`dit_convert.convert_decoder_weights`): strip `decoder.`; drop any name containing `rotary_emb`;
    /// `proj_in.1.*` -> `proj_in.*` with the weight `[out, in, K]` -> `[out, K, in]` (swap axes 1, 2);
    /// `proj_out.1.*` -> `proj_out.*` with the weight `[in, out, K]` -> `[out, K, in]` (transpose 1, 2, 0). Every other
    /// tensor is untouched. Upstream also upcasts to f32; we keep the stored dtype (R3: bf16 stays bf16).
    static func convertDecoder(_ tensors: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for key in tensors.keys.sorted() {
            guard key.hasPrefix("decoder."), let value = tensors[key] else { throw AceStepWeightsError.unexpectedTensor(key) }
            let name = String(key.dropFirst("decoder.".count))
            if name.hasPrefix("proj_in.1.") {
                let newName = name.replacingOccurrences(of: "proj_in.1.", with: "proj_in.")
                if newName.hasSuffix(".weight") {
                    guard value.ndim == 3 else { throw AceStepWeightsError.badShape(key, value.shape) }
                    out[newName] = value.swappedAxes(1, 2)
                } else {
                    out[newName] = value
                }
            } else if name.hasPrefix("proj_out.1.") {
                let newName = name.replacingOccurrences(of: "proj_out.1.", with: "proj_out.")
                if newName.hasSuffix(".weight") {
                    guard value.ndim == 3 else { throw AceStepWeightsError.badShape(key, value.shape) }
                    out[newName] = value.transposed(1, 2, 0)
                } else {
                    out[newName] = value
                }
            } else if name.contains("rotary_emb") {
                continue
            } else {
                out[name] = value
            }
        }
        return out
    }

    /// weight-norm fuse (`vae_convert._fuse_weight_norm`), float32: `w = g * v / (norm + 1e-9)`, where `norm` is the L2 norm of
    /// `v` over every axis except 0, reshaped to `g`'s shape. The epsilon is ADDED to the norm, not clamped.
    static func fuseWeightNorm(g: MLXArray, v: MLXArray, eps: Float = 1e-9) -> MLXArray {
        let g32 = g.asType(.float32)
        let v32 = v.asType(.float32)
        let flat = v32.reshaped(v32.dim(0), -1)
        let norm = sqrt(flat.square().sum(axis: 1)).reshaped(g32.shape)
        return g32 * v32 / (norm + eps)
    }

    /// VAE (`vae_convert.convert_vae_weights`): for each `X.weight_g` + `X.weight_v` pair, FUSE (float32) and THEN lay out
    /// for MLX: a ConvTranspose1d (`conv_t1` in the name) `[in, out, K]` -> `[out, K, in]` via transpose(1, 2, 0), any other
    /// Conv1d `[out, in, K]` -> `[out, K, in]` via swap(1, 2); the result is `X.weight`. Snake `.alpha` / `.beta`
    /// `[1, C, 1]` -> `[C]`. Everything else (biases) passes through.
    ///
    /// Every output is **float32**, as upstream's (`.float()`). The VAE runs f16 in v1, so MC5-3h casts after loading.
    /// Fuse-then-transpose, never the other order: the norm is per dim-0 row of the PyTorch layout.
    /// Deviation from upstream only on a malformed file: a lone `weight_g` or `weight_v` throws `missingTensor` where
    /// upstream silently drops it.
    static func convertVAE(_ tensors: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var done: Set<String> = []
        for key in tensors.keys.sorted() {
            if done.contains(key) { continue }
            guard let value = tensors[key] else { continue }
            if key.hasSuffix(".weight_g") || key.hasSuffix(".weight_v") {
                let base = String(key.dropLast(".weight_g".count))
                let gKey = base + ".weight_g", vKey = base + ".weight_v"
                guard let g = tensors[gKey] else { throw AceStepWeightsError.missingTensor(gKey) }
                guard let v = tensors[vKey] else { throw AceStepWeightsError.missingTensor(vKey) }
                guard v.ndim == 3 else { throw AceStepWeightsError.badShape(vKey, v.shape) }
                let w = fuseWeightNorm(g: g, v: v)
                out[base + ".weight"] = base.contains("conv_t1") ? w.transposed(1, 2, 0) : w.swappedAxes(1, 2)
                done.insert(gKey)
                done.insert(vKey)
            } else if key.hasSuffix(".alpha") || key.hasSuffix(".beta") {
                guard value.ndim == 3, value.dim(0) == 1, value.dim(2) == 1 else {
                    throw AceStepWeightsError.badShape(key, value.shape)
                }
                out[key] = value.asType(.float32).reshaped(value.dim(1))
                done.insert(key)
            } else {
                out[key] = value.asType(.float32)
                done.insert(key)
            }
        }
        return out
    }

    // MARK: Strict names

    /// Throws for the FIRST missing tensor (sorted order) and, if none is missing, the first unused one. Deterministic.
    static func checkStrict(actual: some Collection<String>, expected: some Collection<String>) throws {
        let have = Set(actual), want = Set(expected)
        if let missing = want.subtracting(have).sorted().first { throw AceStepWeightsError.missingTensor(missing) }
        if let unused = have.subtracting(want).sorted().first { throw AceStepWeightsError.unusedTensor(unused) }
    }
}
