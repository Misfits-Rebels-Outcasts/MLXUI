import Foundation

// CFM-R12-6 — the retrieval cluster (`catflow-mlx/src/catflow/tools/index_store.py`): the
// flat on-disk index directory (`manifest.json` + `vectors.bin` + `chunks.jsonl`), plus
// Store Index / Read Index / Retrieve (exact cosine) / Keyword Search (BM25). An `index`-
// kind `Item.path` points at that **directory** (never a single file, per SPEC-Q15).

// MARK: - The on-disk vector format (SPEC-Q15)

/// `.npy` (float32) read/write — the single on-disk format every `vector`-kind item uses, so
/// `EmbeddingStage`'s output and the index tools agree. Matches `np.save`/`np.load`.
nonisolated enum NpyCodec {
    static func save(_ values: [Float], to url: URL) throws {
        // CFM-R12-FIX-6: match `np.save` exactly — `\x93NUMPY` + version 1.0 + u16 header
        // length + the dict, **then space padding, then a trailing `\n`**, so the payload
        // starts on a 64-byte boundary. (numpy's own layout: `{...}` + spaces + `\n` — the
        // old newline-before-spaces order was a SyntaxError for `ast.literal_eval`, and the
        // NUL padding numpy rejected outright.)
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(values.count),), }"
        var headerLen = dict.utf8.count + 1          // + the trailing `\n`
        while (10 + headerLen) % 64 != 0 { headerLen += 1 }
        let header = dict
            + String(repeating: " ", count: headerLen - dict.utf8.count - 1)
            + "\n"
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        data.append(contentsOf: withUnsafeBytes(of: UInt16(header.utf8.count).littleEndian) { Array($0) })
        data.append(Data(header.utf8))
        var floats = values
        floats.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
    }

    /// Load a `.npy` float32 vector. Tolerates numpy v1/v2 headers and reads the `descr`
    /// from the dict, so a float64 array the Python saved is still loadable (converted to
    /// float32, matching `np.load(...).astype(np.float32)`).
    static func load(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 8, data[0] == 0x93,
              String(data: data[1..<6], encoding: .ascii) == "NUMPY" else {
            throw FlowError.stageFailure(row: "vector", message: "not a .npy file")
        }
        let version = data[6]
        let headerStart: Int
        let headerLen: Int
        if version == 1 {
            headerStart = 10
            headerLen = Int(UInt16(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt16.self) }))
        } else {
            headerStart = 12
            headerLen = Int(UInt32(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self) }))
        }
        let dict = String(data: data[headerStart..<(headerStart + min(headerLen, 4096))], encoding: .ascii) ?? ""
        let isFloat64 = dict.contains("<f8") || dict.contains("float64")
        let bodyStart = headerStart + headerLen
        let bytesPerValue = isFloat64 ? 8 : 4
        guard bodyStart <= data.count, (data.count - bodyStart).isMultiple(of: bytesPerValue) else {
            throw FlowError.stageFailure(row: "vector", message: "malformed .npy payload")
        }
        let count = (data.count - bodyStart) / bytesPerValue
        if isFloat64 {
            return data.withUnsafeBytes { raw -> [Float] in
                var out = [Float](repeating: 0, count: count)
                for i in 0..<count {
                    let v = raw.loadUnaligned(fromByteOffset: bodyStart + i * 8, as: Double.self)
                    out[i] = Float(v)
                }
                return out
            }
        }
        return data.withUnsafeBytes { raw in
            Array(UnsafeBufferPointer(start: raw.baseAddress!.advanced(by: bodyStart)
                    .assumingMemoryBound(to: Float.self), count: count))
        }
    }
}

// MARK: - The index directory format

/// The index on-disk format (Python `index_store.py`): `vectors.bin` is a row-major float16
/// matrix (count×dims), `chunks.jsonl` is one `{"text": "…"}` per line, `manifest.json` is
/// the fixed key set below, `format_version` 1.
nonisolated enum IndexFormat {
    static let formatVersion = 1

    struct Manifest: Codable {
        let format_version: Int
        let embedder: String
        let dims: Int
        let normalization: String
        let count: Int
    }

    static func manifest(from data: Data) throws -> Manifest {
        try JSONDecoder().decode(Manifest.self, from: data)
    }

    /// Byte-identical to the Python's `json.dumps(manifest, indent=2)` for this fixed key set.
    static func manifestData(embedder: String, dims: Int, normalization: String, count: Int) -> Data {
        let body = """
        {
          "format_version": \(formatVersion),
          "embedder": "\(embedder)",
          "dims": \(dims),
          "normalization": "\(normalization)",
          "count": \(count)
        }
        """
        return Data(body.utf8)
    }

    static func chunks(from data: Data) throws -> [String] {
        var out: [String] = []
        for line in data.split(separator: 0x0A) where !line.isEmpty {
            // CFM-R12-FIX-11: a corrupt line raises — skipping it shifts chunk<->vector
            // alignment and Retrieve returns the wrong text for a vector.
            guard let json = try? JSONSerialization.jsonObject(with: Data(line)),
                  let dict = json as? [String: Any], let text = dict["text"] as? String else {
                throw FlowError.stageFailure(row: "index", message: "corrupt chunks.jsonl line")
            }
            out.append(text)
        }
        return out
    }

    static func chunksData(_ chunks: [String]) -> Data {
        // Python's `json.dumps({"text": text})` separators put a space after the colon, and
        // `ensure_ascii=True` escapes non-ASCII — match byte for byte (CFM-R12-FIX-11).
        var out = Data()
        for chunk in chunks {
            let value = FlowParity.asciiJSON(chunk)
            out.append(Data("{\"text\": \(value)}\n".utf8))
        }
        return out
    }

    /// Read the float16 `vectors.bin` matrix (count×dims) as float32.
    static func loadVectors(indexDir: URL, manifest: Manifest) throws -> [[Float]] {
        let bin = try Data(contentsOf: indexDir.appendingPathComponent("vectors.bin"))
        let count = manifest.count, dims = manifest.dims
        guard bin.count == count * dims * 2 else {
            throw FlowError.stageFailure(row: "index", message: "vectors.bin size doesn't match the manifest")
        }
        return (0..<count).map { row in
            (0..<dims).map { col in
                let i = (row * dims + col) * 2
                let bits = UInt16(littleEndian: bin.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: i, as: UInt16.self) })
                return f16ToF32(bits)
            }
        }
    }

    /// Write the float16 `vectors.bin` matrix, byte-identical to the Python's `.astype(float16).tofile`.
    static func vectorsData(_ vectors: [[Float]]) -> Data {
        var out = Data()
        for vector in vectors {
            for value in vector {
                let halfBits = f32ToF16(value)
                out.append(contentsOf: withUnsafeBytes(of: halfBits.littleEndian) { Array($0) })
            }
        }
        return out
    }

    // Float16 ↔ Float32 via bit manipulation — Float16 type conversions are arm64-only
    // in the Swift stdlib and fail to compile on x86_64 (universal archive builds).
    private static func f16ToF32(_ h: UInt16) -> Float {
        let s = UInt32(h & 0x8000) << 16
        let e = UInt32(h >> 10) & 0x1F
        let m = UInt32(h & 0x3FF)
        switch e {
        case 0:
            if m == 0 { return Float(bitPattern: s) }
            var exp = UInt32(113), man = m
            while man & 0x400 == 0 { exp -= 1; man <<= 1 }
            return Float(bitPattern: s | (exp << 23) | ((man & 0x3FF) << 13))
        case 31:
            return Float(bitPattern: s | 0x7F800000 | (m << 13))
        default:
            return Float(bitPattern: s | ((e + 112) << 23) | (m << 13))
        }
    }

    private static func f32ToF16(_ f: Float) -> UInt16 {
        let b = f.bitPattern
        let s = UInt16(b >> 16) & 0x8000
        let e = Int((b >> 23) & 0xFF)
        let m = b & 0x7FFFFF
        if e == 0xFF { return s | 0x7C00 | UInt16(m >> 13) }
        let h = e - 127 + 15
        if h <= 0 { return s }
        if h >= 31 { return s | 0x7C00 }
        return s | (UInt16(h) << 10) | UInt16(m >> 13)
    }
}

// MARK: - The tools

/// `Store Index` (text+vector → index): persist the bundled `[text, vector]` pairs as an
/// index directory at `name=`/bare, in the Python's exact on-disk format. Like `Diff`, it is
/// a tuple tool — the executor hands it the full `inputs` array.
nonisolated struct StoreIndexTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        guard inputs.count >= 2 else {
            throw FlowError.badInputCardinality(row: "Store Index", expected: "(text, vector)", got: inputs.count)
        }
        let texts = inputs[0].items.map { $0.value ?? "" }
        let vectors = try inputs[1].items.map { item -> [Float] in
            guard let path = item.path else {
                throw FlowError.missingInlineValue(row: "Store Index", kind: .vector)
            }
            return try NpyCodec.load(from: path)
        }
        guard !texts.isEmpty, texts.count == vectors.count else {
            throw FlowError.badInputCardinality(row: "Store Index",
                                                expected: "matching text/vector counts", got: texts.count)
        }
        let dims = vectors[0].count
        guard vectors.allSatisfy({ $0.count == dims }) else {
            throw FlowError.stageFailure(row: "Store Index", message: "every vector must share the same dimensions")
        }
        let s = FlowSettings(settings)
        guard let name = s.value(for: "name") ?? s.firstBare() else {
            throw FlowError.missingInlineValue(row: "Store Index", kind: .file)
        }
        let dir = try workspace.resolve(name, flowID: flowID)
        let fm = FileManager.default
        // FIX-11: an overwrite moves the old index to trash, never deletes (Spec §14.1).
        try FlowParity.moveToTrash(dir)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        try IndexFormat.vectorsData(vectors).write(to: dir.appendingPathComponent("vectors.bin"))
        try IndexFormat.chunksData(texts).write(to: dir.appendingPathComponent("chunks.jsonl"))
        // FIX-11: honor `normalization=` (default none) and `embedder=` (or the embedder.json
        // sidecar the Python's embed engine writes, when every vector came from one place).
        let embedder = s.value(for: "embedder") ?? Self.autoEmbedder(inputs[1].items) ?? "unknown"
        let normalization = s.value(for: "normalization") ?? "none"
        try IndexFormat.manifestData(embedder: embedder, dims: dims,
                                     normalization: normalization, count: texts.count)
            .write(to: dir.appendingPathComponent("manifest.json"))
        return Asset(items: [Item(kind: .index, value: nil, path: dir, sourceText: nil)])
    }
}

/// `Read Index` (file → index): resolve the path, validate it's an index directory, return it.
nonisolated struct ReadIndexTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.index) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let raw = FlowSettings(settings).pathValue() else {
            throw FlowError.missingInlineValue(row: "Read Index", kind: .file)
        }
        let dir = try workspace.resolve(raw, flowID: flowID)
        let manifestURL = dir.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path),
              let data = try? Data(contentsOf: manifestURL) else {
            throw FlowError.fileReadFailed(row: "Read Index", path: raw)
        }
        _ = try IndexFormat.manifest(from: data)
        progress(1.0)
        return Asset(items: [Item(kind: .index, value: nil, path: dir, sourceText: nil)])
    }
}

/// `Retrieve` (index+vector → text list): exact cosine over the index, `top_k=`, `min_score=`.
nonisolated struct RetrieveTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let allItems = inputs.flatMap { $0.items }
        guard let indexItem = allItems.first(where: { $0.kind == .index }),
              let vectorItem = allItems.first(where: { $0.kind == .vector }),
              let indexDir = indexItem.path, let vectorPath = vectorItem.path else {
            throw FlowError.badInputCardinality(row: "Retrieve", expected: "(index, vector)", got: inputs.count)
        }
        let manifestURL = indexDir.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            throw FlowError.fileReadFailed(row: "Retrieve", path: indexDir.lastPathComponent)
        }
        let manifest = try IndexFormat.manifest(from: data)
        let query = try NpyCodec.load(from: vectorPath)
        guard query.count == manifest.dims else {
            throw FlowError.stageFailure(row: "Retrieve",
                                         message: "query has \(query.count) dims, index expects \(manifest.dims)")
        }
        let vectors = try IndexFormat.loadVectors(indexDir: indexDir, manifest: manifest)
        let chunks = try IndexFormat.chunks(from: Data(contentsOf: indexDir.appendingPathComponent("chunks.jsonl")))

        let s = FlowSettings(settings)
        // FIX-11: an invalid `top_k=`/`min_score=` fails the row (Python's ValueError), it
        // never silently defaults or drops the threshold.
        let topK: Int
        if let raw = s.value(for: "top_k") {
            guard let v = Int(raw) else {
                throw FlowError.invalidSettings(row: "Retrieve", setting: "top_k", detail: "isn't an integer")
            }
            topK = v
        } else {
            topK = 5
        }
        let minScore: Double?
        if let raw = s.value(for: "min_score") {
            guard let v = Double(raw) else {
                throw FlowError.invalidSettings(row: "Retrieve", setting: "min_score", detail: "isn't a number")
            }
            minScore = v
        } else {
            minScore = nil
        }

        let scores = Self.cosineScores(vectors: vectors, query: query)
        // FIX-11: explicit index tie-break — the Python's stable argsort keeps original
        // order on equal scores (Swift's sort is stable today but not documented to be).
        let order = (0..<scores.count).sorted {
            scores[$0] > scores[$1] || (scores[$0] == scores[$1] && $0 < $1)
        }
        var results: [String] = []
        for i in order {
            if let minScore, Double(scores[i]) < minScore { continue }
            results.append(chunks[i])
            if results.count >= topK { break }
        }
        return Asset(items: results.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    static func cosineScores(vectors: [[Float]], query: [Float]) -> [Float] {
        let qNorm = sqrt(query.reduce(0) { $0 + $1 * $1 })
        guard qNorm > 0 else { return vectors.map { _ in 0 } }
        return vectors.map { vector in
            let vNorm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            // CFM-R12-FIX-5: replace only *exactly-zero* norms with 1 (Python's
            // `vector_norms[vector_norms == 0.0] = 1.0`) — clamping everything below 1
            // silently changed what ten RAG flows answer with.
            let norm = vNorm == 0 ? 1 : vNorm
            let dot = zip(vector, query).reduce(0) { $0 + $1.0 * $1.1 }
            return Float(Double(dot) / (Double(norm) * Double(qNorm)))
        }
    }
}

/// `Keyword Search` (index+text → text list): BM25 over the chunks, `top_k=`.
nonisolated struct KeywordSearchTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let allItems = inputs.flatMap { $0.items }
        guard let indexItem = allItems.first(where: { $0.kind == .index }),
              let textItem = allItems.first(where: { $0.kind == .text }),
              let indexDir = indexItem.path else {
            throw FlowError.badInputCardinality(row: "Keyword Search", expected: "(index, text)", got: inputs.count)
        }
        let manifestURL = indexDir.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            throw FlowError.fileReadFailed(row: "Keyword Search", path: indexDir.lastPathComponent)
        }
        _ = try IndexFormat.manifest(from: data)
        let chunks = try IndexFormat.chunks(from: Data(contentsOf: indexDir.appendingPathComponent("chunks.jsonl")))
        let query = textItem.value ?? ""
        let topK: Int
        if let raw = FlowSettings(settings).value(for: "top_k") {
            guard let v = Int(raw) else {
                throw FlowError.invalidSettings(row: "Keyword Search", setting: "top_k", detail: "isn't an integer")
            }
            topK = v
        } else {
            topK = 5
        }

        let scores = Self.bm25(chunks: chunks, query: query)
        // FIX-11: stable index tie-break (the Python's argsort).
        let order = (0..<scores.count).sorted {
            scores[$0] > scores[$1] || (scores[$0] == scores[$1] && $0 < $1)
        }
        let results = order.prefix(topK).filter { scores[$0] > 0 }.map { chunks[$0] }
        return Asset(items: results.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    static func tokenize(_ text: String) -> [String] {
        // The Python `_WORD_RE = re.compile(r"\w+")` — word chars, lowercased.
        var out: [String] = []
        var current = ""
        for scalar in text.lowercased().unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "_" {
                current.unicodeScalars.append(scalar)
            } else {
                if !current.isEmpty { out.append(current); current = "" }
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    static func bm25(chunks: [String], query: String, k1: Double = 1.5, b: Double = 0.75) -> [Double] {
        let docs = chunks.map { tokenize($0) }
        let n = docs.count
        guard n > 0 else { return [] }
        let totalTerms = Double(docs.reduce(0) { $0 + $1.count })
        let avgdl = totalTerms > 0 ? totalTerms / Double(n) : 1.0

        var docFreq: [String: Int] = [:]
        for doc in docs { for term in Set(doc) { docFreq[term, default: 0] += 1 } }
        let idf = docFreq.mapValues { freq in
            log(1 + (Double(n) - Double(freq) + 0.5) / (Double(freq) + 0.5))
        }

        let queryTerms = tokenize(query)
        return docs.map { doc in
            var termFreq: [String: Int] = [:]
            for term in doc { termFreq[term, default: 0] += 1 }
            let denomNorm = k1 * (1 - b + b * Double(doc.count) / avgdl)
            var score = 0.0
            for term in queryTerms {
                guard let tf = termFreq[term], let idfValue = idf[term] else { continue }
                score += idfValue * Double(tf) * (k1 + 1) / (Double(tf) + denomNorm)
            }
            return score
        }
    }
}

extension StoreIndexTool {
    /// SPEC-Q22: read the `embedder.json` sidecar when every vector came from the same
    /// directory (the Python's `engines/embed.py` writes it); nil otherwise.
    fileprivate static func autoEmbedder(_ vectorItems: [Item]) -> String? {
        guard let first = vectorItems.first?.path else { return nil }
        let parents = Set(vectorItems.compactMap { $0.path?.deletingLastPathComponent() })
        guard parents.count == 1, let dir = parents.first else { return nil }
        let sidecar = dir.appendingPathComponent("embedder.json")
        guard let data = try? Data(contentsOf: sidecar),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedder = json["embedder"] as? String else { return nil }
        return embedder
    }
}
