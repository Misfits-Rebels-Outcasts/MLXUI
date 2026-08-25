import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-6 — the retrieval cluster (`Store Index` / `Read Index` / `Retrieve` / `Keyword
/// Search`), tested against the Python-generated golden index (`Fixtures/CatFlow/tools/
/// idxsmall/`, 4 chunks × 8 dims) byte-for-byte where the format is binary.
struct CatFlowIndexStoreTests {

    /// `Fixtures/CatFlow/tools/idxsmall` (repo root, outside any target) — the Python's
    /// `store_index` output for 4 known chunks and 4 one-hot 8-dim vectors.
    private func goldenIndexDir() throws -> URL {
        let filePath = #filePath
        let url = URL(fileURLWithPath: filePath)
        var dir = url.deletingLastPathComponent()
        while dir.lastPathComponent != "MLXUITests" { dir = dir.deletingLastPathComponent() }
        return dir.deletingLastPathComponent().appendingPathComponent("Fixtures/CatFlow/tools/idxsmall")
    }

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-idx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    // MARK: - Store Index writes the Python format byte for byte

    @Test func storeIndexWritesThePythonFormatByteForByte() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let chunks = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning models run on apple silicon",
            "rotate the api keys before they expire",
            "the meeting minutes were sent to everyone",
        ]
        let oneHots: [[Float]] = [
            [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
        ]
        // Vector items as .npy in the flow folder.
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let vectorItems = try oneHots.enumerated().map { i, v in
            let url = flowDir.appendingPathComponent("v\(i).npy")
            try NpyCodec.save(v, to: url)
            return Item(kind: .vector, value: nil, path: url, sourceText: nil)
        }
        let textItems = chunks.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) }

        let tool = StoreIndexTool(workspace: ws, flowID: "f", settings: "myindex")
        _ = try await tool.run(inputs: [Asset(items: textItems), Asset(items: vectorItems)])

        let out = ws.directory(for: "f").appendingPathComponent("myindex")
        let golden = try goldenIndexDir()
        for name in ["vectors.bin", "chunks.jsonl", "manifest.json"] {
            let written = try Data(contentsOf: out.appendingPathComponent(name))
            let expected = try Data(contentsOf: golden.appendingPathComponent(name))
            #expect(written == expected, "\(name) differs from the Python golden")
        }
    }

    // MARK: - Retrieve against the Python golden

    @Test func retrieveMatchesThePythonGolden() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let idx = try goldenIndexDir()
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let queryURL = flowDir.appendingPathComponent("q.npy")
        try NpyCodec.save([0, 0.9, 0.1, 0, 0, 0, 0, 0], to: queryURL)

        let tool = RetrieveTool(workspace: ws, flowID: "f", settings: "top_k=2")
        let out = try await tool.run(inputs: [
            Asset(items: [Item(kind: .index, value: nil, path: idx, sourceText: nil)]),
            Asset(items: [Item(kind: .vector, value: nil, path: queryURL, sourceText: nil)]),
        ])
        let texts = out.items.map { $0.value ?? "" }
        // The Python's golden for this query: chunk 1 then chunk 2.
        #expect(texts == [
            "machine learning models run on apple silicon",
            "rotate the api keys before they expire",
        ])
    }

    @Test func keywordSearchMatchesThePythonGolden() async throws {
        let idx = try goldenIndexDir()
        let tool = KeywordSearchTool(workspace: FlowWorkspace(root: URL(fileURLWithPath: "/tmp")),
                                     flowID: "f", settings: "top_k=2")
        let out = try await tool.run(inputs: [
            Asset(items: [Item(kind: .index, value: nil, path: idx, sourceText: nil)]),
            Asset(items: [Item(kind: .text, value: "api keys", path: nil, sourceText: nil)]),
        ])
        #expect(out.items.map { $0.value } == ["rotate the api keys before they expire"])
    }

    // MARK: - The format helpers

    @Test func npyRoundTrips() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-npy-\(UUID().uuidString).npy")
        defer { try? FileManager.default.removeItem(at: url) }
        let values: [Float] = [0.1, -0.5, 1.0, 0.0]
        try NpyCodec.save(values, to: url)
        let loaded = try NpyCodec.load(from: url)
        #expect(loaded.count == values.count)
        for (a, b) in zip(loaded, values) { #expect(abs(a - b) < 0.0001) }
    }

    /// CFM-R12-FIX-6: the `.npy` writer matches `np.save` — 64-byte-aligned header, dict
    /// then **space padding then a trailing `\n`** (numpy's exact layout), no NUL bytes.
    @Test func npyWriterMatchesNumpyLayout() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-npy-\(UUID().uuidString).npy")
        defer { try? FileManager.default.removeItem(at: url) }
        try NpyCodec.save([1, 2, 3], to: url)
        let data = try Data(contentsOf: url)
        let headerLen = Int(UInt16(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt16.self) }))
        // Magic + version + len + header must land the payload on a 64-byte boundary.
        #expect((10 + headerLen) % 64 == 0)
        let header = data[10..<(10 + headerLen)]
        #expect(!header.contains(0x00))          // no NUL padding (numpy rejects it)
        #expect(header.last == 0x0A)             // trailing \n after the spaces
        #expect(header[header.count - 2] == 0x20) // the padding is spaces before it
        let dict = String(data: header, encoding: .ascii) ?? ""
        #expect(dict.contains("'<f4'"))
        #expect(dict.contains("(3,)"))
        // Total matches numpy's own 3-vector file (128-byte header + 12 data).
        #expect(data.count == 140)
    }

    /// CFM-R12-FIX-6: loading a float64 `.npy` the Python saved converts to float32.
    @Test func npyLoadsFloat64() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-npy64-\(UUID().uuidString).npy")
        defer { try? FileManager.default.removeItem(at: url) }
        let dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }\n"
        var headerLen = UInt16(dict.utf8.count)
        while (10 + Int(headerLen)) % 64 != 0 { headerLen += 1 }
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        data.append(contentsOf: withUnsafeBytes(of: headerLen.littleEndian) { Array($0) })
        data.append(Data(dict.utf8))
        data.append(Data(repeating: 0x20, count: Int(headerLen) - dict.utf8.count))
        var doubles: [Double] = [1.5, -2.25]
        doubles.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
        let loaded = try NpyCodec.load(from: url)
        #expect(loaded == [1.5, -2.25])
    }

    /// CFM-R12-FIX-5: only exactly-zero norms are clamped — the reviewer's case.
    @Test func retrieveDoesNotClampNormsBelowOne() {
        let vectors: [[Float]] = [[0.1, 0.1, 0.1], [0.6, 0, 0], [0.05, 0.05, 0]]
        let query: [Float] = [1, 0, 0]
        let scores = RetrieveTool.cosineScores(vectors: vectors, query: query)
        let order = (0..<scores.count).sorted { scores[$0] > scores[$1] }
        // Python: scores [0.57735, 1.0, 0.707107], ranking [1, 2, 0].
        #expect(abs(scores[0] - 0.57735) < 0.001)
        #expect(abs(scores[1] - 1.0) < 0.001)
        #expect(abs(scores[2] - 0.707107) < 0.001)
        #expect(order == [1, 2, 0])
    }

    @Test func readIndexValidatesTheDirectory() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let idx = try goldenIndexDir()

        let good = ReadIndexTool(workspace: ws, flowID: "f", settings: idx.lastPathComponent)
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: idx, to: flowDir.appendingPathComponent(idx.lastPathComponent))
        let out = try await good.run(Asset(items: [])) { _ in }
        #expect(out.items.first?.kind == .index)

        // A non-index directory refuses.
        let plain = flowDir.appendingPathComponent("plain")
        try FileManager.default.createDirectory(at: plain, withIntermediateDirectories: true)
        let bad = ReadIndexTool(workspace: ws, flowID: "f", settings: "plain")
        await #expect(throws: FlowError.self) {
            _ = try await bad.run(Asset(items: [])) { _ in }
        }
    }

    /// R12-6's done-when: the RAG family reads `library.index` (bundled) + Retrieve. The
    /// flows whose only blockers those were are runnable now: 17/19/44. 41/42/45/20 also
    /// need Calculate (R12-7 group a), 18 needs Save Context, 46 Read Context (group b).
    @Test func ragFamilyUnblockedByRetrievalIsRunnable() throws {
        let runnable = ["17-AskYourDocs", "19-CorrectiveRag", "44-FrontierEscalate"]
        for fid in runnable {
            let doc = try GalleryLoader.loadDocument(flowID: fid)
            #expect(FlowRunner.canRun(doc) == .runnable, "\(fid) should be runnable after R12-6")
        }
    }

    /// CFM-R12-FIX-11: invalid `top_k=` fails the row instead of silently defaulting.
    @Test func retrieveRefusesInvalidTopK() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let idx = try goldenIndexDir()
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let queryURL = flowDir.appendingPathComponent("q.npy")
        try NpyCodec.save([1, 0, 0, 0, 0, 0, 0, 0], to: queryURL)
        let tool = RetrieveTool(workspace: ws, flowID: "f", settings: "top_k=abc")
        await #expect(throws: FlowError.self) {
            _ = try await tool.run(inputs: [
                Asset(items: [Item(kind: .index, value: nil, path: idx, sourceText: nil)]),
                Asset(items: [Item(kind: .vector, value: nil, path: queryURL, sourceText: nil)]),
            ])
        }
    }

    /// CFM-R12-FIX-11: corrupt chunks.jsonl raises (alignment must not shift).
    @Test func corruptChunkLineRaises() async throws {
        let data = Data("{\"text\": \"ok\"}\ngarbage\n{\"text\": \"two\"}".utf8)
        await #expect(throws: FlowError.self) {
            _ = try IndexFormat.chunks(from: data)
        }
    }

    /// CFM-R12-FIX-11: `asciiJSON` escapes non-ASCII like Python's ensure_ascii.
    @Test func asciiJSONEscapesNonASCII() {
        #expect(FlowParity.asciiJSON("café") == "\"caf\\u00e9\"")
        #expect(FlowParity.asciiJSON("plain") == "\"plain\"")
    }

}
