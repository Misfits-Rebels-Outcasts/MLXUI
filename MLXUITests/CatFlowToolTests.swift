import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R2-4: the four instant tools as `AssetStage`s, round-tripping through a temp
/// workspace, with the CFM-R1-4 path boundary re-asserted at this layer. See
/// `RSI/DelegateMergeBacklog.md` CFM-R2-4.
struct CatFlowToolTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-tools-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    // MARK: - Read Audio

    @Test func readAudioProducesAudioAsset() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        // A tiny real WAV in the flow dir.
        let buffer = AudioBuffer(samples: [0.1, 0.2, -0.1], sampleRate: 24_000)
        let flowDir = ws.directory(for: "01-SpokenSummary")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let wav = flowDir.appendingPathComponent("memo.m4a")
        try AudioWriter.writeWAV(buffer, to: wav)

        let tool = ReadAudioTool(workspace: ws, flowID: "01-SpokenSummary", settings: "memo.m4a")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let item = try #require(out.items.first)
        #expect(item.kind == .audio)
        let readBack = try AudioFileReader.read(try #require(item.path))
        #expect(readBack.sampleRate == 24_000)
    }

    // MARK: - Save Audio

    @Test func saveAudioWritesReadableWAVAtSameRate() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        // The flow produces a file-backed audio blob (24 kHz, as Kokoro does).
        let buffer = AudioBuffer(samples: [0.5, -0.5, 0.25], sampleRate: 24_000)
        let blobDir = ws.directory(for: "01-SpokenSummary")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        let blob = blobDir.appendingPathComponent("speak.blob.wav")
        try AudioWriter.writeWAV(buffer, to: blob)

        let tool = SaveAudioTool(workspace: ws, flowID: "01-SpokenSummary", settings: "memo-tldr.wav")
        let input = Asset(items: [Item(kind: .audio, value: nil, path: blob, sourceText: nil)])
        let out = try await tool.run(input) { _ in }
        let status = try #require(out.items.first)
        #expect(status.kind == .status)
        #expect(status.value?.contains("memo-tldr.wav") == true)

        // Round-trip the saved WAV back at the same rate.
        let saved = try AudioFileReader.read(blobDir.appendingPathComponent("memo-tldr.wav"))
        #expect(saved.sampleRate == 24_000)
        #expect(saved.samples.count == buffer.samples.count)
    }

    // MARK: - Save Text

    @Test func saveTextWritesUTF8() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "01-SpokenSummary")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)

        let tool = SaveTextTool(workspace: ws, flowID: "01-SpokenSummary", settings: "memo-transcript.txt")
        let input = Asset(items: [Item(kind: .text, value: "the transcript", path: nil, sourceText: nil)])
        let out = try await tool.run(input) { _ in }
        let status = try #require(out.items.first)
        #expect(status.kind == .status)
        let written = try String(contentsOf: flowDir.appendingPathComponent("memo-transcript.txt"), encoding: .utf8)
        #expect(written == "the transcript")
    }

    // MARK: - Read Text

    @Test func readTextReadsUTF8() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "08-PolicyDiff")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        try "policy text".write(to: flowDir.appendingPathComponent("policy-2025.md"), atomically: true, encoding: .utf8)

        let tool = ReadTextTool(workspace: ws, flowID: "08-PolicyDiff", settings: "policy-2025.md")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let item = try #require(out.items.first)
        #expect(item.kind == .text)
        #expect(item.value == "policy text")
    }

    // MARK: - Path escape throws at this layer too (CFM-R1-4 boundary re-asserted)

    @Test func readAudioRejectsPathEscape() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let tool = ReadAudioTool(workspace: ws, flowID: "01-SpokenSummary", settings: "../../../etc/passwd")
        await #expect(throws: FlowWorkspaceError.self) {
            _ = try await tool.run(Asset(items: [])) { _ in }
        }
    }

    @Test func saveTextRejectsAbsolutePath() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let tool = SaveTextTool(workspace: ws, flowID: "01-SpokenSummary", settings: "/tmp/escape.txt")
        let input = Asset(items: [Item(kind: .text, value: "x", path: nil, sourceText: nil)])
        await #expect(throws: FlowWorkspaceError.self) {
            _ = try await tool.run(input) { _ in }
        }
    }

    // MARK: - FlowSettings

    @Test func settingsParsesBareAndKeyed() {
        let s = FlowSettings("memo.m4a; lang=en")
        #expect(s.pathValue() == "memo.m4a")
        #expect(s.value(for: "lang") == "en")
    }

    @Test func settingsPathOverridesBare() {
        let s = FlowSettings("bare.txt; path=explicit.txt")
        #expect(s.pathValue() == "explicit.txt")
    }

    // MARK: - FlowSettings tokenizer golden (CFM-FIX-2 / H5)

    @Test func settingsMatchesPythonTokenizerGolden() throws {
        let filePath = #filePath
        let url = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow/tools/settings_golden.json")
        let data = try Data(contentsOf: url)
        struct Root: Decodable { var cases: [String: Case] }
        struct Case: Decodable {
            var pairs: [String: String]
            var multi: [String: [String]]
            var bare: [String]
        }
        let root = try JSONDecoder().decode(Root.self, from: data)
        #expect(root.cases.count == 23)

        for (raw, c) in root.cases {
            let s = FlowSettings(raw)
            for (key, expected) in c.pairs {
                #expect(s.value(for: key) == expected,
                        "\(raw): value(for: \(key)) — Swift \(s.value(for: key) ?? "nil") vs Python \(expected)")
            }
            for (key, expected) in c.multi {
                #expect(s.multi(for: key) == expected,
                        "\(raw): multi(\(key)) — Swift \(s.multi(for: key)) vs Python \(expected)")
            }
            #expect(s.testKeys == Set(c.multi.keys),
                    "\(raw): key set — Swift \(s.testKeys) vs Python \(Set(c.multi.keys))")
            // bare tokens must match exactly (order preserved)
            #expect(s.testBare == c.bare,
                    "\(raw): bare — Swift \(s.testBare) vs Python \(c.bare)")
        }
    }
}
