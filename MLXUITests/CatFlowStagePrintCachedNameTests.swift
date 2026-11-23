import Testing
import Foundation
@testable import MLXUI

/// ER-FIX-7 (owner, Boo, 2026-11-23): when `Make PDF`'s result is served from the flow's saved results, the file on disk is the cache's
/// `<row>.<i>.file.<hash>.bin` copy; the `Stage Print` outbox entry must still be labelled with the row's real name. The entry's id is
/// the content's alone and never reads the name.
struct CatFlowStagePrintCachedNameTests {
    private func scratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("erfix7-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private struct Staged { var id: String; var fileName: String; var text: String; var stored: String; var storedBytes: Data }

    private func onlyEntry(_ outbox: URL) throws -> Staged {
        let jsons = try FileManager.default.contentsOfDirectory(at: outbox, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
        #expect(jsons.count == 1)
        let url = try #require(jsons.first)
        let entry = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let stored = try #require(entry["stored_file"] as? String)
        return Staged(id: try #require(entry["id"] as? String), fileName: try #require(entry["file_name"] as? String),
                      text: try #require(entry["text"] as? String), stored: stored,
                      storedBytes: try Data(contentsOf: outbox.appendingPathComponent("files").appendingPathComponent(stored)))
    }

    @Test func aCacheServedMakePDFIsLabelledByItsOwnNameNotTheBlobs() async throws {
        let base = try scratch()
        defer { try? FileManager.default.removeItem(at: base) }
        let blobs = base.appendingPathComponent("blobs")
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let flowFolder = workspace.directory(for: "erfix7")
        try FileManager.default.createDirectory(at: flowFolder, withIntermediateDirectories: true)
        try "Ship to: A. Example\n12 Example Road".write(to: flowFolder.appendingPathComponent("label.txt"), atomically: true, encoding: .utf8)
        let doc = try CatParser.parse("mlxflow 0.8\n1. Read Text  label.txt\n2. Make PDF\n3. Stage Print\n")
        let real = RealExecutor(workspace: workspace, flowID: "erfix7", blobDirectory: blobs,
                                makeModelStage: { _, _ in throw FlowError.unsupportedTask(row: "-", task: "model") },
                                installedModelIDs: [], catalog: [])
        let caching = CachingExecutor(inner: real, store: FlowCacheStore(root: base.appendingPathComponent("cache")), cacheTier: "real",
                                      catalog: [], runSeed: 0, workspace: workspace, flowID: "erfix7")
        let context = FlowRunner.RunContext(flowID: "erfix7", workspace: workspace, blobDirectory: blobs, executor: caching)
        let outbox = OutboxStore.directory(workspace: workspace, flowID: "erfix7")

        func run() async -> [FlowEvent] {
            var events: [FlowEvent] = []
            for await event in FlowRunner().run(doc, context: context) { events.append(event) }
            return events
        }
        let first = await run()
        #expect(!first.contains { if case .failed = $0 { return true }; return false }, "\(first)")
        let fresh = try onlyEntry(outbox)
        #expect(fresh.fileName.hasPrefix("makepdf-") && fresh.fileName.hasSuffix(".pdf"), "\(fresh.fileName)")

        // The same flow again: Make PDF is a cache hit, so Stage Print is handed the cache's `.bin` copy.
        try FileManager.default.removeItem(at: outbox)
        let second = await run()
        #expect(second.contains { if case .cacheHit = $0 { return true }; return false }, "premise: served from the saved results")
        let cached = try onlyEntry(outbox)
        #expect(!cached.fileName.hasSuffix(".bin") && !cached.fileName.contains(".file."), "\(cached.fileName)")
        #expect(cached.fileName == fresh.fileName && cached.text == fresh.text)  // the row's real name, carried through the cache
        #expect(cached.id == fresh.id)                                           // the id is unchanged
        #expect(cached.storedBytes == fresh.storedBytes && cached.stored == "\(cached.id).pdf")
    }

    @Test func aNamelessCacheFileFallsBackToTheSniffedExtensionAndTheIdIgnoresTheName() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let pdf = Data("%PDF-1.7\n1 0 obj<<>>endobj\n%%EOF\n".utf8)
        func prepared(_ name: String, originalName: String? = nil) throws -> StagePrintTool.Prepared {
            let url = dir.appendingPathComponent(name)
            try pdf.write(to: url)
            let asset = Asset(items: [Item(kind: .file, value: nil, path: url, sourceText: nil, originalName: originalName)])
            return try StagePrintTool.prepare(settings: nil, inputs: [asset], path: "3")
        }
        let named = try prepared("makepdf-1234.pdf")
        let blob = try prepared("3.0.file.0123456789abcdef.bin")
        let carried = try prepared("3.0.file.0123456789abcdef.bin", originalName: "makepdf-1234.pdf")
        #expect(named.name == "makepdf-1234.pdf" && carried.name == "makepdf-1234.pdf")
        #expect(blob.name.hasSuffix(".pdf") && !blob.name.hasSuffix(".bin"))
        #expect(named.id == blob.id && blob.id == carried.id)
    }

    /// A recorded file name is provenance only: the row's cache key is the same with and without it.
    @Test func theCacheKeyIgnoresARecordedFileName() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("3.0.file.0123456789abcdef.bin")
        try Data("%PDF-1.7\n%%EOF\n".utf8).write(to: url)
        let bare = Asset(items: [Item(kind: .file, value: nil, path: url, sourceText: nil)])
        let named = Asset(items: [Item(kind: .file, value: nil, path: url, sourceText: nil, originalName: "makepdf-1234.pdf")])
        let a = try CacheKey.cacheKey(task: "Stage Print", model: nil, settings: nil, inputs: [bare], realism: "real")
        let b = try CacheKey.cacheKey(task: "Stage Print", model: nil, settings: nil, inputs: [named], realism: "real")
        #expect(a == b)
        #expect(try bare.contentHash() == named.contentHash())
    }
}
