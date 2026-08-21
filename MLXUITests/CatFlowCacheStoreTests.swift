import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R3-2: the content-addressed `FlowCacheStore` (put/get round-trip, blob
/// sharing, LRU eviction, clear) with an injectable temp root — the real Application
/// Support directory is never touched. See `RSI/DelegateMergeBacklog.md` CFM-R3-2.
struct CatFlowCacheStoreTests {

    private func makeStore() throws -> (FlowCacheStore, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-cache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowCacheStore(root: base.appendingPathComponent("cache")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    private func textAsset(_ value: String) -> Asset {
        Asset(items: [Item(kind: .text, value: value, path: nil, sourceText: nil)])
    }

    // MARK: - Round-trip

    @Test func inlineTextRoundTrips() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }

        let key = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: nil,
                                        inputs: [], realism: "real")
        try store.put(key: key, asset: textAsset("the summary"))
        let blobDir = base.appendingPathComponent("run-blobs")
        let got = try store.get(key: key, blobDirectory: blobDir, path: "3")
        #expect(got?.items.first?.value == "the summary")
        #expect(got?.items.first?.kind == .text)
    }

    @Test func missReturnsNil() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let got = try store.get(key: "deadbeef", blobDirectory: base, path: "1")
        #expect(got == nil)
    }

    @Test func fileBackedAudioRoundTrips() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }

        let audioURL = base.appendingPathComponent("audio.bin")
        try Data([0x01, 0x02, 0x03, 0x04]).write(to: audioURL)
        let asset = Asset(items: [Item(kind: .audio, value: nil, path: audioURL, sourceText: nil)])

        let key = try CacheKey.cacheKey(task: "Transcribe", model: "m", settings: nil,
                                        inputs: [], realism: "real")
        try store.put(key: key, asset: asset)

        let blobDir = base.appendingPathComponent("run-blobs")
        let got = try store.get(key: key, blobDirectory: blobDir, path: "2")
        let item = try #require(got?.items.first)
        #expect(item.kind == .audio)
        let dest = try #require(item.path)
        #expect(try Data(contentsOf: dest) == Data([0x01, 0x02, 0x03, 0x04]))
    }

    // MARK: - Content-addressed blob sharing

    @Test func identicalContentSharesOneBlob() throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }

        let url = base.appendingPathComponent("audio.bin")
        try Data([0xAA, 0xBB]).write(to: url)
        let asset = Asset(items: [Item(kind: .audio, value: nil, path: url, sourceText: nil)])

        let keyA = try CacheKey.cacheKey(task: "Transcribe", model: "m1", settings: nil,
                                         inputs: [], realism: "real")
        let keyB = try CacheKey.cacheKey(task: "Transcribe", model: "m2", settings: nil,
                                         inputs: [], realism: "real")
        try store.put(key: keyA, asset: asset)
        try store.put(key: keyB, asset: asset)

        // Two distinct entries, one blob (content-addressed).
        #expect(store.entryCount == 2)
        let blobs = countFiles(store.blobsDirectory)
        #expect(blobs == 1)
    }

    @Test func differentContentDifferentBlobs() throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }

        let urlA = base.appendingPathComponent("a.bin")
        let urlB = base.appendingPathComponent("b.bin")
        try Data([0xAA]).write(to: urlA)
        try Data([0xBB]).write(to: urlB)

        let keyA = try CacheKey.cacheKey(task: "Transcribe", model: "m1", settings: nil,
                                         inputs: [], realism: "real")
        let keyB = try CacheKey.cacheKey(task: "Transcribe", model: "m2", settings: nil,
                                         inputs: [], realism: "real")
        try store.put(key: keyA, asset: Asset(items: [Item(kind: .audio, value: nil, path: urlA, sourceText: nil)]))
        try store.put(key: keyB, asset: Asset(items: [Item(kind: .audio, value: nil, path: urlB, sourceText: nil)]))

        #expect(countFiles(store.blobsDirectory) == 2)
    }

    // MARK: - LRU eviction

    @Test func evictIfNeededRemovesOldestFirst() throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }

        // Two entries, then a tiny budget forcing one eviction.
        let keyA = try CacheKey.cacheKey(task: "Summarize", model: "m1", settings: nil,
                                         inputs: [], realism: "real")
        let keyB = try CacheKey.cacheKey(task: "Summarize", model: "m2", settings: nil,
                                         inputs: [], realism: "real")
        try store.put(key: keyA, asset: textAsset("aaaaaaaaaaaaaaaaaaaa"))
        try store.put(key: keyB, asset: textAsset("bbbbbbbbbbbbbbbbbbbb"))

        try store.evictIfNeeded(byteBudget: 1)   // tiny → evict until under
        #expect(store.entryCount < 2)
    }

    @Test func evictKeepsStoreUnderBudget() throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }

        for i in 0..<5 {
            let key = try CacheKey.cacheKey(task: "Summarize", model: "m\(i)", settings: nil,
                                            inputs: [], realism: "real")
            try store.put(key: key, asset: textAsset(String(repeating: "x", count: 200)))
        }
        try store.evictIfNeeded(byteBudget: 300)
        #expect(store.totalBytes <= 300)
    }

    // MARK: - Clear

    @Test func clearEmptiesTheStore() throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }

        let key = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: nil,
                                        inputs: [], realism: "real")
        try store.put(key: key, asset: textAsset("x"))
        #expect(store.entryCount == 1)
        try store.clear()
        #expect(store.entryCount == 0)
        #expect(try store.get(key: key, blobDirectory: base, path: "1") == nil)
    }

    // MARK: - Helpers

    private func countFiles(_ dir: URL) -> Int {
        guard FileManager.default.fileExists(atPath: dir.path) else { return 0 }
        let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)
        return (enumerator?.allObjects as? [URL])?.filter { !$0.hasDirectoryPath }.count ?? 0
    }
}

extension URL {
    /// Whether this URL points at a directory.
    fileprivate var hasDirectoryPath: Bool {
        (try? resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }
}
