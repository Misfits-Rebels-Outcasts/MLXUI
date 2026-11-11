import Testing
import Foundation
@testable import MLXUI

/// MT1-3 checkpoint (a): the per-conversation asset store and the `@TaskLocal` host.
@Suite struct ChatAssetStoreTests {

    private func tempRoot() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-assets-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func handlesNumberFromA1AndAreNeverReused() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatAssetStore(conversationID: UUID(), in: root)
        let a = try store.addText("one two", name: "first")
        let b = try store.addText("three", name: "second")
        #expect(a.handle == "a1")
        #expect(b.handle == "a2")
        #expect(a.words == 2)
        #expect(store.record("A2")?.name == "second")           // case-insensitive
        #expect(store.record(" a1 ")?.name == "first")
        #expect(store.record("a9") == nil)
    }

    @Test func indexSurvivesAFreshStoreInstance() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let first = ChatAssetStore(conversationID: id, in: root)
        let prov = ChatAssetProvenance(rowText: "1. Transcribe   Parakeet TDT 0.6B v3", model: "Parakeet TDT 0.6B v3",
                                       inputs: ["a1"], milliseconds: 812)
        let rec = try first.addText("hello world again", name: "transcript", provenance: prov)

        let second = ChatAssetStore(conversationID: id, in: root)
        #expect(second.all == [rec])
        #expect(second.record("a1")?.provenance == prov)
        #expect(second.text(for: "a1") == "hello world again")
        // numbering continues after a relaunch
        #expect(try second.addText("x", name: "n").handle == "a2")
    }

    @Test func adoptMovesTheBlobIntoTheAssetsFolderAsAN() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatAssetStore(conversationID: UUID(), in: root)
        try FileManager.default.createDirectory(at: store.blobDirectory, withIntermediateDirectories: true)
        let blob = store.blobDirectory.appendingPathComponent("model.\(UUID().uuidString).png")
        try Data([1, 2, 3, 4]).write(to: blob)
        let rec = try store.adopt(file: blob, kind: "image", name: "picture")
        #expect(rec.handle == "a1")
        #expect(rec.fileName == "a1.png")
        #expect(rec.bytes == 4)
        #expect(!FileManager.default.fileExists(atPath: blob.path))
        #expect(FileManager.default.fileExists(atPath: store.url(for: rec).path))
        #expect(store.url(for: rec).deletingLastPathComponent().lastPathComponent == "assets")
    }

    @Test func importCopiesAndLeavesTheSource() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("receipt.jpg")
        try Data([9, 9]).write(to: source)
        let store = ChatAssetStore(conversationID: UUID(), in: root)
        let rec = try store.importFile(source, kind: "image")
        #expect(rec.name == "receipt.jpg")
        #expect(FileManager.default.fileExists(atPath: source.path))
    }

    @Test func aMissingFileDropsOutOfTheIndexOnReload() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let store = ChatAssetStore(conversationID: id, in: root)
        let rec = try store.addText("gone", name: "g")
        try FileManager.default.removeItem(at: store.url(for: rec))
        #expect(ChatAssetStore(conversationID: id, in: root).all.isEmpty)
    }

    @Test func aVanishedNewestAssetsHandleIsNeverReused() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let store = ChatAssetStore(conversationID: id, in: root)
        _ = try store.addText("one", name: "n")
        let newest = try store.addText("two", name: "n")
        try FileManager.default.removeItem(at: store.url(for: newest))
        let reloaded = ChatAssetStore(conversationID: id, in: root)
        #expect(reloaded.all.map(\.handle) == ["a1"])
        #expect(try reloaded.addText("three", name: "n").handle == "a3")
    }

    @Test func theFirstShapeOfIndexJSONStillLoads() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let store = ChatAssetStore(conversationID: id, in: root)
        let rec = try store.addText("old", name: "n")
        let array = try JSONEncoder().encode([rec])   // a bare array, dates as numbers: the encoder default differs, so re-encode iso8601
        _ = array
        let enc = JSONEncoder(); enc.dateEncodingStrategy = .iso8601
        try enc.encode([rec]).write(to: store.directory.appendingPathComponent("index.json"))
        let again = ChatAssetStore(conversationID: id, in: root)
        #expect(again.all == [rec])
        #expect(try again.addText("new", name: "n").handle == "a2")
    }

    @Test func deletingAConversationRemovesItsAssets() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let store = ChatAssetStore(conversationID: id, in: root)
        try store.addText("keep me not", name: "t")
        #expect(FileManager.default.fileExists(atPath: store.directory.path))
        ConversationStore.delete(id: id, in: root)
        #expect(!FileManager.default.fileExists(atPath: store.directory.path))
        #expect(ChatAssetStore(conversationID: id, in: root).all.isEmpty)
    }

    @Test func handleDetection() {
        #expect(ChatAssetStore.looksLikeHandle("a3"))
        #expect(ChatAssetStore.looksLikeHandle(" A12 "))
        #expect(!ChatAssetStore.looksLikeHandle("a"))
        #expect(!ChatAssetStore.looksLikeHandle("abc"))
        #expect(!ChatAssetStore.looksLikeHandle("https://example.com/a1"))
    }

    // MARK: host

    @Test func noHostIsBoundByDefault() {
        #expect(ModelToolHost.current == nil)
    }

    @Test func hostIsVisibleInsideWithValueOnly() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root))
        let inside = await ModelToolHost.$current.withValue(host) { ModelToolHost.current != nil }
        #expect(inside)
        #expect(ModelToolHost.current == nil)
    }

    @Test func mt1InstallRequestAlwaysDeclines() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root))
        let need = ModelInstallNeed(toolName: "transcribe_audio", modelDisplayName: "Parakeet TDT 0.6B v3",
                                    modelID: "m", downloadGB: 0.6)
        #expect(await host.installRequest(need) == false)
        #expect(ModelToolHost.declineSentence(for: "Parakeet TDT 0.6B v3")
                == "The tool did not run and produced nothing: the user chose not to download Parakeet TDT 0.6B v3, so no result exists. Do not say the task was done. Tell the user it was not done, and that they can ask again and choose Download & continue to install Parakeet TDT 0.6B v3, or ask for something that needs no download.")
    }
}
