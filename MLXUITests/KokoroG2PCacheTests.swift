import Testing
import Foundation
import HuggingFace
import MLXAudioCore
@testable import MLXUI

/// KG-2 (R34): before synthesis the engine makes the mlx-audio-swift cache folder a current copy of `<Kokoro dir>/g2p/` plus a
/// stub `config.json`, only when it is missing or differs, and fails with one plain sentence when `g2p/` is missing. No test here
/// touches a real cache folder or the network.
@Suite(.serialized)
struct KokoroG2PCacheTests {

    private let names = ["us_bart.safetensors", "us_bart_config.json", "us_gold.json", "us_silver.json"]

    /// A Kokoro model folder with (optionally) a `g2p/` of four small files, and an empty cache root.
    private func fixture(g2p: Bool = true) throws -> (model: URL, cache: URL, root: URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("kg2-\(UUID().uuidString)", isDirectory: true)
        let model = root.appendingPathComponent("model", isDirectory: true)
        let cache = root.appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        if g2p {
            let dir = model.appendingPathComponent("g2p", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for (i, name) in names.enumerated() {
                try Data((0..<(2_000 + i * 100)).map { UInt8(($0 &* 31 &+ i) % 251) }).write(to: dir.appendingPathComponent(name))
            }
        }
        return (model, cache, root)
    }

    private func folder(_ cache: URL) -> URL { KokoroG2PCache.cacheFolder(cacheRoot: cache) }

    // MARK: copy only when missing or mismatched

    @Test func theFirstCallCopiesAndTheSecondDoesNot() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .copied)
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .alreadyCurrent)
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .alreadyCurrent)
        for name in names {
            #expect(try Data(contentsOf: folder(f.cache).appendingPathComponent(name)) == Data(contentsOf: f.model.appendingPathComponent("g2p/\(name)")))
        }
    }

    @Test func theStubConfigIsExactlyAnEmptyObjectAndANewline() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache)
        #expect(try Data(contentsOf: folder(f.cache).appendingPathComponent("config.json")) == Data([0x7B, 0x7D, 0x0A]))
    }

    @Test func aCorruptedFileOfTheSameSizeIsDetectedAndRecreated() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache)
        let victim = folder(f.cache).appendingPathComponent("us_gold.json")
        var bytes = try Data(contentsOf: victim)
        bytes[10] ^= 0xFF                                            // same size, different content
        try bytes.write(to: victim)
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .copied)
        #expect(try Data(contentsOf: victim) == Data(contentsOf: f.model.appendingPathComponent("g2p/us_gold.json")))
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .alreadyCurrent)
    }

    @Test func aTruncatedMissingOrUnstubbedCacheIsRecreated() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache)
        try Data([1, 2, 3]).write(to: folder(f.cache).appendingPathComponent("us_silver.json"))      // truncated
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .copied)
        try FileManager.default.removeItem(at: folder(f.cache).appendingPathComponent("us_bart.safetensors"))   // missing
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .copied)
        try FileManager.default.removeItem(at: folder(f.cache).appendingPathComponent("config.json"))           // no stub
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .copied)
        try Data("{\"a\":1}".utf8).write(to: folder(f.cache).appendingPathComponent("config.json"))            // wrong stub
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .copied)
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .alreadyCurrent)
    }

    @Test func extraFilesInTheCacheFolderAreIgnored() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache)
        let extra = folder(f.cache).appendingPathComponent("gb_gold.json")
        try Data("x".utf8).write(to: extra)
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .alreadyCurrent)
        #expect(FileManager.default.fileExists(atPath: extra.path), "a current copy is left exactly as it is")
    }

    @Test func aStaleCopyFromAnEarlierPackageRunIsReplacedAndNoStagingFolderIsLeft() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try FileManager.default.createDirectory(at: folder(f.cache), withIntermediateDirectories: true)
        try Data("old".utf8).write(to: folder(f.cache).appendingPathComponent("us_gold.json"))
        try Data("{}".utf8).write(to: folder(f.cache).appendingPathComponent(".mlx-audio-snapshot.json"))
        #expect(try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) == .copied)
        #expect(!FileManager.default.fileExists(atPath: folder(f.cache).appendingPathComponent(".mlx-audio-snapshot.json").path))
        let siblings = try FileManager.default.contentsOfDirectory(atPath: folder(f.cache).deletingLastPathComponent().path)
        #expect(siblings == [KokoroG2PCache.repoFolderName], "no temp folder left behind: \(siblings)")
    }

    // MARK: missing g2p/

    private let sentence = "Kokoro is missing some files it needs to read text aloud. In the sidebar under Installed, click Kokoro-82M, then click Uninstall, then click Install."

    @Test func aMissingG2PFolderGivesTheOnePlainSentenceAndCreatesNothing() throws {
        let f = try fixture(g2p: false); defer { try? FileManager.default.removeItem(at: f.root) }
        do {
            try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache)
            Issue.record("expected a refusal")
        } catch {
            #expect(error.localizedDescription == sentence)
        }
        #expect(!FileManager.default.fileExists(atPath: folder(f.cache).path), "nothing is created in the cache")
    }

    @Test func anIncompleteOrEmptyG2PFolderGivesTheSameSentence() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try FileManager.default.removeItem(at: f.model.appendingPathComponent("g2p/us_silver.json"))
        #expect(throws: KokoroG2PCache.G2PError.notInstalled) { try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) }
        try Data().write(to: f.model.appendingPathComponent("g2p/us_silver.json"))                    // zero bytes
        #expect(throws: KokoroG2PCache.G2PError.notInstalled) { try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache) }
    }

    @Test func theSentenceHasNoPathAndNoJargonAndSurvivesTheErrorGuardAsWritten() {
        let text = KokoroG2PCache.G2PError.notInstalled.localizedDescription
        #expect(text == sentence)
        #expect(!text.contains("/") && !text.contains("g2p") && !text.contains("~") && !text.lowercased().contains("huggingface"))
        #expect(MLXErrorGuard.sentence(for: KokoroG2PCache.G2PError.notInstalled) == sentence)
    }

    /// The shared engine entry (flow rows, the Run screen) raises it before loading anything or touching the network.
    @Test func synthesizeFailsWithTheSentenceWhenTheInstallLacksG2P() async throws {
        let f = try fixture(g2p: false); defer { try? FileManager.default.removeItem(at: f.root) }
        do {
            _ = try await KokoroEngine.synthesize("Hello there.", modelDirectory: f.model)
            Issue.record("expected a refusal")
        } catch {
            #expect(error.localizedDescription == sentence, "got: \(error)")
        }
    }

    // MARK: the package dependency (R35)

    /// Guards the assumption KG-2 rests on: with the seeded folder, mlx-audio-swift's resolve path (the call
    /// `MisakiTextProcessor.prepare()` makes) takes its "Using cached model" branch and never asks the Hub. The client's host is a
    /// loopback port nothing listens on, so any request would fail the call. If a package bump changes `ModelUtils` (R35), this fails first.
    @Test func theSeededFolderMakesThePackageResolveWithoutTheNetwork() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache)
        let cache = HubCache(cacheDirectory: f.cache)
        let client = HubClient(host: URL(string: "http://127.0.0.1:9")!, cache: cache)
        let dir = try await ModelUtils.resolveOrDownloadModel(
            client: client, cache: cache,
            repoID: Repo.ID(namespace: "beshkenadze", name: "kitten-tts-g2p"), requiredExtension: "safetensors")
        #expect(dir.standardizedFileURL == folder(f.cache).standardizedFileURL)
    }

    /// Control: the same folder without the stub `config.json` is NOT trusted and the call goes to the (unreachable) host, so it throws.
    @Test func withoutTheStubConfigThePackageAsksTheHubAndFailsOffline() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try KokoroG2PCache.ensure(modelDirectory: f.model, cacheRoot: f.cache)
        try FileManager.default.removeItem(at: folder(f.cache).appendingPathComponent("config.json"))
        let cache = HubCache(cacheDirectory: f.cache)
        let client = HubClient(host: URL(string: "http://127.0.0.1:9")!, cache: cache)
        await #expect(throws: (any Error).self) {
            _ = try await ModelUtils.resolveOrDownloadModel(
                client: client, cache: cache,
                repoID: Repo.ID(namespace: "beshkenadze", name: "kitten-tts-g2p"), requiredExtension: "safetensors")
        }
    }
}
