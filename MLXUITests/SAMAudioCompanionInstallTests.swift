import Testing
import Foundation
@testable import MLXUI

/// MC1-3: the companion-repo install — T5-base lands in `t5/` inside SAM-Audio's model folder and leaves with it,
/// and MusicGen's EnCodec companion keeps `encodec/`. The file lists and sizes are the real HF repos (fetched
/// 2026-11-02, `api/models/{id}?blobs=true`), pinned so a change in what the installer fetches is a test failure.
@MainActor
struct SAMAudioCompanionInstallTests {

    struct RemoteFile { let name: String; let size: Int64 }

    /// `mlx-community/sam-audio-base` @ sha `614abc87`.
    static let samAudio: [RemoteFile] = [
        .init(name: ".gitattributes", size: 1_519),
        .init(name: "README.md", size: 2_041),
        .init(name: "config.json", size: 754),
        .init(name: "model-00001-of-00001.safetensors", size: 5_032_052_232),
        .init(name: "model.safetensors.index.json", size: 55_842),
    ]

    /// `google-t5/t5-base`, the companion. It also ships PyTorch / TF / Flax / Rust weights, which must not come down.
    static let t5: [RemoteFile] = [
        .init(name: ".gitattributes", size: 537),
        .init(name: "README.md", size: 8_477),
        .init(name: "config.json", size: 1_208),
        .init(name: "flax_model.msgpack", size: 891_625_348),
        .init(name: "generation_config.json", size: 147),
        .init(name: "model.safetensors", size: 891_646_390),
        .init(name: "pytorch_model.bin", size: 891_691_430),
        .init(name: "rust_model.ot", size: 891_679_884),
        .init(name: "spiece.model", size: 791_656),
        .init(name: "tf_model.h5", size: 892_146_080),
        .init(name: "tokenizer.json", size: 1_389_353),
    ]

    /// What the installer actually fetches from each repo, and the total the catalog entry shows.
    @Test func theSelectedFilesAndTheirTotalAreTheOnesTheCatalogShows() throws {
        let sam = ModelFileSelector.filesToDownload(siblings: Self.samAudio.map(\.name))
        let t5 = ModelFileSelector.filesToDownload(siblings: Self.t5.map(\.name))
        #expect(Set(sam) == ["config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json"])
        #expect(Set(t5) == ["config.json", "generation_config.json", "model.safetensors", "spiece.model", "tokenizer.json"])
        let samByName = Dictionary(uniqueKeysWithValues: Self.samAudio.map { ($0.name, $0.size) })
        let t5ByName = Dictionary(uniqueKeysWithValues: Self.t5.map { ($0.name, $0.size) })
        let samBytes = sam.reduce(Int64(0)) { $0 + (samByName[$1] ?? 0) }
        let t5Bytes = t5.reduce(Int64(0)) { $0 + (t5ByName[$1] ?? 0) }
        #expect(samBytes == 5_032_108_828)
        #expect(t5Bytes == 893_828_754)
        let totalGB = (Double(samBytes + t5Bytes) / 1e9 * 100).rounded() / 100
        #expect(totalGB == 5.93)
        let entry = try SAMAudioCatalogTests.samAudio()
        #expect(entry.downloadSizeGB == totalGB, "the catalog figure must be what the installer downloads")
    }

    /// The companion's destination is its own subfolder: MusicGen keeps `encodec/`, SAM-Audio gets `t5/`.
    @Test func eachCompanionLandsInItsOwnSubfolder() throws {
        let musicgens = ModelFileSelector.companionRepos(for: "jasonvassallo/mlx-musicgen-small")
        #expect(musicgens.count == 1)
        let musicgen = try #require(musicgens.first)
        #expect(musicgen.repository == "mlx-community/encodec-32khz-float32")
        #expect(musicgen.subfolder == "encodec/")
        let sams = ModelFileSelector.companionRepos(for: "mlx-community/sam-audio-base")
        #expect(sams.count == 1)
        let sam = try #require(sams.first)
        #expect(sam.repository == "google-t5/t5-base")
        #expect(sam.subfolder == "t5/")
        // A model with no companion keeps no subfolder.
        #expect(ModelFileSelector.companionRepos(for: "mlx-community/DeepFilterNet-mlx").isEmpty)
    }

    /// Removing SAM-Audio removes its folder — and `t5/` lives inside it. The engine then reports the missing encoder
    /// rather than trying to download one.
    @Test func uninstallingSAMAudioRemovesT5WithIt() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("sam-uninstall-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(baseDirectory: root)
        let entry = try SAMAudioCatalogTests.samAudio()
        let modelDir = store.directory(forHFModelID: entry.hfModelId)
        let t5Dir = modelDir.appendingPathComponent("t5", isDirectory: true)
        try FileManager.default.createDirectory(at: t5Dir, withIntermediateDirectories: true)
        try Data("config".utf8).write(to: modelDir.appendingPathComponent("config.json"))
        try Data("config".utf8).write(to: t5Dir.appendingPathComponent("config.json"))
        try Data(count: 16).write(to: t5Dir.appendingPathComponent("model.safetensors"))
        FileManager.default.createFile(atPath: store.installedMarker(forHFModelID: entry.hfModelId).path, contents: nil)
        #expect(FileManager.default.fileExists(atPath: t5Dir.path))

        InstallManager(store: store).uninstall(entry, catalog: [entry], installedModelIDs: [entry.id])

        #expect(!FileManager.default.fileExists(atPath: modelDir.path))
        #expect(!FileManager.default.fileExists(atPath: t5Dir.path))
    }
}
