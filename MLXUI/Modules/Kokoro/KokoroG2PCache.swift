import Foundation
import CryptoKit
import HuggingFace

/// KG-2 (R34, `RSI/DelegateKokoroG2PBacklog.md`): Kokoro's English G2P files are fetched by `InstallManager` into
/// `<Kokoro dir>/g2p/` (KG-1) and never at run time. mlx-audio-swift reads them from its own cache folder
/// (`<HubCache.default>/mlx-audio/beshkenadze_kitten-tts-g2p/`), and asks huggingface.co on every text chunk unless that folder
/// looks like a finished download -- which, for this repo, means a valid `config.json` (`ModelUtils.swift:105-117`; the repo ships
/// none). So before synthesis the engine makes that folder a copy of `g2p/` plus a `config.json` of exactly `{}\n`.
///
/// **The cache copy is disposable (R32).** It is recreated whenever it is missing or any of the four files differs from `g2p/` in
/// size or SHA-256, or `config.json` is not the stub. Extra files in it are ignored. Never run per chunk: once per `synthesize`.
/// **Sizes + SHA-256**: the files total 8.7 MB, so hashing both copies costs a few milliseconds per synthesis and is
/// independent of timestamps, which a plain size check or an mtime check is not.
/// **Atomic**: the copy is built in a sibling temp folder and swapped in with `replaceItemAt`, so a half-copy is never left
/// where the package would trust it.
/// The cache location comes from the same `HubCache.default` the package uses (it differs in the sandbox), unless a test hands one in.
nonisolated enum KokoroG2PCache {
    nonisolated static let repoFolderName = ModelFileSelector.kokoroG2PRepository.replacingOccurrences(of: "/", with: "_")
    nonisolated static let requiredFiles = ModelFileSelector.kokoroG2PFiles
    nonisolated static let stubConfig = Data("{}\n".utf8)

    enum Result: Equatable { case alreadyCurrent, copied }

    enum G2PError: LocalizedError, Equatable {
        /// `g2p/` is absent or incomplete: an install made before KG-1 (or a damaged one).
        case notInstalled
        /// Copying into the cache folder failed (disk full, permissions).
        case couldNotPrepare

        var errorDescription: String? {
            switch self {
            case .notInstalled:
                return "Kokoro is missing some files it needs to read text aloud. In the sidebar under Installed, click Kokoro-82M, then click Uninstall, then click Install."
            case .couldNotPrepare:
                return "Kokoro could not get its speech files ready. Free some disk space and try again."
            }
        }
    }

    /// `<cache root>/mlx-audio/beshkenadze_kitten-tts-g2p/`
    nonisolated static func cacheFolder(cacheRoot: URL = HubCache.default.cacheDirectory) -> URL {
        cacheRoot.appendingPathComponent("mlx-audio", isDirectory: true).appendingPathComponent(repoFolderName, isDirectory: true)
    }

    /// Makes the package's cache folder a current copy of `<modelDirectory>/g2p/`. Throws `G2PError` -- never touches the network.
    @discardableResult
    nonisolated static func ensure(modelDirectory: URL, cacheRoot: URL = HubCache.default.cacheDirectory) throws -> Result {
        let fm = FileManager.default
        let source = modelDirectory.appendingPathComponent(ModelFileSelector.kokoroG2PSubfolder, isDirectory: true)
        // The installed copy must be whole before anything is copied or trusted.
        for name in requiredFiles {
            let size = (try? fm.attributesOfItem(atPath: source.appendingPathComponent(name).path)[.size] as? Int64) ?? nil
            guard let size, size > 0 else { throw G2PError.notInstalled }
        }
        let target = cacheFolder(cacheRoot: cacheRoot)
        if isCurrent(target: target, source: source) { return .alreadyCurrent }

        let parent = target.deletingLastPathComponent()
        let staging = parent.appendingPathComponent(".kg-g2p-\(UUID().uuidString)", isDirectory: true)
        do {
            try fm.createDirectory(at: staging, withIntermediateDirectories: true)
            for name in requiredFiles {
                try fm.copyItem(at: source.appendingPathComponent(name), to: staging.appendingPathComponent(name))
            }
            try stubConfig.write(to: staging.appendingPathComponent("config.json"))
            if fm.fileExists(atPath: target.path) {
                _ = try fm.replaceItemAt(target, withItemAt: staging)
            } else {
                try fm.moveItem(at: staging, to: target)
            }
        } catch {
            try? fm.removeItem(at: staging)
            throw G2PError.couldNotPrepare
        }
        return .copied
    }

    /// True when all four files match `source` by size and SHA-256 and `config.json` is the stub.
    nonisolated static func isCurrent(target: URL, source: URL) -> Bool {
        guard let config = try? Data(contentsOf: target.appendingPathComponent("config.json")), config == stubConfig else { return false }
        for name in requiredFiles {
            let a = source.appendingPathComponent(name), b = target.appendingPathComponent(name)
            guard let sa = fileSize(a), let sb = fileSize(b), sa == sb, let ha = digest(a), let hb = digest(b), ha == hb else { return false }
        }
        return true
    }

    private nonisolated static func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? nil
    }

    /// Streams the file through SHA-256 in 1 MiB blocks (no 3 MB file held twice).
    private nonisolated static func digest(_ url: URL) -> SHA256.Digest? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while let block = try? handle.read(upToCount: 1_048_576), !block.isEmpty { hasher.update(data: block) }
        return hasher.finalize()
    }
}
