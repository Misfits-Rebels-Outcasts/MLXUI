import Testing
import Foundation
import CryptoKit
@testable import MLXUI

/// Covers CFM-R5-1: the Python conformance corpus + goldens are synced into
/// `Fixtures/CatFlow/` (repo root, outside any target) by
/// `scripts/sync-catflow-fixtures.sh`, which records the source commit SHA and a
/// per-file sha256 in `manifest.json`. This suite makes drift visible three ways:
///
///   1. **Corpus advance** — the recorded `source_sha` must equal `pinnedSourceSHA`.
///      When catflow-mlx advances, re-running the script bumps the manifest and this
///      test fails, forcing a reviewer to inspect the new corpus before updating the pin.
///   2. **Hand-edited fixtures** — every file's sha256 must match the manifest, so a
///      fixture edited without re-syncing is caught.
///   3. **Broken corpus shape** — every `conformance/*.cat` must have a sibling
///      `.expect.json`, and the golden subdirectories must be non-empty.
struct CatFlowFixtureSyncTests {

    /// The catflow-mlx commit the fixtures were synced from. Bump only after reviewing
    /// a re-synced corpus. Source: `git -C catflow-mlx rev-parse HEAD` at sync time.
    private static let pinnedSourceSHA = "9f682aed944a2c728388cad3e27a2174b209e304"

    // MARK: - Fixtures dir (repo-relative, mirroring the other CatFlow tests)

    private var fixturesDir: URL {
        let filePath = #filePath                     // .../MLXUITests/CatFlowFixtureSyncTests.swift
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow")
    }

    private struct Manifest: Decodable {
        var source_repo: String
        var source_sha: String
        var files: [String: String]
    }

    private func loadManifest() throws -> Manifest {
        let data = try Data(contentsOf: fixturesDir.appendingPathComponent("manifest.json"))
        return try JSONDecoder().decode(Manifest.self, from: data)
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Corpus SHA

    @Test func recordedSHAIsPinned() throws {
        let manifest = try loadManifest()
        #expect(manifest.source_repo == "catflow-mlx")
        #expect(manifest.source_sha == Self.pinnedSourceSHA,
                "catflow-mlx advanced — re-run scripts/sync-catflow-fixtures.sh, review the new corpus, then update pinnedSourceSHA. Current recorded: \(manifest.source_sha)")
    }

    // MARK: - No hand-edited fixtures

    @Test func everyFixtureMatchesItsRecordedHash() throws {
        let manifest = try loadManifest()
        var checked = 0
        for (relativePath, expectedHash) in manifest.files {
            let url = fixturesDir.appendingPathComponent(relativePath)
            #expect(FileManager.default.fileExists(atPath: url.path),
                    "manifest lists \(relativePath) but the file is missing")
            let data = try Data(contentsOf: url)
            let actual = sha256(data)
            #expect(actual == expectedHash,
                    "\(relativePath) was edited without re-syncing — run scripts/sync-catflow-fixtures.sh to restore the corpus copy")
            checked += 1
        }
        #expect(checked > 0, "manifest is empty — the sync script must have failed")
    }

    @Test func noUndeclaredFixtureFiles() throws {
        let manifest = try loadManifest()
        let resourceKeys: Set<URLResourceKey> = [.isRegularFileKey]
        let urls = try FileManager.default.contentsOfDirectory(
            at: fixturesDir,
            includingPropertiesForKeys: [.isRegularFileKey]
        )
        var discovered: [String] = []
        for url in urls {
            if try url.resourceValues(forKeys: resourceKeys).isRegularFile == true,
               url.pathExtension == "json", url.lastPathComponent != "manifest.json" {
                discovered.append(url.lastPathComponent)
            }
        }
        #expect(discovered.isEmpty,
                "unexpected top-level files in the fixtures dir — the sync script owns it: \(discovered.joined(separator: ", "))")
    }

    // MARK: - MLXUI-generated fixtures the sync deliberately does not track

    /// Subtrees under `Fixtures/CatFlow/` that have **no upstream source** — added in
    /// `3c346a2`, regenerated from the Python by other means, and covered by their own
    /// suites. `sync-catflow-fixtures.sh` skips them when building `manifest.json`, so a
    /// re-sync doesn't reclassify ~150 local files as "synced from catflow-mlx". Keep this
    /// list identical to `UNMANAGED_PREFIXES` / `UNMANAGED_EXACT` in the script.
    /// See `RSI/DelegateMergeBacklog.md` Phase R18.
    private static let unmanagedPrefixes = ["gallery/", "goldens/canonicalize/", "traces/", "tools/", "validator/"]
    private static let unmanagedExact = ["goldens/check/uses_golden.json", "nonascii_canonical.cat"]

    @Test func unmanagedFixturesExistButAreNotInTheManifest() throws {
        let manifest = try loadManifest()
        for prefix in Self.unmanagedPrefixes {
            let dir = fixturesDir.appendingPathComponent(String(prefix.dropLast()))
            let files = (try? FileManager.default.subpathsOfDirectory(atPath: dir.path)) ?? []
            let regular = files.filter { !$0.hasSuffix("/") && !$0.contains(".DS_Store") }
            #expect(!regular.isEmpty, "\(prefix) is an MLXUI-local fixture tree but is empty or gone")
            for rel in manifest.files.keys {
                #expect(!rel.hasPrefix(prefix),
                        "\(rel) is under the unmanaged prefix \(prefix) but the manifest tracks it — the sync script's skip-list drifted")
            }
        }
        for exact in Self.unmanagedExact {
            #expect(FileManager.default.fileExists(atPath: fixturesDir.appendingPathComponent(exact).path),
                    "\(exact) is a named MLXUI-local fixture but is missing")
            #expect(manifest.files[exact] == nil,
                    "\(exact) is meant to be unmanaged but the manifest tracks it")
        }
    }

    // MARK: - Corpus shape

    @Test func everyConformanceCatHasAnExpectFile() throws {
        let conformanceDir = fixturesDir.appendingPathComponent("conformance")
        let cats = try FileManager.default.contentsOfDirectory(atPath: conformanceDir.path)
            .filter { $0.hasSuffix(".cat") }
        #expect(cats.count == 41, "expected the 41 conformance flows, found \(cats.count)")
        for cat in cats {
            let expectName = String(cat.dropLast(".cat".count)) + ".expect.json"
            #expect(FileManager.default.fileExists(atPath: conformanceDir.appendingPathComponent(expectName).path),
                    "\(cat) has no sibling .expect.json")
        }
    }

    @Test func goldenSubdirectoriesArePopulated() throws {
        let goldensDir = fixturesDir.appendingPathComponent("goldens")
        for sub in ["parse", "check", "errors", "fmt"] {
            let url = goldensDir.appendingPathComponent(sub)
            let contents = try FileManager.default.contentsOfDirectory(atPath: url.path)
            #expect(!contents.isEmpty, "goldens/\(sub) synced empty — the source corpus may be missing it")
        }
    }
}
