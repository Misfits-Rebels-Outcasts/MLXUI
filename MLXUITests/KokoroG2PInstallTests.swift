import Testing
import Foundation
@testable import MLXUI

/// KG-1 (R34, `RSI/DelegateKokoroG2PBacklog.md`): `InstallManager` installs Kokoro's English G2P files
/// (`beshkenadze/kitten-tts-g2p`: `us_bart.safetensors`, `us_bart_config.json`, `us_gold.json`, `us_silver.json`) into
/// `<Kokoro dir>/g2p/` at install time, so the app never fetches them at run time. Every request goes to a stub `URLProtocol`
/// on a private session; there is no network.
@Suite(.serialized)
struct KokoroG2PInstallTests {

    // MARK: Stub network

    /// Maps URL path -> response body. Records every request path. Per-session (set in `protocolClasses`), never registered globally.
    private final class StubNet: URLProtocol, @unchecked Sendable {
        nonisolated(unsafe) static var routes: [String: Data] = [:]
        nonisolated(unsafe) static var seen: [String] = []
        private static let lock = NSLock()
        static func reset(_ routes: [String: Data]) { lock.withLock { self.routes = routes; seen = [] } }
        static var paths: [String] { lock.withLock { seen } }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() {
            let url = request.url ?? URL(fileURLWithPath: "/")
            let key = url.path
            let (body, status): (Data, Int) = Self.lock.withLock {
                Self.seen.append("\(url.host ?? "")\(key)")
                if url.host == "huggingface.co", let b = Self.routes[key] { return (b, 200) }
                return (Data(), 404)
            }
            let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1",
                                           headerFields: ["Content-Length": "\(body.count)"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: body)
            client?.urlProtocolDidFinishLoading(self)
        }
        override func stopLoading() {}
    }

    private static func session() -> URLSession {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [StubNet.self]
        return URLSession(configuration: c)
    }

    private static func listing(_ files: [String: Int]) -> Data {
        let siblings = files.map { ["rfilename": $0.key, "size": $0.value] as [String: Any] }
        return try! JSONSerialization.data(withJSONObject: ["siblings": siblings])
    }

    private static let g2pFiles: [String: Int] = [
        "us_bart.safetensors": 11, "us_bart_config.json": 12, "us_gold.json": 13, "us_silver.json": 14,
    ]

    private static func routesForKokoro(g2p: [String: Int]? = g2pFiles) -> [String: Data] {
        var r: [String: Data] = [:]
        let main: [String: Int] = ["config.json": 5, "kokoro-v1_0.safetensors": 6, "voices/af_heart.safetensors": 7, "README.md": 3]
        r["/api/models/mlx-community/Kokoro-82M-bf16"] = listing(main)
        for (name, size) in main { r["/mlx-community/Kokoro-82M-bf16/resolve/main/\(name)"] = Data(repeating: 0x41, count: size) }
        if let g2p {
            r["/api/models/beshkenadze/kitten-tts-g2p"] = listing(g2p.merging(["README.md": 3, ".gitattributes": 2, "config.json.bak": 4]) { a, _ in a })
            for (name, size) in g2p { r["/beshkenadze/kitten-tts-g2p/resolve/main/\(name)"] = Data(repeating: 0x47, count: size) }
        }
        return r
    }

    private func tempStore() -> (ModelStore, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("kg1-\(UUID().uuidString)", isDirectory: true)
        return (ModelStore(baseDirectory: dir), dir)
    }

    private func kokoroEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--Kokoro-82M-bf16", displayName: "Kokoro 82M", modelType: .tts,
                  hfModelId: "mlx-community/Kokoro-82M-bf16")
    }

    @MainActor private func waitForTerminal(_ manager: InstallManager, _ id: String) async throws -> InstallState? {
        let deadline = ContinuousClock.now + .seconds(30)
        while ContinuousClock.now < deadline {
            switch manager.modelStates[id] {
            case .installed, .error: return manager.modelStates[id]
            default: try await Task.sleep(for: .milliseconds(20))
            }
        }
        return manager.modelStates[id]
    }

    // MARK: The companion list

    @Test func kokoroHasTheG2PCompanionAndTheOthersAreUnchanged() {
        let kokoro = ModelFileSelector.companionRepos(for: "mlx-community/Kokoro-82M-bf16")
        #expect(kokoro == [ModelFileSelector.CompanionRepo(
            repository: "beshkenadze/kitten-tts-g2p", subfolder: "g2p/",
            only: ["us_bart.safetensors", "us_bart_config.json", "us_gold.json", "us_silver.json"])])
        #expect(ModelFileSelector.companionRepos(for: "jasonvassallo/mlx-musicgen-small")
                == [ModelFileSelector.CompanionRepo(repository: "mlx-community/encodec-32khz-float32", subfolder: "encodec/")])
        #expect(ModelFileSelector.companionRepos(for: "mlx-community/sam-audio-base")
                == [ModelFileSelector.CompanionRepo(repository: "google-t5/t5-base", subfolder: "t5/")])
        #expect(ModelFileSelector.companionRepos(for: "mlx-community/Qwen3-4B-4bit").isEmpty)
    }

    /// Why the companion names its files: the general selector does not know three of the four.
    @Test func theGeneralSelectorAloneWouldMissThreeOfTheG2PFiles() {
        let chosen = ModelFileSelector.filesToDownload(siblings: Array(Self.g2pFiles.keys) + ["README.md", ".gitattributes"])
        #expect(Set(chosen) == ["us_bart.safetensors"])
    }

    // MARK: Install end to end (stub network)

    @Test @MainActor func aNewKokoroInstallPutsTheFourG2PFilesInG2PAndNothingElseFromThatRepo() async throws {
        StubNet.reset(Self.routesForKokoro())
        let (store, dir) = tempStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manager = InstallManager(store: store, session: Self.session(), apiSession: Self.session())
        let entry = kokoroEntry()
        manager.install(entry)
        let end = try await waitForTerminal(manager, entry.id)
        #expect(end == .installed, "ended \(String(describing: end))")

        let modelDir = store.directory(forHFModelID: entry.hfModelId)
        let g2p = modelDir.appendingPathComponent("g2p")
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: g2p.path)) ?? []).sorted()
        #expect(names == ["us_bart.safetensors", "us_bart_config.json", "us_gold.json", "us_silver.json"], "g2p/ holds \(names)")
        #expect(try Data(contentsOf: g2p.appendingPathComponent("us_gold.json")) == Data(repeating: 0x47, count: 13))
        // the model's own files are where they always were
        #expect(FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("voices/af_heart.safetensors").path))
        #expect(FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("kokoro-v1_0.safetensors").path))
        #expect(!FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("README.md").path))
        #expect(FileManager.default.fileExists(atPath: store.installedMarker(forHFModelID: entry.hfModelId).path))
        // only huggingface.co, only the two repos
        let hosts = Set(StubNet.paths.map { $0.prefix { $0 != "/" } })
        #expect(hosts == ["huggingface.co"])
        #expect(StubNet.paths.allSatisfy { $0.contains("Kokoro-82M-bf16") || $0.contains("kitten-tts-g2p") }, "\(StubNet.paths)")
    }

    @Test @MainActor func aG2PRepoMissingAFileFailsTheInstallAndLeavesNoInstalledModel() async throws {
        var partial = Self.g2pFiles
        partial.removeValue(forKey: "us_silver.json")
        StubNet.reset(Self.routesForKokoro(g2p: partial))
        let (store, dir) = tempStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manager = InstallManager(store: store, session: Self.session(), apiSession: Self.session())
        let entry = kokoroEntry()
        manager.install(entry)
        let end = try await waitForTerminal(manager, entry.id)
        guard case .error(let message, _)? = end else { Issue.record("expected an error, got \(String(describing: end))"); return }
        #expect(message.contains("us_silver.json"), "names the missing file: \(message)")
        #expect(!FileManager.default.fileExists(atPath: store.installedMarker(forHFModelID: entry.hfModelId).path),
                "a Kokoro install without g2p/ must not look complete")
    }

    /// MusicGen's companion still installs as before: selector-chosen files, into `encodec/`.
    @Test @MainActor func musicGenStillInstallsItsEncodecCompanionBySelector() async throws {
        var r: [String: Data] = [:]
        r["/api/models/jasonvassallo/mlx-musicgen-small"] = Self.listing(["config.json": 4, "model.safetensors": 5, "README.md": 2])
        r["/jasonvassallo/mlx-musicgen-small/resolve/main/config.json"] = Data(repeating: 1, count: 4)
        r["/jasonvassallo/mlx-musicgen-small/resolve/main/model.safetensors"] = Data(repeating: 2, count: 5)
        r["/api/models/mlx-community/encodec-32khz-float32"] = Self.listing(["config.json": 6, "model.safetensors": 7, "extra.json": 3, "README.md": 2])
        r["/mlx-community/encodec-32khz-float32/resolve/main/config.json"] = Data(repeating: 3, count: 6)
        r["/mlx-community/encodec-32khz-float32/resolve/main/model.safetensors"] = Data(repeating: 4, count: 7)
        StubNet.reset(r)
        let (store, dir) = tempStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manager = InstallManager(store: store, session: Self.session(), apiSession: Self.session())
        let entry = makeEntry(id: "jasonvassallo--mlx-musicgen-small", modelType: .tts, hfModelId: "jasonvassallo/mlx-musicgen-small")
        manager.install(entry)
        let end = try await waitForTerminal(manager, entry.id)
        #expect(end == .installed, "ended \(String(describing: end))")
        let encodec = store.directory(forHFModelID: entry.hfModelId).appendingPathComponent("encodec")
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: encodec.path)) ?? []).sorted()
        #expect(names == ["config.json", "model.safetensors"], "encodec/ holds \(names)")
    }

    /// SAM-Audio's companion likewise lands in `t5/` (only the lookup is exercised end to end here; its size/figures are pinned by `SAMAudioCompanionInstallTests`).
    @Test @MainActor func samAudioStillInstallsItsT5CompanionBySelector() async throws {
        var r: [String: Data] = [:]
        r["/api/models/mlx-community/sam-audio-base"] = Self.listing(["config.json": 4, "model.safetensors": 5])
        r["/mlx-community/sam-audio-base/resolve/main/config.json"] = Data(repeating: 1, count: 4)
        r["/mlx-community/sam-audio-base/resolve/main/model.safetensors"] = Data(repeating: 2, count: 5)
        r["/api/models/google-t5/t5-base"] = Self.listing(["config.json": 6, "model.safetensors": 7, "spiece.model": 3, "tokenizer.json": 2, "flax_model.msgpack": 9])
        for (n, c) in [("config.json", 6), ("model.safetensors", 7), ("spiece.model", 3), ("tokenizer.json", 2)] {
            r["/google-t5/t5-base/resolve/main/\(n)"] = Data(repeating: 5, count: c)
        }
        StubNet.reset(r)
        let (store, dir) = tempStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manager = InstallManager(store: store, session: Self.session(), apiSession: Self.session())
        let entry = makeEntry(id: "mlx-community--sam-audio-base", modelType: .tts, hfModelId: "mlx-community/sam-audio-base")
        manager.install(entry)
        let end = try await waitForTerminal(manager, entry.id)
        #expect(end == .installed, "ended \(String(describing: end))")
        let t5 = store.directory(forHFModelID: entry.hfModelId).appendingPathComponent("t5")
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: t5.path)) ?? []).sorted()
        #expect(names == ["config.json", "model.safetensors", "spiece.model", "tokenizer.json"], "t5/ holds \(names)")
    }

    // MARK: Existing installs

    /// An install made before KG-1 has the marker and no `g2p/`. Launch reconciliation keeps it (KG-2 turns the missing folder into a
    /// plain error at run time); nothing here uninstalls it.
    @Test @MainActor func anExistingKokoroInstallWithoutG2PStillReconcilesAsInstalled() throws {
        let (store, dir) = tempStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let entry = kokoroEntry()
        let modelDir = store.directory(forHFModelID: entry.hfModelId)
        try FileManager.default.createDirectory(at: modelDir.appendingPathComponent("voices"), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forHFModelID: entry.hfModelId).path, contents: nil)
        let manager = InstallManager(store: store, session: Self.session(), apiSession: Self.session())
        let installed = manager.loadInstalled(modelIDs: [entry.id], catalog: [entry])
        #expect(installed == [entry.id])
        #expect(manager.isInstalled(entry))
        #expect(!FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("g2p").path), "reconcile does not create or require it")
    }

    /// R38: the sentence tells the user to Uninstall then Install. That path must leave a complete install: uninstall removes the
    /// whole model folder including `g2p/`, install recreates it.
    @Test @MainActor func uninstallThenInstallLeavesACompleteInstallWithG2P() async throws {
        StubNet.reset(Self.routesForKokoro())
        let (store, dir) = tempStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manager = InstallManager(store: store, session: Self.session(), apiSession: Self.session())
        let entry = kokoroEntry()
        let modelDir = store.directory(forHFModelID: entry.hfModelId)
        // an old install: weights and marker, no g2p/
        try FileManager.default.createDirectory(at: modelDir.appendingPathComponent("voices"), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forHFModelID: entry.hfModelId).path, contents: nil)
        manager.uninstall(entry, catalog: [entry], installedModelIDs: [entry.id])
        #expect(!FileManager.default.fileExists(atPath: modelDir.path), "Uninstall removes the whole folder")
        manager.install(entry)
        let end = try await waitForTerminal(manager, entry.id)
        #expect(end == .installed, "ended \(String(describing: end))")
        let g2p = ((try? FileManager.default.contentsOfDirectory(atPath: modelDir.appendingPathComponent("g2p").path)) ?? []).sorted()
        #expect(g2p == ["us_bart.safetensors", "us_bart_config.json", "us_gold.json", "us_silver.json"])
    }

    /// R38: an install with weights but no `g2p/` fails with the exact sentence and makes zero network requests.
    @Test func synthesizeOnAnOldInstallGivesTheSentenceAndMakesNoRequest() async throws {
        StubNet.reset([:])
        let (store, dir) = tempStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = store.directory(forHFModelID: "mlx-community/Kokoro-82M-bf16")
        try FileManager.default.createDirectory(at: modelDir.appendingPathComponent("voices"), withIntermediateDirectories: true)
        try Data([1]).write(to: modelDir.appendingPathComponent("kokoro-v1_0.safetensors"))
        do {
            _ = try await KokoroEngine.synthesize("Hello.", modelDirectory: modelDir)
            Issue.record("expected a refusal")
        } catch {
            #expect(error.localizedDescription == "Kokoro is missing some files it needs to read text aloud. In the sidebar under Installed, click Kokoro-82M, then click Uninstall, then click Install.")
        }
        #expect(StubNet.paths.isEmpty, "requests: \(StubNet.paths)")
    }
}
