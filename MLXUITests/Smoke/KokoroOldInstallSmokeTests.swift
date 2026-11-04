import Foundation
import AVFoundation
import HuggingFace
import Testing
@testable import MLXUI

/// KG-3 prep (R30(1), R38, R39): an OLD Kokoro install (weights, no `g2p/`) is repaired by the same `InstallManager.uninstall` + `install`
/// calls the Uninstall / Install buttons make, and the repaired install then speaks flow 9 with the network off.
///
/// **Never touches the real store beyond reading it.** The real Kokoro folder in `ModelStore.shared` (the Direct store,
/// `~/Library/Application Support/AI Browser/`) is only the source of an APFS clone (`cp -Rc`); everything else happens under a
/// scratch base directory. **Not run by its author**; it skips unless opted in.
///
/// Environment (all through `TEST_RUNNER_`, unsandboxed host, app closed):
///   `MLXUI_KOKORO_OLDINSTALL_PHASE`  `1` or `2`
///   `MLXUI_KOKORO_OLDINSTALL_SCRATCH` scratch base directory, default `/private/tmp/kokoro-oldinstall-scratch`
///
/// PHASE 1 (ONLINE): clone the real install into scratch minus `g2p/`; assert the engine gives the exact sentence; `uninstall`
/// then `install` into the scratch store (real Hugging Face fetch, the whole Kokoro repo again, about 0.33 GB for the weights and
/// voices plus 8.7 MB of G2P files; the uninstall removes the whole folder, so there is no way to keep the weights and still make
/// the path genuine); assert `g2p/` holds the 4 files.
///
/// PHASE 2 (OFFLINE): `TEST_RUNNER_HF_HUB_CACHE` must name an EMPTY folder and Wi-Fi must be off. Runs flow 9 against the scratch
/// store's repaired install, asserts the WAV is 20-60 s, and that the seeded G2P cache folder appeared in that empty cache.
///
///   S=/private/tmp/kokoro-oldinstall-scratch
///   # PHASE 1
///   TEST_RUNNER_MLXUI_KOKORO_OLDINSTALL_PHASE=1 TEST_RUNNER_MLXUI_KOKORO_OLDINSTALL_SCRATCH=$S \
///   xcodebuild test -project MLXUI.xcodeproj -scheme MLXUI -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO \
///     CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements ENABLE_APP_SANDBOX=NO -only-testing:MLXUITests/KokoroOldInstallSmokeTests
///   # PHASE 2 (Wi-Fi off)
///   mkdir -p /private/tmp/kokoro-empty-hub-cache
///   TEST_RUNNER_MLXUI_KOKORO_OLDINSTALL_PHASE=2 TEST_RUNNER_MLXUI_KOKORO_OLDINSTALL_SCRATCH=$S TEST_RUNNER_HF_HUB_CACHE=/private/tmp/kokoro-empty-hub-cache \
///   xcodebuild test ... (same flags) -only-testing:MLXUITests/KokoroOldInstallSmokeTests
///
/// Lines: `SMOKE|KG3|<check>|PASS|FAIL|INFO|<detail>`.
nonisolated enum KokoroOldInstall {
    static let env = ProcessInfo.processInfo.environment
    static let phase = env["MLXUI_KOKORO_OLDINSTALL_PHASE"]
    static let row = "KG3"
    static let hfID = "mlx-community/Kokoro-82M-bf16"
    static let entryID = "mlx-community--Kokoro-82M-bf16"
    static let scratch = URL(fileURLWithPath: env["MLXUI_KOKORO_OLDINSTALL_SCRATCH"].flatMap { $0.isEmpty ? nil : $0 }
                                              ?? "/private/tmp/kokoro-oldinstall-scratch", isDirectory: true)
    static var scratchStore: ModelStore { ModelStore(baseDirectory: scratch) }
    static let g2pNames = ["us_bart.safetensors", "us_bart_config.json", "us_gold.json", "us_silver.json"]
    static let sentence = "Kokoro is missing some files it needs to read text aloud. In the sidebar under Installed, click Kokoro-82M, then click Uninstall, then click Install."

    static func entry() -> ModelEntry? { BasicShowcaseRealLoad.catalog().first { $0.id == entryID } }
    static func line(_ name: String, _ ok: Bool, _ detail: String) { ModelSmoke.check(row, name, ok, detail) }
}

@Suite(.serialized)
@MainActor
struct KokoroOldInstallSmokeTests {

    private static let phase1Ready = KokoroOldInstall.phase == "1"
        && FileManager.default.fileExists(atPath: ModelStore.shared.installedMarker(forHFModelID: KokoroOldInstall.hfID).path)
    private static let phase2Ready = KokoroOldInstall.phase == "2"
        && FileManager.default.fileExists(atPath: KokoroOldInstall.scratchStore.installedMarker(forHFModelID: KokoroOldInstall.hfID).path)

    // MARK: Phase 1 (online)

    @Test(.enabled(if: phase1Ready, "needs TEST_RUNNER_MLXUI_KOKORO_OLDINSTALL_PHASE=1 and Kokoro installed in the Direct store (unsandboxed host)"))
    func phase1_oldInstallGivesTheSentenceThenUninstallInstallRepairsIt() async throws {
        typealias K = KokoroOldInstall
        let entry = try #require(K.entry(), "Kokoro is not in the catalog")
        let fm = FileManager.default
        let scratchStore = K.scratchStore
        try? fm.removeItem(at: K.scratch)
        try fm.createDirectory(at: scratchStore.modelsDirectory, withIntermediateDirectories: true)

        // 1. A clone of the real install (read-only on the source), minus g2p/.
        let source = ModelStore.shared.directory(forHFModelID: K.hfID)
        let target = scratchStore.directory(forHFModelID: K.hfID)
        let cp = Process()
        cp.executableURL = URL(fileURLWithPath: "/bin/cp")
        cp.arguments = ["-Rc", source.path, target.path]
        try cp.run(); cp.waitUntilExit()
        K.line("1.cloned", cp.terminationStatus == 0 && fm.fileExists(atPath: target.path), "cp -Rc exit \(cp.terminationStatus)")
        try? fm.removeItem(at: target.appendingPathComponent("g2p"))
        K.line("1.clone-has-no-g2p", !fm.fileExists(atPath: target.appendingPathComponent("g2p").path), "removed from the clone only")
        K.line("1.clone-is-marked-installed", fm.fileExists(atPath: scratchStore.installedMarker(forHFModelID: K.hfID).path), "")

        // 2. The engine refuses with the exact sentence before anything else (it raises before the audio thread and any package call).
        do {
            _ = try await KokoroEngine.synthesize("Hello.", modelDirectory: target)
            K.line("2.exact-sentence", false, "synthesize returned audio on an install without g2p/")
        } catch {
            K.line("2.exact-sentence", error.localizedDescription == K.sentence, "got: \(error.localizedDescription)")
        }

        // 3. The buttons' calls: InstallManager.uninstall, then install, online, into the scratch store.
        let manager = InstallManager(store: scratchStore)
        manager.uninstall(entry, catalog: [entry], installedModelIDs: [entry.id])
        K.line("3.uninstall-removed-the-folder", !fm.fileExists(atPath: target.path), "")
        let started = ContinuousClock.now
        manager.install(entry)
        let deadline = ContinuousClock.now + .seconds(3600)
        var state: InstallState?
        while ContinuousClock.now < deadline {
            state = manager.modelStates[entry.id]
            if case .installed = state { break }
            if case .error = state { break }
            if case .needsAuth = state { break }
            try await Task.sleep(for: .milliseconds(500))
        }
        K.line("3.install-finished", state == .installed, "state \(String(describing: state)), \(started.duration(to: .now))")
        let names = ((try? fm.contentsOfDirectory(atPath: target.appendingPathComponent("g2p").path)) ?? []).sorted()
        K.line("3.g2p-has-the-four-files", names == K.g2pNames, "g2p/ holds \(names)")
        var bytes = 0
        if let walker = fm.enumerator(at: target, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let url as URL in walker { bytes += (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        }
        ModelSmoke.info(K.row, "3.scratch-install-size", String(format: "%.1f MB on disk (what was downloaded)", Double(bytes) / 1_000_000))
        ModelSmoke.info(K.row, "next", "Phase 2: Wi-Fi off, HF_HUB_CACHE at an empty folder, scratch = \(K.scratch.path)")
    }

    // MARK: Phase 2 (offline)

    @Test(.enabled(if: phase2Ready, "needs TEST_RUNNER_MLXUI_KOKORO_OLDINSTALL_PHASE=2 and a repaired install in the scratch store from phase 1"))
    func phase2_flow9SpeaksOfflineOnTheRepairedInstall() async throws {
        typealias K = KokoroOldInstall
        let fm = FileManager.default
        let entry = try #require(K.entry(), "Kokoro is not in the catalog")
        let target = K.scratchStore.directory(forHFModelID: K.hfID)

        // Preconditions that make a pass mean something.
        let names = ((try? fm.contentsOfDirectory(atPath: target.appendingPathComponent("g2p").path)) ?? []).sorted()
        K.line("0.scratch-install-is-repaired", names == K.g2pNames, "g2p/ holds \(names)")
        let hubEnv = K.env["HF_HUB_CACHE"] ?? ""
        let cacheDir = URL(fileURLWithPath: hubEnv, isDirectory: true)
        let cacheRoot = HubCache.default.cacheDirectory
        K.line("0.hub-cache-is-the-env-folder", !hubEnv.isEmpty && cacheRoot.standardizedFileURL.path == cacheDir.standardizedFileURL.path,
               "HF_HUB_CACHE=\(hubEnv), the package would use \(cacheRoot.path)")
        let before = (try? fm.contentsOfDirectory(atPath: cacheRoot.path)) ?? []
        K.line("0.hub-cache-is-empty", before.isEmpty, "contains \(before)")
        let reachable = await Self.hubReachable()
        K.line("0.network-is-off", !reachable, reachable ? "huggingface.co answered: turn Wi-Fi off, this run proves nothing" : "huggingface.co unreachable")

        // Flow 9 through the real interpreter and RealExecutor, the Kokoro stage bound to the SCRATCH install.
        let flowID = "9-ReadItAloud"
        let base = fm.temporaryDirectory.appendingPathComponent("kg3-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: base) }
        let ws = try #require(FlowWorkspace(root: base.appendingPathComponent("flows")))
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory, bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let catalog = BasicShowcaseRealLoad.catalog()
        let makeStage: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            guard model.id == K.entryID else { throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind) }
            let voice = config.voice ?? KokoroEngine.defaultVoice
            let speed = config.speed, seed = config.seed, trim = config.trimSilence
            return TTSStage(id: "kokoro.\(model.id).\(voice)", name: "Kokoro TTS (\(voice))") { text in
                try await KokoroEngine.synthesize(text, voice: voice, speed: speed, seed: seed, trimSilence: trim, modelDirectory: target)
            }
        }
        let executor = RealExecutor(workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
                                    makeModelStage: makeStage, installedModelIDs: [entry.id], catalog: catalog)
        let doc = try GalleryLoader.loadDocument(flowID: flowID)
        let events = try await FlowInterpreter.run(doc, executor: executor)
        let failed = events.filter { $0.kind == .rowFailed }
        K.line("1.no-row-failed", failed.isEmpty, failed.map { "\($0.path): \($0.message ?? "")" }.joined(separator: "; "))

        let wav = ws.directory(for: flowID).appendingPathComponent("note.wav")
        if let file = try? AVAudioFile(forReading: wav) {
            let seconds = Double(file.length) / file.processingFormat.sampleRate
            K.line("1.wav-20-to-60-s", (20...60).contains(seconds), String(format: "%.1f s", seconds))
            let keep = K.scratch.appendingPathComponent("flow9-offline-note.wav")
            try? fm.removeItem(at: keep); try? fm.copyItem(at: wav, to: keep)
            ModelSmoke.info(K.row, "1.wav-kept", keep.path)
        } else {
            K.line("1.wav-20-to-60-s", false, "no readable note.wav in \(ws.directory(for: flowID).path)")
        }
        let seeded = cacheRoot.appendingPathComponent("mlx-audio/beshkenadze_kitten-tts-g2p/us_bart.safetensors")
        K.line("2.g2p-cache-was-seeded-from-the-install", fm.fileExists(atPath: seeded.path), seeded.path)
    }

    private static func hubReachable() async -> Bool {
        var request = URLRequest(url: URL(string: "https://huggingface.co/")!)
        request.timeoutInterval = 5
        request.httpMethod = "HEAD"
        return (try? await URLSession(configuration: .ephemeral).data(for: request)) != nil
    }
}
