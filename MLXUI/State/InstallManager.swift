import Foundation
import Observation

enum InstallState: Equatable {
    case idle
    case resolving
    case downloading(progress: Double, downloaded: Int64, total: Int64)
    case verifying
    case installed
    case error(String, canRetry: Bool)
    /// Install was refused because the model is gated and no valid HF token is set.
    /// Distinct from `.error` so the UI can route the user to the token entry in Settings.
    case needsAuth(String)

    var isActive: Bool {
        switch self {
        case .idle, .installed, .error, .needsAuth: false
        default: true
        }
    }
}

@Observable
final class InstallManager {
    /// MoC-5-1: injectable so tests can point storage at a temp directory rather than the
    /// real `Application Support/AI Browser/`. `ModelStore` stays the one place per-model
    /// paths are built — every path below routes through it, never hand-rolled here.
    private let store: ModelStore
    private let modelsDir: URL
    private let downloadsDir: URL
    private let installedURL: URL
    private let session: URLSession
    /// MoC-5-4: download tracking is keyed by **repo** slug, not card id — one physical
    /// download per repo, however many cards are watching it. `modelStates` (below) stays
    /// card-keyed for the UI; `setState(_:forRepo:)` fans a repo-level update out to every
    /// card in `cardIDsByRepo[repo]`.
    private var downloadTasksByRepo: [String: Task<Void, Never>] = [:]
    /// Every catalog card id currently interested in a repo — refreshed on each `install`
    /// call. Lets one download's progress reach every card naming the same repo, and lets a
    /// second `install` for a sibling card join an in-flight download instead of starting a
    /// duplicate.
    private var cardIDsByRepo: [String: Set<String>] = [:]

    var modelStates: [String: InstallState] = [:]
    var onInstallComplete: ((String) -> Void)?

    init(store: ModelStore = .shared) {
        self.store = store
        modelsDir = store.modelsDirectory
        downloadsDir = store.downloadsDirectory
        installedURL = store.installedRegistryURL

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 3600
        session = URLSession(configuration: config)

        do {
            try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: downloadsDir, withIntermediateDirectories: true)
            print("[Install] Storage ready: \(modelsDir.path)")
        } catch {
            print("[Install] Storage init failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Public

    func install(_ model: ModelEntry, onComplete: @escaping (String) -> Void = { _ in }) {
        guard !modelStates.keys.contains(model.id) || modelStates[model.id] == .idle
                || isError(model.id) || isNeedsAuth(model.id) else { return }

        let repo = ModelStore.repoSlug(for: model.hfModelId)
        cardIDsByRepo[repo, default: []].insert(model.id)

        // MoC-5-FIX-2: the repo's files are already on disk — a sibling card sharing this
        // download installed them (MoC-6), or this card did in an earlier session and the
        // in-memory state was lost. Mark installed and fire completion; never re-fetch bytes
        // that are already present. Defence in depth: independent of how the read path
        // resolves the marker, so it holds even if that path regresses again.
        if FileManager.default.fileExists(
            atPath: store.installedMarker(forHFModelID: model.hfModelId).path) {
            modelStates[model.id] = .installed
            onComplete(model.id)
            return
        }

        // MoC-5-4: another card is already downloading this repo — join it rather than
        // starting a duplicate. Mirror whichever state that download is currently in;
        // every future update for `repo` fans out to every id in `cardIDsByRepo[repo]`,
        // this one now included.
        if downloadTasksByRepo[repo] != nil {
            if let watcherState = cardIDsByRepo[repo]?
                .first(where: { $0 != model.id })
                .flatMap({ modelStates[$0] }) {
                modelStates[model.id] = watcherState
            }
            return
        }

        // Pre-flight: refuse to start a download that can't fit on disk, rather than
        // failing partway through after writing gigabytes of temp files.
        if let diskError = DiskSpace.preflightError(
            downloadSizeGB: model.downloadSizeGB,
            availableDiskGB: currentAvailableDiskGB()
        ) {
            setState(.error(diskError, canRetry: true), forRepo: repo)
            return
        }

        setState(.resolving, forRepo: repo)
        let task = Task { await downloadModel(model, repo: repo, onComplete: onComplete) }
        downloadTasksByRepo[repo] = task
    }

    /// Free space (GB) on the volume that holds the models directory. Measured fresh at
    /// install time so the check reflects current conditions, not a stale launch reading.
    private func currentAvailableDiskGB() -> Double {
        let free = try? modelsDir.resourceValues(forKeys: [.volumeAvailableCapacityKey])
            .volumeAvailableCapacity
        return Double(free ?? 0) / 1_000_000_000.0
    }

    /// MoC-5-4: writes `state` into every card currently watching `repo` — one download's
    /// progress reaching every card that shares it.
    private func setState(_ state: InstallState, forRepo repo: String) {
        for cardID in cardIDsByRepo[repo] ?? [] {
            modelStates[cardID] = state
        }
    }

    func cancel(_ model: ModelEntry) {
        let repo = ModelStore.repoSlug(for: model.hfModelId)
        downloadTasksByRepo[repo]?.cancel()
        downloadTasksByRepo[repo] = nil
        setState(.idle, forRepo: repo)
        // MoC-5-2: the download's scratch directory is repo-keyed (see `downloadModel`),
        // matching where it was actually written.
        let downloadDir = store.downloadDirectory(forHFModelID: model.hfModelId)
        try? FileManager.default.removeItem(at: downloadDir)
    }

    /// MoC-5-2: reference-counted uninstall. `catalog` + `installedModelIDs` (both readily
    /// available at every real call site — `AppState` always has the live catalog and its
    /// own installed-ids set) are what let this be answered from the catalog alone, per the
    /// backlog's own bar: deleting the repo directory because *this* card was removed would
    /// break every other still-installed card naming the same repo.
    func uninstall(_ model: ModelEntry, catalog: [ModelEntry], installedModelIDs: Set<String>) {
        modelStates[model.id] = .idle
        let repo = ModelStore.repoSlug(for: model.hfModelId)
        let anotherInstalledCardSharesThisRepo = catalog.contains { other in
            other.id != model.id
                && installedModelIDs.contains(other.id)
                && ModelStore.repoSlug(for: other.hfModelId) == repo
        }
        guard !anotherInstalledCardSharesThisRepo else {
            print("[Install] Uninstall \(model.id): another installed card shares \(repo) — directory kept")
            return
        }
        let modelDir = store.directory(forHFModelID: model.hfModelId)
        try? FileManager.default.removeItem(at: modelDir)
    }

    func isError(_ modelId: String) -> Bool {
        if case .error = modelStates[modelId] { return true }
        return false
    }

    func isNeedsAuth(_ modelId: String) -> Bool {
        if case .needsAuth = modelStates[modelId] { return true }
        return false
    }

    /// MoC-5-FIX-1 (`RSI/DelegateMoCBacklog.md`): the `.installed` marker is *written*
    /// repo-keyed (`downloadModel` → `installedMarker(forHFModelID:)`), so it must be *read*
    /// the same way. `isInstalled` therefore takes the whole `ModelEntry` — for a card whose
    /// id equals its repo slug (38 of 39 shipping entries) this is byte-identical to the old
    /// `forModelID:` lookup; for MoC-6's `Qwen3.5 9B Vision`, whose id deliberately diverges
    /// from the repo slug it shares with the `llm` card, it is the difference between the
    /// card reading "Installed" and it offering a 6 GB re-download of files already on disk.
    func isInstalled(_ model: ModelEntry) -> Bool {
        if case .installed = modelStates[model.id] { return true }
        return FileManager.default.fileExists(
            atPath: store.installedMarker(forHFModelID: model.hfModelId).path)
    }

    /// MoC-5-FIX-1: resolve each registry id to its HF repo via `catalog` before checking the
    /// marker, matching where `downloadModel` wrote it. `catalog` can be empty — at launch
    /// this runs from `AppState.init()` before `browser.json` is decoded, and
    /// `AppState.loadBrowserData()` re-runs the reconciliation once the catalog exists. An id
    /// absent from the catalog (pulled from `browser.json` in a later release, still on disk)
    /// falls back to the card-keyed path, which is also its repo path since a single-card
    /// entry always has `id == repoSlug(hfModelId)`.
    func loadInstalled(modelIDs: Set<String>, catalog: [ModelEntry]) -> Set<String> {
        let hfModelIDByCardID = Dictionary(
            catalog.map { ($0.id, $0.hfModelId) }, uniquingKeysWith: { first, _ in first })
        var installed = Set<String>()
        for id in modelIDs {
            let marker = hfModelIDByCardID[id].map { store.installedMarker(forHFModelID: $0) }
                ?? store.installedMarker(forModelID: id)
            if FileManager.default.fileExists(atPath: marker.path) {
                installed.insert(id)
                modelStates[id] = .installed
            } else {
                print("[Registry] Stale entry \(id): marker missing at \(marker.path)")
            }
        }
        print("[Registry] loadInstalled: \(installed.count)/\(modelIDs.count) verified")
        return installed
    }

    // MARK: - Download Logic

    /// `repo` is the slug `install` already computed — threaded through rather than
    /// recomputed, and used for every `modelStates` update (`setState(_:forRepo:)`) so a
    /// download started by one card reaches every card watching the same repo (MoC-5-4).
    private func downloadModel(_ model: ModelEntry, repo: String, onComplete: @escaping (String) -> Void) async {
        let modelId = model.id
        let hfModelId = model.hfModelId
        let variant = model.variants?.first?.hfModelId ?? hfModelId

        print("[Install] Starting download for \(model.displayName) → \(variant)")

        do {
            // 1. Resolve files from HF API
            let files = try await resolveFiles(for: variant)
            guard !files.isEmpty else {
                await MainActor.run {
                    setState(.error("No downloadable files found", canRetry: false), forRepo: repo)
                    downloadTasksByRepo[repo] = nil
                }
                return
            }

            // 1b. Some models need weights from a second HF repo bundled into the install
            // (e.g. MusicGen's EnCodec codec is a separate repo — MG-DL1). Their files land
            // under `encodec/` inside the model dir, so the engine's A2-pattern load path
            // finds them next to the model's own files.
            let companion = ModelFileSelector.companionRepo(for: hfModelId)
            var companionFiles: [HFRemoteFile] = []
            if let companion {
                companionFiles = try await resolveFiles(for: companion)
            }

            print("[Install] Resolved \(files.count) files" + (companion.map { " + \($0)" } ?? ""))

            await MainActor.run {
                setState(.downloading(progress: 0, downloaded: 0, total: 1), forRepo: repo)
            }

            // 2. Create download directory. MoC-5-2: repo-keyed (not card-keyed) so two
            // cards naming the same repo (MoC-6) land in the same place.
            let downloadDir = store.downloadDirectory(forHFModelID: hfModelId)
            if FileManager.default.fileExists(atPath: downloadDir.path) {
                try FileManager.default.removeItem(at: downloadDir)
            }
            try FileManager.default.createDirectory(at: downloadDir, withIntermediateDirectories: true)
            print("[Install] Download dir: \(downloadDir.path)")

            // 3. Download files sequentially. A "plan" entry carries its own source repo (a
            // companion repo's files come from a different HF id) and a destination subdir
            // (`encodec/` for the companion, empty = model root).
            struct DownloadPlanEntry {
                let file: HFRemoteFile
                let repo: String
                let subdir: String
            }
            var plan: [DownloadPlanEntry] = files.map { .init(file: $0, repo: variant, subdir: "") }
            if let companion {
                plan += companionFiles.map { .init(file: $0, repo: companion, subdir: "encodec/") }
            }

            var downloadedSoFar: Int64 = 0
            var totalExpectedSoFar: Int64 = 0
            var tempFiles: [(file: HFRemoteFile, localURL: URL)] = []

            for (entryIndex, entry) in plan.enumerated() {
                guard !Task.isCancelled else { return }
                let file = entry.file
                let localName = entry.subdir + file.filename
                guard let downloadURL = buildDownloadURL(model: entry.repo, filename: file.filename) else {
                    await MainActor.run {
                        setState(.error("Invalid URL for \(file.filename)", canRetry: false), forRepo: repo)
                        downloadTasksByRepo[repo] = nil
                    }
                    return
                }
                print("[Install] Downloading \(file.filename) from \(downloadURL.absoluteString)")
                let localURL = downloadDir.appendingPathComponent(localName)

                // Track this file's bytes separately
                var fileBytesExpected: Int64 = file.size
                // Catalog sizes of files not yet started, so the total denominator stays stable.
                let remainingCatalogSize = plan.dropFirst(entryIndex + 1).reduce(0) { $0 + $1.file.size }

                do {
                    let (tempURL, response) = try await session.download(from: downloadURL) { bytesWritten, totalExpected in
                        if totalExpected > 0 { fileBytesExpected = totalExpected }
                        let total = totalExpectedSoFar + max(fileBytesExpected, totalExpected) + remainingCatalogSize
                        let downloaded = downloadedSoFar + bytesWritten
                        Task { @MainActor in
                            self.setState(.downloading(
                                progress: DownloadProgress.fraction(downloaded: downloaded, total: total),
                                downloaded: downloaded,
                                total: max(total, 1)
                            ), forRepo: repo)
                        }
                    }
                    // Get actual file size from response if API didn't provide it
                    let actualSize: Int64 = {
                        if file.size > 0 { return file.size }
                        if let httpResponse = response as? HTTPURLResponse,
                           let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
                           let size = Int64(contentLength) { return size }
                        return (try? FileManager.default.attributesOfItem(atPath: tempURL.path)[.size] as? Int64) ?? file.size
                    }()
                    // Ensure the destination directory exists, including nested paths for
                    // files in subfolders (e.g. Kokoro's `voices/af_heart.safetensors`).
                    try FileManager.default.createDirectory(
                        at: localURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                    if FileManager.default.fileExists(atPath: localURL.path) {
                        try FileManager.default.removeItem(at: localURL)
                    }
                    // Copy from temp to destination (moveItem can fail across sandbox boundaries)
                    try FileManager.default.copyItem(at: tempURL, to: localURL)
                    try FileManager.default.removeItem(at: tempURL)
                    downloadedSoFar += actualSize
                    totalExpectedSoFar += fileBytesExpected > 0 ? fileBytesExpected : actualSize
                    tempFiles.append((HFRemoteFile(filename: file.filename, size: actualSize), localURL))
                } catch {
                    if Task.isCancelled { return }
                    let msg = error.localizedDescription
                    print("[Install] Download failed: \(msg)")
                    await MainActor.run {
                        setState(.error(msg, canRetry: true), forRepo: repo)
                        downloadTasksByRepo[repo] = nil
                    }
                    return
                }
            }

            // 4. Verify
            await MainActor.run { setState(.verifying, forRepo: repo) }
            guard verifyFiles(tempFiles) else {
                await MainActor.run {
                    setState(.error("File verification failed", canRetry: true), forRepo: repo)
                    downloadTasksByRepo[repo] = nil
                }
                return
            }

            // 5. Atomic move. MoC-5-2: repo-keyed final directory and marker — this is the
            // one place a model's actual weights land, so it must match `isInstalled`/
            // `uninstall`'s repo-path resolution.
            let modelDir = store.directory(forHFModelID: hfModelId)
            if FileManager.default.fileExists(atPath: modelDir.path) {
                try FileManager.default.removeItem(at: modelDir)
            }
            try FileManager.default.copyItem(at: downloadDir, to: modelDir)
            try FileManager.default.removeItem(at: downloadDir)
            FileManager.default.createFile(atPath: store.installedMarker(forHFModelID: hfModelId).path, contents: nil)

            print("[Install] Success: \(model.displayName)")
            await MainActor.run {
                setState(.installed, forRepo: repo)
                downloadTasksByRepo[repo] = nil
                // MoC-5-4: one `onComplete` closure exists per repo-download (the
                // originating `install` call's) — invoke it once per card watching this
                // repo, so every sibling's `AppState.installedModelIDs` picks it up, not
                // only the card that happened to start the download.
                for cardID in cardIDsByRepo[repo] ?? [modelId] {
                    onComplete(cardID)
                }
            }

        } catch InstallError.needsAuth {
            print("[Install] Gated model — needs HF token: \(modelId)")
            await MainActor.run {
                setState(.needsAuth(
                    "This model is gated. Add a HuggingFace token in Settings, then retry."), forRepo: repo)
                downloadTasksByRepo[repo] = nil
            }
        } catch {
            if Task.isCancelled { return }
            print("[Install] Error: \(error.localizedDescription)")
            await MainActor.run {
                setState(.error(error.localizedDescription, canRetry: true), forRepo: repo)
                downloadTasksByRepo[repo] = nil
            }
        }
    }

    private func buildDownloadURL(model: String, filename: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(model)/resolve/main/\(filename)"
        return components.url
    }

    // MARK: - HF API

    private func resolveFiles(for modelId: String) async throws -> [HFRemoteFile] {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/api/models/\(modelId)"
        guard let apiURL = components.url else {
            throw InstallError.downloadFailed("Invalid API URL for \(modelId)")
        }
        print("[Install] Resolving files from \(apiURL.absoluteString)")
        var request = URLRequest(url: apiURL)
        request.setValue("ai-browser/1.0", forHTTPHeaderField: "User-Agent")

        if let token = KeychainHelper.getToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await URLSession.shared.data(for: request)

        if let httpResponse = response as? HTTPURLResponse,
           let statusError = InstallError.fromHTTPStatus(httpResponse.statusCode, modelId: modelId) {
            throw statusError
        }

        struct SiblingInfo: Codable {
            let rfilename: String
            let size: Int64?
        }
        struct ModelInfoResponse: Codable {
            let siblings: [SiblingInfo]
        }

        let info = try JSONDecoder().decode(ModelInfoResponse.self, from: data)

        // Which files to download is pure, unit-tested logic (see `ModelFileSelector`).
        let sizeByName = Dictionary(
            info.siblings.map { ($0.rfilename, $0.size ?? 0) }, uniquingKeysWith: { first, _ in first })
        var files = ModelFileSelector.filesToDownload(siblings: info.siblings.map(\.rfilename))
            .map { HFRemoteFile(filename: $0, size: sizeByName[$0] ?? 0) }

        // Sort: small metadata first, large weights (.safetensors) last.
        files.sort { f1, f2 in
            let w1 = f1.filename.lowercased().hasSuffix(".safetensors")
            let w2 = f2.filename.lowercased().hasSuffix(".safetensors")
            if w1 != w2 { return !w1 }
            return f1.size < f2.size
        }

        return files
    }

    private func verifyFiles(_ files: [(file: HFRemoteFile, localURL: URL)]) -> Bool {
        for (file, localURL) in files {
            guard FileManager.default.fileExists(atPath: localURL.path) else { return false }
            let attrs = try? FileManager.default.attributesOfItem(atPath: localURL.path)
            let actualSize = (attrs?[.size] as? Int64) ?? 0
            if file.size > 0 && abs(actualSize - file.size) > 1024 { return false }
        }
        return true
    }

    // MARK: - Installed Registry

    func saveRegistry(installedIDs: Set<String>, browserData: BrowserData?) {
        let catalog = browserData?.domains.flatMap { $0.allModels } ?? []
        let entryByID = Dictionary(catalog.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        var models: [String: InstalledModel] = [:]
        var found = 0
        for id in installedIDs {
            // MoC-5-FIX-1: the marker, the model directory and the recorded `path` all resolve
            // by repo — the slug the files were actually written under. `id` itself for a card
            // not in the catalog (which is also its own repo path).
            let repoID = entryByID[id].map { ModelStore.repoSlug(for: $0.hfModelId) } ?? id
            let marker = store.installedMarker(forModelID: repoID)
            guard FileManager.default.fileExists(atPath: marker.path) else {
                print("[Registry] Marker not found for \(id) at \(marker.path)")
                continue
            }
            found += 1
            let modelDir = store.directory(forModelID: repoID)
            let totalSize = (try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: [.fileSizeKey], options: .skipsHiddenFiles))?
                .compactMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }
                .reduce(0, +) ?? 0

            let entry = entryByID[id]
            models[id] = InstalledModel(
                installedAt: ISO8601DateFormatter().string(from: Date()),
                variant: entry?.variants?.first?.hfModelId ?? entry?.hfModelId ?? id,
                path: "models/\(repoID)",
                sizeBytes: totalSize
            )
        }

        let registry = InstalledModels(version: 1, models: models)
        do {
            let data = try JSONEncoder().encode(registry)
            try data.write(to: installedURL)
            print("[Registry] Saved \(found)/\(installedIDs.count) models to \(installedURL.path)")
        } catch {
            print("[Registry] Failed to save: \(error.localizedDescription)")
        }
    }
}

// MARK: - Supporting Types

struct HFRemoteFile {
    let filename: String
    let size: Int64
}

/// Pure disk-space pre-flight logic. Kept free of any I/O so it can be unit-tested
/// directly (see RSI/evals/eval-plan.md, gate G2).
enum DiskSpace {
    /// Installing copies the downloaded files from the temp `downloads/` dir into the
    /// `models/` dir before deleting the temp copy, so peak usage is ~2× the download size.
    static let installOverheadMultiplier = 2.0
    /// Extra headroom so we never fill the volume to the brim.
    static let safetyBufferGB = 2.0

    /// Total free space (GB) needed to install a model of the given download size.
    static func requiredGB(downloadSizeGB: Double) -> Double {
        downloadSizeGB * installOverheadMultiplier + safetyBufferGB
    }

    /// Returns a user-facing error message if `availableDiskGB` is insufficient, else `nil`.
    static func preflightError(downloadSizeGB: Double, availableDiskGB: Double) -> String? {
        let required = requiredGB(downloadSizeGB: downloadSizeGB)
        guard availableDiskGB < required else { return nil }
        return String(
            format: "Not enough disk space — this model needs about %.1f GB free (%.1f GB to download), but only %.1f GB is available.",
            required, downloadSizeGB, availableDiskGB
        )
    }
}

/// Pure progress-fraction math for the download UI. `downloaded` can transiently exceed
/// `total` because `total` is estimated and revised mid-download, so the ratio is clamped
/// into the 0...1 range SwiftUI's `ProgressView(value:)` requires (avoids an out-of-bounds
/// runtime warning). Kept I/O-free for unit testing (see RSI/evals/eval-plan.md, gate G2).
enum DownloadProgress {
    static func fraction(downloaded: Int64, total: Int64) -> Double {
        guard total > 0 else { return 0 }
        return min(max(Double(downloaded) / Double(total), 0), 1)
    }
}

enum InstallError: LocalizedError {
    case needsAuth
    case modelNotFound(String)
    case downloadFailed(String)

    var errorDescription: String? {
        switch self {
        case .needsAuth:
            return "This model requires a HuggingFace token."
        case .modelNotFound(let id):
            return "Model not found: \(id)"
        case .downloadFailed(let msg):
            return "Download failed: \(msg)"
        }
    }

    /// Maps an HF API HTTP status code to the install error it represents, or `nil` if the
    /// status isn't one we special-case. Pure (no I/O) so it can be unit-tested directly
    /// (see RSI/evals/eval-plan.md, gate G2). 401/403 → gated model needs a token.
    static func fromHTTPStatus(_ code: Int, modelId: String) -> InstallError? {
        switch code {
        case 401, 403: return .needsAuth
        case 404: return .modelNotFound(modelId)
        default: return nil
        }
    }
}

// MARK: - URLSession + Progress

extension URLSession {
    func download(from url: URL, progress: @escaping (Int64, Int64) -> Void) async throws -> (URL, URLResponse) {
        var request = URLRequest(url: url)
        request.setValue("ai-browser/1.0", forHTTPHeaderField: "User-Agent")
        if let token = KeychainHelper.getToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // Create a safe temp directory that won't be cleaned up
        let safeDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: safeDir, withIntermediateDirectories: true)

        let delegate = ProgressDelegate(onProgress: progress, safeDir: safeDir)
        let downloadSession = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        let task = downloadSession.downloadTask(with: request)

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                delegate.onComplete = { location, response in
                    continuation.resume(returning: (location, response))
                }
                delegate.onError = { error in
                    try? FileManager.default.removeItem(at: safeDir)
                    continuation.resume(throwing: error)
                }
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }
}

private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate {
    let onProgress: (Int64, Int64) -> Void
    let safeDir: URL
    var onComplete: ((URL, URLResponse) -> Void)?
    var onError: ((Error) -> Void)?

    init(onProgress: @escaping (Int64, Int64) -> Void, safeDir: URL) {
        self.onProgress = onProgress
        self.safeDir = safeDir
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // Copy to safe location before URLSession deletes the temp file
        let dest = safeDir.appendingPathComponent(location.lastPathComponent)
        do {
            try FileManager.default.copyItem(at: location, to: dest)
            onComplete?(dest, downloadTask.response ?? URLResponse())
        } catch {
            onError?(error)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error = error {
            onError?(error)
        }
    }
}

// MARK: - Keychain Helper

enum KeychainHelper {
    private static let service = "com.ai-browser"
    private static let account = "huggingface-token"

    static func getToken() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func saveToken(_ token: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: token.data(using: .utf8)!,
        ]
        SecItemDelete(query as CFDictionary)
        SecItemAdd(query as CFDictionary, nil)
    }

    static func deleteToken() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
