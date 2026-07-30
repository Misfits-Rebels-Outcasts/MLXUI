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
    private let modelsDir: URL
    private let downloadsDir: URL
    private let installedURL: URL
    private let session: URLSession
    private var activeTasks: [String: [URLSessionDownloadTask]] = [:]

    var modelStates: [String: InstallState] = [:]
    var onInstallComplete: ((String) -> Void)?

    init() {
        let store = ModelStore.shared
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

        // Pre-flight: refuse to start a download that can't fit on disk, rather than
        // failing partway through after writing gigabytes of temp files.
        if let diskError = DiskSpace.preflightError(
            downloadSizeGB: model.downloadSizeGB,
            availableDiskGB: currentAvailableDiskGB()
        ) {
            modelStates[model.id] = .error(diskError, canRetry: true)
            return
        }

        modelStates[model.id] = .resolving
        Task { await downloadModel(model, onComplete: onComplete) }
    }

    /// Free space (GB) on the volume that holds the models directory. Measured fresh at
    /// install time so the check reflects current conditions, not a stale launch reading.
    private func currentAvailableDiskGB() -> Double {
        let free = try? modelsDir.resourceValues(forKeys: [.volumeAvailableCapacityKey])
            .volumeAvailableCapacity
        return Double(free ?? 0) / 1_000_000_000.0
    }

    func cancel(_ modelId: String) {
        activeTasks[modelId]?.forEach { $0.cancel() }
        activeTasks[modelId] = nil
        modelStates[modelId] = .idle
        let downloadDir = downloadsDir.appendingPathComponent(modelId)
        try? FileManager.default.removeItem(at: downloadDir)
    }

    func uninstall(_ modelId: String) {
        let modelDir = modelsDir.appendingPathComponent(modelId)
        try? FileManager.default.removeItem(at: modelDir)
        modelStates[modelId] = .idle
    }

    func isError(_ modelId: String) -> Bool {
        if case .error = modelStates[modelId] { return true }
        return false
    }

    func isNeedsAuth(_ modelId: String) -> Bool {
        if case .needsAuth = modelStates[modelId] { return true }
        return false
    }

    func isInstalled(_ modelId: String) -> Bool {
        if case .installed = modelStates[modelId] { return true }
        return FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent(modelId).appendingPathComponent(".installed").path)
    }

    func loadInstalled(modelIDs: Set<String>) -> Set<String> {
        var installed = Set<String>()
        for id in modelIDs {
            let marker = modelsDir.appendingPathComponent(id).appendingPathComponent(".installed")
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

    private func downloadModel(_ model: ModelEntry, onComplete: @escaping (String) -> Void) async {
        let modelId = model.id
        let hfModelId = model.hfModelId
        let variant = model.variants?.first?.hfModelId ?? hfModelId

        print("[Install] Starting download for \(model.displayName) → \(variant)")

        do {
            // 1. Resolve files from HF API
            let files = try await resolveFiles(for: variant)
            guard !files.isEmpty else {
                await MainActor.run {
                    modelStates[modelId] = .error("No downloadable files found", canRetry: false)
                }
                return
            }

            print("[Install] Resolved \(files.count) files")

            await MainActor.run {
                modelStates[modelId] = .downloading(progress: 0, downloaded: 0, total: 1)
            }

            // 2. Create download directory
            let downloadDir = downloadsDir.appendingPathComponent(modelId)
            if FileManager.default.fileExists(atPath: downloadDir.path) {
                try FileManager.default.removeItem(at: downloadDir)
            }
            try FileManager.default.createDirectory(at: downloadDir, withIntermediateDirectories: true)
            print("[Install] Download dir: \(downloadDir.path)")

            // 3. Download files sequentially
            var downloadedSoFar: Int64 = 0
            var totalExpectedSoFar: Int64 = 0
            var tempFiles: [(file: HFRemoteFile, localURL: URL)] = []

            for file in files {
                guard let downloadURL = buildDownloadURL(model: variant, filename: file.filename) else {
                    await MainActor.run {
                        modelStates[modelId] = .error("Invalid URL for \(file.filename)", canRetry: false)
                    }
                    return
                }
                print("[Install] Downloading \(file.filename) from \(downloadURL.absoluteString)")
                let localURL = downloadDir.appendingPathComponent(file.filename)

                // Track this file's bytes separately
                var fileBytesDownloaded: Int64 = 0
                var fileBytesExpected: Int64 = file.size

                do {
                    let (tempURL, response) = try await session.download(from: downloadURL) { bytesWritten, totalExpected in
                        fileBytesDownloaded = bytesWritten
                        if totalExpected > 0 { fileBytesExpected = totalExpected }
                        let total = totalExpectedSoFar + max(fileBytesExpected, totalExpected)
                        let downloaded = downloadedSoFar + bytesWritten
                        Task { @MainActor in
                            self.modelStates[modelId] = .downloading(
                                progress: DownloadProgress.fraction(downloaded: downloaded, total: total),
                                downloaded: downloaded,
                                total: max(total, 1)
                            )
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
                    let msg = error.localizedDescription
                    print("[Install] Download failed: \(msg)")
                    await MainActor.run {
                        modelStates[modelId] = .error(msg, canRetry: true)
                    }
                    return
                }
            }

            // 4. Verify
            await MainActor.run { modelStates[modelId] = .verifying }
            guard verifyFiles(tempFiles) else {
                await MainActor.run {
                    modelStates[modelId] = .error("File verification failed", canRetry: true)
                }
                return
            }

            // 5. Atomic move
            let modelDir = modelsDir.appendingPathComponent(modelId)
            if FileManager.default.fileExists(atPath: modelDir.path) {
                try FileManager.default.removeItem(at: modelDir)
            }
            try FileManager.default.copyItem(at: downloadDir, to: modelDir)
            try FileManager.default.removeItem(at: downloadDir)
            FileManager.default.createFile(atPath: modelDir.appendingPathComponent(".installed").path, contents: nil)

            print("[Install] Success: \(model.displayName)")
            await MainActor.run {
                modelStates[modelId] = .installed
                onComplete(modelId)
            }

        } catch InstallError.needsAuth {
            print("[Install] Gated model — needs HF token: \(modelId)")
            await MainActor.run {
                modelStates[modelId] = .needsAuth(
                    "This model is gated. Add a HuggingFace token in Settings, then retry.")
            }
        } catch {
            print("[Install] Error: \(error.localizedDescription)")
            await MainActor.run {
                modelStates[modelId] = .error(error.localizedDescription, canRetry: true)
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
        var models: [String: InstalledModel] = [:]
        var found = 0
        for id in installedIDs {
            let marker = modelsDir.appendingPathComponent(id).appendingPathComponent(".installed")
            guard FileManager.default.fileExists(atPath: marker.path) else {
                print("[Registry] Marker not found for \(id) at \(marker.path)")
                continue
            }
            found += 1
            let modelDir = modelsDir.appendingPathComponent(id)
            let totalSize = (try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: [.fileSizeKey], options: .skipsHiddenFiles))?
                .compactMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }
                .reduce(0, +) ?? 0

            let entry = browserData?.domains.flatMap { $0.allModels }.first(where: { $0.id == id })
            models[id] = InstalledModel(
                installedAt: ISO8601DateFormatter().string(from: Date()),
                variant: entry?.variants?.first?.hfModelId ?? entry?.hfModelId ?? id,
                path: "models/\(id)",
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
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

        return try await withCheckedThrowingContinuation { continuation in
            delegate.onComplete = { location, response in
                continuation.resume(returning: (location, response))
            }
            delegate.onError = { error in
                try? FileManager.default.removeItem(at: safeDir)
                continuation.resume(throwing: error)
            }
            let task = session.downloadTask(with: request)
            task.resume()
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
