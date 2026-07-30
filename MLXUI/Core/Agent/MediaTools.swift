//
//  MediaTools.swift
//  MLXUI — agentic chat tools, slice AG4c.
//
//  Media tools that let a text chat model reach into audio/image modalities:
//    • transcribe_audio — audio URL → text via MLXWhisperEngine (ASR)
//    • ocr_image        — image URL → extracted text via the OCR/VLM engines
//
//  Input arrives as an http(s) URL (fetched over the app's `network.client` entitlement, like
//  AG2's `fetch_url`) — NOT a local file path, which would need AG5's human-approved
//  `files.user-selected.read-write`. So these stay inside the current sandbox posture.
//
//  Decode + inference reuse the app's existing nonisolated engines/helpers (AudioFileReader +
//  AudioResampler, ImageLoader, MLXWhisperEngine, VLM/Paddle/Dots/DeepSeek OCR engines). The
//  network transfer + heavy decode can't be unit-tested (offline CI, no weights); the pure bits
//  (OCR engine routing) live in a `static` helper with synchronous tests.
//

import Foundation
import CoreGraphics
import MLXLMCommon

/// Shared raw-bytes fetch for the media tools: validate an http(s) URL, GET the body with time +
/// size caps, and return the bytes plus the URL's file extension. Read-only.
nonisolated enum MediaFetch {
    /// Largest media file accepted. Unlike `fetch_url`, media is rejected (not truncated) past
    /// the cap — a partial audio/image file would just fail to decode.
    static let maxBytes = 25 * 1024 * 1024
    /// Per-request + whole-resource timeout (seconds); the real runaway guard.
    static let timeout: TimeInterval = 30

    /// Outcome of `fetchData` — the bytes + file extension, or a human-readable failure reason.
    enum Outcome {
        case success(data: Data, fileExtension: String)
        case failure(String)
    }

    /// GET the bytes at `raw`, or a human-readable failure reason.
    static func fetchData(_ raw: String) async -> Outcome {
        let url: URL
        switch FetchURLTool.validate(raw) {
        case .ok(let u): url = u
        case .invalid(let message): return .failure(message)
        }

        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = timeout
        config.timeoutIntervalForResource = timeout
        config.httpCookieStorage = nil
        config.urlCache = nil
        let session = URLSession(configuration: config)
        defer { session.finishTasksAndInvalidate() }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Mozilla/5.0 (compatible; AI-Browser-Agent/1.0)", forHTTPHeaderField: "User-Agent")

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return .failure("non-HTTP response from \(url.absoluteString).")
            }
            guard (200..<300).contains(http.statusCode) else {
                return .failure("HTTP \(http.statusCode) fetching \(url.absoluteString).")
            }
            guard data.count <= maxBytes else {
                return .failure("file is too large (\(data.count / 1_048_576) MB > \(maxBytes / 1_048_576) MB limit).")
            }
            return .success(data: data, fileExtension: url.pathExtension)
        } catch let error as URLError where error.code == .timedOut {
            return .failure("request to \(url.absoluteString) timed out after \(Int(timeout))s.")
        } catch {
            return .failure("could not fetch \(url.absoluteString): \(error.localizedDescription)")
        }
    }
}

/// Transcribes speech from an audio file at an http(s) URL using a locally-installed ASR model.
nonisolated struct TranscribeAudioTool: AgentTool {
    let name = "transcribe_audio"
    let toolDescription = """
        Transcribe speech from an audio file at an http(s) URL into text, using a locally-installed \
        speech-to-text model. Supports common audio formats (wav, mp3, m4a, …). Read-only.
        """
    let parameters: [ToolParameter] = [
        .required("url", type: .string, description: "The absolute http(s) URL of the audio file.")
    ]

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let raw = (arguments.string("url") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return "Error: missing required 'url' argument." }
        guard let modelID = InstalledModelIndex.loadInstalled().best(kind: .asr) else {
            return "Error: no speech-to-text model is installed. Install one from the Browse tab first."
        }

        let data: Data
        let ext: String
        switch await MediaFetch.fetchData(raw) {
        case .failure(let message): return "Error: \(message)"
        case .success(let bytes, let fileExtension): (data, ext) = (bytes, fileExtension)
        }

        // AVFoundation sniffs by file extension, so keep the URL's extension on the temp file.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("agent-audio-\(UUID().uuidString)")
            .appendingPathExtension(ext.isEmpty ? "audio" : ext)
        do {
            try data.write(to: tmp)
            defer { try? FileManager.default.removeItem(at: tmp) }
            let buffer = try AudioFileReader.read(tmp)
            let samples = try AudioResampler.resample(buffer, to: 16_000).samples
            let text = try await MLXWhisperEngine.transcribe(
                samples,
                modelDirectory: MLXWhisperEngine.installedModelDirectory(id: modelID),
                language: nil)
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? "(no speech detected in the audio)" : trimmed
        } catch {
            return "Error: could not process the audio: \(error.localizedDescription)"
        }
    }
}

/// Extracts text from an image at an http(s) URL using a locally-installed OCR (or vision) model.
nonisolated struct OCRImageTool: AgentTool {
    let name = "ocr_image"
    let toolDescription = """
        Extract the text from an image at an http(s) URL using a locally-installed OCR or vision \
        model. Returns the transcribed text in reading order. Read-only.
        """
    let parameters: [ToolParameter] = [
        .required("url", type: .string, description: "The absolute http(s) URL of the image.")
    ]

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let raw = (arguments.string("url") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return "Error: missing required 'url' argument." }
        // Prefer a dedicated OCR model; fall back to a general vision model.
        let index = InstalledModelIndex.loadInstalled()
        guard let modelID = index.best(kind: .ocr) ?? index.best(kind: .vision) else {
            return "Error: no OCR or vision model is installed. Install one from the Browse tab first."
        }

        let data: Data
        switch await MediaFetch.fetchData(raw) {
        case .failure(let message): return "Error: \(message)"
        case .success(let bytes, _): data = bytes
        }
        guard let image = ImageLoader.decodedCGImage(from: data) else {
            return "Error: could not decode an image from \(raw)."
        }

        do {
            let text: String
            switch Self.backend(forModelID: modelID) {
            case .paddle:
                text = try await PaddleOCREngine.generate(
                    image: image,
                    modelDir: PaddleOCREngine.installedModelDirectory(id: modelID),
                    maxTokens: OCRSDK.ocrMaxTokens)
            case .deepseek:
                text = try await DeepSeekOCREngine.generate(
                    image: image,
                    modelDir: DeepSeekOCREngine.installedModelDirectory(id: modelID),
                    maxTokens: OCRSDK.ocrMaxTokens)
            case .dots:
                text = try await DotsOCREngine.generate(
                    image: image,
                    modelDir: DotsOCREngine.installedModelDirectory(id: modelID),
                    maxTokens: OCRSDK.ocrMaxTokens)
            case .vlm:
                text = try await VLMEngine.generate(
                    prompt: OCRSDK.ocrPrompt,
                    image: image,
                    modelDir: VLMEngine.installedModelDirectory(id: modelID),
                    maxTokens: OCRSDK.ocrMaxTokens)
            }
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? "(no text found in the image)" : trimmed
        } catch {
            return "Error: could not read text from the image: \(error.localizedDescription)"
        }
    }

    /// Which engine runs a given OCR/vision model, mirroring the per-module `ModelSDK.claim`
    /// predicates (which match on the model id/repo). Pure, so it is unit-testable.
    enum Backend: Equatable { case paddle, deepseek, dots, vlm }

    static func backend(forModelID id: String) -> Backend {
        let h = id.lowercased()
        if h.contains("paddleocr") { return .paddle }
        if h.contains("deepseek-ocr") { return .deepseek }
        if h.contains("dots") { return .dots }
        return .vlm  // olmOCR / qwen2.5-vl OCR / any vision model → shared MLXVLM engine
    }
}
