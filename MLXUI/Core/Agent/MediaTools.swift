//
//  MediaTools.swift
//  MLXUI — agentic chat tools, slice AG4c.
//
//  Media tools that let a text chat model reach into audio/image modalities:
//    • transcribe_audio — audio → text
//    • ocr_image        — image → extracted text
//
//  MT1-3: both are now thin names over `ModelTool` (Core/ModelTools). The model is chosen by
//  `CapabilityResolver` (not `InstalledModelIndex`, and not an id substring) and the work runs as a
//  one-row flow through `RealExecutor`, so a Parakeet-only Mac transcribes with Parakeet and an OCR
//  model is picked by its claim. The input is an http(s) `url` (fetched over `network.client`, like
//  `fetch_url`, via `MediaFetch` below), a granted `path`, or an `asset` handle in a Model Tools chat.
//  With no `ModelToolHost` bound the result is the same plain text as before.
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

/// Transcribes speech from an audio file (`url`, `path` or `asset`) using a locally-installed ASR model.
nonisolated struct TranscribeAudioTool: AgentTool {
    let name = "transcribe_audio"
    /// MT1-FIX-1: `asset` is advertised only when MT4 builds the tool for a host-bound chat.
    var advertisesAssets = false
    private var tool: ModelTool? {
        ModelToolSpec.spec(named: name).map { ModelTool(spec: $0, advertisesAssets: advertisesAssets) }
    }
    var toolDescription: String { tool?.toolDescription ?? "" }
    var parameters: [ToolParameter] { tool?.parameters ?? [] }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let tool else { return "Error: \(name) isn't available." }
        return try await tool.execute(arguments: arguments)
    }
}

/// Extracts text from an image (`url`, `path` or `asset`) using a locally-installed OCR (or vision) model.
nonisolated struct OCRImageTool: AgentTool {
    let name = "ocr_image"
    /// MT1-FIX-1: `asset` is advertised only when MT4 builds the tool for a host-bound chat.
    var advertisesAssets = false
    private var tool: ModelTool? {
        ModelToolSpec.spec(named: name).map { ModelTool(spec: $0, advertisesAssets: advertisesAssets) }
    }
    var toolDescription: String { tool?.toolDescription ?? "" }
    var parameters: [ToolParameter] { tool?.parameters ?? [] }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let tool else { return "Error: \(name) isn't available." }
        return try await tool.execute(arguments: arguments)
    }

    /// CW-3's entry point for the attachment importer: OCR **a local file** (the container copy) through the same
    /// resolver and row path as the tool, returning `nil` when no OCR/vision model is *installed* (so the "no OCR" note
    /// still shows) or the read fails. Never offers a download.
    static func recognizeText(in file: URL, runtime: ModelToolRuntime?) async -> String? {
        guard let runtime, let spec = ModelToolSpec.spec(named: "ocr_image") else { return nil }
        return await ModelToolRunner(spec: spec, runtime: runtime).recognize(file: file)
    }
}
