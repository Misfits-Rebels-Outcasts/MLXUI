import Foundation

/// Instant tools for the Spoken-Summary-shaped flows (CFM-R2-4), as `AssetStage`s. They
/// reuse the existing `Modules/Tools/` helpers (`AudioFileReader`, `AudioWriter`,
/// `AudioResampler`) rather than reimplementing them. All paths resolve through
/// `FlowWorkspace.resolve` — the CFM-R1-4 security boundary is re-asserted here, and the
/// sample-rate conversion lives in the **stage**, never the caller.

/// `Read Audio` (file → audio): resolve the settings path (bare or `path=`) and read it
/// into an `AudioBuffer` at its own sample rate. `AudioBuffer` carries the rate, so
/// downstream stages resample as needed (`Transcribe` → 16 kHz, `Speak` output → 24 kHz).
nonisolated struct ReadAudioTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String
    /// FIP-3: see `ReadImageTool.path` (`ReadTools.swift`).
    var path: String = "1"

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.audio) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        // FIP-3: Read Audio never checked an upstream `.file` item before this — verified
        // against the prior code, not assumed; `checksUpstream: false` keeps that unchanged.
        let url = try ReadPath.resolve(workspace: workspace, flowID: flowID, path: path, settings: settings,
                                       inputs: [input], kind: .file, row: "Read Audio", checksUpstream: false)
        progress(0.3)
        _ = try AudioFileReader.read(url)
        progress(1.0)
        return Asset(items: [Item(kind: .audio, value: nil, path: url, sourceText: nil)])
    }
}

/// `Save Audio` (any → status): write the input's audio to the resolved WAV path, returning
/// a `status` item whose value is the plain sentence the UI shows. The WAV keeps the
/// buffer's own sample rate (a resample to a target is the producer's job, not ours).
nonisolated struct SaveAudioTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .anyKind }
    var produces: Shape { .single(.status) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let item = input.items.first else {
            throw FlowError.badInputCardinality(row: "Save Audio", expected: "an audio input", got: 0)
        }
        guard item.kind == .audio else {
            throw FlowError.unsupportedKind(row: "Save Audio", kind: item.kind)
        }
        guard let src = item.path else {
            throw FlowError.missingInlineValue(row: "Save Audio", kind: .audio)
        }
        guard let path = FlowSettings(settings).pathValue() else {
            throw FlowError.missingInlineValue(row: "Save Audio", kind: .file)
        }
        let url = try workspace.resolve(path, flowID: flowID)

        // Read at the blob's own rate, write WAV at that same rate (16-bit PCM).
        let buffer = try AudioFileReader.read(src)
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            try AudioWriter.writeWAV(buffer, to: url)
        } catch {
            throw FlowError.writeFailed(row: "Save Audio", path: path)
        }
        progress(1.0)
        return Asset(items: [Item(kind: .status, value: "saved to \(path)", path: nil, sourceText: nil)])
    }
}

/// `Save Text` (any → status): write UTF-8 to the resolved path. `SaveTextStage` exists but
/// is a `PipelineStage` bound to a fixed URL; this flow version takes the path per run.
/// Reuses the same writing logic (atomically, UTF-8).
nonisolated struct SaveTextTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .anyKind }
    var produces: Shape { .single(.status) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let item = input.items.first else {
            throw FlowError.badInputCardinality(row: "Save Text", expected: "a text input", got: 0)
        }
        guard let path = FlowSettings(settings).pathValue() else {
            throw FlowError.missingInlineValue(row: "Save Text", kind: .file)
        }
        let url = try workspace.resolve(path, flowID: flowID)

        guard let text = item.value else {
            throw FlowError.missingInlineValue(row: "Save Text", kind: .text)
        }
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            throw FlowError.writeFailed(row: "Save Text", path: path)
        }
        progress(1.0)
        return Asset(items: [Item(kind: .status, value: "saved to \(path)", path: nil, sourceText: nil)])
    }
}

/// `Read Text` (file → text): resolve the settings path and read it as UTF-8. Needed by
/// `08-PolicyDiff`; cheap to add now (CFM-R2-4).
nonisolated struct ReadTextTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String
    /// FIP-3: see `ReadImageTool.path` (`ReadTools.swift`).
    var path: String = "1"

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        // FIP-3: Read Text never checked an upstream `.file` item before this — verified
        // against the prior code, not assumed.
        let url = try ReadPath.resolve(workspace: workspace, flowID: flowID, path: path, settings: settings,
                                       inputs: [input], kind: .file, row: "Read Text", checksUpstream: false)
        progress(0.3)
        do {
            let text = try String(contentsOf: url, encoding: .utf8)
            progress(1.0)
            return Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
        } catch {
            // resolve() already refused a missing file; a present-but-undecodable one is its
            // own, unrelated failure.
            throw FlowError.fileReadFailed(row: "Read Text", path: FlowSettings(settings).pathValue() ?? url.path)
        }
    }
}
