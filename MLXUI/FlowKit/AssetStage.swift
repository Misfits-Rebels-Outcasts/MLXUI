import Foundation
import CoreGraphics
import AppKit

/// A FlowKit stage operating on `Asset` values — FlowKit's own `PipelineStage` analogue,
/// with `Shape` in/out so the flow layer checks compatibility with the shared shape logic.
/// Ported in spirit from `catflow-mlx`'s tool/engine contract; `SingleMediaStage` adapts the
/// 16 shipped `PipelineStage` modules to this protocol **without changing any of them**
/// (`PipelineStage.accepts`/`produces` stay a single `MediaKind` each — do not widen it).
/// See `RSI/DelegateMergeBacklog.md` CFM-R2-3.
nonisolated protocol AssetStage: Sendable {
    var accepts: Shape { get }
    var produces: Shape { get }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset
}

/// The `Kind ↔ MediaKind` mapping — one place, so the adapter and the tools can't drift.
/// `text↔.text`, `audio↔.audio`, `image↔.image`, `vector↔.embedding`. **Every other `Kind`
/// has no `MediaKind`** and must produce a "this row needs a capability this version
/// doesn't have yet" refusal rather than a silent fallthrough.
nonisolated enum MediaKindMapping {
    static func mediaKind(for kind: Kind) -> MediaKind? {
        switch kind {
        case .text:   return .text
        case .audio:  return .audio
        case .image:  return .image
        case .vector: return .embedding
        default:      return nil
        }
    }

    static func kind(for mediaKind: MediaKind) -> Kind {
        switch mediaKind {
        case .text:      return .text
        case .audio:     return .audio
        case .image:     return .image
        case .embedding: return .vector
        }
    }
}

/// Adapts any shipped `PipelineStage` (a `Media → Media` transform) into an `AssetStage`.
/// Unwraps a one-`Item` `Asset` into a `Media`, runs the stage, wraps the result back.
/// Throws a clear FlowKit error when the `Asset` has zero or more than one item, or when
/// the `Kind` doesn't map to a `MediaKind`. All file access goes through
/// `FlowWorkspace.resolve` (the caller passes already-resolved paths).
///
/// **Result persistence.** The Python's `Asset` model keeps heavy payloads (audio/image/…)
/// file-backed; `Media.audio`/`Media.image` carry their payload in memory. So the adapter
/// persists a heavy result into `blobDirectory` (the flow's working directory, resolved by
/// the runner) as a WAV/PNG and returns a path-backed `Item` — the flow model stays
/// file-backed and inspectable, exactly as the Python's run-store does.
nonisolated struct SingleMediaStage: AssetStage {
    let id: String
    let name: String
    let inner: any PipelineStage
    /// The flow row this stage belongs to, for error voice ("Row 2 …").
    let rowLabel: String
    /// Directory file-backed results are written into (per-run, inside the flow workspace).
    let blobDirectory: URL

    var accepts: Shape {
        .single(MediaKindMapping.kind(for: inner.accepts))
    }

    var produces: Shape {
        .single(MediaKindMapping.kind(for: inner.produces))
    }

    init(id: String, name: String, inner: any PipelineStage,
         rowLabel: String = "this row", blobDirectory: URL) {
        self.id = id
        self.name = name
        self.inner = inner
        self.rowLabel = rowLabel
        self.blobDirectory = blobDirectory
    }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard input.items.count == 1, let item = input.items.first else {
            throw FlowError.badInputCardinality(row: rowLabel, expected: "exactly one item",
                                                got: input.items.count)
        }
        guard let mediaKind = MediaKindMapping.mediaKind(for: item.kind) else {
            throw FlowError.unsupportedKind(row: rowLabel, kind: item.kind)
        }
        let media = try materialize(item, as: mediaKind, row: rowLabel)
        let result = try await inner.run(media, progress: progress)
        return Asset(items: [try persist(result, row: rowLabel)])
    }

    // MARK: - Result → Item persistence

    /// Wrap a result `Media` as an `Item`: text stays inline, heavy payloads are written
    /// to `blobDirectory` and returned file-backed.
    private func persist(_ media: Media, row: String) throws -> Item {
        switch media {
        case .text(let s):
            return Item(kind: .text, value: s, path: nil, sourceText: nil)
        case .audio(let buffer):
            let url = try blobURL(extension: "wav")
            do {
                try AudioWriter.writeWAV(buffer, to: url)
            } catch {
                throw FlowError.writeFailed(row: row, path: url.lastPathComponent)
            }
            return Item(kind: .audio, value: nil, path: url, sourceText: nil)
        case .image(let image):
            let url = try blobURL(extension: "png")
            guard let data = PNGEncoder.pngData(from: image.cgImage) else {
                throw FlowError.writeFailed(row: row, path: url.lastPathComponent)
            }
            do {
                try data.write(to: url)
            } catch {
                throw FlowError.writeFailed(row: row, path: url.lastPathComponent)
            }
            return Item(kind: .image, value: nil, path: url, sourceText: nil)
        case .embedding(let vectors):
            // CFM-R12-6: persist each vector as a `.npy` (the SPEC-Q15 on-disk format the
            // index tools read), so `Store Index`/`Retrieve` consume `Embed`'s output.
            // The FIRST vector is the returned item (single-media adapter); a batch is
            // written to the blob dir too, and the item names the first file.
            guard let first = vectors.first else {
                throw FlowError.unsupportedKind(row: row, kind: .vector)
            }
            let url = try blobURL(extension: "npy")
            do {
                try NpyCodec.save(first, to: url)
            } catch {
                throw FlowError.writeFailed(row: row, path: url.lastPathComponent)
            }
            return Item(kind: .vector, value: nil, path: url, sourceText: nil)
        }
    }

    private func blobURL(extension ext: String) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
        return blobDirectory.appendingPathComponent("\(id).\(UUID().uuidString).\(ext)")
    }

    // MARK: - Item ↔ Media materialization

    /// Turn a file-backed or inline `Item` into the `Media` the stage consumes.
    private func materialize(_ item: Item, as mediaKind: MediaKind, row: String) throws -> Media {
        switch (item.kind, item.value, item.path) {
        case (.text, let value?, _):
            return .text(value)
        case (.text, nil, _):
            throw FlowError.missingInlineValue(row: row, kind: .text)
        case (.audio, _, let path?):
            let buffer = try AudioFileReader.read(path)
            return .audio(buffer)
        case (.image, _, let path?):
            guard let data = try? Data(contentsOf: path),
                  let cgImage = ImageLoader.decodedCGImage(from: data) else {
                throw FlowError.fileReadFailed(row: row, path: path.lastPathComponent)
            }
            return .image(ImageMedia(cgImage: cgImage))
        case (.vector, _, let path?):
            // CFM-R12-6: a vector item is a `.npy` blob (the SPEC-Q15 format) — load it back
            // into a `Media.embedding` batch of one.
            do {
                let values = try NpyCodec.load(from: path)
                return .embedding([values])
            } catch {
                throw FlowError.fileReadFailed(row: row, path: path.lastPathComponent)
            }
        case (let kind, _, nil) where MediaKindMapping.mediaKind(for: kind) == .audio:
            throw FlowError.fileReadFailed(row: row, path: "audio item")
        default:
            throw FlowError.unsupportedKind(row: row, kind: item.kind)
        }
    }

    /// The inline value of a result `Media` (text → value; audio/image/embedding → nil).
    private func inlineValue(for media: Media) -> String? {
        if case .text(let s) = media { return s }
        return nil
    }
}

/// PNG encoding for file-backed image results (the same `NSBitmapImageRep` path the run
/// views' Save PNG uses).
nonisolated enum PNGEncoder {
    static func pngData(from cgImage: CGImage) -> Data? {
        NSBitmapImageRep(cgImage: cgImage).representation(using: .png, properties: [:])
    }
}

/// FlowKit-stage errors. Error voice: name the row, one plain sentence, imply the fix.
nonisolated enum FlowError: Error, CustomStringConvertible, Equatable {
    case badInputCardinality(row: String, expected: String, got: Int)
    case unsupportedKind(row: String, kind: Kind)
    case missingInlineValue(row: String, kind: Kind)
    case fileReadFailed(row: String, path: String)
    case writeFailed(row: String, path: String)
    case frameInputOutOfRange(index: Int, count: Int)
    case unknownTask(row: String)
    case unsupportedReference(row: String)
    case referenceNotFound(row: String)
    case stageFailure(row: String, message: String)
    case modelNotRunnable(row: String, display: String, reason: String)
    case unsupportedTask(row: String, task: String)
    case invalidSettings(row: String, setting: String, detail: String)
    case budgetExceeded(row: String, visitsLeq: Int)

    var description: String {
        switch self {
        case .badInputCardinality(let row, let expected, let got):
            return "\(row) needs \(expected), but got \(got) — check what feeds it."
        case .unsupportedKind(let row, let kind):
            return "\(row) produces a \(kind.rawValue), which this version of Flows doesn't run yet."
        case .missingInlineValue(let row, let kind):
            return "\(row) is missing its \(kind.rawValue) content — the input wasn't produced."
        case .fileReadFailed(let row, let path):
            return "\(row) couldn't read '\(path)' — make sure it's in the flow's folder."
        case .writeFailed(let row, let path):
            return "\(row) couldn't write '\(path)' — the flow's folder may be read-only."
        case .frameInputOutOfRange(let index, let count):
            return "A frame references input[\(index)], but only \(count) item(s) were given."
        case .unknownTask(let row):
            return "Row \(row) uses a task this version of Flows doesn't know — update the flow."
        case .unsupportedReference(let row):
            return "Row \(row) uses a reference type this version of Flows can't follow."
        case .referenceNotFound(let row):
            return "Row \(row) references a row that hasn't run yet — check the flow's order."
        case .stageFailure(let row, let message):
            return "Row \(row) failed: \(message)"
        case .modelNotRunnable(let row, let display, let reason):
            return "Row \(row) needs \(display), which isn't runnable: \(reason)"
        case .unsupportedTask(let row, let task):
            return "Row \(row) uses \(task), which this version of Flows doesn't run yet."
        case .invalidSettings(let row, let setting, let detail):
            return "\(row)'s \(setting) setting is malformed — \(detail), then run again."
        case .budgetExceeded(let row, let visitsLeq):
            return "Row \(row) hit its budget of \(visitsLeq) visits with `on_budget=fail` — no forced edge to take."
        }
    }
}
