import Foundation
import CoreGraphics
import AppKit

// CFM-R12-7 group d — the image tools (`tools/media.py`, the Pillow-free CoreGraphics port):
// Resize, Crop, Convert, Watermark, Overlay Text, Contact Sheet. Each produces a file-backed
// `.image` item in the flow's blob directory.

/// Shared CGImage plumbing for the image tools.
nonisolated enum ImageTools {
    /// Load a decoded CGImage from a file.
    static func load(_ url: URL) throws -> CGImage {
        guard let data = try? Data(contentsOf: url),
              let cg = ImageLoader.decodedCGImage(from: data) else {
            throw FlowError.fileReadFailed(row: "image", path: url.lastPathComponent)
        }
        return cg
    }

    /// Draw `cg` at `size` (scaled), returning a new CGImage.
    static func scaled(_ cg: CGImage, to size: CGSize) throws -> CGImage {
        guard let ctx = CGContext(data: nil, width: Int(size.width.rounded()), height: Int(size.height.rounded()),
                                  bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw FlowError.stageFailure(row: "image", message: "couldn't create the canvas")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(origin: .zero, size: size))
        guard let out = ctx.makeImage() else {
            throw FlowError.stageFailure(row: "image", message: "couldn't render")
        }
        return out
    }

    /// Save a CGImage in the flow's blob directory, honoring a `format=` (jpeg/png).
    static func save(_ cg: CGImage, format: String?, blobDirectory: URL,
                     row: String) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
        let ext = (format ?? "png").lowercased() == "jpeg" || (format ?? "").lowercased() == "jpg" ? "jpeg" : "png"
        let url = blobDirectory.appendingPathComponent("img-\(UUID().uuidString).\(ext)")
        let rep = NSBitmapImageRep(cgImage: cg)
        let data: Data?
        if ext == "jpeg" {
            data = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
        } else {
            data = rep.representation(using: .png, properties: [:])
        }
        guard let data else {
            throw FlowError.writeFailed(row: row, path: url.lastPathComponent)
        }
        try data.write(to: url)
        return url
    }

    /// Fit `cg` into `maxDim`×`maxDim`, preserving aspect (Pillow's `thumbnail`).
    static func thumbnail(_ cg: CGImage, max maxDim: Int) throws -> CGImage {
        let w = CGFloat(cg.width), h = CGFloat(cg.height)
        guard w > 0, h > 0 else { return cg }
        let scale = min(CGFloat(maxDim) / w, CGFloat(maxDim) / h, 1)
        guard scale < 1 else { return cg }
        return try scaled(cg, to: CGSize(width: w * scale, height: h * scale))
    }
}

/// `Resize` (image → image): `max=`/bare, thumbnail to the max dimension; `format=` output.
nonisolated struct ResizeTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.image) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let path = try ImageTools.requireImagePath(input, row: "Resize")
        let s = FlowSettings(settings)
        let maxRaw = s.value(for: "max") ?? s.firstBare()
        let cg = try ImageTools.load(path)
        let resized: CGImage
        if let maxRaw, let max = Int(maxRaw) {
            resized = (try? ImageTools.thumbnail(cg, max: max)) ?? cg
        } else {
            resized = cg
        }
        let out = try ImageTools.save(resized, format: s.value(for: "format"),
                                      blobDirectory: blobDir, row: "Resize")
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: out, sourceText: nil)])
    }

    private var blobDir: URL { workspace.directory(for: flowID).appendingPathComponent(".blobs") }
}

/// `Crop` (image → image): `box=left,top,right,bottom`.
nonisolated struct CropTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.image) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let path = try ImageTools.requireImagePath(input, row: "Crop")
        let s = FlowSettings(settings)
        let boxRaw = s.value(for: "box") ?? s.firstBare()
        let parts = (boxRaw ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count == 4, parts.allSatisfy({ Int($0) != nil }),
              let left = Int(parts[0]), let top = Int(parts[1]),
              let right = Int(parts[2]), let bottom = Int(parts[3]) else {
            throw FlowError.invalidSettings(row: "Crop", setting: "box",
                                            detail: "needs 4 comma-separated integers")
        }
        let cg = try ImageTools.load(path)
        guard let cropped = cg.cropping(to: CGRect(x: left, y: top,
                                                   width: right - left, height: bottom - top)) else {
            throw FlowError.stageFailure(row: "Crop", message: "box is outside the image")
        }
        let out = try ImageTools.save(cropped, format: nil,
                                      blobDirectory: workspace.directory(for: flowID).appendingPathComponent(".blobs"),
                                      row: "Crop")
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: out, sourceText: nil)])
    }
}

/// `Convert` (image → image): re-encode in `format=` (jpeg/png).
nonisolated struct ConvertTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.image) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let path = try ImageTools.requireImagePath(input, row: "Convert")
        let s = FlowSettings(settings)
        let fmt = s.value(for: "format") ?? s.firstBare()
        guard let fmt else {
            throw FlowError.missingInlineValue(row: "Convert", kind: .file)
        }
        let cg = try ImageTools.load(path)
        let out = try ImageTools.save(cg, format: fmt,
                                      blobDirectory: workspace.directory(for: flowID).appendingPathComponent(".blobs"),
                                      row: "Convert")
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: out, sourceText: nil)])
    }
}

/// `Watermark` (image → image): composite the watermark image at `position=` (default
/// bottom-right, 10px margin).
nonisolated struct WatermarkTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.image) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let basePath = try ImageTools.requireImagePath(input, row: "Watermark")
        let s = FlowSettings(settings)
        let markSetting = s.value(for: "asset") ?? s.firstBare()
        guard let markSetting else {
            throw FlowError.missingInlineValue(row: "Watermark", kind: .file)
        }
        let markPath = try workspace.resolve(markSetting, flowID: flowID)
        let position = s.value(for: "position") ?? "bottom-right"

        let base = try ImageTools.load(basePath)
        let mark = try ImageTools.load(markPath)
        let size = CGSize(width: base.width, height: base.height)
        let markSize = CGSize(width: mark.width, height: mark.height)
        let xy = Self.positionXY(position, baseSize: size, markSize: markSize, margin: 10)

        guard let ctx = CGContext(data: nil, width: base.width, height: base.height,
                                  bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw FlowError.stageFailure(row: "Watermark", message: "couldn't create the canvas")
        }
        ctx.draw(base, in: CGRect(origin: .zero, size: size))
        ctx.draw(mark, in: CGRect(origin: xy, size: markSize))
        guard let composed = ctx.makeImage() else {
            throw FlowError.stageFailure(row: "Watermark", message: "couldn't composite")
        }
        let out = try ImageTools.save(composed, format: nil,
                                      blobDirectory: workspace.directory(for: flowID).appendingPathComponent(".blobs"),
                                      row: "Watermark")
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: out, sourceText: nil)])
    }

    static func positionXY(_ position: String, baseSize: CGSize, markSize: CGSize,
                           margin: CGFloat = 10) -> CGPoint {
        let x: CGFloat, y: CGFloat
        switch position {
        case "top-left": x = margin; y = margin
        case "top-right": x = baseSize.width - markSize.width - margin; y = margin
        case "bottom-left": x = margin; y = baseSize.height - markSize.height - margin
        case "center": x = (baseSize.width - markSize.width) / 2; y = (baseSize.height - markSize.height) / 2
        default: x = baseSize.width - markSize.width - margin; y = baseSize.height - markSize.height - margin
        }
        return CGPoint(x: max(0, x), y: max(0, y))
    }
}

/// `Overlay Text` (image → image): draw `text=`/bare at `position=`, default font.
nonisolated struct OverlayTextTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.image) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let basePath = try ImageTools.requireImagePath(input, row: "Overlay Text")
        let s = FlowSettings(settings)
        let text = s.value(for: "text") ?? s.firstBare()
        guard let text else {
            throw FlowError.missingInlineValue(row: "Overlay Text", kind: .text)
        }
        let position = s.value(for: "position") ?? "bottom-right"
        let base = try ImageTools.load(basePath)
        let size = CGSize(width: base.width, height: base.height)

        guard let ctx = CGContext(data: nil, width: base.width, height: base.height,
                                  bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw FlowError.stageFailure(row: "Overlay Text", message: "couldn't create the canvas")
        }
        ctx.draw(base, in: CGRect(origin: .zero, size: size))
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 18),
            .foregroundColor: NSColor.white,
        ]
        let bounds = (text as NSString).size(withAttributes: attrs)
        let xy = WatermarkTool.positionXY(position, baseSize: size,
                                          markSize: CGSize(width: bounds.width, height: bounds.height))
        (text as NSString).draw(at: xy, withAttributes: attrs)
        guard let outImage = ctx.makeImage() else {
            throw FlowError.stageFailure(row: "Overlay Text", message: "couldn't render")
        }
        let out = try ImageTools.save(outImage, format: nil,
                                      blobDirectory: workspace.directory(for: flowID).appendingPathComponent(".blobs"),
                                      row: "Overlay Text")
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: out, sourceText: nil)])
    }
}

/// `Contact Sheet` (images + text → image): tile N images into one, each labelled with the
/// matching text item. `cols=` (4), `size=` (256). A tuple tool — the executor hands it the
/// full `inputs` array.
nonisolated struct ContactSheetTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let allItems = inputs.flatMap { $0.items }
        let images = allItems.filter { $0.kind == .image && $0.path != nil }
        let labels = allItems.filter { $0.kind == .text }.map { $0.value ?? "" }
        guard !images.isEmpty else {
            throw FlowError.badInputCardinality(row: "Contact Sheet", expected: "image inputs", got: 0)
        }
        let s = FlowSettings(settings)
        let cols = max(1, Int(s.value(for: "cols") ?? "4") ?? 4)
        let size = max(1, Int(s.value(for: "size") ?? "256") ?? 256)

        let thumbs = try images.map { item -> CGImage in
            try ImageTools.thumbnail(try ImageTools.load(item.path!), max: size)
        }
        let font = NSFont.systemFont(ofSize: 12)
        let labelHeight: CGFloat = labels.isEmpty ? 0 : 18
        let rows = max(1, (thumbs.count + cols - 1) / cols)
        let cellW = CGFloat(thumbs.map(\.width).max() ?? size)
        let cellH = CGFloat(thumbs.map(\.height).max() ?? size) + labelHeight
        let sheetW = Int(CGFloat(cols) * cellW), sheetH = Int(CGFloat(rows) * cellH)

        guard let ctx = CGContext(data: nil, width: sheetW, height: sheetH,
                                  bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw FlowError.stageFailure(row: "Contact Sheet", message: "couldn't create the canvas")
        }
        ctx.setFillColor(CGColor(gray: 20.0/255.0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: sheetW, height: sheetH))
        for (index, thumb) in thumbs.enumerated() {
            let col = index % cols, row = index / cols
            let x = CGFloat(col) * cellW + (cellW - CGFloat(thumb.width)) / 2
            let y = CGFloat(row) * cellH + (cellH - labelHeight - CGFloat(thumb.height)) / 2
            ctx.draw(thumb, in: CGRect(x: x, y: y, width: CGFloat(thumb.width), height: CGFloat(thumb.height)))
            if index < labels.count {
                let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
                let text = labels[index] as NSString
                let tw = text.size(withAttributes: attrs).width
                text.draw(at: CGPoint(x: CGFloat(col) * cellW + (cellW - tw) / 2,
                                      y: CGFloat(row) * cellH + cellH - labelHeight + 3),
                          withAttributes: attrs)
            }
        }
        guard let sheet = ctx.makeImage() else {
            throw FlowError.stageFailure(row: "Contact Sheet", message: "couldn't render")
        }
        let out = try ImageTools.save(sheet, format: nil,
                                      blobDirectory: workspace.directory(for: flowID).appendingPathComponent(".blobs"),
                                      row: "Contact Sheet")
        return Asset(items: [Item(kind: .image, value: nil, path: out, sourceText: nil)])
    }
}

private nonisolated extension ImageTools {
    /// The single image input's file path (the image tools' input contract).
    static func requireImagePath(_ input: Asset, row: String) throws -> URL {
        guard let item = input.items.first, let path = item.path else {
            throw FlowError.missingInlineValue(row: row, kind: .image)
        }
        return path
    }
}
