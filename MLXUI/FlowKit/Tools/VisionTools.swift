import Foundation
import CoreGraphics
import Vision
import AppKit

// CFM-R12-7 group f — the vision tools. `Detect Edges` is arithmetic (Sobel gradient +
// Canny-style hysteresis, no model); `Detect Pose` uses Apple's **Vision** framework (no
// model weights to ship — the dependency check the reviewer asked for came back clean).

// MARK: - Detect Edges

/// `Detect Edges` (image → image): Sobel gradient magnitude with `low=`/`high=` hysteresis
/// (`method=canny`, default) or a plain `method=sobel` pass. A grayscale image where edges
/// are bright.
nonisolated struct DetectEdgesTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.image) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let path = input.items.first?.path else {
            throw FlowError.missingInlineValue(row: "Detect Edges", kind: .image)
        }
        let s = FlowSettings(settings)
        let method = s.value(for: "method") ?? "canny"
        let low = Double(Int(s.value(for: "low") ?? "80") ?? 80)
        let high = Double(Int(s.value(for: "high") ?? "200") ?? 200)
        guard method == "canny" || method == "sobel" else {
            throw FlowError.invalidSettings(row: "Detect Edges", setting: "method",
                                            detail: "use `canny` or `sobel`")
        }
        let cg = try ImageTools.load(path)
        let gray = try Self.grayPixels(cg)
        let gx = Self.sobel(gray, axis: 1)
        let gy = Self.sobel(gray, axis: 0)
        let magnitude: [[Float]] = zip(gx, gy).map { rowX, rowY in
            zip(rowX, rowY).map { hypotf($0, $1) }
        }

        let edges: [[Float]]
        if method == "sobel" {
            edges = magnitude.map { $0.map { min(max($0, 0), 1) } }
        } else {
            let strong = magnitude.map { $0.map { $0 >= Float(high / 255.0) } }
            let weak = magnitude.map { $0.map { $0 >= Float(low / 255.0) } }
            let kept = Self.hysteresis(strong: strong, weak: weak)
            edges = kept.map { $0.map { $0 ? 1 : 0 } }
        }

        let height = gray.count, width = gray[0].count
        let out = try Self.renderGray(edges, width: width, height: height)
        let blobDir = workspace.directory(for: flowID).appendingPathComponent(".blobs")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        let url = blobDir.appendingPathComponent("edges-\(UUID().uuidString).png")
        guard let data = PNGEncoder.pngData(from: out) else {
            throw FlowError.writeFailed(row: "Detect Edges", path: url.lastPathComponent)
        }
        try data.write(to: url)
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: url, sourceText: nil)])
    }

    /// Grayscale 0…1 pixels (H×W).
    static func grayPixels(_ cg: CGImage) throws -> [[Float]] {
        let width = cg.width, height = cg.height
        var data = [UInt8](repeating: 0, count: width * height)
        let ctx = CGContext(data: &data, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                            bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return (0..<height).map { y in
            (0..<width).map { x in Float(data[y * width + x]) / 255.0 }
        }
    }

    /// Sobel gradient along `axis` (0 = y-gradient, 1 = x-gradient), zero-padded.
    static func sobel(_ array: [[Float]], axis: Int) -> [[Float]] {
        let h = array.count, w = array[0].count
        let kernel: [[Float]] = axis == 1
            ? [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            : [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        var out = [[Float]](repeating: [Float](repeating: 0, count: w), count: h)
        for y in 0..<h {
            for x in 0..<w {
                var sum: Float = 0
                for dy in 0..<3 {
                    for dx in 0..<3 {
                        let ny = y - 1 + dy, nx = x - 1 + dx
                        guard ny >= 0, ny < h, nx >= 0, nx < w else { continue }
                        sum += kernel[dy][dx] * array[ny][nx]
                    }
                }
                out[y][x] = sum
            }
        }
        return out
    }

    /// Flood-fill over 8-neighbours: a weak edge survives only if it connects to a strong one.
    static func hysteresis(strong: [[Bool]], weak: [[Bool]]) -> [[Bool]] {
        let h = strong.count, w = strong[0].count
        var kept = [[Bool]](repeating: [Bool](repeating: false, count: w), count: h)
        var stack: [(Int, Int)] = []
        for y in 0..<h { for x in 0..<w where strong[y][x] { stack.append((y, x)) } }
        while let (y, x) = stack.popLast() {
            guard !kept[y][x] else { continue }
            kept[y][x] = true
            for dy in -1...1 {
                for dx in -1...1 {
                    let ny = y + dy, nx = x + dx
                    guard ny >= 0, ny < h, nx >= 0, nx < w,
                          !kept[ny][nx], weak[ny][nx] else { continue }
                    stack.append((ny, nx))
                }
            }
        }
        return kept
    }

    /// Render a 0…1 grayscale array as a CGImage.
    static func renderGray(_ values: [[Float]], width: Int, height: Int) throws -> CGImage {
        var data = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                data[y * width + x] = UInt8(min(max(values[y][x] * 255, 0), 255))
            }
        }
        guard let ctx = CGContext(data: &data, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let out = ctx.makeImage() else {
            throw FlowError.stageFailure(row: "Detect Edges", message: "couldn't render")
        }
        return out
    }
}

// MARK: - Detect Pose

/// `Detect Pose` (image → image): Apple Vision body-pose → the OpenPose COCO-18 stick
/// figure (the exact limb/colour convention ControlNet pose models expect).
nonisolated struct DetectPoseTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.image) }
    var produces: Shape { .single(.image) }

    /// Vision joint names → COCO-18 index (the `pose.py` mapping).
    static let visionToCOCO: [VNHumanBodyPoseObservation.JointName: Int] = [
        .nose: 0, .neck: 1, .rightShoulder: 2, .rightElbow: 3, .rightWrist: 4,
        .leftShoulder: 5, .leftElbow: 6, .leftWrist: 7, .rightHip: 8, .rightKnee: 9,
        .rightAnkle: 10, .leftHip: 11, .leftKnee: 12, .leftAnkle: 13,
        .rightEye: 14, .leftEye: 15, .rightEar: 16, .leftEar: 17,
    ]

    static let limbs: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (0, 14), (14, 16), (0, 15), (15, 17),
    ]

    /// OpenPose's colour gradient down the body (head red through legs blue).
    static let limbColours: [(r: Int, g: Int, b: Int)] = [
        (255, 0, 0), (255, 60, 0), (255, 120, 0), (255, 180, 0), (230, 200, 0),
        (180, 220, 0), (120, 240, 0), (0, 255, 40), (0, 220, 120), (0, 180, 180),
        (0, 140, 220), (0, 100, 240), (0, 60, 255), (255, 0, 120), (255, 0, 60),
        (255, 60, 180), (255, 60, 120),
    ]

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let path = input.items.first?.path else {
            throw FlowError.missingInlineValue(row: "Detect Pose", kind: .image)
        }
        let cg = try ImageTools.load(path)
        let width = cg.width, height = cg.height

        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cgImage: cg)
        try handler.perform([request])

        guard let observation = request.results?.first else {
            throw FlowError.stageFailure(row: "Detect Pose", message: "found no person in the image")
        }
        // COCO-18 ordered pixel keypoints (nil when Vision didn't see the joint).
        var keypoints: [(x: Double, y: Double)?] = Array(repeating: nil, count: 18)
        if let points = try? observation.recognizedPoints(.all) {
            for (name, point) in points {
                guard let coco = Self.visionToCOCO[name] else { continue }
                keypoints[coco] = (point.location.x * Double(width),
                                   (1 - point.location.y) * Double(height))
            }
        }

        let rendered = Self.renderPose(keypoints, width: width, height: height)
        let blobDir = workspace.directory(for: flowID).appendingPathComponent(".blobs")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        let url = blobDir.appendingPathComponent("pose-\(UUID().uuidString).png")
        guard let data = PNGEncoder.pngData(from: rendered) else {
            throw FlowError.writeFailed(row: "Detect Pose", path: url.lastPathComponent)
        }
        try data.write(to: url)
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: url, sourceText: nil)])
    }

    /// Render one person's COCO-18 keypoints as OpenPose's stick figure on black.
    static func renderPose(_ keypoints: [(x: Double, y: Double)?], width: Int, height: Int) -> CGImage {
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        for (limb, colour) in zip(Self.limbs, Self.limbColours) {
            guard let a = Self.visible(keypoints, limb.0), let b = Self.visible(keypoints, limb.1) else { continue }
            ctx.setStrokeColor(CGColor(srgbRed: CGFloat(colour.r) / 255, green: CGFloat(colour.g) / 255,
                                       blue: CGFloat(colour.b) / 255, alpha: 1))
            ctx.setLineWidth(2)
            ctx.strokeLineSegments(between: [CGPoint(x: a.0, y: a.1), CGPoint(x: b.0, y: b.1)])
        }
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        for point in keypoints {
            guard let point else { continue }
            let p = CGPoint(x: point.x, y: point.y)
            ctx.fillEllipse(in: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6))
        }
        return ctx.makeImage()!
    }

    private static func visible(_ keypoints: [(x: Double, y: Double)?], _ index: Int) -> (CGFloat, CGFloat)? {
        guard index < keypoints.count, let p = keypoints[index] else { return nil }
        return (CGFloat(p.x), CGFloat(p.y))
    }
}
