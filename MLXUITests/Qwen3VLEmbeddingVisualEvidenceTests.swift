import Foundation
import Testing
import AppKit
import SwiftUI
@testable import MLXUI

/// MC4-6 — the visual evidence for the PR. **Not part of the normal suite**: gated like the
/// real-weights tests, it renders the actual SwiftUI views to PNGs under `/private/tmp` (never
/// committed) — the "Before" Embeddings Run screen as it ships for every text embedder, and the new
/// Images tab with reallly ranked images for two queries.
@MainActor
struct Qwen3VLEmbeddingVisualEvidenceTests {

    /// The sandboxed test host cannot write `/private/tmp`; its temporary directory is inside the
    /// app container, which the evidence script reads afterwards
    /// (`~/Library/Containers/net.connectcode.mlxui/Data/tmp/qwen3vl-evidence`).
    private static var outputDirectory: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("qwen3vl-evidence")
    }

    /// Render a SwiftUI view off-screen to `path` (the test host is the app, so AppKit is live).
    private func render<V: View>(_ view: V, size: CGSize, to path: String) throws {
        let hosting = NSHostingView(rootView: view)
        hosting.frame = CGRect(origin: .zero, size: size)
        let window = NSWindow(
            contentRect: hosting.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = hosting
        window.setIsVisible(false)
        hosting.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        hosting.layoutSubtreeIfNeeded()
        guard let rep = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) else {
            throw CocoaError(.fileWriteUnknown)
        }
        hosting.cacheDisplay(in: hosting.bounds, to: rep)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        try data.write(to: URL(fileURLWithPath: path))
        print("QWEN3VLEMBEDSHOT \(path) \(Int(size.width))x\(Int(size.height))")
    }

    /// "Before": the Embeddings Run screen exactly as it ships for every text embedder (no tab).
    @Test(.enabled(if: Qwen3VLEmbedRealLoad.canRun))
    func renderBeforeScreenshot() throws {
        try FileManager.default.createDirectory(
            at: Self.outputDirectory, withIntermediateDirectories: true)
        try render(
            EmbeddingRunView(
                modelDisplayName: "BGE-M3", license: "MIT",
                modelID: "mlx-community--bge-m3-mlx-fp16", family: "bge"),
            size: CGSize(width: 620, height: 620),
            to: Self.outputDirectory.appendingPathComponent("before-text-embeddings-run-screen.png").path)
    }

    /// "After": the new Images tab, with 20 generated images ranked by the real model for two
    /// queries. The two PNGs are attached to the PR conversation (not committed).
    @Test(.enabled(if: Qwen3VLEmbedRealLoad.canRun))
    func renderAfterImagesTabScreenshots() async throws {
        let directory = try #require(Qwen3VLEmbedRealLoad.directory)
        try FileManager.default.createDirectory(
            at: Self.outputDirectory, withIntermediateDirectories: true)

        let photos = try Self.makePhotoSet()
        defer { for url in photos { try? FileManager.default.removeItem(at: url) } }

        let session = try await Qwen3VLEmbeddingSession.load(from: directory)
        let model = EmbeddingImagesRunModel(
            embedImage: { image in try await session.embedImage(image) },
            embedText: { text in try await session.embedText(text) })
        await model.addImages(at: photos)
        print("QWEN3VLEMBEDSHOT images=\(model.entries.count)")

        await model.rank(query: "a red square")
        print("QWEN3VLEMBEDSHOT query=red-square top=\(model.ranked.prefix(5).map { $0.entry.url.lastPathComponent + String(format: " %.3f", $0.score) })")
        try render(
            EmbeddingImagesView(model: model).frame(width: 620, height: 700),
            size: CGSize(width: 620, height: 700),
            to: Self.outputDirectory.appendingPathComponent("after-images-tab-query-red-square.png").path)

        await model.rank(query: "a blue circle")
        print("QWEN3VLEMBEDSHOT query=blue-circle top=\(model.ranked.prefix(5).map { $0.entry.url.lastPathComponent + String(format: " %.3f", $0.score) })")
        try render(
            EmbeddingImagesView(model: model).frame(width: 620, height: 700),
            size: CGSize(width: 620, height: 700),
            to: Self.outputDirectory.appendingPathComponent("after-images-tab-query-blue-circle.png").path)
    }

    /// Twenty generated "photos": red/blue/green shapes on a light background, each with a label.
    private static func makePhotoSet() throws -> [URL] {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3vl-photos-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let shapes = ["red square", "blue circle", "green triangle", "yellow star"]
        let colors: [String: NSColor] = [
            "red": .systemRed, "blue": .systemBlue, "green": .systemGreen, "yellow": .systemYellow,
        ]
        var urls: [URL] = []
        for index in 0..<20 {
            let label = shapes[index % shapes.count]
            let parts = label.split(separator: " ")
            let color = colors[String(parts[0])] ?? .systemGray
            let shape = String(parts[1])
            let url = directory.appendingPathComponent(
                String(format: "photo-%02d-%@.png", index, label.replacingOccurrences(of: " ", with: "-")))
            guard let rep = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: 320, pixelsHigh: 240,
                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else {
                throw CocoaError(.fileWriteUnknown)
            }
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
            NSColor(white: 0.96, alpha: 1).setFill()
            NSRect(x: 0, y: 0, width: 320, height: 240).fill()
            color.setFill()
            switch shape {
            case "square":
                NSRect(x: 80, y: 60, width: 160, height: 120).fill()
            case "circle":
                NSBezierPath(ovalIn: NSRect(x: 90, y: 55, width: 140, height: 130)).fill()
            case "triangle":
                let path = NSBezierPath()
                path.move(to: NSPoint(x: 160, y: 190))
                path.line(to: NSPoint(x: 85, y: 60))
                path.line(to: NSPoint(x: 235, y: 60))
                path.close()
                path.fill()
            default:
                let path = NSBezierPath()
                let center = NSPoint(x: 160, y: 125)
                for point in 0..<10 {
                    let radius = point % 2 == 0 ? 70.0 : 28.0
                    let angle = Double(point) * .pi / 5 - .pi / 2
                    let next = NSPoint(
                        x: center.x + CGFloat(cos(angle) * radius),
                        y: center.y + CGFloat(sin(angle) * radius))
                    if point == 0 { path.move(to: next) } else { path.line(to: next) }
                }
                path.close()
                path.fill()
            }
            NSGraphicsContext.restoreGraphicsState()
            guard let data = rep.representation(using: .png, properties: [:]) else {
                throw CocoaError(.fileWriteUnknown)
            }
            try data.write(to: url)
            urls.append(url)
        }
        return urls
    }
}
