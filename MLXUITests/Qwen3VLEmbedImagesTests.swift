import Foundation
import Testing
import AppKit
@testable import MLXUI

/// MC4-6 — the Images tab's run model, exercised with stub embedders (no weights, no MLX):
/// ranking by cosine similarity, the 200-image cap, an unreadable image skipped with a note,
/// Stop between images, one-plain-sentence errors, and that nothing is written to disk.
@MainActor
struct Qwen3VLEmbedImagesTests {

    // MARK: - helpers

    private func makeTempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3vl-images-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A solid-colour PNG of `size`; the stub embedder keys off the width.
    private func writeImage(width: Int, height: Int, to url: URL) throws {
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else {
            throw CocoaError(.fileWriteUnknown)
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.systemTeal.setFill()
        NSRect(x: 0, y: 0, width: width, height: height).fill()
        NSGraphicsContext.restoreGraphicsState()
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        try data.write(to: url)
    }

    private func writeImages(count: Int, in directory: URL) throws -> [URL] {
        var urls: [URL] = []
        for index in 0..<count {
            let url = directory.appendingPathComponent("image-\(index).png")
            try writeImage(width: 3 + index % 5, height: 3 + index % 5, to: url)
            urls.append(url)
        }
        return urls
    }

    private func makeModel(
        embedImage: @escaping @Sendable (CGImage) async throws -> [Float],
        embedText: @escaping @Sendable (String) async throws -> [Float]
    ) -> EmbeddingImagesRunModel {
        EmbeddingImagesRunModel(embedImage: embedImage, embedText: embedText)
    }

    // MARK: - ranking

    @Test func ranksImagesByCosineSimilarity() async throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let small = dir.appendingPathComponent("small.png")     // width 10 → [0, 1, 0]
        let medium = dir.appendingPathComponent("medium.png")   // width 20 → [0.6, 0.8, 0]
        let large = dir.appendingPathComponent("large.png")     // width 30 → [1, 0, 0]
        try writeImage(width: 10, height: 10, to: small)
        try writeImage(width: 20, height: 20, to: medium)
        try writeImage(width: 30, height: 30, to: large)

        let model = makeModel(embedImage: { image in
            switch image.width {
            case 30: return [1, 0, 0]
            case 20: return [0.6, 0.8, 0]
            default: return [0, 1, 0]
            }
        }, embedText: { _ in [1, 0, 0] })
        await model.addImages(at: [small, medium, large])
        await model.rank(query: "a teal square")

        #expect(model.entries.count == 3)
        #expect(model.ranked.count == 3)
        #expect(model.ranked.map(\.entry.url.lastPathComponent) == ["large.png", "medium.png", "small.png"])
        #expect(abs(model.ranked[0].score - 1.0) < 1e-5)
        #expect(abs(model.ranked[1].score - 0.6) < 1e-5)
        #expect(abs(model.ranked[2].score - 0.0) < 1e-5)
        #expect(model.dimensions == 3)
        #expect(model.errorText == nil)
    }

    @Test func folderDropAddsTheImagesInsideIt() async throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try writeImages(count: 3, in: dir)
        let model = makeModel(embedImage: { _ in [1, 0] }, embedText: { _ in [1, 0] })
        await model.addImages(at: [dir])
        #expect(model.entries.count == 3)
    }

    // MARK: - the cap

    @Test func capsTheSessionAtTwoHundredImages() async throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try writeImages(count: 201, in: dir)
        let model = makeModel(embedImage: { _ in [1] }, embedText: { _ in [1] })
        await model.addImages(at: [dir])
        #expect(model.entries.count == EmbeddingImagesRunModel.maxImages)
        #expect(model.entries.count == 200)
        #expect(model.note?.contains("Only the first 200 images") == true)
        // Adding more later is refused by the same cap, not by reloading.
        await model.addImages(at: [dir.appendingPathComponent("image-0.png"), dir.appendingPathComponent("image-1.png")])
        #expect(model.entries.count == 200)
    }

    // MARK: - unreadable images

    @Test func anUnreadableImageIsSkippedWithANote() async throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let good = dir.appendingPathComponent("good.png")
        try writeImage(width: 8, height: 8, to: good)
        let broken = dir.appendingPathComponent("broken.png")
        try Data("this is not a PNG".utf8).write(to: broken)

        let model = makeModel(embedImage: { _ in [1, 0] }, embedText: { _ in [1, 0] })
        await model.addImages(at: [good, broken])
        #expect(model.entries.count == 1)
        #expect(model.entries.first?.url.lastPathComponent == "good.png")
        let note = try #require(model.note)
        #expect(note.contains("1 image could not be read and was skipped"))
    }

    // MARK: - stop

    /// Stop is checked between images: a 20-image run stopped after a few images does not finish
    /// the set or produce a ranking.
    @Test func stopEndsTheRunBetweenImages() async throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let urls = try writeImages(count: 20, in: dir)
        let counter = EmbeddingCallCounter()
        let model = makeModel(embedImage: { _ in
            _ = await counter.next()
            try await Task.sleep(for: .milliseconds(10))
            return [1, 0]
        }, embedText: { _ in [1, 0] })

        await model.addImages(at: urls)
        let run = Task { await model.rank(query: "anything") }
        try await Task.sleep(for: .milliseconds(70))
        #expect(model.isEmbedding)
        model.stop()
        await run.value
        let calls = await counter.value
        #expect(calls < 20, "Stop did not interrupt the image loop (embedded \(calls) images)")
        #expect(model.ranked.isEmpty, "a stopped run produces no ranking")
        #expect(!model.isEmbedding)
    }

    // MARK: - errors

    /// The Run screen shows one plain sentence, never a raw error dump.
    @Test func errorsAreOnePlainSentence() async throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let urls = try writeImages(count: 1, in: dir)
        let model = makeModel(embedImage: { _ in [1, 0] }, embedText: { _ in
            throw Qwen3VLEmbedError.inputTooLong(tokens: 9_000)
        })
        await model.addImages(at: urls)
        await model.rank(query: "a query")
        let message = try #require(model.errorText)
        #expect(message == "This model accepts up to 8,192 tokens after rendering; this input rendered to 9000. Shorten it and try again.")
        #expect(!message.contains("Error Domain"))
        #expect(!message.contains("UserInfo"))
        #expect(model.ranked.isEmpty)
    }

    // MARK: - nothing on disk

    @Test func nothingIsWrittenToDisk() async throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let urls = try writeImages(count: 3, in: dir)
        let before = try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()

        let model = makeModel(embedImage: { _ in [0.5, 0.5] }, embedText: { _ in [0.5, 0.5] })
        await model.addImages(at: urls)
        await model.rank(query: "a query")

        let after = try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()
        #expect(after == before, "the Images run wrote to the picked folder: \(after)")
        // Clearing is pure memory too.
        model.clear()
        #expect(try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted() == before)
    }
}

/// Counts stub embedder calls across the run's task.
private actor EmbeddingCallCounter {
    private(set) var value = 0

    @discardableResult
    func next() -> Int {
        value += 1
        return value
    }
}
