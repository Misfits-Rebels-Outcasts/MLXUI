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

    // MARK: - Stop then Rank: one loop at a time (MC4 review)

    /// Stop followed straight away by Rank must never leave two embedding loops alive. The first
    /// loop is held inside its first image; Stop retires it; a second Rank starts but must wait for
    /// the first loop to leave the embedder (never two calls at once); when the first call is released
    /// its result is discarded (not written to `vectors`: the new loop has to embed that image again),
    /// the stale loop's exit does not clear the new run's `isEmbedding`, and only the new loop's
    /// results land.
    @Test func stopThenRankRunsOnlyOneLoop() async throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let small = dir.appendingPathComponent("small.png")     // width 10 -> [0, 1, 0]
        let medium = dir.appendingPathComponent("medium.png")   // width 20 -> [0.6, 0.8, 0]
        let large = dir.appendingPathComponent("large.png")     // width 30 -> [1, 0, 0]
        try writeImage(width: 10, height: 10, to: small)
        try writeImage(width: 20, height: 20, to: medium)
        try writeImage(width: 30, height: 30, to: large)

        // Calls 1 (the first loop's first image) and 2 (the second loop's first image) are held.
        let gate = EmbedGate(hold: [1, 2])
        let model = makeModel(embedImage: { image in
            _ = await gate.begin()
            let vector: [Float]
            switch image.width {
            case 30: vector = [1, 0, 0]
            case 20: vector = [0.6, 0.8, 0]
            default: vector = [0, 1, 0]
            }
            await gate.end()
            return vector
        }, embedText: { _ in
            _ = await gate.begin()
            await gate.end()
            return [1, 0, 0]
        })
        await model.addImages(at: [small, medium, large])

        // 1. First loop starts and is blocked inside its first image.
        let first = Task { await model.rank(query: "a teal square") }
        #expect(await waitUntil { await gate.calls == 1 })
        #expect(model.isEmbedding)

        // 2. Stop retires it; the flag drops at once and nothing is ranked.
        model.stop()
        #expect(!model.isEmbedding)
        #expect(model.ranked.isEmpty)

        // 3. A second Rank starts. It owns `isEmbedding`, but it must not enter the embedder while the
        //    first loop is still inside it.
        let second = Task { await model.rank(query: "a teal square") }
        #expect(await waitUntil { model.isEmbedding })
        try await Task.sleep(for: .milliseconds(100))
        #expect(await gate.calls == 1, "the second loop entered the embedder while the first was still in it")
        #expect(await gate.running == 1)

        // 4. Release the first call. Its loop is stale: it exits without writing. The second loop then
        //    enters the embedder (call 2, held) -- and `isEmbedding` is still the second run's.
        await gate.release(1)
        await first.value
        #expect(await waitUntil { await gate.calls == 2 })
        #expect(model.isEmbedding, "the stale loop's exit cleared the new run's isEmbedding")
        #expect(model.ranked.isEmpty, "the stale loop wrote a ranking")

        // 5. Let the second loop finish.
        await gate.release(2)
        await second.value

        #expect(!model.isEmbedding)
        #expect(model.errorText == nil)
        #expect(model.ranked.map(\.entry.url.lastPathComponent) == ["large.png", "medium.png", "small.png"])
        #expect(await gate.maxRunning == 1, "two embedder calls overlapped")
        // 4 image calls (the discarded stale one + 3 for the new loop: the stale result was not cached)
        // and 1 text call.
        #expect(await gate.calls == 5)
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

/// A stub-embedder gate: `begin()` counts a call and suspends the calls named in `hold` until they are
/// released; it also records how many calls are inside the embedder at once.
private actor EmbedGate {
    private let hold: Set<Int>
    private var released: Set<Int> = []
    private var blocked: [Int: CheckedContinuation<Void, Never>] = [:]
    private(set) var calls = 0
    private(set) var running = 0
    private(set) var maxRunning = 0

    init(hold: Set<Int>) {
        self.hold = hold
    }

    func begin() async -> Int {
        calls += 1
        let number = calls
        running += 1
        maxRunning = max(maxRunning, running)
        if hold.contains(number), !released.contains(number) {
            await withCheckedContinuation { blocked[number] = $0 }
        }
        return number
    }

    func end() {
        running -= 1
    }

    func release(_ number: Int) {
        released.insert(number)
        blocked.removeValue(forKey: number)?.resume()
    }
}

/// Poll `condition` (up to 5 s) until it holds.
@MainActor
private func waitUntil(_ condition: @MainActor () async -> Bool) async -> Bool {
    for _ in 0..<500 {
        if await condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return false
}

// MARK: - One loaded session per model directory (MC4 smoke fix)

/// The Text tab, the `Embed` row and the Images tab all reach the checkpoint through
/// `Qwen3VLEmbeddingSessions.shared`, a `DirectoryLoadCache`. With a stub load function (no weights)
/// this proves the sharing: any mix of text and image embeds for one directory loads once, another
/// directory loads separately, concurrent first calls share one load, and eviction forces a reload.
struct Qwen3VLEmbeddingSessionCacheTests {

    /// A stand-in session: counts nothing itself, it only has to be the same object each time.
    private final class StubSession: Sendable {
        let directory: URL
        init(directory: URL) { self.directory = directory }
        func embedText(_ text: String) -> [Float] { [1, 0] }
        func embedImage() -> [Float] { [0, 1] }
    }

    private actor LoadCounter {
        private(set) var loads: [URL] = []
        func record(_ url: URL) { loads.append(url) }
    }

    @Test func twoTextEmbedsAndAnImageEmbedLoadTheDirectoryOnce() async throws {
        let counter = LoadCounter()
        let cache = DirectoryLoadCache<StubSession> { directory in
            await counter.record(directory)
            try await Task.sleep(for: .milliseconds(20))
            return StubSession(directory: directory)
        }
        let dir = URL(fileURLWithPath: "/tmp/qwen3vl-a")
        let first = try await cache.value(for: dir)
        _ = first.embedText("one")
        let second = try await cache.value(for: dir)
        _ = second.embedText("two")
        let third = try await cache.value(for: dir)
        _ = third.embedImage()
        #expect(await counter.loads.count == 1, "the checkpoint was loaded more than once")
        #expect(first === second && second === third)

        // A different directory loads separately; the first stays cached.
        let other = try await cache.value(for: URL(fileURLWithPath: "/tmp/qwen3vl-b"))
        #expect(other !== first)
        #expect(await counter.loads.count == 2)
        _ = try await cache.value(for: dir)
        #expect(await counter.loads.count == 2)
    }

    @Test func concurrentFirstCallsShareOneLoad() async throws {
        let counter = LoadCounter()
        let cache = DirectoryLoadCache<StubSession> { directory in
            await counter.record(directory)
            try await Task.sleep(for: .milliseconds(50))
            return StubSession(directory: directory)
        }
        let dir = URL(fileURLWithPath: "/tmp/qwen3vl-c")
        async let a = cache.value(for: dir)
        async let b = cache.value(for: dir)
        async let c = cache.value(for: dir.appendingPathComponent("../qwen3vl-c"))   // same directory, unstandardized
        let (x, y, z) = try await (a, b, c)
        #expect(x === y && y === z)
        #expect(await counter.loads.count == 1)
    }

    @Test func evictionForcesAReloadAndAFailedLoadIsNotCached() async throws {
        let counter = LoadCounter()
        let cache = DirectoryLoadCache<StubSession> { directory in
            await counter.record(directory)
            if await counter.loads.count == 1 { throw CocoaError(.fileReadUnknown) }
            return StubSession(directory: directory)
        }
        let dir = URL(fileURLWithPath: "/tmp/qwen3vl-d")
        await #expect(throws: (any Error).self) { _ = try await cache.value(for: dir) }
        let loaded = try await cache.value(for: dir)          // retried, not cached as a failure
        #expect(await counter.loads.count == 2)
        await cache.evict(dir)
        let reloaded = try await cache.value(for: dir)
        #expect(reloaded !== loaded)
        #expect(await counter.loads.count == 3)
    }
}
