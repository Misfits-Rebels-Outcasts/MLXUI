import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// MC4-6 — the Images tab of the Qwen3-VL Embedding 2B Run screen (decision 6).
///
/// Pick up to 200 images (Open… or drop a folder), type a query, and the images are ranked by
/// cosine similarity and shown as thumbnails with their scores. Images are decoded on demand and
/// only a 128 px thumbnail is kept per image; every vector lives in memory for the session, no
/// index is built and nothing is written to disk.
struct EmbeddingImagesView: View {
    @Bindable var model: EmbeddingImagesRunModel

    @State private var query = ""
    @State private var showImporter = false
    @State private var isTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            controls
            if model.isEmbedding { progressRow }
            if let note = model.note {
                Text(note).font(.caption).foregroundStyle(.secondary)
            }
            if let error = model.errorText {
                RunErrorBar(message: error)
            }
            if model.ranked.isEmpty {
                emptyState
            } else {
                rankedList
            }
            queryRow
        }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.image, .folder],
            allowsMultipleSelection: true
        ) { result in
            guard case let .success(urls) = result, !urls.isEmpty else { return }
            Task { await model.addImages(at: urls) }
        }
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            loadDropped(providers)
        }
    }

    // MARK: - Sections

    private var controls: some View {
        HStack(spacing: 10) {
            Button("Open Images…") { showImporter = true }
            if !model.entries.isEmpty {
                Button("Clear") { model.clear() }
            }
            Spacer()
            Text(model.entries.isEmpty ? "No images" : "\(model.entries.count) image\(model.entries.count == 1 ? "" : "s")")
                .font(.callout).foregroundStyle(.secondary).monospacedDigit()
        }
    }

    private var emptyState: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(
                    isTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                    style: StrokeStyle(lineWidth: 1.5, dash: [6]))
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
            VStack(spacing: 6) {
                Image(systemName: "photo.on.rectangle.angled").font(.title2)
                Text("Drop up to 200 images or a folder here, or open them")
                    .font(.callout).foregroundStyle(.secondary)
                Text("Then type a query to rank them").font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 180)
    }

    private var progressRow: some View {
        HStack(spacing: 8) {
            ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
            Text("Embedding images…").font(.callout).foregroundStyle(.secondary)
            Spacer()
            Text(String(format: "%.0f%%", model.progress * 100))
                .font(.callout).foregroundStyle(.secondary).monospacedDigit()
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var rankedList: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(model.ranked) { item in
                    HStack(spacing: 8) {
                        Image(decorative: item.entry.thumbnail, scale: 1)
                            .resizable().scaledToFill()
                            .frame(width: 44, height: 44)
                            .clipShape(RoundedRectangle(cornerRadius: 5))
                        Text(item.entry.url.lastPathComponent)
                            .font(.callout).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Text(String(format: "%.3f", item.score))
                            .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxHeight: .infinity)
    }

    private var queryRow: some View {
        HStack(spacing: 8) {
            TextField("Search this image set…", text: $query)
                .textFieldStyle(.roundedBorder)
                .onSubmit(rank)
                .disabled(model.entries.isEmpty || model.isEmbedding)
            Button(model.isEmbedding ? "Stop" : "Rank") {
                if model.isEmbedding { model.stop() } else { rank() }
            }
            .keyboardShortcut(.defaultAction)
            .disabled(model.entries.isEmpty || (query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !model.isEmbedding))
        }
    }

    // MARK: - Actions

    private func rank() {
        let text = query
        Task { await model.rank(query: text) }
    }

    private func loadDropped(_ providers: [NSItemProvider]) -> Bool {
        let group = DispatchGroup()
        let collector = URLCollector()
        for provider in providers {
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url { collector.append(url) }
                group.leave()
            }
        }
        group.notify(queue: .main) {
            let urls = collector.urls
            guard !urls.isEmpty else { return }
            Task { await model.addImages(at: urls) }
        }
        return true
    }
}

/// Collects dropped file URLs across the item providers' callbacks (thread-safe).
private final class URLCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [URL] = []

    func append(_ url: URL) {
        lock.lock()
        defer { lock.unlock() }
        storage.append(url)
    }

    var urls: [URL] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

// MARK: - The run model

/// Drives one Images run: holds the picked images and their vectors in memory, ranks a typed
/// query by cosine similarity (the embedder L2-normalises, so it is a dot product), and never
/// writes to disk. The embedder is two injected closures so the ranking, the cap, the skip note
/// and Stop are unit-testable without a model or real weights.
@MainActor
@Observable
final class EmbeddingImagesRunModel {

    struct Entry: Identifiable {
        let id: URL
        let url: URL
        /// A small decoded preview (128 px); the full image is decoded again only while it is
        /// being embedded, so 200 images do not hold 200 full bitmaps in memory.
        let thumbnail: CGImage
    }

    struct Ranked: Identifiable {
        let entry: Entry
        let score: Double
        var id: URL { entry.id }
    }

    /// Decision 6: up to 200 images.
    static let maxImages = 200

    private(set) var entries: [Entry] = []
    private(set) var ranked: [Ranked] = []
    private(set) var isEmbedding = false
    private(set) var progress = 0.0
    private(set) var note: String?
    private(set) var dimensions: Int?
    var errorText: String?

    /// One L2-normalised vector per image/text, kept for the session. Keyed by image URL.
    private var vectors: [URL: [Float]] = [:]
    private var stopRequested = false

    private let embedImage: @Sendable (CGImage) async throws -> [Float]
    private let embedText: @Sendable (String) async throws -> [Float]

    init(
        embedImage: @escaping @Sendable (CGImage) async throws -> [Float],
        embedText: @escaping @Sendable (String) async throws -> [Float]
    ) {
        self.embedImage = embedImage
        self.embedText = embedText
    }

    /// The real embedder. One session is loaded once and shared by every image and query, so a
    /// 200-image run reads the 1.8 GB checkpoint once (see `Qwen3VLEmbeddingSessionLoader`).
    convenience init(modelID: String) {
        let loader = Qwen3VLEmbeddingSessionLoader(
            directory: EmbeddingEngine.installedModelDirectory(id: modelID))
        self.init(
            embedImage: { image in
                try await loader.load().embedImage(image)
            },
            embedText: { text in
                try await loader.load().embedText(text)
            })
    }

    /// Add images (files and/or folders of images): dedupe, cap at 200, note unreadable files.
    func addImages(at urls: [URL]) async {
        errorText = nil
        var notes: [String] = []

        // Folders from the importer are security-scoped; hold the scope while their children are
        // decoded (a plain URL answers `false` and needs no release).
        var scopedFolders: [URL] = []
        func openFolder(_ folder: URL) {
            if folder.startAccessingSecurityScopedResource() { scopedFolders.append(folder) }
        }
        defer {
            for folder in scopedFolders { folder.stopAccessingSecurityScopedResource() }
        }

        var candidates: [URL] = []
        var unreadable = 0
        for url in urls {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
                unreadable += 1
                continue
            }
            if isDirectory.boolValue {
                openFolder(url)
                candidates.append(contentsOf: Self.imageFiles(in: url))
            } else {
                candidates.append(url)
            }
        }

        var seen = Set(entries.map(\.url))
        var unique: [URL] = []
        for url in candidates where !seen.contains(url) {
            seen.insert(url)
            unique.append(url)
        }
        let room = max(0, Self.maxImages - entries.count)
        if unique.count > room {
            notes.append("Only the first \(Self.maxImages) images are used.")
            unique = Array(unique.prefix(room))
        }

        for url in unique {
            guard let full = ImageLoader.decodedCGImage(fromSecurityScoped: url),
                  let thumbnail = Self.thumbnail(of: full) else {
                unreadable += 1
                continue
            }
            entries.append(Entry(id: url, url: url, thumbnail: thumbnail))
        }
        if unreadable > 0 {
            notes.append("\(unreadable) image\(unreadable == 1 ? "" : "s") could not be read and \(unreadable == 1 ? "was" : "were") skipped.")
        }
        note = notes.isEmpty ? nil : notes.joined(separator: " ")
        ranked = []
        vectors = vectors.filter { key, _ in entries.contains { $0.url == key } }
    }

    /// Embed any not-yet-embedded images (progress between them), then the query, then rank.
    /// `stopRequested` is checked between images, so Stop takes effect without waiting for a whole
    /// 200-image run.
    func rank(query: String) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !entries.isEmpty else { return }
        errorText = nil
        stopRequested = false
        isEmbedding = true
        progress = 0
        defer { isEmbedding = false }

        do {
            var done = 0
            for entry in entries {
                if stopRequested || Task.isCancelled { return }
                if vectors[entry.url] == nil {
                    guard let image = ImageLoader.decodedCGImage(fromSecurityScoped: entry.url) else {
                        note = "An image could not be read and was skipped."
                        done += 1
                        continue
                    }
                    vectors[entry.url] = try await embedImage(image)
                }
                done += 1
                progress = Double(done) / Double(entries.count)
            }
            if stopRequested || Task.isCancelled { return }
            let queryVector = try await embedText(trimmed)
            dimensions = queryVector.count
            applyRanking(queryVector: queryVector)
        } catch {
            errorText = MLXErrorGuard.sentence(for: error)
        }
    }

    /// Cosine similarity is the dot product (the embedder L2-normalises). Pure, so the ranking is
    /// testable without an embedder.
    func applyRanking(queryVector: [Float]) {
        ranked = entries.compactMap { entry -> Ranked? in
            guard let vector = vectors[entry.url], vector.count == queryVector.count else { return nil }
            let dot = zip(vector, queryVector).reduce(Float(0)) { partial, pair in
                partial + pair.0 * pair.1
            }
            return Ranked(entry: entry, score: Double(dot))
        }
        .sorted { $0.score > $1.score }
    }

    /// Stop between images (decision 6): the in-flight image finishes/decode returns, then the
    /// loop exits and no ranking is produced.
    func stop() {
        stopRequested = true
        isEmbedding = false
    }

    /// Drop every image and vector; nothing was on disk, so nothing to delete.
    func clear() {
        stop()
        entries = []
        ranked = []
        vectors = [:]
        note = nil
        errorText = nil
        dimensions = nil
        progress = 0
    }

    // MARK: - Helpers

    private static func imageFiles(in folder: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: folder, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]) else { return [] }
        var urls: [URL] = []
        for case let url as URL in enumerator {
            guard let type = UTType(filenameExtension: url.pathExtension),
                  type.conforms(to: .image) else { continue }
            urls.append(url)
        }
        return urls.sorted { $0.path < $1.path }
    }

    /// A 128 px longest-edge preview, so the session holds thumbnails, not 200 full bitmaps.
    private static func thumbnail(of image: CGImage) -> CGImage? {
        let longest = max(image.width, image.height)
        guard longest > 0 else { return nil }
        let scale = min(1, 128 / CGFloat(longest))
        let width = max(1, Int((CGFloat(image.width) * scale).rounded()))
        let height = max(1, Int((CGFloat(image.height) * scale).rounded()))
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }
}
