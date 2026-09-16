import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// CFM-R11-0c + CFM-R11-0b — the four newly-ported `Read *` tools and the sample seeding
/// that makes a freshly-added `Read *` row run with no further input.
struct CatFlowReadToolsTests {

    // MARK: - Helpers

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-read-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    private func writeWAV(named name: String, in dir: URL) throws -> URL {
        let url = dir.appendingPathComponent(name)
        try AudioWriter.writeWAV(AudioBuffer(samples: [0.1, 0.2, -0.1], sampleRate: 24_000), to: url)
        return url
    }

    private func redPixelPNG() -> Data? {
        let ctx = CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.setFillColor(CGColor(srgbRed: 1, green: 0, blue: 0, alpha: 1))
        ctx?.fill(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard let image = ctx?.makeImage() else { return nil }
        return PNGEncoder.pngData(from: image)
    }

    // MARK: - Read Image

    @Test func readImageProducesFileBackedImageItem() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        // A real decodable PNG (1×1 red pixel).
        let png = try #require(redPixelPNG())
        try png.write(to: flowDir.appendingPathComponent("photo.png"))

        let tool = ReadImageTool(workspace: ws, flowID: "sample-flow", settings: "photo.png")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let item = try #require(out.items.first)
        #expect(item.kind == .image)
        #expect(item.path?.lastPathComponent == "photo.png")
    }

    @Test func readImageRefusesMissingFile() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let tool = ReadImageTool(workspace: ws, flowID: "sample-flow", settings: "nope.png", path: "4")
        do {
            _ = try await tool.run(Asset(items: [])) { _ in }
            Issue.record("expected a missing-input refusal")
        } catch let error as FlowError {
            // FIP-3: the shared, house-voice R903 sentence, naming the row and the file —
            // not each tool's own ad-hoc wording.
            guard case .missingInput(let row, let message) = error else {
                Issue.record("unexpected error \(error)")
                return
            }
            #expect(row == "4")
            #expect(message.contains("Row 4"))
            #expect(message.contains("nope.png"))
        }
    }

    // MARK: - FIP-3: ReadPath's shared missing-input error (R903)

    @Test func readPathResolveThrowsMissingInputWithTheR903Sentence() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let dir = ws.directory(for: "sample-flow")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        do {
            _ = try ReadPath.resolve(workspace: ws, flowID: "sample-flow", path: "3", settings: "notes.txt",
                                     inputs: [], kind: .file, row: "Read Text", checksUpstream: false)
            Issue.record("expected a missing-input refusal")
        } catch let error as FlowError {
            guard case .missingInput(let row, let message) = error else {
                Issue.record("unexpected error \(error)")
                return
            }
            #expect(row == "3")
            #expect(message == "Row 3 couldn't read notes.txt: no such file. Fix or re-point the row; rows 1–2 are cached.")
        }
    }

    /// The precedence rule this whole item exists to get right: an upstream `.file` item's
    /// path is real by construction (it came from an actual `Read Files` enumeration), so
    /// `resolve` must return it without any existence check of its own — never a false refusal
    /// on a correct flow.
    @Test func readPathResolveSkipsTheExistenceCheckWhenUpstreamSupplies() throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let phantom = URL(fileURLWithPath: "/does/not/exist-in-this-test.png")
        let upstream = Asset(items: [Item(kind: .file, value: nil, path: phantom, sourceText: nil)])
        let url = try ReadPath.resolve(workspace: ws, flowID: "sample-flow", path: "2", settings: "unused.png",
                                       inputs: [upstream], kind: .file, row: "Read Image")
        #expect(url == phantom)
    }

    // MARK: - Read Images

    @Test func readImagesEnumeratesImageExtensionsSorted() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("vacation", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try writeWAV(named: "b.m4a", in: folder)     // not an image
        try writeWAV(named: "a.png", in: folder)
        try writeWAV(named: "c.jpg", in: folder)

        let tool = ReadImagesTool(workspace: ws, flowID: "sample-flow", settings: "vacation")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let names = out.items.map { $0.path?.lastPathComponent ?? "?" }
        #expect(names == ["a.png", "c.jpg"])
    }

    @Test func readImagesHonorsPattern() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("vacation", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try writeWAV(named: "a-1.png", in: folder)
        try writeWAV(named: "a-2.png", in: folder)
        try writeWAV(named: "b.png", in: folder)

        let tool = ReadImagesTool(workspace: ws, flowID: "sample-flow", settings: "vacation; pattern=a-*.png")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let names = out.items.map { $0.path?.lastPathComponent ?? "?" }
        #expect(names == ["a-1.png", "a-2.png"])
    }

    // FILE-2 / SPEC-Q219: a folder that resolves fine but has nothing to read must say so, not
    // complete silently — an enclosing `<each>` over an empty list would otherwise run zero
    // times and the flow would finish green having done nothing.
    @Test func readImagesThrowsOnAnEmptyFolderRatherThanReturningNothing() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("vacation", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let tool = ReadImagesTool(workspace: ws, flowID: "sample-flow", settings: "vacation")
        do {
            _ = try await tool.run(Asset(items: [])) { _ in }
            Issue.record("expected throw")
        } catch let error as FlowError {
            guard case .emptyFolder(let row, let path) = error else {
                Issue.record("wrong FlowError case: \(error)")
                return
            }
            #expect(row == "Read Images")
            #expect(path == folder.path)
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    @Test func readImagesThrowsWhenNothingMatchesThePattern() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("vacation", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try writeWAV(named: "b.m4a", in: folder)   // present, but not an image

        let tool = ReadImagesTool(workspace: ws, flowID: "sample-flow", settings: "vacation")
        await #expect(throws: FlowError.self) {
            _ = try await tool.run(Asset(items: [])) { _ in }
        }
    }

    // MARK: - Read Files

    @Test func readFilesEnumeratesPattern() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("docs", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("one".utf8).write(to: folder.appendingPathComponent("a.txt"))
        try Data("two".utf8).write(to: folder.appendingPathComponent("b.txt"))
        try Data("three".utf8).write(to: folder.appendingPathComponent("c.md"))

        let tool = ReadFilesTool(workspace: ws, flowID: "sample-flow", settings: "docs; pattern=*.txt")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let names = out.items.map { $0.path?.lastPathComponent ?? "?" }
        #expect(names == ["a.txt", "b.txt"])
    }

    @Test func readFilesDefaultsToEverythingWhenPatternIsStar() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("docs", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("one".utf8).write(to: folder.appendingPathComponent("a.txt"))
        try Data("two".utf8).write(to: folder.appendingPathComponent("b.md"))

        // A bare folder with no pattern globs the folder's own name (Python parity) — the
        // seeded row always carries an explicit `pattern=*` to enumerate everything.
        let tool = ReadFilesTool(workspace: ws, flowID: "sample-flow", settings: "docs; pattern=*")
        let out = try await tool.run(Asset(items: [])) { _ in }
        #expect(out.items.count == 2)
    }

    @Test func readFilesRecursive() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("docs", isDirectory: true)
        let nested = folder.appendingPathComponent("sub", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data("one".utf8).write(to: folder.appendingPathComponent("a.txt"))
        try Data("two".utf8).write(to: nested.appendingPathComponent("b.txt"))

        let tool = ReadFilesTool(workspace: ws, flowID: "sample-flow", settings: "docs; recursive=true; pattern=*.txt")
        let out = try await tool.run(Asset(items: [])) { _ in }
        #expect(out.items.count == 2)
    }

    @Test func readFilesRecursiveSkipsTrash() async throws {
        // R13-8: the recursive branch got the same `.skipsHiddenFiles` as the flat one —
        // a `Save *` overwrite leaves `.trash/` beside the file, and a re-run's glob must
        // not enumerate the trashed copies.
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("docs", isDirectory: true)
        let trash = folder.appendingPathComponent(".trash", isDirectory: true)
        try FileManager.default.createDirectory(at: trash, withIntermediateDirectories: true)
        try Data("one".utf8).write(to: folder.appendingPathComponent("a.txt"))
        try Data("stale".utf8).write(to: trash.appendingPathComponent("a.txt.1728000000000"))

        let tool = ReadFilesTool(workspace: ws, flowID: "sample-flow", settings: "docs; recursive=true; pattern=*.txt")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let names = out.items.map { $0.path?.lastPathComponent ?? "?" }
        #expect(names == ["a.txt"])
    }

    // FILE-2 / SPEC-Q219 — same rule as Read Images: a resolved-but-empty folder throws.
    @Test func readFilesThrowsOnAnEmptyFolderRatherThanReturningNothing() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        let folder = flowDir.appendingPathComponent("docs", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let tool = ReadFilesTool(workspace: ws, flowID: "sample-flow", settings: "docs; pattern=*")
        do {
            _ = try await tool.run(Asset(items: [])) { _ in }
            Issue.record("expected throw")
        } catch let error as FlowError {
            guard case .emptyFolder(let row, let path) = error else {
                Issue.record("wrong FlowError case: \(error)")
                return
            }
            #expect(row == "Read Files")
            #expect(path == folder.path)
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    // MARK: - Read PDF

    @Test func readPDFExtractsSelectableText() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }

        let flowDir = ws.directory(for: "sample-flow")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        // The bundled sample PDF (a one-page selectable-text document).
        let sample = try #require(Bundle.main.url(forResource: "sample-pdf", withExtension: "pdf"))
        let pdfURL = flowDir.appendingPathComponent("sample.pdf")
        try Data(contentsOf: sample).write(to: pdfURL)

        let tool = ReadPDFTool(workspace: ws, flowID: "sample-flow", settings: "sample.pdf")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let item = try #require(out.items.first)
        #expect(item.kind == .text)
        let text = try #require(item.value)
        #expect(text.contains("AI Browser Sample Document"))
    }

    // MARK: - GlobMatch

    @Test func globMatchHandlesWildcards() {
        #expect(GlobMatch.matches("a.txt", glob: "*.txt"))
        #expect(GlobMatch.matches("b.m4a", glob: "*"))
        #expect(GlobMatch.matches("ab.txt", glob: "a?.txt"))
        #expect(!GlobMatch.matches("a.txt", glob: "*.png"))
        #expect(!GlobMatch.matches("x.txt", glob: "a*.txt"))
    }

    // MARK: - CFM-R11-0b seeding

    private func editorWithSamples(_ base: URL, flowID: String = "sample-flow") throws -> (FlowEditorModel, URL) {
        let samples = base.appendingPathComponent("samples")
        try FileManager.default.createDirectory(at: samples, withIntermediateDirectories: true)
        try writeWAV(named: "sample-audio.m4a", in: samples)
        try Data("sample text".utf8).write(to: samples.appendingPathComponent("sample-text.txt"))
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let model = FlowEditorModel(name: "Flow", flowID: flowID, workspace: ws, sampleSourceDir: samples)
        return (model, samples)
    }

    @Test func addingSeededReadAudioSeedsTheRow() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let (model, _) = try editorWithSamples(base)

        model.add(task: "Read Audio")
        let row = try #require(model.document.rows.first)
        #expect(row.settings == "sample-audio.m4a")
        // The sample was copied into the flow folder (prepare is idempotent).
        let flowDir = ws.directory(for: "sample-flow")
        #expect(FileManager.default.fileExists(atPath: flowDir.appendingPathComponent("sample-audio.m4a").path))
    }

    @Test func addingSeededReadFilesSeedsFolder() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let samples = base.appendingPathComponent("samples")
        try FileManager.default.createDirectory(at: samples, withIntermediateDirectories: true)
        // The bundle flattens `Resources/Samples/**` — sources are flat names, exactly as
        // `SampleSeed.seedAssets` declares them.
        try Data("one".utf8).write(to: samples.appendingPathComponent("sample-note-01.txt"))
        let model = FlowEditorModel(name: "Flow", flowID: "sample-flow",
                                    workspace: FlowWorkspace(root: base.appendingPathComponent("flows")),
                                    sampleSourceDir: samples)

        model.add(task: "Read Files")
        let row = try #require(model.document.rows.first)
        #expect(row.settings == "sample-files; pattern=*")
        #expect(FlowSettings(row.settings).pathValue() == "sample-files")
        // prepare copies the flat source to the `.cat`-relative destination.
        let folder = ws.directory(for: "sample-flow").appendingPathComponent("sample-files")
        #expect(FileManager.default.fileExists(atPath: folder.appendingPathComponent("sample-note-01.txt").path))
    }

    @Test func addingTwiceSeedsTheSameFile() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-seed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { teardown(base) }
        let (model, _) = try editorWithSamples(base)

        model.add(task: "Read Audio")
        model.add(task: "Read Audio")
        let rows = model.document.rows
        #expect(rows.count == 2)
        // Second seeded row points at the same file, not a minted copy.
        #expect(rows[0].settings == "sample-audio.m4a")
        #expect(rows[1].settings == "sample-audio.m4a")
    }

    @Test func unportedReadTasksAreNotSeeded() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-seed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { teardown(base) }
        let (model, _) = try editorWithSamples(base)

        model.add(task: "Read Video")
        model.add(task: "Read Index")
        model.add(task: "Read Context")
        #expect(model.document.rows.allSatisfy { $0.settings == nil })
    }

    /// The one-list rule: a seeded task is runnable, and every seeded task is a `Read *` task.
    @Test func seedingAndToolAvailabilityShareOneList() {
        let readTasks = Set(SampleSeed.readTaskNames)
        for task in SampleSeed.samples.keys {
            #expect(SampleSeed.isRunnable(task))
            #expect(readTasks.contains(task))
            #expect(SampleSeed.seedValue(for: task) != nil)
        }
        // The unported three are unseeded by the same list.
        #expect(!SampleSeed.isRunnable("Read Video"))
        #expect(!SampleSeed.isRunnable("Read Index"))
        #expect(!SampleSeed.isRunnable("Read Context"))
        #expect(SampleSeed.seedValue(for: "Read Index") == nil)
    }

    // MARK: - The chooser's bare-name write-back (setPath / replacePath)

    @Test func replacePathWritesBareTokenOverFirstBare() {
        // A seeded row's settings is a bare filename; replacing it splices that token.
        let edited = FlowSettingsEditor.replacePath("mine.m4a", in: "sample-audio.m4a")
        #expect(edited == "mine.m4a")
    }

    @Test func replacePathRewritesPathPair() {
        let edited = FlowSettingsEditor.replacePath("mine.m4a", in: "path=sample-audio.m4a; lang=en")
        #expect(edited == "path=mine.m4a; lang=en")
    }

    @Test func replacePathAppendsWhenMissing() {
        let edited = FlowSettingsEditor.replacePath("mine.m4a", in: "lang=en")
        #expect(edited == "lang=en; mine.m4a")
    }

    @Test func setPathRoundTripsThroughSettings() {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-path-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { teardown(base) }
        let (model, _) = try! editorWithSamples(base)
        model.add(task: "Read Audio")

        let rowID = model.document.rows[0].id
        model.setPath("mine.m4a", for: rowID)
        #expect(model.document.rows[0].settings == "mine.m4a")
        #expect(FlowSettings(model.document.rows[0].settings).pathValue() == "mine.m4a")
    }
}
