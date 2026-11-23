import Testing
import Foundation
import ImageIO
import CryptoKit
import UniformTypeIdentifiers
@testable import MLXUI

/// ER-11 (SPEC-Q258, owner rulings R8 and R10): `Stage Print`. Staging, the idempotent id, the PDF-by-magic-bytes rule, the image types,
/// the E722 refusals, the approval path through an injected fake print panel (no test opens a real one), and the entitlement.
struct CatFlowStagePrintTests {

    private func scratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("er11-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static let pdfBytes = Data("%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n".utf8)

    private func file(_ name: String, _ data: Data, in dir: URL, kind: Kind = .file) throws -> Asset {
        let url = dir.appendingPathComponent(name)
        try data.write(to: url)
        return Asset(items: [Item(kind: kind, value: nil, path: url, sourceText: nil)])
    }

    private func imageData(_ type: String) -> Data? {
        guard let ctx = CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.setFillColor(red: 0.2, green: 0.5, blue: 0.8, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        guard let image = ctx.makeImage() else { return nil }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, type as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest) ? data as Data : nil
    }

    private func executor(_ dir: URL) -> RealExecutor {
        RealExecutor(workspace: FlowWorkspace(root: dir.appendingPathComponent("flows")), flowID: "er11",
                     blobDirectory: dir.appendingPathComponent("blobs"),
                     makeModelStage: { _, _ in PrintStub() }, installedModelIDs: [], catalog: [])
    }

    private func outbox(_ dir: URL) -> URL {
        OutboxStore.directory(workspace: FlowWorkspace(root: dir.appendingPathComponent("flows")), flowID: "er11")
    }

    private func refusal(_ body: () throws -> Void) -> String? {
        do { try body(); return nil } catch let e as FlowError {
            if case .stageFailure(_, let message) = e { return message }
            return "\(e)"
        } catch { return "\(error)" }
    }

    private func entry(_ outbox: URL, _ id: String, committed: Bool = false) throws -> [String: Any] {
        let url = (committed ? outbox.appendingPathComponent("committed") : outbox).appendingPathComponent("\(id).json")
        return try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    // MARK: - the catalog tables

    @Test func theTaskIsAStagedEffectAcceptingFileOrImage() throws {
        let desc = try #require(TaskCatalog.get("Stage Print"))
        #expect(desc.taskClass == .staged && desc.refKind == .tool)
        #expect(desc.refName == "tools.outbox.stage_print")
        #expect(desc.accepts == .unionOf([.file, .image]) && desc.gives == .single(.status))
        #expect(TaskFamilies.families["Staged effects"] == ["Stage Send", "Stage Post", "Stage Print"])
        #expect(CacheKey.neverCache.contains("Stage Print"))
        #expect(FlowValidator.pipelineRefused["Stage Print"] != nil)
        #expect(CatParser.taskNames.contains("Stage Print"))
        #expect(TaskAvailability.state(for: desc, catalog: [], claimableModelIDs: []) == .available)
    }

    // MARK: - E722

    private func issues(_ text: String) throws -> [FlowIssue] {
        FlowValidator.checkFlow(try CatParser.parseForValidation(text))
    }

    @Test func settingsAreRefusedByNameStatically() throws {
        for (setting, shown) in [("printer=office", "`printer=`"), ("copies=2", "`copies=`"), ("\"Office\"", "`Office`")] {
            let found = try issues("mlxflow 0.8\n1. Read Text  label.txt\n2. Make PDF  page=a4\n3. Stage Print  \(setting)\n")
            #expect(found.first { $0.code == "E722" }?.message
                    == "Row 3's `Stage Print` doesn't take settings -- the file goes to the system's default printer, and v1 has no printer or option settings; remove \(shown).",
                    "\(found)")
        }
        let clean = try issues("mlxflow 0.8\n1. Read Text  label.txt\n2. Make PDF  page=a4\n3. Stage Print\n")
        #expect(clean.isEmpty, "\(clean)")
    }

    // MARK: - staging

    @Test func stagingAPDFWritesTheEntryAndItsOwnCopyAndPrintsNothing() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let input = try file("label.pdf", Self.pdfBytes, in: dir)
        let staged = try StagePrintTool.stage(settings: nil, inputs: [input], path: "3",
                                              workspace: FlowWorkspace(root: dir.appendingPathComponent("flows")), flowID: "er11")
        let digest = SHA256.hash(data: Self.pdfBytes).map { String(format: "%02x", $0) }.joined()
        let expectedID = String(SHA256.hash(data: Data("Stage Print|\(digest)".utf8)).map { String(format: "%02x", $0) }.joined().prefix(10))
        #expect(staged.id == expectedID)
        #expect(staged.status == "staged for print (\(expectedID))")
        #expect(staged.summary == "to default printer: label.pdf")
        let box = outbox(dir)
        let e = try entry(box, expectedID)
        #expect(e["kind"] as? String == "print" && e["status"] as? String == "pending")
        #expect(e["destination"] as? String == "default printer" && e["text"] as? String == "label.pdf")
        #expect(e["file_name"] as? String == "label.pdf" && e["mime"] as? String == "application/pdf")
        #expect(e["sha256"] as? String == digest && e["stored_file"] as? String == "\(expectedID).pdf")
        #expect(try Data(contentsOf: box.appendingPathComponent("files/\(expectedID).pdf")) == Self.pdfBytes)
        // The listing the outbox UI reads exposes the print fields.
        let listed = try #require(OutboxStore.entries(workspace: FlowWorkspace(root: dir.appendingPathComponent("flows")), flowID: "er11").first)
        #expect(listed.kind == "print" && listed.fileName == "label.pdf" && listed.storedFile == "\(expectedID).pdf" && listed.lastError == nil)
    }

    @Test func theSameBytesFromAnotherNameAreOnePendingEntry() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let ws = FlowWorkspace(root: dir.appendingPathComponent("flows"))
        let a = try StagePrintTool.stage(settings: nil, inputs: [try file("a.pdf", Self.pdfBytes, in: dir)], path: "3", workspace: ws, flowID: "er11")
        let b = try StagePrintTool.stage(settings: nil, inputs: [try file("elsewhere.bin", Self.pdfBytes, in: dir)], path: "3", workspace: ws, flowID: "er11")
        #expect(a.id == b.id)
        let jsons = try FileManager.default.contentsOfDirectory(atPath: outbox(dir).path).filter { $0.hasSuffix(".json") }
        #expect(jsons.count == 1)
        #expect(try entry(outbox(dir), a.id)["file_name"] as? String == "a.pdf", "the first staging stands")
    }

    @Test func aPDFIsDecidedByItsBytesNotItsName() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let msg = refusal { _ = try StagePrintTool.prepare(settings: nil, inputs: [try file("notes.pdf", Data("just text".utf8), in: dir)], path: "2") }
        #expect(msg == "Stage Print can't print notes.pdf -- it takes a PDF or an image (PNG, JPEG, GIF, TIFF, BMP); this file is neither")
        let ok = try StagePrintTool.prepare(settings: nil, inputs: [try file("report.bin", Self.pdfBytes, in: dir)], path: "2")
        #expect(ok.mime == "application/pdf" && ok.ext == ".pdf")
    }

    @Test func pngJpegGifTiffAndBmpAreAcceptedAsTheyAre() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let cases: [(UTType, String, String)] = [(.png, "image/png", ".png"), (.jpeg, "image/jpeg", ".jpg"), (.gif, "image/gif", ".gif"),
                                                 (.tiff, "image/tiff", ".tiff"), (.bmp, "image/bmp", ".bmp")]
        for (type, mime, ext) in cases {
            let data = try #require(imageData(type.identifier), "could not write a \(type.identifier) test image")
            let p = try StagePrintTool.prepare(settings: nil, inputs: [try file("p\(ext)", data, in: dir, kind: .image)], path: "2")
            #expect(p.mime == mime && p.ext == ext && p.data == data, "\(type.identifier)")
        }
    }

    @Test func otherImageFormatsAndOtherFilesAreRefusedByName() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        if let other = imageData("public.jpeg-2000") {
            let msg = refusal { _ = try StagePrintTool.prepare(settings: nil, inputs: [try file("scan.jp2", other, in: dir, kind: .image)], path: "2") }
            #expect(msg?.hasPrefix("Stage Print can't print a ") == true && msg?.contains("(scan.jp2)") == true && msg?.contains("`Convert` format=png") == true, "\(msg ?? "nil")")
        }
        let jp = try #require(imageData(UTType.png.identifier))
        // Nothing, two items, a text item, a missing and an empty file, and a setting.
        func m(_ inputs: [Asset], _ settings: String? = nil) -> String? {
            refusal { _ = try StagePrintTool.prepare(settings: settings, inputs: inputs, path: "2") }
        }
        #expect(m([])?.hasPrefix("Stage Print got nothing to print") == true)
        let two = Asset(items: try (file("a.png", jp, in: dir, kind: .image).items + file("b.png", jp, in: dir, kind: .image).items))
        #expect(m([two]) == "Stage Print got 2 items -- it prints one file at a time; wrap it in `<each>`")
        #expect(m([Asset(items: [Item(kind: .text, value: "hi", path: nil, sourceText: nil)])]) == "Stage Print takes a PDF file or an image, not a text")
        let ghost = Asset(items: [Item(kind: .file, value: nil, path: dir.appendingPathComponent("gone.pdf"), sourceText: nil)])
        #expect(m([ghost]) == "Stage Print can't read gone.pdf -- check the file exists")
        #expect(m([try file("empty.pdf", Data(), in: dir)]) == "Stage Print can't print empty.pdf -- the file is empty")
        #expect(m([try file("ok.pdf", Self.pdfBytes, in: dir)], "printer=x")?.hasPrefix("Stage Print doesn't take settings") == true)
    }

    // MARK: - executors

    @Test func realExecutorStagesAndRecordsTheEffect() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let ex = executor(dir)
        let out = try await ex.execute(path: "3", row: Row(task: "Stage Print"), inputs: [try file("l.pdf", Self.pdfBytes, in: dir)])
        let status = try #require(out.items.first?.value)
        #expect(out.items.first?.kind == .status && status.hasPrefix("staged for print ("))
        let staged = try #require(ex.lastStaged)
        #expect(staged.kind == "print" && status == "staged for print (\(staged.id))")
        #expect(FileManager.default.fileExists(atPath: outbox(dir).appendingPathComponent("\(staged.id).json").path))
    }

    @Test func theMockValidatesLikeTheRealOneAndStandsInForItsOwnFiles() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let blobs = dir.appendingPathComponent("blobs")
        let mock = MockExecutor(blobDirectory: blobs)
        // A real .txt still refuses under the mock.
        await #expect(throws: (any Error).self) {
            _ = try await mock.execute(path: "3", row: Row(task: "Stage Print"), inputs: [try file("notes.txt", Data("hi".utf8), in: dir)],
                                       transcript: nil, context: nil, usedFlowContent: nil)
        }
        // A file the mock fabricated under its own blob directory stands in as a minimal PDF.
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let fabricated = try file("0123456789abcdef.bin", Data("0123456789abcdef".utf8), in: blobs)
        let out = try await mock.execute(path: "3", row: Row(task: "Stage Print"), inputs: [fabricated],
                                         transcript: nil, context: nil, usedFlowContent: nil)
        #expect(out.items.first?.value?.hasPrefix("staged for print (") == true)
        #expect(mock.lastStaged?.kind == "print")
    }

    // MARK: - approval (injected fake print panel)

    @MainActor
    private final class FakePanel: PrintPresenting {
        var outcome: PrintPanelOutcome
        private(set) var shown: [(url: URL, mime: String, title: String)] = []
        init(_ outcome: PrintPanelOutcome) { self.outcome = outcome }
        func present(fileURL: URL, mime: String, title: String) async -> PrintPanelOutcome {
            shown.append((fileURL, mime, title))
            return outcome
        }
    }

    private func staged(_ dir: URL) throws -> (id: String, box: URL) {
        let r = try StagePrintTool.stage(settings: nil, inputs: [try file("l.pdf", Self.pdfBytes, in: dir)], path: "3",
                                         workspace: FlowWorkspace(root: dir.appendingPathComponent("flows")), flowID: "er11")
        return (r.id, outbox(dir))
    }

    @MainActor @Test func completingThePrintCommitsTheEntry() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let (id, box) = try staged(dir)
        let panel = FakePanel(.completed)
        let result = await PrintOutbox.approve(id: id, outbox: box, presenter: panel)
        #expect(result == .committed)
        #expect(panel.shown.count == 1 && panel.shown[0].mime == "application/pdf" && panel.shown[0].title == "l.pdf")
        #expect(panel.shown[0].url.lastPathComponent == "\(id).pdf", "the stored copy is what is shown")
        #expect(!FileManager.default.fileExists(atPath: box.appendingPathComponent("\(id).json").path))
        let done = try entry(box, id, committed: true)
        #expect(done["status"] as? String == "committed" && done["committed_at"] is String)
        #expect(FileManager.default.fileExists(atPath: box.appendingPathComponent("files/\(id).pdf").path), "the stored copy stays, for audit")
        #expect(OutboxStore.entries(workspace: FlowWorkspace(root: dir.appendingPathComponent("flows")), flowID: "er11").isEmpty)
        // The same file can be staged again once committed.
        let again = try staged(dir)
        #expect(again.id == id && FileManager.default.fileExists(atPath: box.appendingPathComponent("\(id).json").path))
    }

    @MainActor @Test func cancellingLeavesItPendingWithNoError() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let (id, box) = try staged(dir)
        #expect(await PrintOutbox.approve(id: id, outbox: box, presenter: FakePanel(.cancelled)) == .cancelled)
        let e = try entry(box, id)
        #expect(e["status"] as? String == "pending" && e["last_error"] == nil && e["attempts"] == nil)
    }

    @MainActor @Test func aFailedPrintStaysPendingAndRecordsTheErrorUntilALaterSuccess() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let (id, box) = try staged(dir)
        let panel = FakePanel(.failed("the printer is offline"))
        #expect(await PrintOutbox.approve(id: id, outbox: box, presenter: panel) == .failed("the printer is offline"))
        _ = await PrintOutbox.approve(id: id, outbox: box, presenter: panel)
        let e = try entry(box, id)
        #expect(e["status"] as? String == "pending" && e["last_error"] as? String == "the printer is offline")
        #expect(e["attempts"] as? Int == 2 && e["last_attempt_at"] is String)
        #expect(OutboxStore.entries(workspace: FlowWorkspace(root: dir.appendingPathComponent("flows")), flowID: "er11").first?.lastError == "the printer is offline")
        panel.outcome = .completed
        #expect(await PrintOutbox.approve(id: id, outbox: box, presenter: panel) == .committed)
        #expect(try entry(box, id, committed: true)["last_error"] == nil, "a later success clears the error")
    }

    @MainActor @Test func aMissingStoredCopyOrAnUnknownEntryNeverReachesThePanel() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let (id, box) = try staged(dir)
        let panel = FakePanel(.completed)
        #expect(await PrintOutbox.approve(id: "../etc", outbox: box, presenter: panel) == .notFound)
        #expect(await PrintOutbox.approve(id: "nope", outbox: box, presenter: panel) == .notFound)
        // A hand-edited entry cannot point the print elsewhere: a non-bare stored name is treated as missing.
        let url = box.appendingPathComponent("\(id).json")
        var e = try entry(box, id)
        e["stored_file"] = "../../../etc/hosts"
        try JSONSerialization.data(withJSONObject: e).write(to: url)
        if case .failed(let message) = await PrintOutbox.approve(id: id, outbox: box, presenter: panel) {
            #expect(message.contains("stored copy"))
        } else { Issue.record("a non-bare stored_file should fail") }
        #expect(panel.shown.isEmpty)
    }

    // MARK: - the entitlement (R10) and the seam

    private func plist(_ name: String) throws -> [String: Any] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/\(name)"))
        return try #require(try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any])
    }

    @Test func printingIsAllowedInTheSandboxedEditionOnly() throws {
        let appStore = try plist("MLXUI-AppStore.entitlements")
        #expect(appStore["com.apple.security.app-sandbox"] as? Bool == true)
        #expect(appStore["com.apple.security.print"] as? Bool == true)
        // The Direct edition is un-sandboxed (Hardened Runtime only): the key is neither needed nor present.
        let direct = try plist("MLXUI-Direct.entitlements")
        #expect(direct["com.apple.security.app-sandbox"] == nil)
        #expect(direct["com.apple.security.print"] == nil)
    }

    @Test func noPrintCodePathRunsAProcess() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        for name in ["MLXUI/FlowKit/Tools/PrintOutbox.swift", "MLXUI/Views/Flows/SystemPrintPresenter.swift"] {
            let text = try String(contentsOf: root.appendingPathComponent(name), encoding: .utf8)
            for banned in ["Process(", "NSTask", "posix_spawn", "\"lp\"", "/usr/bin/lp"] {
                #expect(!text.contains(banned), "\(name) mentions \(banned)")
            }
        }
    }
}

private nonisolated struct PrintStub: PipelineStage {
    let id = "stub.print", name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}
