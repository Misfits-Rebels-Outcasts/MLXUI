import Testing
import Foundation
import AppKit
import SwiftUI
import MLXLMCommon
@testable import MLXUI

/// MT4-4 fixtures shared by the chip, listing and hosting suites.
enum MT44Fixtures {
    /// A silent 8 kHz mono 16-bit WAV of `seconds`.
    static func wav(seconds: Double = 0.5) -> Data {
        let rate = 8000, samples = Int(Double(rate) * seconds), dataBytes = samples * 2
        var d = Data()
        func u32(_ v: Int) { var x = UInt32(v).littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: Int) { var x = UInt16(v).littleEndian; d.append(Data(bytes: &x, count: 2)) }
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + dataBytes)
        d.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16); u16(1); u16(1); u32(rate); u32(rate * 2); u16(2); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(dataBytes)
        d.append(Data(count: dataBytes))
        return d
    }

    static func png() throws -> Data {
        let ctx = CGContext(data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 16,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.setFillColor(CGColor(srgbRed: 0, green: 0.5, blue: 1, alpha: 1))
        ctx?.fill(CGRect(x: 0, y: 0, width: 4, height: 4))
        let image = try #require(ctx?.makeImage())
        return try #require(PNGEncoder.pngData(from: image))
    }

    static func tempRoot() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mt44-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

/// MT4-4: the chip is read from a tool card's result and the conversation's assets. The real-path tests run the real
/// `ModelToolRunner` (fake executor) with a host, so the result string the chip parses is the one a turn really produces.
@MainActor @Suite struct ModelToolChipTests {

    // MARK: pure

    @Test func durationsAreOneDecimalUnderAMinute() {
        #expect(ModelToolChip.duration(milliseconds: 800) == "0.8 s")
        #expect(ModelToolChip.duration(milliseconds: 2100) == "2.1 s")
        #expect(ModelToolChip.duration(milliseconds: 0) == "0.0 s")
        #expect(ModelToolChip.duration(milliseconds: 72_000) == "1 m 12 s")
        #expect(ModelToolChip.duration(milliseconds: -5) == "0.0 s")
    }

    @Test func theRowTextLosesItsIndexAndExtraSpaces() {
        #expect(ModelToolChip.displayRow("1. OCR   PaddleOCR-VL") == "OCR PaddleOCR-VL")
        #expect(ModelToolChip.displayRow("1. Speak   Kokoro 82M; voice=\"af_heart\"") == "Speak Kokoro 82M; voice=\"af_heart\"")
        #expect(ModelToolChip.displayRow("OCR Model") == "OCR Model")
    }

    @Test func theOutputHandleIsReadFromTheResultJSON() {
        #expect(ModelToolChip.outputHandle(inResult: #"{"asset":"a3","kind":"audio","model":"M"}"#) == "a3")
        #expect(ModelToolChip.outputHandle(inResult: #"{"asset":"A3"}"#) == "a3")
        #expect(ModelToolChip.outputHandle(inResult: "Error: nope") == nil)
        #expect(ModelToolChip.outputHandle(inResult: "plain words") == nil)
        #expect(ModelToolChip.outputHandle(inResult: #"{"asset":"report.txt"}"#) == nil)
        #expect(ModelToolChip.outputHandle(inResult: "{not json") == nil)
    }

    private func record(_ handle: String, kind: String = "audio", model: String? = "Kokoro 82M", ms: Int? = 800,
                        inputs: [String] = ["a1"], row: String? = "1. Speak   Kokoro 82M") -> ChatAssetRecord {
        ChatAssetRecord(handle: handle, kind: kind, name: "speak result", fileName: "\(handle).wav", bytes: 10, words: nil,
                        duration: nil, provenance: ChatAssetProvenance(rowText: row, model: model, inputs: inputs, milliseconds: ms),
                        createdAt: Date())
    }

    @Test func aFinishedCallIsModelTaskTime() throws {
        let chip = try #require(ModelToolChip.make(
            toolName: "speak", id: UUID(), status: .finished,
            result: #"{"asset":"a2","kind":"audio","model":"Kokoro 82M"}"#, records: [record("a2")]))
        #expect(chip.title == "Kokoro 82M · Speak · 0.8 s")
        #expect(chip.state == .done)
        #expect(chip.outputHandle == "a2" && chip.outputKind == "audio" && chip.inputHandles == ["a1"])
        let lines = chip.details.map { "\($0.label): \($0.value)" }
        #expect(lines == ["Input: a1", "Output: a2 (audio)", "Row: Speak Kokoro 82M", "Time: 0.8 s"])
    }

    @Test func aMissingRecordStillShowsTheTaskNotAMadeUpModel() throws {
        let chip = try #require(ModelToolChip.make(
            toolName: "ocr_image", id: UUID(), status: .finished, result: #"{"asset":"a9","kind":"text"}"#, records: []))
        #expect(chip.title == "OCR")
        #expect(chip.model == nil && chip.milliseconds == nil)
    }

    @Test func runningFailedAndDeclinedCalls() throws {
        let running = try #require(ModelToolChip.make(toolName: "transcribe_audio", id: UUID(), status: .running, result: nil, records: []))
        #expect(running.state == .running && running.title == "Transcribe · running…")
        let failed = try #require(ModelToolChip.make(
            toolName: "ocr_image", id: UUID(), status: .finished, result: "Error: no OCR or vision model is installed.", records: []))
        #expect(failed.state == .failed("Error: no OCR or vision model is installed.") && failed.title == "OCR · failed")
        let declined = try #require(ModelToolChip.make(
            toolName: "transcribe_audio", id: UUID(), status: .finished,
            result: ModelToolHost.declineSentence(for: "Parakeet TDT 0.6B v3"), records: []))
        #expect(declined.state == .declined && declined.title == "Transcribe · not downloaded")
    }

    @Test func otherCardsKeepTheirOwnCard() {
        // not a model tool
        #expect(ModelToolChip.make(toolName: "read_file", id: UUID(), status: .finished, result: #"{"asset":"a1"}"#, records: []) == nil)
        // the two tools that run no model
        #expect(ModelToolChip.make(toolName: "read_asset", id: UUID(), status: .finished, result: "text", records: []) == nil)
        #expect(ModelToolChip.make(toolName: "list_capabilities", id: UUID(), status: .finished, result: "x", records: []) == nil)
        // a plain-text result from before Model Tools has no asset behind it
        #expect(ModelToolChip.make(toolName: "ocr_image", id: UUID(), status: .finished, result: "TOTAL 84.20", records: []) == nil)
        #expect(ModelToolChip.make(toolName: "ocr_image", id: UUID(), status: .unknownTool, result: nil, records: []) == nil)
    }

    // MARK: the real path: runner + host -> result -> chip -> listing

    private func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    private func runtime(installed hf: [String], executor: MT13Executor) throws -> ModelToolRuntime {
        let cat = try catalog()
        let ids = Set(try hf.map { name in try #require(cat.first { $0.hfModelId == name }, "\(name)").id })
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(cat.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        return ModelToolRuntime(catalog: cat, installedModelIDs: ids, claimableModelIDs: claimable, totalRAMGB: 19,
                                makeExecutor: { blob, _ in executor.made(blob: blob); return executor })
    }

    private func call(_ tool: String, _ args: [String: JSONValue], runtime rt: ModelToolRuntime, host: ModelToolHost) async throws -> String {
        let runner = ModelToolRunner(spec: try #require(ModelToolSpec.spec(named: tool)), runtime: rt)
        return await ModelToolHost.$current.withValue(host) { await runner.run(arguments: args) }
    }

    @Test func aRealTurnsResultsBecomeChipsAndFilesWithTheirHandles() async throws {
        let root = MT44Fixtures.tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatAssetStore(conversationID: UUID(), in: root)
        let host = ModelToolHost(assets: store)
        // the user's receipt is a1
        let receipt = root.appendingPathComponent("receipt.png"); try MT44Fixtures.png().write(to: receipt)
        let a1 = try store.importFile(receipt, kind: "image")
        #expect(a1.handle == "a1")
        // OCR it (a2, text), then speak the text (a3, audio)
        let ocr = try await call("ocr_image", ["asset": .string("a1")],
                                 runtime: try runtime(installed: ["mlx-community/GLM-OCR-4bit"], executor: .text("TOTAL RM 84.20")), host: host)
        let speak = try await call("speak", ["asset": .string("a2")],
                                   runtime: try runtime(installed: ["mlx-community/Kokoro-82M-bf16"],
                                                        executor: .file(kind: .audio, ext: "wav")), host: host)
        let ocrActivity = ToolActivity(name: "ocr_image", arguments: ["asset": .string("a1")], status: .finished, result: ocr)
        let speakActivity = ToolActivity(name: "speak", arguments: ["asset": .string("a2")], status: .finished, result: speak)

        let records = store.all
        let ocrChip = try #require(ModelToolChip.make(toolName: "ocr_image", id: ocrActivity.id, status: .finished, result: ocr, records: records))
        #expect(ocrChip.outputHandle == "a2" && ocrChip.inputHandles == ["a1"] && ocrChip.outputKind == "text")
        #expect(ocrChip.title.contains(" · OCR · ") && ocrChip.title.hasSuffix(" s") && ocrChip.model != nil)
        let speakChip = try #require(ModelToolChip.make(toolName: "speak", id: speakActivity.id, status: .finished, result: speak, records: records))
        #expect(speakChip.outputHandle == "a3" && speakChip.inputHandles == ["a2"] && speakChip.outputKind == "audio")
        #expect(speakChip.title.hasPrefix("Kokoro 82M · Speak · "))
        #expect(speakChip.details.contains { $0.label == "Row" && $0.value.hasPrefix("Speak Kokoro 82M") })

        // and the inspector's Files
        let listing = ChatAssetListing.make(records: records, items: [.tool(ocrActivity), .tool(speakActivity)])
        #expect(listing.rows.map(\.handle) == ["a1", "a2", "a3"])
        #expect(listing.rows[0].origin == "attached" && listing.rows[0].activityID == nil)
        #expect(listing.rows[1].activityID == ocrActivity.id && listing.rows[2].activityID == speakActivity.id)
        #expect(listing.rows[2].origin == "Kokoro 82M · Speak" && listing.rows[2].kind == "audio")
        #expect(listing.rows[1].origin.hasSuffix(" · OCR"))
        #expect(listing.modelsUsed.count == 2 && listing.modelsUsed.contains("Kokoro 82M"))
    }

    @Test func theChipsSurviveARelaunch() async throws {
        let root = MT44Fixtures.tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let store = ChatAssetStore(conversationID: id, in: root)
        let result = try await call("speak", ["text": .string("hello")],
                                    runtime: try runtime(installed: ["mlx-community/Kokoro-82M-bf16"], executor: .file(kind: .audio, ext: "wav")),
                                    host: ModelToolHost(assets: store))
        // a new store over the same folder is what the next launch reads
        let reopened = ChatAssetStore(conversationID: id, in: root)
        let chip = try #require(ModelToolChip.make(toolName: "speak", id: UUID(), status: .finished, result: result, records: reopened.all))
        #expect(chip.title.hasPrefix("Kokoro 82M · Speak · "))
        #expect(chip.outputHandle == "a1")
    }

    @Test func aRefusedRealCallIsAFailedChipNotARecord() async throws {
        let root = MT44Fixtures.tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatAssetStore(conversationID: UUID(), in: root)
        // the model is installed but the handle is not an asset: the real runner answers with its own sentence
        let result = try await call("ocr_image", ["asset": .string("a9")],
                                    runtime: try runtime(installed: ["mlx-community/GLM-OCR-4bit"], executor: .text("x")),
                                    host: ModelToolHost(assets: store))
        let chip = try #require(ModelToolChip.make(toolName: "ocr_image", id: UUID(), status: .finished, result: result, records: store.all))
        guard case .failed(let sentence) = chip.state else { Issue.record("not failed: \(chip.state) from \(result)"); return }
        #expect(sentence == result && result.hasPrefix("Error:"))
        #expect(store.all.isEmpty)
    }

    @Test func aDeclinedRealDownloadIsADeclinedChip() async throws {
        let root = MT44Fixtures.tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatAssetStore(conversationID: UUID(), in: root)
        let audio = root.appendingPathComponent("m.wav"); try MT44Fixtures.wav().write(to: audio)
        _ = try store.importFile(audio, kind: "audio")
        // nothing installed for ASR: the host's install card is answered "no"
        let result = try await call("transcribe_audio", ["asset": .string("a1")],
                                    runtime: try runtime(installed: [], executor: .text("x")), host: ModelToolHost(assets: store))
        #expect(result.hasPrefix(ModelToolHost.declineSentencePrefix))
        let chip = try #require(ModelToolChip.make(toolName: "transcribe_audio", id: UUID(), status: .finished, result: result, records: store.all))
        #expect(chip.state == .declined)
        #expect(store.all.count == 1)
    }
}

@MainActor @Suite struct ChatAssetListingTests {
    @Test func aToolResultThatIsNotFinishedDoesNotClaimAnAsset() {
        let record = ChatAssetRecord(handle: "a1", kind: "audio", name: "speak result", fileName: "a1.wav", bytes: 2048, words: nil,
                                     duration: nil, provenance: ChatAssetProvenance(rowText: "1. Speak   Kokoro 82M", model: "Kokoro 82M", inputs: [], milliseconds: 5),
                                     createdAt: Date())
        let running = ToolActivity(name: "speak", arguments: [:], status: .running, result: #"{"asset":"a1"}"#)
        let listing = ChatAssetListing.make(records: [record], items: [.tool(running)])
        #expect(listing.rows.first?.activityID == nil)
        #expect(listing.rows.first?.size.isEmpty == false)
    }

    @Test func modelsUsedAreDistinctAndInFirstUseOrder() {
        func rec(_ h: String, _ model: String?) -> ChatAssetRecord {
            ChatAssetRecord(handle: h, kind: "text", name: "n", fileName: "\(h).txt", bytes: 1, words: 1, duration: nil,
                            provenance: ChatAssetProvenance(rowText: nil, model: model, inputs: [], milliseconds: nil), createdAt: Date())
        }
        let listing = ChatAssetListing.make(records: [rec("a1", nil), rec("a2", "B"), rec("a3", "A"), rec("a4", "B")], items: [])
        #expect(listing.modelsUsed == ["B", "A"])
        #expect(listing.rows.map(\.origin) == ["attached", "B", "A", "B"])
    }
}

/// The players are real AppKit / AVFoundation views; host them in a window (the VideoPreview lesson, 2026-10-29).
@MainActor @Suite struct InlineMediaHostingTests {

    private func layOut<V: View>(_ view: V, size: NSSize) async {
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: view)
        window.contentView = host
        window.orderFront(nil)
        host.layoutSubtreeIfNeeded()
        for _ in 0 ..< 8 {
            try? await Task.sleep(for: .milliseconds(50))
            host.layoutSubtreeIfNeeded()
            host.displayIfNeeded()
        }
        #expect(host.fittingSize.width > 0)
        window.orderOut(nil)
    }

    private func chip(_ tool: String, handle: String, kind: String) -> ModelToolChip {
        ModelToolChip(id: UUID(), toolName: tool, task: ModelToolSpec.spec(named: tool)?.task ?? "", state: .done, model: "M",
                      milliseconds: 1200, outputHandle: handle, outputKind: kind, inputHandles: ["a1"], rowText: "1. X   M")
    }

    private func record(_ handle: String, kind: String, file: String) -> ChatAssetRecord {
        ChatAssetRecord(handle: handle, kind: kind, name: "n", fileName: file, bytes: 1, words: nil, duration: nil,
                        provenance: ChatAssetProvenance(), createdAt: Date())
    }

    @Test func theAudioPlayerLoadsADurationAndSeeks() throws {
        let root = MT44Fixtures.tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("a.wav"); try MT44Fixtures.wav(seconds: 1.5).write(to: file)
        let model = InlineAudioPlayerModel(url: file)
        #expect(model.failure == nil)
        #expect(abs(model.duration - 1.5) < 0.05)
        model.seek(to: 1.0)
        #expect(abs(model.position - 1.0) < 0.05)
        model.seek(to: 99)
        #expect(model.position <= model.duration + 0.01)
        #expect(InlineAudioPlayerModel.clock(6.4) == "0:06" && InlineAudioPlayerModel.clock(125) == "2:05")
    }

    @Test func aMissingOrBrokenAudioFileSaysSoInsteadOfCrashing() throws {
        #expect(InlineAudioPlayerModel(url: URL(fileURLWithPath: "/nonexistent/x.wav")).failure == "This audio file is no longer on disk.")
        let root = MT44Fixtures.tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let bad = root.appendingPathComponent("bad.wav"); try Data([1, 2, 3]).write(to: bad)
        #expect(InlineAudioPlayerModel(url: bad).failure != nil)
    }

    @Test func anAudioOutputRendersAPlayerInAWindow() async throws {
        let root = MT44Fixtures.tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("a3.wav"); try MT44Fixtures.wav().write(to: file)
        await layOut(ModelToolChipView(chip: chip("speak", handle: "a3", kind: "audio"),
                                       output: (record("a3", kind: "audio", file: "a3.wav"), file)),
                     size: NSSize(width: 480, height: 160))
    }

    @Test func anImageOutputRendersInAWindow() async throws {
        let root = MT44Fixtures.tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("a2.png"); try MT44Fixtures.png().write(to: file)
        await layOut(ModelToolChipView(chip: chip("generate_image", handle: "a2", kind: "image"),
                                       output: (record("a2", kind: "image", file: "a2.png"), file)),
                     size: NSSize(width: 480, height: 360))
    }

    @Test func aVideoOutputRendersThroughVideoPreviewInAWindow() async throws {
        let url = try await RIFETestVideo.make(frames: 8, fps: 8)
        defer { try? FileManager.default.removeItem(at: url) }
        await layOut(ModelToolChipView(chip: chip("generate_image", handle: "a2", kind: "video"),
                                       output: (record("a2", kind: "video", file: url.lastPathComponent), url)),
                     size: NSSize(width: 480, height: 360))
    }

    @Test func aGoneFileRendersItsMessageNotABlankOrACrash() async throws {
        let gone = URL(fileURLWithPath: "/nonexistent/a2.png")
        await layOut(ModelToolChipView(chip: chip("generate_image", handle: "a2", kind: "image"),
                                       output: (record("a2", kind: "image", file: "a2.png"), gone)),
                     size: NSSize(width: 480, height: 200))
    }
}
