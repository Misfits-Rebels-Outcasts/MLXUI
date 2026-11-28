import Testing
import Foundation
import SwiftUI
import AppKit
@testable import MLXUI

/// The aligner's Run sheet view in a window's `.sheet(item:)`, over a `LeasingStage` as the app's `RunSheetHost` hands
/// it, driven through its own buttons and keys. `RunSheetHost` itself, the module registry and the engine are not in it.
/// No weights: the run's stage is a stub that, like the engine's forward pass, does not end when cancelled — the test
/// releases it. These tests put a window on screen while they run.
@Suite(.serialized)
@MainActor
struct AlignRunSheetTests {

    // MARK: - Done, then reopen at once (owner rulings 15, 21 and 40)

    /// Owner ruling 40 (Boo, 2026-11-28): "Done during a run, then reopening at once (tried twice, with a 90 s clip):
    /// no 'Waiting for the last alignment to finish…' message, and Choose Audio and Open… are enabled. Check with a log
    /// whether the step was still running. If it was, rulings 15 and 21 aren't met: fix it, and add a test that drives
    /// the real sheet lifecycle."
    @Test func doneThenReopenIsDisabledAndSaysWhyUntilTheStepReturns() async throws {
        let rig = try SheetRig()
        defer { rig.close() }
        let first = try await rig.present()
        try await rig.pressReturnToAlign(first)

        try await rig.clickDone()
        #expect(first.model.isStopping, "Done takes the Stop path")

        let reopened = try await rig.present()
        #expect(reopened.model !== first.model, "a reopened sheet makes its own model")
        #expect(rig.stage.gate.isEntered && first.model.isRunning, "the first step is still in flight")
        #expect(reopened.model.blockedText == AlignRunModel.waitingText)
        #expect(!reopened.model.canChangeInputs, "Choose Audio…, Open…, drop and Align all ask this")

        rig.stage.gate.open()
        try await rig.wait("the step to return") { !first.model.isRunning }
        #expect(reopened.model.blockedText == nil)
        #expect(reopened.model.canChangeInputs)
        #expect(first.model.result == nil, "the stopped run shows no result")
        #expect(rig.stage.gate.entries == 1, "no second step ran")
    }
}

// MARK: - The harness

/// A window that presents the aligner's sheet as `MLXUIApp` does, and the events a person would send it.
@MainActor
private final class SheetRig {
    struct Presented {
        let sheet: NSWindow
        let model: AlignRunModel
    }

    let stage = SheetStubStage()
    private let host = SheetHost()
    private let window: NSWindow
    private let audio: URL
    /// A model id of this rig's own, so the app-wide step tracker the sheet uses sees only this rig's steps.
    private let modelID = "align-sheet-\(UUID().uuidString)"
    private var models: [UUID: AlignRunModel] = [:]
    /// The test's own budget and engine cache, so the sheet's lease never touches the app's.
    private let cache = EngineCache(budget: MemoryBudget(
        capacityBytes: 20 * 1_073_741_824,
        cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1))), countsLLMWeights: true)

    init() throws {
        audio = FileManager.default.temporaryDirectory.appendingPathComponent("fa-sheet-\(UUID().uuidString).wav")
        try AudioWriter.writeWAV(AudioBuffer(samples: [Float](repeating: 0.1, count: 16_000), sampleRate: 16_000), to: audio)
        window = NSWindow(contentRect: NSRect(x: 80, y: 60, width: 900, height: 880),
                          styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: SheetParent(host: host) { [weak self] in self?.view(for: $0) })
        window.makeKeyAndOrderFront(nil)
    }

    private func view(for presentation: SheetPresentation) -> ForcedAlignerRunView {
        // One model per presentation (the closure runs on every body pass), made as the view itself would make it. It is
        // handed in only so the test can read it, and given the recording a file panel would have chosen.
        let model = models[presentation.id] ?? {
            let made = AlignRunModel(modelID: modelID)
            if models.isEmpty {
                made.chooseAudio(audio)
                made.transcript = "Guten Morgen"
            }
            models[presentation.id] = made
            return made
        }()
        let stage = self.stage
        // As `RunSheetHost` builds it for the aligner, which books its own weights.
        let handed = LeasingStage(model: makeEntry(id: modelID, modelType: .align, ramGB: 3), config: .default,
                                  probe: stage, booksWeights: false, cache: cache, maxWait: .milliseconds(300))
        return ForcedAlignerRunView(modelDisplayName: "Qwen3 Aligner 0.6B", license: nil, modelID: modelID,
                                    footprintBytes: 1, handedStage: handed, model: model, runStage: { _ in stage })
    }

    func present() async throws -> Presented {
        try await wait("the last sheet to close") { self.window.attachedSheet == nil }
        let presentation = SheetPresentation()
        host.item = presentation
        try await wait("the sheet") { self.window.attachedSheet != nil && self.models[presentation.id] != nil }
        try await Task.sleep(for: .milliseconds(500))   // the presentation settles
        return Presented(sheet: try #require(window.attachedSheet), model: try #require(models[presentation.id]))
    }

    /// Align is the sheet's default button.
    func pressReturnToAlign(_ presented: Presented) async throws {
        try await offer("\r", 36, to: presented.sheet)
        try await wait("the run to reach the stage") { self.stage.gate.isEntered }
        try await Task.sleep(for: .milliseconds(300))   // the running row and the Stop button appear
    }

    /// A mouse click on Done, bottom left: Done has no key while a run is going (ruling 13), and SwiftUI exposes no
    /// button object to press. The point is the button's middle: the view's 18-point padding plus half a button.
    func clickDone() async throws {
        let sheet = try #require(window.attachedSheet)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            let event = try #require(NSEvent.mouseEvent(
                with: type, location: NSPoint(x: 43, y: 29), modifierFlags: [],
                timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: sheet.windowNumber, context: nil,
                eventNumber: 0, clickCount: 1, pressure: 1))
            NSApp.postEvent(event, atStart: false)
        }
        try await wait("Done to close the sheet") { self.window.attachedSheet == nil }
        host.item = nil
    }

    /// Offers a key equivalent until a button takes it: SwiftUI installs them a moment after the view that has them
    /// appears, and an offer nothing takes does nothing.
    func offer(_ characters: String, _ code: UInt16, to sheet: NSWindow) async throws {
        let event = try key(characters, code, in: sheet)
        try await wait("a button to take key \(code)") { sheet.performKeyEquivalent(with: event) }
    }

    func key(_ characters: String, _ code: UInt16, in window: NSWindow) throws -> NSEvent {
        try #require(NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [],
                                      timestamp: ProcessInfo.processInfo.systemUptime,
                                      windowNumber: window.windowNumber, context: nil, characters: characters,
                                      charactersIgnoringModifiers: characters, isARepeat: false, keyCode: code))
    }

    func wait(_ what: String, until condition: () -> Bool) async throws {
        for _ in 0..<400 where !condition() { try await Task.sleep(for: .milliseconds(25)) }
        try #require(condition(), "timed out waiting for \(what)")
    }

    func close() {
        stage.gate.open()
        host.item = nil
        if let sheet = window.attachedSheet { window.endSheet(sheet) }
        window.orderOut(nil)
        try? FileManager.default.removeItem(at: audio)
    }
}

private struct SheetPresentation: Identifiable { let id = UUID() }

@MainActor @Observable private final class SheetHost { var item: SheetPresentation? }

private struct SheetParent: View {
    @Bindable var host: SheetHost
    let content: @MainActor (SheetPresentation) -> ForcedAlignerRunView?
    var body: some View {
        Color.clear.frame(width: 900, height: 880).sheet(item: $host.item) { content($0) }
    }
}

/// Deaf to cancellation, as the engine's synchronous forward pass is.
private struct SheetStubStage: PipelineStage, @unchecked Sendable {
    let gate = AlignRunGate()
    var id: String { "align-sheet-stub" }
    var name: String { "Qwen3 Aligner 0.6B" }
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        await gate.wait()
        return .text("text,start,end\nGuten,0.000,0.400\n")
    }
}
