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

    // MARK: - Esc (owner rulings 13 and 41)

    /// Owner ruling 41 (Boo, 2026-11-28): "Esc during a run closes the sheet instead of acting as Stop (ruling 13).
    /// Make Esc while running mean Stop, and test it."
    ///
    /// Owner ruling 44 (Boo, 2026-11-28): "Fix step 4 on fa-f: Esc during a run must go through exactly the path the Stop
    /// button uses, wherever keyboard focus is (button, transcript, nothing): "Stopping…" stays and every control stays
    /// disabled until the step returns (rulings 11 and 14). Find which handler made the sheet look idle. Test with focus
    /// on the Align button, in the transcript, and nowhere."
    ///
    /// "Every control" is read as the ones ruling 14 disables (choosing, dropping and opening files, and Align), which
    /// `canChangeInputs` decides; the transcript box and the language picker stay editable (ruling 16). *Implementer's
    /// call, pending owner confirmation.*
    @Test(arguments: EscRoute.allCases, FocusSpot.allCases)
    func escDuringARunStopsAndStaysStoppingUntilTheStepReturns(route: EscRoute, spot: FocusSpot) async throws {
        // A button takes the focus only with the system's keyboard navigation on; the test reports it on.
        let restore = spot == .alignButton ? try forceKeyboardNavigation() : {}
        defer { restore() }
        let rig = try SheetRig()
        defer { rig.close() }
        let presented = try await rig.present()
        if spot == .alignButton { try rig.focus(presented.sheet, on: spot) }   // as a person who tabbed to Align
        try await rig.pressReturnToAlign(presented)
        if spot != .alignButton { try rig.focus(presented.sheet, on: spot) }
        #expect(rig.focused(presented.sheet) == spot, "the focus is where the case says when Esc is pressed")

        try await rig.pressEscape(presented.sheet, by: route)
        try await rig.expectStopping(presented, for: .seconds(1))

        try await rig.pressEscape(presented.sheet, by: route)
        try await rig.expectStopping(presented, for: .milliseconds(300))

        rig.stage.gate.open()
        try await rig.wait("the step to return") { !presented.model.isRunning }
        #expect(presented.model.runButtonTitle == "Align")
        #expect(presented.model.canChangeInputs)
        #expect(presented.model.result == nil)
        #expect(presented.model.errorText == nil)
        #expect(rig.isShowing(presented.sheet))

        // SwiftUI takes a moment after a run ends to give Done its key back, and an Esc inside that moment can go
        // unanswered (seen once, `.eventQueue` with nothing focused, with no wait here).
        // Owner ruling 45 (Boo, 2026-11-28): "Confirmed: the step log records the model id only, never audio or
        // transcript content. Confirmed: an Esc landing the instant a run ends may do nothing."
        try await Task.sleep(for: .seconds(1))
        try await rig.pressEscape(presented.sheet, by: route)
        try await rig.wait("Esc to close the idle sheet") { !rig.isShowing(presented.sheet) }
    }

    @Test(arguments: EscRoute.allCases, [FocusSpot.transcript, .nowhere])
    func escClosesASheetThatHasRunNothing(route: EscRoute, spot: FocusSpot) async throws {
        let rig = try SheetRig()
        defer { rig.close() }
        let presented = try await rig.present()
        try rig.focus(presented.sheet, on: spot)
        try await rig.pressEscape(presented.sheet, by: route)
        try await rig.wait("Esc to close the idle sheet") { !rig.isShowing(presented.sheet) }
        #expect(!rig.stage.gate.isEntered)
    }
}

/// The two ways Esc reaches a sheet.
nonisolated enum EscRoute: CaseIterable, Sendable {
    /// Offered to the window as a key equivalent: the Stop or Done button answers it.
    case keyEquivalent
    /// Delivered by the app's event dispatch. In a test host that isn't the active app it travels the responder chain
    /// as `cancelOperation:`, which a sheet answers by closing unless its view takes it.
    case eventQueue
}

/// Where the keyboard focus is when Esc is pressed.
nonisolated enum FocusSpot: CaseIterable, Sendable {
    case alignButton, transcript, nowhere
}

// MARK: - The harness

/// Makes AppKit report keyboard navigation as on, in this process, until the returned closure runs. Nothing is written
/// to any preference.
@MainActor private func forceKeyboardNavigation() throws -> () -> Void {
    let method = try #require(class_getInstanceMethod(
        NSApplication.self, #selector(getter: NSApplication.isFullKeyboardAccessEnabled)))
    let on: @convention(block) (AnyObject) -> Bool = { _ in true }
    let original = method_setImplementation(method, imp_implementationWithBlock(on))
    return { method_setImplementation(method, original) }
}

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

    func isShowing(_ sheet: NSWindow) -> Bool { window.attachedSheet === sheet }

    func focus(_ sheet: NSWindow, on spot: FocusSpot) throws {
        let target: NSView? = switch spot {
        case .alignButton: try #require(alignFocusView(in: sheet))
        case .transcript: try #require(views(in: sheet).first { $0 is NSTextView })
        case .nowhere: nil
        }
        #expect(sheet.makeFirstResponder(target))
    }

    /// Which of the three spots holds the focus, if any.
    func focused(_ sheet: NSWindow) -> FocusSpot? {
        if sheet.firstResponder === sheet { return .nowhere }
        if sheet.firstResponder is NSTextView { return .transcript }
        if let align = alignFocusView(in: sheet), sheet.firstResponder === align { return .alignButton }
        return nil
    }

    /// The view SwiftUI focuses for the Align button. It carries no title, so it is found by place: the one button's
    /// focus view on the left, below the language picker and above the footer.
    private func alignFocusView(in sheet: NSWindow) -> NSView? {
        let found = views(in: sheet).filter { view in
            let frame = view.convert(view.bounds, to: nil)
            return String(describing: type(of: view)) == "KeyViewProxy" && frame.minX < 20 && frame.width < 100
                && frame.minY > 100 && frame.minY < 450
        }
        return found.count == 1 ? found[0] : nil
    }

    private func views(in sheet: NSWindow) -> [NSView] {
        func all(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap { all($0) } }
        return sheet.contentView.map { all($0) } ?? []
    }

    /// The sheet stays and shows "Stopping…", with ruling 14's controls disabled, for all of `duration`.
    func expectStopping(_ presented: Presented, for duration: Duration) async throws {
        let model = presented.model
        let until = ContinuousClock.now + duration
        repeat {
            try #require(isShowing(presented.sheet), "the sheet stays")
            try #require(stage.gate.isEntered && model.isRunning && model.isStopping, "Esc is Stop, and it stays Stop")
            try #require(model.runButtonTitle == "Stopping…" && model.progressText == "Stopping…")
            try #require(!model.canChangeInputs, "Choose Audio…, Open…, drop and Align all ask this")
            try await Task.sleep(for: .milliseconds(50))
        } while ContinuousClock.now < until
    }

    /// Sends Esc and gives the sheet time to answer (a closing sheet animates away).
    func pressEscape(_ sheet: NSWindow, by route: EscRoute) async throws {
        switch route {
        case .keyEquivalent: try await offer("\u{1b}", 53, to: sheet)
        case .eventQueue: NSApp.sendEvent(try key("\u{1b}", 53, in: sheet))
        }
        try await Task.sleep(for: .milliseconds(800))
    }

    func key(_ characters: String, _ code: UInt16, in window: NSWindow) throws -> NSEvent {
        try #require(NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [],
                                      timestamp: ProcessInfo.processInfo.systemUptime,
                                      windowNumber: window.windowNumber, context: nil, characters: characters,
                                      charactersIgnoringModifiers: characters, isARepeat: false, keyCode: code))
    }

    /// Returns at the first `true` and asks no more: `offer`'s condition sends a key each time it is asked.
    func wait(_ what: String, until condition: () -> Bool) async throws {
        for _ in 0..<400 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(25))
        }
        try #require(Bool(false), "timed out waiting for \(what)")
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
