import Testing
import AppKit
import SwiftUI
import MLXLMCommon
@testable import MLXUI

/// MT4-7-FIX-5: the REAL approval window (`AppKitApprovalHost`, a sheet on the topmost window/sheet, the SwiftUI content), no fake host.
/// Opt-in: it opens a window and runs a modal loop, so it needs `TEST_RUNNER_MLXUI_TEST_REAL_PANEL=1`. It prints
/// `APPROVAL-PANEL ...` lines (what the panel shows, whether Allow is enabled before and after the 0.7 s delay) and writes PNGs of
/// the panel to `TEST_RUNNER_MLXUI_TEST_REAL_PANEL_DIR` when set.
@MainActor @Suite(.serialized) struct ApprovalPanelRealTests {

    private static let enabled = ProcessInfo.processInfo.environment["MLXUI_TEST_REAL_PANEL"] == "1"

    private static func walk(_ element: Any, depth: Int = 0, into lines: inout [String], buttons: inout [(String, Bool)]) {
        guard depth < 40, let node = element as? NSAccessibilityProtocol else { return }
        let role = node.accessibilityRole()?.rawValue ?? ""
        let label = node.accessibilityLabel() ?? ""
        let title = node.accessibilityTitle() ?? ""
        let value = (node.accessibilityValue() as? String) ?? ""
        if !label.isEmpty || !title.isEmpty || !value.isEmpty { lines.append("\(role) title=\(title) label=\(label) value=\(value)") }
        if role == NSAccessibility.Role.button.rawValue { buttons.append((title.isEmpty ? label : title, node.isAccessibilityEnabled())) }
        for child in node.accessibilityChildren() ?? [] { walk(child, depth: depth + 1, into: &lines, buttons: &buttons) }
        if let view = element as? NSView, (node.accessibilityChildren() ?? []).isEmpty {
            for sub in view.subviews { walk(sub, depth: depth + 1, into: &lines, buttons: &buttons) }
        }
    }

    private static func snapshot(_ window: NSWindow, name: String) {
        guard let dir = ProcessInfo.processInfo.environment["MLXUI_TEST_REAL_PANEL_DIR"], let view = window.contentView else { return }
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name))
    }

    @Test(.enabled(if: ApprovalPanelRealTests.enabled, "needs MLXUI_TEST_REAL_PANEL=1"))
    func theRealPanelShowsTheArgumentsAndAllowWorksAfterTheDelay() async throws {
        // A window with a "Run UI" sheet up, as in smoke step 27: the approval must stack over the sheet.
        let appWindow = NSWindow(contentRect: NSRect(x: 200, y: 200, width: 700, height: 500), styleMask: [.titled, .closable],
                                 backing: .buffered, defer: false)
        appWindow.isReleasedWhenClosed = false
        appWindow.makeKeyAndOrderFront(nil)
        let runSheet = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 400, height: 240), styleMask: [.titled], backing: .buffered, defer: false)
        runSheet.isReleasedWhenClosed = false
        runSheet.title = "Run UI"
        appWindow.beginSheet(runSheet, completionHandler: nil)
        defer { appWindow.endSheet(runSheet); appWindow.orderOut(nil) }
        let host = AppKitApprovalHost()
        let presenter = ApprovalPresenter(host: host)
        let answers = Answers()
        presenter.resolve = { _, ok in answers.values.append(ok) }
        let pending = PendingApproval(
            toolName: "run_shell", toolDescription: "Run a shell command.",
            arguments: ["command": .string("ls ~/Documents/mt-smoke"), "working_directory": .string("/Users/test")], respond: { _ in })
        let report = Report()
        func capture(_ tag: String) {
            report.lines.append("APPROVAL-PANEL \(tag): approval stacked over the Run sheet = \(runSheet.attachedSheet?.title ?? "nil")")
            guard let window = NSApp.windows.first(where: { $0.title == "Allow tool call?" }), let content = window.contentView else {
                report.lines.append("APPROVAL-PANEL \(tag): no panel window"); return
            }
            var lines: [String] = [], buttons: [(String, Bool)] = []
            Self.walk(content, into: &lines, buttons: &buttons)
            report.lines.append("APPROVAL-PANEL \(tag): frame=\(window.frame.size) fitting=\(content.fittingSize) isVisible=\(window.isVisible) presenterAllow=\(presenter.isAllowEnabled())")
            report.lines.append("APPROVAL-PANEL \(tag): contentLayoutHeight=\(window.contentLayoutRect.height) fits=\(window.contentLayoutRect.height >= content.fittingSize.height - 1)")
            report.fits.append(window.contentLayoutRect.height >= content.fittingSize.height - 1)
            report.lines.append("APPROVAL-PANEL \(tag): buttons=\(buttons)")
            for line in lines { report.lines.append("APPROVAL-PANEL \(tag): \(line)") }
            Self.snapshot(window, name: "panel-\(tag).png")
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { capture("t0.25") }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { capture("t1.5") }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { capture("t3.0"); _ = presenter.allow() }
        DispatchQueue.main.asyncAfter(deadline: .now() + 6.0) { if presenter.shownID != nil { report.lines.append("APPROVAL-PANEL watchdog: still up, denying"); presenter.deny() } }
        presenter.sync(pending: pending, holdsLease: true)
        try await Task.sleep(for: .seconds(8))
        for line in report.lines { print(line) }
        print("APPROVAL-PANEL answers=\(answers.values)")
        #expect(!report.fits.isEmpty && !report.fits.contains(false), "the window is shorter than its content")
        #expect(report.lines.contains { $0.contains("value=ls ~/Documents/mt-smoke") } && report.lines.contains { $0.contains("value=/Users/test") })
        #expect(answers.values == [true])
    }

    private final class Answers { var values: [Bool] = [] }
    private final class Report { var lines: [String] = []; var fits: [Bool] = [] }
}
