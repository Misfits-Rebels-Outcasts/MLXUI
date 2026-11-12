//
//  ApprovalPanel.swift
//  MLXUI — MT4-7-FIX-4 (A2): the approval question as an app-modal window (the AppKit host) and its SwiftUI content.
//  Logic lives in `ApprovalPresenter`; this file is the presentation only.
//

import SwiftUI
import AppKit

/// The approval window. Escape denies.
private final class ApprovalPanel: NSPanel {
    var onCancel: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override func cancelOperation(_ sender: Any?) { onCancel?() }
}

/// Presents the question as a sheet on the TOPMOST window or sheet of the app, so it is frontmost over a Run UI sheet and the
/// window under it cannot be clicked until it is answered.
///
/// It is deliberately NOT `NSApp.runModal`: a modal run loop does not service the main dispatch queue or main-actor tasks
/// (MT4-7-FIX-5: with `runModal` the panel showed no arguments and Allow never armed, because the SwiftUI task that arms it and
/// the layout pass never ran; a real-panel test hung inside the loop). A sheet leaves the run loop alone.
/// With no window to attach to, it is a floating panel above everything.
@MainActor
final class AppKitApprovalHost: ApprovalWindowHosting {
    private var panel: ApprovalPanel?
    private weak var parent: NSWindow?

    func showModal(for pending: PendingApproval, presenter: ApprovalPresenter) {
        closeModal()
        let hosting = NSHostingController(rootView: ApprovalContentView(pending: pending, presenter: presenter))
        let panel = ApprovalPanel(contentViewController: hosting)
        panel.styleMask = [.titled]
        panel.title = "Allow tool call?"
        panel.isReleasedWhenClosed = false
        panel.onCancel = { [weak presenter] in presenter?.deny() }
        self.panel = panel
        // MT4-7-FIX-5: the window is sized to the content, and again once the first layout pass has run. Left alone it kept the
        // height of the first (empty ScrollView) layout, so the arguments and the buttons sat below the fold.
        Self.fit(panel, to: hosting)
        DispatchQueue.main.async { [weak panel, weak hosting] in
            guard let panel, let hosting else { return }
            Self.fit(panel, to: hosting)
        }
        NSApp.activate()
        if let top = Self.topmostWindow() {
            parent = top
            top.beginSheet(panel, completionHandler: nil)
        } else {
            panel.level = .modalPanel
            panel.center()
            panel.makeKeyAndOrderFront(nil)
        }
    }

    static func fit(_ panel: NSWindow, to hosting: NSHostingController<ApprovalContentView>) {
        hosting.view.layoutSubtreeIfNeeded()
        let size = hosting.view.fittingSize
        guard size.width > 0, size.height > 0 else { return }
        panel.setContentSize(size)
    }

    /// The end of the deepest sheet chain among the visible windows (a Run UI sheet, or a sheet on a sheet), else the key or main
    /// window: the approval goes over whatever is frontmost, so nothing under it can be clicked.
    static func topmostWindow() -> NSWindow? {
        func end(of window: NSWindow) -> (window: NSWindow, depth: Int) {
            var top = window, depth = 0
            while let sheet = top.attachedSheet { top = sheet; depth += 1 }
            return (top, depth)
        }
        let visible = NSApp.windows.filter { $0.isVisible && !($0 is ApprovalPanel) }
        let preferred = [NSApp.keyWindow, NSApp.mainWindow].compactMap { $0 } + visible
        var best: (window: NSWindow, depth: Int)?
        for window in preferred {
            let candidate = end(of: window)
            if let current = best, current.depth >= candidate.depth { continue }
            best = candidate
        }
        return best?.window
    }

    func closeModal() {
        guard let panel else { return }
        self.panel = nil
        if let sheetParent = panel.sheetParent ?? parent, sheetParent.attachedSheet === panel { sheetParent.endSheet(panel) }
        panel.orderOut(nil)
        parent = nil
    }
}

/// The question: the tool, what it was called with (the same sections the sheet showed), Deny and Allow.
/// Deny is the default (Return) and the escape action; Allow has no shortcut and is disabled for the first moments.
struct ApprovalContentView: View {
    let pending: PendingApproval
    let presenter: ApprovalPresenter

    var body: some View {
        let sections = ApprovalPreview.sections(toolName: pending.toolName, arguments: pending.arguments)
        VStack(alignment: .leading, spacing: 14) {
            Label("Allow tool call?", systemImage: "wrench.and.screwdriver")
                .font(.headline)
            VStack(alignment: .leading, spacing: 4) {
                Text(pending.toolName)
                    .font(.callout.monospaced().weight(.semibold))
                Text(pending.toolDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if sections.isEmpty {
                Text("The model wants to run this tool. Allow it to proceed?")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("The model wants to run this tool with the arguments below.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(sections.indices, id: \.self) { index in
                            section(sections[index])
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 320)
            }
            HStack {
                Spacer()
                Button("Deny") { presenter.deny() }
                    .keyboardShortcut(.defaultAction)
                // Re-evaluated on a timeline against the presenter's own clock (the same flag `allow()` honours), so the button
                // cannot stay greyed because some other task did not run (MT4-7-FIX-5).
                TimelineView(.periodic(from: .now, by: 0.1)) { context in
                    Button("Allow") { presenter.allow() }
                        .buttonStyle(.borderedProminent)
                        .disabled(!presenter.isAllowEnabled(at: context.date))
                }
            }
        }
        .padding(20)
        .frame(width: sections.isEmpty ? 360 : 560)
    }

    /// One labelled argument block: monospaced and selectable, red for a removal and green for an addition (D10).
    private func section(_ section: ApprovalPreview.Section) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(section.title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(section.body)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(tint(for: section.style))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func tint(for style: ApprovalPreview.Style) -> Color {
        switch style {
        case .removed: return .red
        case .added: return .green
        case .plain, .code: return .primary
        }
    }
}
