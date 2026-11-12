//
//  ApprovalPanel.swift
//  MLXUI — MT4-7-FIX-4 (A2): the approval question as an app-modal window (the AppKit host) and its SwiftUI content.
//  Logic lives in `ApprovalPresenter`; this file is the presentation only.
//

import SwiftUI
import AppKit

/// App-modal, so it is frontmost over any sheet and blocks the windows (and sheets) under it. Escape denies.
private final class ApprovalPanel: NSPanel {
    var onCancel: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override func cancelOperation(_ sender: Any?) { onCancel?() }
}

@MainActor
final class AppKitApprovalHost: ApprovalWindowHosting {
    private var panel: ApprovalPanel?

    func showModal(for pending: PendingApproval, presenter: ApprovalPresenter) {
        closeModal()
        let panel = ApprovalPanel(contentViewController: NSHostingController(
            rootView: ApprovalContentView(pending: pending, presenter: presenter)))
        panel.styleMask = [.titled]
        panel.title = "Allow tool call?"
        panel.isReleasedWhenClosed = false
        panel.onCancel = { [weak presenter] in presenter?.deny() }
        panel.center()
        self.panel = panel
        NSApp.activate()
        // Run the modal loop after the SwiftUI update that asked for it has finished.
        Task { @MainActor [weak self] in
            guard let self, self.panel === panel else { return }
            NSApp.runModal(for: panel)
            panel.orderOut(nil)
            if self.panel === panel { self.panel = nil }
        }
    }

    func closeModal() {
        guard let panel else { return }
        self.panel = nil
        if NSApp.modalWindow === panel { NSApp.abortModal() }
        panel.orderOut(nil)
    }
}

/// The question: the tool, what it was called with (the same sections the sheet showed), Deny and Allow.
/// Deny is the default (Return) and the escape action; Allow has no shortcut and is disabled for the first moments.
struct ApprovalContentView: View {
    let pending: PendingApproval
    let presenter: ApprovalPresenter
    @State private var armed = false

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
                Button("Allow") { presenter.allow() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!armed)
            }
        }
        .padding(20)
        .frame(width: sections.isEmpty ? 360 : 560)
        .task {
            let wait = presenter.secondsUntilArmed()
            if wait > 0 { try? await Task.sleep(for: .seconds(wait)) }
            armed = presenter.isAllowEnabled()
        }
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
