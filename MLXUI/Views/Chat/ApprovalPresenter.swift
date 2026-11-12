//
//  ApprovalPresenter.swift
//  MLXUI — MT4-7-FIX-4 (A2): where the tool-approval question appears, and when Allow is armed.
//
//  The approval used to be a SwiftUI `.sheet(item:)` on `ChatView`. A window shows one sheet at a time, so with a Run UI sheet
//  open (a `.sheet` on the root window) the approval was queued behind it: invisible, then popping up under the pointer the
//  moment the Run sheet closed. A click meant for the Run sheet could approve a `run_shell`. The question is now an app-modal
//  window (`AppKitApprovalHost`), frontmost over any sheet, and Allow stays disabled for a moment after it appears.
//
//  This file holds the part a unit test can cover: which question is on screen, when the window is shown or closed, the
//  click-through guard's clock, and that a question is answered once. The AppKit modality itself is not unit-tested.
//

import Foundation

/// The click-through guard: Allow is disabled for `delay` seconds after the window appears.
struct ApprovalArming: Equatable, Sendable {
    static let defaultDelay: TimeInterval = 0.7
    let shownAt: Date
    let delay: TimeInterval

    func isArmed(at now: Date) -> Bool { now.timeIntervalSince(shownAt) >= delay }
    func remaining(at now: Date) -> TimeInterval { max(0, delay - now.timeIntervalSince(shownAt)) }
}

/// What the presenter needs from the window system (`AppKitApprovalHost` in the app, a fake in tests).
@MainActor
protocol ApprovalWindowHosting: AnyObject {
    /// Put the question in an app-modal window above every sheet.
    func showModal(for pending: PendingApproval, presenter: ApprovalPresenter)
    /// Take it down (no-op when none is up).
    func closeModal()
}

@MainActor
final class ApprovalPresenter {
    private let host: any ApprovalWindowHosting
    private let now: () -> Date
    private let delay: TimeInterval
    /// Answers the runner: called at most once per question, only for the question that is on screen.
    var resolve: (_ id: UUID, _ approved: Bool) -> Void = { _, _ in }

    /// The question on screen now.
    private(set) var shownID: UUID?
    private(set) var arming: ApprovalArming?

    init(host: any ApprovalWindowHosting, delay: TimeInterval = ApprovalArming.defaultDelay, now: @escaping () -> Date = { Date() }) {
        self.host = host
        self.delay = delay
        self.now = now
    }

    /// Bring the window in line with the runner: show a new question once, close when it is gone or this surface no longer
    /// holds the runner. Safe to call on every change.
    func sync(pending: PendingApproval?, holdsLease: Bool) {
        guard holdsLease, let pending else {
            if shownID != nil { closeShown() }
            return
        }
        if shownID == pending.id { return }
        if shownID != nil { closeShown() }                      // a newer question replaces the old one
        shownID = pending.id
        arming = ApprovalArming(shownAt: now(), delay: delay)
        host.showModal(for: pending, presenter: self)
    }

    /// Whether the Allow button is live now.
    func isAllowEnabled() -> Bool { arming?.isArmed(at: now()) ?? false }

    func secondsUntilArmed() -> TimeInterval { arming?.remaining(at: now()) ?? 0 }

    /// Allow. Ignored until armed (a click that was meant for something under the window) and when nothing is on screen.
    @discardableResult
    func allow() -> Bool {
        guard let id = shownID, isAllowEnabled() else { return false }
        answer(id, approved: true)
        return true
    }

    /// Deny (also Return and Escape): always available.
    func deny() {
        guard let id = shownID else { return }
        answer(id, approved: false)
    }

    /// The view that showed the question went away: deny it (nothing runs), so the turn is not left waiting on a hidden window.
    func dismissDenying() { deny() }

    private func answer(_ id: UUID, approved: Bool) {
        closeShown()
        resolve(id, approved)
    }

    private func closeShown() {
        shownID = nil
        arming = nil
        host.closeModal()
    }
}
