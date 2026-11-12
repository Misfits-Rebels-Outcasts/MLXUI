import Testing
import Foundation
@testable import MLXUI

/// MT4-7-FIX-4 (A2): which question is on screen and when Allow is armed. The AppKit app-modal window itself (frontmost over a
/// sheet, blocking the sheet's controls) is not unit-testable and is covered by the owner's smoke, not here.
@MainActor @Suite struct ApprovalPresenterTests {

    private final class FakeHost: ApprovalWindowHosting {
        private(set) var shown: [UUID] = []
        private(set) var closes = 0
        func showModal(for pending: PendingApproval, presenter: ApprovalPresenter) { shown.append(pending.id) }
        func closeModal() { closes += 1 }
    }

    private final class Clock: @unchecked Sendable {
        var time = Date(timeIntervalSince1970: 1_000)
        func advance(_ seconds: TimeInterval) { time = time.addingTimeInterval(seconds) }
    }

    private func pending() -> PendingApproval {
        PendingApproval(toolName: "run_shell", toolDescription: "Run a command", arguments: [:], respond: { _ in })
    }

    private func rig() -> (ApprovalPresenter, FakeHost, Clock, answers: Box) {
        let host = FakeHost(), clock = Clock(), box = Box()
        let presenter = ApprovalPresenter(host: host, now: { clock.time })
        presenter.resolve = { id, ok in box.answers.append((id, ok)) }
        return (presenter, host, clock, box)
    }

    private final class Box { var answers: [(UUID, Bool)] = [] }

    @Test func aQuestionIsShownOnceInAnAppModalWindowAndNotAgainOnEveryChange() {
        let (p, host, _, _) = rig()
        let q = pending()
        p.sync(pending: q, holdsLease: true)
        p.sync(pending: q, holdsLease: true)
        p.sync(pending: q, holdsLease: true)
        #expect(host.shown == [q.id] && p.shownID == q.id)
    }

    @Test func aSurfaceWithoutTheLeaseShowsNothingAndWithdrawsWhenItLosesIt() {
        let (p, host, _, _) = rig()
        let q = pending()
        p.sync(pending: q, holdsLease: false)
        #expect(host.shown.isEmpty && p.shownID == nil)
        p.sync(pending: q, holdsLease: true)
        p.sync(pending: q, holdsLease: false)
        #expect(p.shownID == nil && host.closes == 1)
    }

    @Test func theWindowClosesWhenTheRunnerDropsTheQuestion() {
        let (p, host, _, answers) = rig()
        p.sync(pending: pending(), holdsLease: true)
        p.sync(pending: nil, holdsLease: true)                      // Stop or a conversation switch released it
        #expect(p.shownID == nil && host.closes == 1 && answers.answers.isEmpty)
    }

    @Test func aNewerQuestionReplacesTheOldOne() {
        let (p, host, _, _) = rig()
        let first = pending(), second = pending()
        p.sync(pending: first, holdsLease: true)
        p.sync(pending: second, holdsLease: true)
        #expect(host.shown == [first.id, second.id] && p.shownID == second.id)
    }

    @Test func allowIsDisabledForTheFirstSeventyHundredthsOfASecondThenArms() {
        let (p, _, clock, answers) = rig()
        let q = pending()
        p.sync(pending: q, holdsLease: true)
        #expect(!p.isAllowEnabled())
        #expect(abs(p.secondsUntilArmed() - 0.7) < 0.001)
        #expect(p.allow() == false)                                   // a click that was meant for the sheet under it
        clock.advance(0.69)
        #expect(!p.isAllowEnabled() && p.allow() == false)
        #expect(answers.answers.isEmpty && p.shownID == q.id)
        clock.advance(0.02)
        #expect(p.isAllowEnabled())
        #expect(p.allow() == true)
        #expect(answers.answers.count == 1 && answers.answers[0].0 == q.id && answers.answers[0].1 == true)
        #expect(p.shownID == nil)
    }

    @Test func denyIsAlwaysAvailableAndAnswersOnce() {
        let (p, host, _, answers) = rig()
        let q = pending()
        p.sync(pending: q, holdsLease: true)
        p.deny()
        p.deny()
        #expect(answers.answers.count == 1 && answers.answers[0].1 == false)
        #expect(host.closes == 1 && p.shownID == nil)
        #expect(p.allow() == false)                                   // nothing left to allow
    }

    @Test func aQuestionShownAfterAnotherGetsItsOwnArmingClock() {
        let (p, _, clock, _) = rig()
        p.sync(pending: pending(), holdsLease: true)
        clock.advance(5)
        #expect(p.isAllowEnabled())
        p.sync(pending: pending(), holdsLease: true)
        #expect(!p.isAllowEnabled())
    }

    @Test func dismissingTheHostViewDeniesTheOpenQuestion() {
        let (p, _, _, answers) = rig()
        p.sync(pending: pending(), holdsLease: true)
        p.dismissDenying()
        #expect(answers.answers.map(\.1) == [false])
    }

    @Test func armingMath() {
        let a = ApprovalArming(shownAt: Date(timeIntervalSince1970: 0), delay: 0.7)
        #expect(!a.isArmed(at: Date(timeIntervalSince1970: 0.5)) && a.isArmed(at: Date(timeIntervalSince1970: 0.7)))
        #expect(a.remaining(at: Date(timeIntervalSince1970: 5)) == 0)
    }
}
