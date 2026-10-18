import Foundation

/// SC-6/SC-7 (gates F, G, H — **implementer's call, pending owner confirmation**: none of the three is ruled yet).
///
/// **Dedicated server mode** is for a Mac whose job is serving. While on (and the Local Server enabled):
/// the budget is raised to `MemoryBudget.dedicatedFraction` of RAM, every reachable served model is preloaded and kept
/// ready, and nothing else on the Mac may load a model (`RunAdmission`). The setting is the only thing this stream
/// persists (backlog rule 8); it is **off by default**.
///
/// Like `LocalServerOption`, the store is read directly so a reader on any actor sees the same answer, with no
/// controller to hand. Tests swap in an isolated suite.
nonisolated enum DedicatedMode {
    nonisolated(unsafe) static var store: UserDefaults = .standard
    static let key = "localServerDedicatedMode"

    /// The stored switch alone — what the toggle shows. Use `isActive` to ask whether the mode is in force.
    static func isOn(in defaults: UserDefaults? = nil) -> Bool {
        (defaults ?? store).bool(forKey: key)
    }

    static func set(_ on: Bool, in defaults: UserDefaults? = nil) {
        (defaults ?? store).set(on, forKey: key)
    }

    /// In force: the switch is on **and** serving is enabled (rule 7 — with the gate closed the app behaves as if
    /// serving didn't exist, so nothing is blocked and no budget is raised).
    static var isActive: Bool { LocalServerGate.isEnabled && isOn() }
}

/// SC-7 (gate F): the one question every model-loading surface asks first. A refusal carries gate F's sentence.
/// Browse, install and delete never ask. The server's own loads never ask (they are not a "surface").
nonisolated enum RunAdmission {
    /// Where a load comes from — for the tests; the sentence is the same everywhere.
    enum Surface: String, Sendable {
        case chat, runWindow, flowRow
    }

    nonisolated struct Refusal: Error, LocalizedError, Equatable, Sendable {
        let surface: Surface
        var sentence: String { RunAdmission.sentence }
        var errorDescription: String? { sentence }
    }

    static let sentence = "This Mac is in Dedicated server mode. Turn it off on the Local Server screen to run models here."

    /// `nil` = go ahead.
    static func canLoadModel(surface: Surface) -> Refusal? {
        DedicatedMode.isActive ? Refusal(surface: surface) : nil
    }
}
