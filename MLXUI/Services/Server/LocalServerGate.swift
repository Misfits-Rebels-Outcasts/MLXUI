import Foundation

/// The **one** question every Local Server surface asks (owner ruling R14, design §5a): "is the
/// Local Server available in this build?" It answers `!AppState.hideLocalServer`. No surface —
/// not the server, not a Serve toggle, not the Connect sheet, not the Settings pane, not the
/// stay-running decision — reads `AppState.hideLocalServer` directly (backlog rule 14).
///
/// `overrideForTesting` is the test seam so the suite covers both states without touching the
/// committed flag. Production code never sets it.
nonisolated enum LocalServerGate {
    nonisolated(unsafe) static var overrideForTesting: Bool?

    static var isAvailable: Bool {
        overrideForTesting ?? !AppState.hideLocalServer
    }
}
