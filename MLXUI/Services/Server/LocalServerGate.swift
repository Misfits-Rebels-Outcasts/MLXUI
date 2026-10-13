import Foundation

/// The questions every Local Server surface asks (owner ruling R14, design §5a; gate J, ruled 2026-10-08).
///
/// - `isAvailable` — "is the Local Server in this build?" (`!AppState.hideLocalServer`). **Only the Settings →
///   Local Server pane asks it**, so the user can find the "Enable Local Server" option.
/// - `isEnabled` — available **and** the user turned the option on. Every other serving surface asks this: the
///   per-model Serve toggle, the Connect sheet, auto-start, stay-running, preload, pinning, the listener.
///
/// No surface reads `AppState.hideLocalServer` or the stored option directly (backlog rule 14).
///
/// `overrideForTesting` is the test seam so the suite covers both states without touching the committed flag.
/// Production code never sets any of these. For tests, `overrideForTesting = true` means "available **and**
/// enabled" (what every pre-SC-4a test assumed); `optionOverrideForTesting` pins the option on its own, and
/// `usesStoredOptionInTests` makes the gate read `LocalServerOption.store` the way production does.
nonisolated enum LocalServerGate {
    nonisolated(unsafe) static var overrideForTesting: Bool?
    nonisolated(unsafe) static var optionOverrideForTesting: Bool?
    nonisolated(unsafe) static var usesStoredOptionInTests = false

    static var isAvailable: Bool {
        overrideForTesting ?? !AppState.hideLocalServer
    }

    static var isEnabled: Bool {
        guard isAvailable else { return false }
        if let forced = optionOverrideForTesting { return forced }
        if overrideForTesting != nil, !usesStoredOptionInTests { return true }
        return LocalServerOption.isOn()
    }
}

/// The persisted "Enable Local Server" option (SC-4a, gate J). **Off by default.**
nonisolated enum LocalServerOption {
    /// Where the option lives. Tests swap in an isolated suite.
    nonisolated(unsafe) static var store: UserDefaults = .standard

    static let optionKey = "localServerOptionEnabled"
    static let migratedKey = "localServerOptionMigrated"
    // Keys written by `LocalServerController` / `ServeSettings` in 0.20/0.21, which the migration reads.
    private static let masterSwitchKey = "localServerEnabled"
    private static let connectShownKey = "localServerConnectShown"

    /// Whether the option is on. Runs the one-time migration first, so no reader sees the pre-migration state.
    static func isOn(in defaults: UserDefaults? = nil) -> Bool {
        let defaults = defaults ?? store
        migrateIfNeeded(in: defaults)
        return defaults.bool(forKey: optionKey)
    }

    static func set(_ on: Bool, in defaults: UserDefaults? = nil) {
        let defaults = defaults ?? store
        migrateIfNeeded(in: defaults)
        defaults.set(on, forKey: optionKey)
    }

    /// One time only (recorded in `migratedKey`): the option starts **on** for anyone who has used serving —
    /// a non-empty served list, a master switch the user stored as on, or the Connect sheet ever shown — so 0.20/0.21
    /// users lose nothing. The master switch counts only when it was *stored*: an unset switch reads as on in the
    /// controller, which says nothing about this user. Everyone else starts off.
    static func migrateIfNeeded(in defaults: UserDefaults) {
        guard !defaults.bool(forKey: migratedKey) else { return }
        defer { defaults.set(true, forKey: migratedKey) }
        guard defaults.object(forKey: optionKey) == nil else { return }
        let usedServing = !ServeSettings.load(from: defaults).servedModelIDs.isEmpty
            || (defaults.object(forKey: masterSwitchKey) as? Bool) == true
            || defaults.bool(forKey: connectShownKey)
        if usedServing { defaults.set(true, forKey: optionKey) }
    }
}
