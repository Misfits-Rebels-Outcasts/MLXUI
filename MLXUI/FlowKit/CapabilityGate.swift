import Foundation

/// The App Store distribution tier's capability gate — ported from
/// `catflow-mlx/src/catflow/core/capabilities.py` (CFM-R5-6).
///
/// `APP_STORE_REFUSED_FLAGS` mirrors the Python's frozenset exactly: a flow whose header
/// declares `code` or `improvise` is refused by the App Store build outright
/// (2.4.5(iv): no installing "code or resources to add functionality"). Refusal is a
/// property of the **channel**, never of the runtime — the Direct build accepts these
/// flags (its fenced execution is a later phase). The flags always parse and validate;
/// only *running* is refused here.
nonisolated enum CapabilityGate {

    /// Whether this binary is the App Store tier. Defined by `Config/AppStore.xcconfig`'s
    /// `APPSTORE_BUILD`; the Direct target uses `DIRECT_BUILD` instead and never sees it.
    static var isAppStoreBuild: Bool {
        #if APPSTORE_BUILD
        return true
        #else
        return false
        #endif
    }

    /// Mirrors `capabilities.py::APP_STORE_REFUSED_FLAGS`.
    static let appStoreRefusedFlags: Set<String> = ["code", "improvise"]

    /// A refusal message if *flow*'s header declares a flag the App Store tier refuses,
    /// else `nil`. Byte-for-byte `capabilities.py::app_store_refusal`.
    static func appStoreRefusal(flags: [String]) -> String? {
        let refused = appStoreRefusedFlags.filter { flags.contains($0) }.sorted()
        if refused.isEmpty { return nil }
        let listed = refused.map { "`\($0)`" }.joined(separator: ", ")
        return "this flow declares \(listed) -- App Store builds refuse it outright (2.4.5 iv); distribute it directly instead."
    }
}
