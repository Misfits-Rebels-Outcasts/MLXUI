import Foundation

/// FIX-1 (`RSI/DelegateFixItBacklog.md`) — the one header-flag repair serving all five
/// "a capability flag is missing from line one" errors (E103/E109/E118/E120/E604). Each of
/// those five templates ends "(fmt will do this for you.)"; MLXUI has no `fmt`, so this is a
/// single-flag header edit through the existing document model instead — never a formatter
/// (E107's renumber, E108's `--upgrade` stay out of scope, per §3 of the backlog).
nonisolated enum FlowHeaderRepair {

    /// `FlowIssue.code` -> the flag whose absence raises it. One table, one test per entry.
    /// E103 has no raise site in MLXUI today (no `FlowIssue` with this code is ever produced —
    /// §4 trap 4 of the backlog forbids adding one here), but it costs one line to map anyway:
    /// it is the next E120, and FIX-3's affordance scan reads this table too.
    static let flagForCode: [String: CapabilityFlag] = [
        "E103": .network,
        "E109": .improvise,
        "E118": .code,
        "E120": .offdevice,
        "E604": .events,
    ]

    /// The flag a repair would add for `code`, or nil when `code` isn't one of the five.
    static func flag(forCode code: String) -> CapabilityFlag? {
        flagForCode[code]
    }

    /// Add `flag` to `doc`'s header. Appends to both `flags` and `flagsOrder` — never
    /// re-sorts `flagsOrder`, which `CatSerializer` renders in source order — and touches
    /// nothing else, so the byte diff of a repair is confined to line one. Idempotent: a
    /// second application (the flag already present) returns `doc` unchanged.
    static func apply(_ flag: CapabilityFlag, to doc: FlowDocument) -> FlowDocument {
        guard !doc.flags.contains(flag) else { return doc }
        var out = doc
        out.flags.insert(flag)
        out.flagsOrder.append(flag)
        return out
    }
}
