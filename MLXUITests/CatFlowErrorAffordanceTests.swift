import Testing
import Foundation
@testable import MLXUI

/// Phase FIX, FIX-3 (`RSI/DelegateFixItBacklog.md`) — the rule that stops the next feature
/// reintroducing the pattern this whole phase exists to fix: a message promising an action
/// the app cannot perform. Scans every `ErrorCatalog` template for a phrase that promises a
/// fix and asserts each hit is either served by a registered affordance or explicitly
/// exempted with a one-line reason. This is the deliverable that makes owner ruling 3 real —
/// without it, Phase FIX is two patches wearing one heading.
struct CatFlowErrorAffordanceTests {

    /// Phrases that promise the app will perform (or has performed) a fix. Any template
    /// containing one of these must be accounted for below.
    static let promisingPhrases = [
        "fmt will do this for you",
        "running fmt",
        "mlxflow fmt",
        "mlxflow undo",
    ]

    /// Codes FIX-1's header-flag repair genuinely serves (E103/E109/E118/E120/E604) — the
    /// same table `FlowRowInspectorView`'s button reads from.
    static var headerRepairCodes: Set<String> { Set(FlowHeaderRepair.flagForCode.keys) }

    /// Codes served by an affordance that isn't FIX-1's header repair, each with a one-line
    /// note of what actually serves it. Corrected 2026-09-13 while writing the handoff
    /// (`DelegateFixItBacklog.md` §3.2): the undo door already exists for all three — "Undo
    /// Improvise" (`FlowMaintenanceMenu.swift`) -> `FlowRunSession.undoImprovise` ->
    /// `FencedRunner.undo`. Only the *wording* names a CLI (`mlxflow undo {run} {n}`); rule 11
    /// forbids rewording it, so these are served, not exempt.
    static let otherServedCodes: [String: String] = [
        "R906": "\"Undo Improvise\" (FlowMaintenanceMenu) -> FlowRunSession.undoImprovise -> FencedRunner.undo performs exactly this; only the wording names a CLI.",
        "R907": "same door as R906 — an improvised action's changes undo the same way.",
        "F009": "same door as R906/R907 — the files-changed report undoes through the same menu item.",
    ]

    /// Codes whose promised action is deliberately out of scope for this phase — porting a
    /// real `fmt` (§3.1 of `DelegateFixItBacklog.md`). Each carries the one-line reason.
    static let exemptCodes: [String: String] = [
        "E107": "no renumber-and-re-aim affordance — a real `fmt` is out of scope (§3.1).",
        "E108": "no `--upgrade` path in-app — a real `fmt` is out of scope (§3.1).",
    ]

    private func promisingCodes(in catalog: [String: CatErrorSpec]) -> Set<String> {
        Set(catalog.values.filter { spec in
            Self.promisingPhrases.contains { spec.template.contains($0) }
        }.map(\.code))
    }

    /// FIX-3's actual scan: every code among `catalogs` that promises a fix and is neither
    /// served (`headerRepairCodes`/`otherServedCodes`) nor exempt (`exemptCodes`). Both tests
    /// below call this — not their own copy of the served/exempt decision — so a real drift in
    /// the scan and a broken proof-of-concept fail the same way.
    private func unaccounted(in catalogs: [String: CatErrorSpec]...) -> [String] {
        let codes = catalogs.reduce(into: Set<String>()) { $0.formUnion(promisingCodes(in: $1)) }
        return codes.sorted().filter { code in
            !Self.headerRepairCodes.contains(code)
                && Self.otherServedCodes[code] == nil
                && Self.exemptCodes[code] == nil
        }
    }

    // MARK: - The scan itself

    @Test func everyPromisingTemplateIsServedOrExempt() {
        let found = unaccounted(in: ErrorCatalog.catalogV07, ErrorCatalog.catalogV08)
        #expect(found.isEmpty,
                "these codes promise a fix with no registered affordance and no exemption: \(found)")
    }

    /// FIX-3's own exit criterion: a deliberately unserved promise fails the scan. Calls the
    /// same `unaccounted(in:)` the real scan above does, so this proves the scan catches the
    /// case — not just that this test's own logic would have.
    @Test func aDeliberatelyUnservedPromiseFailsTheScan() {
        var scratch = ErrorCatalog.catalogV08
        scratch["E999"] = CatErrorSpec(code: "E999", name: "scratch", citation: nil,
                                       template: "Row {n} is scratch. (fmt will do this for you.)")
        #expect(unaccounted(in: scratch) == ["E999"])
    }

    @Test func r906r907f009AreRegisteredAsServedNotExempt() {
        for code in ["R906", "R907", "F009"] {
            #expect(Self.otherServedCodes[code] != nil, "\(code) must be registered as served — the undo door already exists")
            #expect(Self.exemptCodes[code] == nil, "\(code) is not a `fmt`-scope exemption")
        }
    }

    @Test func e107AndE108AreTheOnlyFmtScopeExemptions() {
        #expect(Set(Self.exemptCodes.keys) == ["E107", "E108"])
    }

    /// FIX-1's rule 11 tripwire, read from this file's own table: every header-repair code's
    /// template still ends with the promise the repair keeps (see also
    /// `CatFlowHeaderRepairTests.allFiveFlagMessagesStillPromiseTheFixRule11Tripwire`).
    @Test func headerRepairCodesTemplatesStillPromiseTheFix() throws {
        for code in Self.headerRepairCodes {
            guard let spec = ErrorCatalog.catalogV08[code] else { continue }
            #expect(spec.template.hasSuffix("(fmt will do this for you.)"), "\(code) template must not be softened")
        }
    }
}
