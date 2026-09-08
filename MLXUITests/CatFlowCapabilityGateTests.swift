import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R5-6: the `code`/`improvise` door refuses to run under `APPSTORE_BUILD`.
/// The MLXUI test host builds against `Config/AppStore.xcconfig` (which defines
/// `APPSTORE_BUILD`), so `CapabilityGate.isAppStoreBuild` is true here — the tests assert
/// the gate is live and `FlowRunner.canRun` surfaces its refusal verbatim.
struct CatFlowCapabilityGateTests {

    // MARK: - The message, byte-for-byte

    @Test func refusalMessageMatchesPython() {
        let msg = CapabilityGate.appStoreRefusal(flags: ["code"])
        #expect(msg == "this flow declares `code` -- App Store builds refuse it outright (2.4.5 iv); distribute it directly instead.")
        let both = CapabilityGate.appStoreRefusal(flags: ["improvise", "code", "events"])
        #expect(both == "this flow declares `code`, `improvise` -- App Store builds refuse it outright (2.4.5 iv); distribute it directly instead.")
    }

    @Test func noRefusedFlagsIsNil() {
        #expect(CapabilityGate.appStoreRefusal(flags: []) == nil)
        #expect(CapabilityGate.appStoreRefusal(flags: ["network", "events"]) == nil)
    }

    // MARK: - The test host is the App Store tier

    @Test func testHostIsAppStoreBuild() {
        #expect(CapabilityGate.isAppStoreBuild,
                "the MLXUI test host builds with Config/AppStore.xcconfig, so APPSTORE_BUILD must be defined")
    }

    // MARK: - canRun refuses a flagged flow under the store build

    @Test func canRunRefusesCodeFlag() throws {
        let doc = try CatParser.parseForValidation("mlxflow 0.8; code\n1. Read Text  a.txt\n").flowDocument
        let runnability = FlowRunner.canRun(try #require(doc))
        guard case .notRunnable(let reason) = runnability else {
            Issue.record("expected refusal")
            return
        }
        #expect(reason == "this flow declares `code` -- App Store builds refuse it outright (2.4.5 iv); distribute it directly instead.")
    }

    @Test func canRunRefusesImproviseFlag() throws {
        let doc = try CatParser.parseForValidation("mlxflow 0.8; improvise\n1. Read Text  a.txt\n").flowDocument
        let runnability = FlowRunner.canRun(try #require(doc))
        guard case .notRunnable(let reason) = runnability else {
            Issue.record("expected refusal")
            return
        }
        #expect(reason.contains("`improvise`"))
    }

    @Test func canRunAcceptsOrdinaryFlags() throws {
        // `network`/`events` are not refused — only `code`/`improvise` are.
        let doc = try CatParser.parseForValidation("mlxflow 0.8; network\n1. Read Text  a.txt\n2. Save Text  out.txt\n").flowDocument
        let runnability = FlowRunner.canRun(try #require(doc))
        if case .notRunnable(let reason) = runnability {
            #expect(!reason.contains("App Store builds refuse"),
                    "network shouldn't be door-refused, got: \(reason)")
        }
    }

    @Test func flagsParseAndValidateRegardlessOfTheDoor() throws {
        // The door is run-time only — a `code` flow still parses and validates.
        let parsed = try CatParser.parseForValidation("mlxflow 0.8; code\n1. Read Text  a.txt\n")
        #expect(parsed.flags.contains("code"))
        #expect(parsed.flowDocument?.flags.contains(.code) == true)
    }
}
