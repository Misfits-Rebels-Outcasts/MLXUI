import Testing
@testable import MLXUI

/// FA-R3 — the join the Japanese parity comparison uses. Test-only: the engine and the app keep the library's split.
///
/// Owner ruling 8 (Boo, 2026-11-27): "Option A. Use the 'Japanese' golden. In the parity test only, for Japanese only, join
/// consecutive Swift units whose concatenated text exactly equals one reference unit, then apply ±80 ms to the joined span.
/// If the join doesn't line up exactly, the test fails. The engine and the app's output stay as the library splits them
/// (issue #13 decision 4: don't re-split). Add a test that pins Swift's current Japanese units (こんにちは, 世, 界), so a
/// future upstream splitter shows up as a deliberate change. Note the difference in the fixture README, and add it to the
/// same upstream mlx-audio-swift report as the Chinese drift (ruling 2). Chinese stays as ruling 2: exact units, ±160 ms."
nonisolated enum ForcedAlignerUnitJoin {
    enum Mismatch: Error, Equatable, CustomStringConvertible {
        /// The next Swift units can't make `reference` exactly; `joined` is what had been put together when it failed.
        case doesNotLineUp(reference: String, joined: String)
        /// Every reference unit was built and these Swift units remain.
        case leftover([String])

        var description: String {
            switch self {
            case let .doesNotLineUp(reference, joined):
                return "the reference unit '\(reference)' can't be built exactly from consecutive Swift units (got '\(joined)')"
            case let .leftover(texts):
                return "Swift units are left over after the last reference unit: \(texts)"
            }
        }
    }

    /// One unit per reference text, in order: its text, the first joined unit's start and the last joined unit's end.
    static func join(_ units: [AlignedUnit], toMatch reference: [String]) throws -> [AlignedUnit] {
        var joined: [AlignedUnit] = []
        var next = 0
        for target in reference {
            let first = next
            var text = ""
            var end = 0.0
            while text != target, next < units.count {
                text += units[next].text
                end = units[next].end
                next += 1
                guard target.hasPrefix(text) else { throw Mismatch.doesNotLineUp(reference: target, joined: text) }
            }
            guard text == target, next > first else { throw Mismatch.doesNotLineUp(reference: target, joined: text) }
            joined.append(AlignedUnit(text: target, start: units[first].start, end: end))
        }
        guard next == units.count else { throw Mismatch.leftover(units[next...].map(\.text)) }
        return joined
    }
}

/// The helper's own contract, no weights needed.
struct ForcedAlignerUnitJoinTests {

    private func units(_ pairs: [(String, Double, Double)]) -> [AlignedUnit] {
        pairs.map { AlignedUnit(text: $0.0, start: $0.1, end: $0.2) }
    }

    @Test func consecutiveUnitsThatMakeOneReferenceUnitAreJoinedOverTheWholeSpan() throws {
        let swift = units([("こんにちは", 0.08, 0.8), ("世", 0.96, 1.12), ("界", 1.12, 1.44)])
        let joined = try ForcedAlignerUnitJoin.join(swift, toMatch: ["こんにちは", "世界"])
        #expect(joined == units([("こんにちは", 0.08, 0.8), ("世界", 0.96, 1.44)]))
    }

    @Test func unitsThatAlreadyMatchAreUnchanged() throws {
        let swift = units([("Guten", 0, 0.4), ("Morgen", 0.4, 0.9)])
        #expect(try ForcedAlignerUnitJoin.join(swift, toMatch: ["Guten", "Morgen"]) == swift)
    }

    @Test func aUnitThatCrossesAReferenceBoundaryFails() {
        let swift = units([("Stra", 0, 0.2), ("ßenbahn", 0.2, 0.9)])
        #expect(throws: ForcedAlignerUnitJoin.Mismatch.doesNotLineUp(reference: "Straßen", joined: "Straßenbahn")) {
            _ = try ForcedAlignerUnitJoin.join(swift, toMatch: ["Straßen", "bahn"])
        }
    }

    @Test func differentTextFails() {
        let swift = units([("Bonsoir", 0, 0.5)])
        #expect(throws: ForcedAlignerUnitJoin.Mismatch.doesNotLineUp(reference: "Bonjour", joined: "Bonsoir")) {
            _ = try ForcedAlignerUnitJoin.join(swift, toMatch: ["Bonjour"])
        }
    }

    @Test func runningOutOfUnitsFails() {
        let swift = units([("Straßen", 0, 0.5)])
        #expect(throws: ForcedAlignerUnitJoin.Mismatch.doesNotLineUp(reference: "Straßenbahn", joined: "Straßen")) {
            _ = try ForcedAlignerUnitJoin.join(swift, toMatch: ["Straßenbahn"])
        }
        #expect(throws: ForcedAlignerUnitJoin.Mismatch.doesNotLineUp(reference: "hält", joined: "")) {
            _ = try ForcedAlignerUnitJoin.join(swift, toMatch: ["Straßen", "hält"])
        }
    }

    @Test func leftoverUnitsFail() {
        let swift = units([("Guten", 0, 0.4), ("Morgen", 0.4, 0.9), ("zusammen", 0.9, 1.5)])
        #expect(throws: ForcedAlignerUnitJoin.Mismatch.leftover(["zusammen"])) {
            _ = try ForcedAlignerUnitJoin.join(swift, toMatch: ["Guten", "Morgen"])
        }
    }
}
