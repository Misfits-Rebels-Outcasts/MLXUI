import Foundation
import CryptoKit

/// Seed semantics (CFM-R3-1, port of `catflow-mlx/src/catflow/core/seeds.py`): a stochastic
/// row's `seed` setting has three states — omitted (runtime pins a deterministic seed derived
/// from the row's identity), `seed=auto` (derived per activation as `runSeed + k`), or
/// `seed=N` (author-pinned, passed through). The `_with_seed` substitution (`CachingExecutor`
/// in R3-2) injects the concrete derived number into the row's settings **before** the cache
/// key, mock fingerprint, and engine call are built, so the derived number is all any of them
/// ever see. Only a row whose serving manifest declares a `seed` setting is stochastic (Speak
/// today; diffusion rows later).
nonisolated enum FlowSeed {
    static let seedKey = "seed"
    static let autoValue = "auto"
    static let maxSeed: UInt32 = 0xFFFF_FFFF

    /// `run_seed_for` — the deterministic run-level seed a `seed=auto` row derives from.
    /// A pure function of the flow's own text, so derived seeds are stable across runs,
    /// machines, and mock/real tiers.
    static func runSeed(for flowText: String) -> UInt32 {
        // Python: int.from_bytes(digest[:4], "big").
        SHA256.hash(data: Data(flowText.utf8)).bytes.prefix(4).reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
    }

    /// `derive_pinned` — the deterministic pin an *omitted* seed resolves to.
    static func derivePinned(identity: String) -> UInt32 {
        // Python: int.from_bytes(digest[:4], "big").
        SHA256.hash(data: Data(identity.utf8)).bytes.prefix(4).reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
    }

    /// `derive_auto` — `seed=auto`'s per-activation seed: `runSeed + k`.
    static func deriveAuto(runSeed: UInt32, activationIndex: Int) -> UInt32 {
        (runSeed &+ UInt32(activationIndex)) & maxSeed
    }

    /// `activation_index` — `"4"` → 1; `"4@2"` → 2.
    static func activationIndex(execPath: String) -> Int {
        if let at = execPath.lastIndex(of: "@") {
            return Int(execPath[execPath.index(after: at)...]) ?? 1
        }
        return 1
    }

    /// `row_identity` — the identity an omitted seed is derived from; the `@k` is stripped so
    /// the pin is stable across a row's activations.
    static func rowIdentity(path: String, row: Row, modelID: String) -> String {
        let bare = path.split(separator: "@", maxSplits: 1).map(String.init)[0]
        return "\(bare)|\(row.task ?? "")|\(modelID)"
    }

    /// `resolve_seed_settings` — returns settings with a concrete `seed=N`, or `nil` when it
    /// should be left as-is (already pinned, or already a concrete number).
    static func resolveSeedSettings(
        settings: String?,
        runSeed: UInt32,
        activationIndex: Int,
        identity: String
    ) -> String? {
        let s = FlowSettings(settings)
        let seedValue = s.value(for: seedKey)
        if seedValue == nil {
            let pinned = String(derivePinned(identity: identity))
            if let settings, !settings.isEmpty {
                return "\(settings); \(seedKey)=\(pinned)"
            }
            return "\(seedKey)=\(pinned)"
        }
        if seedValue == autoValue {
            let derived = String(deriveAuto(runSeed: runSeed, activationIndex: activationIndex))
            return replaceSeedToken(in: settings ?? "", with: derived)
        }
        return nil
    }

    /// `_replace_seed_token` — quote-aware replacement of the `seed=auto` token's value.
    static func replaceSeedToken(in settings: String, with value: String) -> String {
        // Blank quoted spans so a literal "seed=auto" inside quotes isn't matched.
        var scan = ""
        var inQuote = false
        for ch in settings {
            if ch == "\"" {
                inQuote.toggle()
                scan.append(" ")
            } else {
                scan.append(inQuote ? " " : ch)
            }
        }
        // Find `seed` `=` value outside quotes (whitespace-tolerant).
        let pattern = NSRegularExpression.compiled(#"\bseed\s*=\s*[^\s;]+"#)
        guard let match = pattern.firstMatch(in: scan, range: NSRange(scan.startIndex..<scan.endIndex, in: scan)),
              let range = Range(match.range, in: settings) else {
            return settings
        }
        guard let eq = settings[range].firstIndex(of: "=") else { return settings }
        let valueStart = settings.distance(from: settings.startIndex, to: range.lowerBound)
            + settings.distance(from: range.lowerBound, to: eq) + 1
        let start = settings.index(settings.startIndex, offsetBy: valueStart)
        return String(settings[..<start]) + value + String(settings[range.upperBound...])
    }
}

nonisolated extension SHA256.Digest {
    var bytes: [UInt8] { Array(self) }
}
