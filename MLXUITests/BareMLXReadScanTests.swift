import Foundation
import Testing

// WW2-3-8 — no bare read of an MLX array in app code (owner ruling 2026-10-23, journal 2026-560).
//
// A failed `eval` leaves an array with no buffer, and `asArray` / `item` on it then dereferences null: a segfault (B10 Kokoro,
// B18 SeedVR2). The engines read through `MLXErrorGuard.read` / `readRows` / `item`, which throw instead. This scan fails on
// any other `.asArray(` / `.item(` in `MLXUI/` or `Vendor/`, so a new engine (the Model Pack's) can't reintroduce it. An
// entry in `allowed` needs a reason that stays true. Skipped where the sources aren't readable (a sandboxed test host).

struct BareMLXReadScanTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    private static var sourcesReadable: Bool {
        FileManager.default.isReadableFile(atPath: repoRoot.appendingPathComponent("MLXUI").path)
    }

    /// file → why its bare reads are safe (or the known gap).
    static let allowed: [String: String] = [
        "MLXUI/Modules/Flux/FluxSampler.swift":
            "`linspace` of two scalar constants (the sampler's schedule) — a host-sized op with no engine graph behind it.",
        "MLXUI/Modules/Flux/FluxCLIPEncoder.swift":
            "`argMax` over the host-built token array to find the EOS position (a host Int is required, LY-5); the encoder's output "
            + "is evaluated through `MLXErrorGuard.evaluate` in `FluxEngine` right after.",
        "MLXUI/Modules/MusicGen/MusicGenEngine.swift":
            "`undoDelay` reads `rows`, which the caller has just evaluated through `MLXErrorGuard.evaluate` (a failure throws before it).",
        "MLXUI/Services/Server/Backends/MLXChatBackend.swift":
            "reads prompt token ids out of the tokenizer's host-built `MLXArray` (no engine graph behind it).",
        "Vendor/PaddleOCRVL/Sources/PaddleOCRVL/Generator.swift":
            "KNOWN GAP: the vendored package can't see `MLXErrorGuard`, and its token loop reads `argMax(logits).item(Int.self)` per step. "
            + "PaddleOCR is tiny (0.22 GB) and passed B6; a recorded MLX error mid-generation would still crash. Backlog: guard the vendored loop.",
        "Vendor/PaddleOCRVL/Sources/PaddleOCRVL/PaddleOCRVLModel.swift":
            "KNOWN GAP (see Generator.swift): `argMax(mask)[0].item(Int.self)` on the host-built attention mask.",
    ]

    @Test(.enabled(if: BareMLXReadScanTests.sourcesReadable))
    func noEngineReadsAnMLXArrayWithoutTheGuard() {
        let pattern = try! NSRegularExpression(pattern: #"(?<!MLXErrorGuard)\.(asArray|item)\("#)
        var hits: [String] = []
        for dir in ["MLXUI", "Vendor"] {
            for file in MixedSubscriptScanner.swiftFiles(under: Self.repoRoot.appendingPathComponent(dir)) {
                let relative = file.path.replacingOccurrences(of: Self.repoRoot.path + "/", with: "")
                if relative == "MLXUI/Core/MLXErrorGuard.swift" || Self.allowed[relative] != nil { continue }
                guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
                for (index, line) in text.components(separatedBy: "\n").enumerated() {
                    let code = line.components(separatedBy: "//").first ?? line
                    if pattern.firstMatch(in: code, range: NSRange(location: 0, length: (code as NSString).length)) != nil {
                        hits.append("\(relative):\(index + 1): \(line.trimmingCharacters(in: .whitespaces))")
                    }
                }
            }
        }
        #expect(hits.isEmpty, "bare asArray/item — use MLXErrorGuard.read / readRows / item (a failed array segfaults):\n\(hits.joined(separator: "\n"))")
    }

    /// Every allow-list entry still has a bare read in it (a stale entry hides nothing and misleads).
    @Test(.enabled(if: BareMLXReadScanTests.sourcesReadable))
    func theAllowListHasNoStaleEntries() {
        let pattern = try! NSRegularExpression(pattern: #"(?<!MLXErrorGuard)\.(asArray|item)\("#)
        for (relative, reason) in Self.allowed {
            let url = Self.repoRoot.appendingPathComponent(relative)
            let text = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
            let code = text.components(separatedBy: "\n").map { $0.components(separatedBy: "//").first ?? $0 }.joined(separator: "\n")
            #expect(pattern.firstMatch(in: code, range: NSRange(location: 0, length: (code as NSString).length)) != nil,
                    "\(relative) has no bare read left — remove it from `allowed` (\(reason.prefix(40))…)")
        }
    }
}
