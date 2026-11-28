import Foundation
import Testing

// FA-R2 — the `Align` row's spec question is SPEC-Q261. SPEC-Q252 belongs to `Redact` (PF) and stays on Redact's lines.
//
// Owner ruling 9 (Boo, 2026-11-27): "Use SPEC-Q261 for Align. Rename every Align mark in one FA-R commit, and add a test or
// grep check that no Align code still carries Q252. In catflow-mlx's SPEC_QUESTIONS.md, add a one-line placeholder for Q261
// ('Align task, MLXUI PR #14, entry to follow') in its own commit. The full entry stays a follow-up."
//
// Skipped where the sources aren't readable (a sandboxed test host), as `BareMLXReadScanTests` is. Files are listed by
// relative path (`enumerator(atPath:)`), so `/tmp` against `/private/tmp` never matters.

struct ForcedAlignerSpecMarkTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .resolvingSymlinksInPath()
    private static var sourcesReadable: Bool {
        FileManager.default.isReadableFile(atPath: repoRoot.appendingPathComponent("MLXUI").path)
    }

    private static let moduleFolder = "MLXUI/Modules/ForcedAligner/"
    private static let rowFile = "MLXUI/FlowKit/AlignRow.swift"
    private static let manifest = "MLXUI/Resources/CatFlow/models/qwen3-forcedaligner-0.6b-8bit.json"

    private static func lines(of relative: String) -> [String]? {
        (try? String(contentsOf: repoRoot.appendingPathComponent(relative), encoding: .utf8))?.components(separatedBy: "\n")
    }

    @Test(.enabled(if: ForcedAlignerSpecMarkTests.sourcesReadable))
    func noAlignCodeStillCarriesTheRedactNumber() throws {
        let appRoot = Self.repoRoot.appendingPathComponent("MLXUI").path
        let walker = try #require(FileManager.default.enumerator(atPath: appRoot))
        var visited: Set<String> = []
        var hits: [String] = []
        for case let path as String in walker {
            let relative = "MLXUI/" + path
            // Folders and binary resources don't decode as text.
            guard let lines = Self.lines(of: relative) else { continue }
            visited.insert(relative)
            let wholeFileIsAlign = relative.hasPrefix(Self.moduleFolder) || relative == Self.rowFile || relative == Self.manifest
            for (index, line) in lines.enumerated() where line.contains("Q252") {
                let namesAlign = ["Align", "Qwen3 Aligner", "FA-"].contains { line.contains($0) }
                if wholeFileIsAlign || namesAlign {
                    hits.append("\(relative):\(index + 1): \(line.trimmingCharacters(in: .whitespaces).prefix(120))")
                }
            }
        }
        // Not vacuous: the walk reached the aligner's own files.
        #expect(visited.contains(Self.rowFile))
        #expect(visited.contains(Self.manifest))
        #expect(visited.contains { $0.hasPrefix(Self.moduleFolder) })
        #expect(hits.isEmpty, "Align code carries SPEC-Q252 (Redact's number) — use SPEC-Q261:\n\(hits.joined(separator: "\n"))")
    }

    @Test(.enabled(if: ForcedAlignerSpecMarkTests.sourcesReadable))
    func theCatalogRowAndTheExecutorBranchCarryTheAlignMark() throws {
        let sites = [("MLXUI/FlowKit/TaskCatalog.swift", #"TaskDescriptor("Align""#),
                     ("MLXUI/FlowKit/RealExecutor.swift", #"if desc.refName == "engines.asr.align""#)]
        for (file, anchor) in sites {
            let lines = try #require(Self.lines(of: file))
            let matches = lines.filter { $0.contains(anchor) }
            #expect(matches.count == 1, "\(file): expected one line with \(anchor)")
            #expect(matches.allSatisfy { $0.contains("// SPEC-Q261") }, "\(file): the Align line lacks `// SPEC-Q261`")
        }
    }
}
