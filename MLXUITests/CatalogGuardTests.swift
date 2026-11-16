import Testing
import Foundation
@testable import MLXUI

/// CAT-GUARD-1 — the catalog size/RAM guard (owner ruling, Boo, 2026-11-16): a model's RAM figure is never below its download,
/// and every variant has a real download size. Enforced here; the add-a-model intake (AM1/AM2) points at it.
/// Reads the raw JSON because `notes` is not a `ModelEntry` field (the decoder ignores it).
struct CatalogGuardTests {
    private struct Row {
        let id: String
        let ram: Double
        let download: Double
        let notes: String
        let variants: [(hf: String, ram: Double, download: Double)]
        /// The marker CAT-SIZE-1 wrote for a curated or measured RAM the weights rule does not describe.
        /// Variants carry no notes of their own, so a variant takes its **parent entry's** notes (implementer's call).
        var exempt: Bool { notes.contains("Floor exempt") }
    }

    private func rows() throws -> [Row] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        let top = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var out: [Row] = []
        func walk(_ nodes: [[String: Any]]) {
            for node in nodes {
                for m in node["models"] as? [[String: Any]] ?? [] {
                    let variants = (m["variants"] as? [[String: Any]] ?? []).map {
                        (hf: $0["hfModelId"] as? String ?? "?", ram: $0["ramGB"] as? Double ?? 0, download: $0["downloadSizeGB"] as? Double ?? 0)
                    }
                    out.append(Row(id: m["id"] as? String ?? "?", ram: m["ramGB"] as? Double ?? 0,
                                   download: m["downloadSizeGB"] as? Double ?? 0, notes: m["notes"] as? String ?? "", variants: variants))
                }
                walk(node["children"] as? [[String: Any]] ?? [])
            }
        }
        walk(top["domains"] as? [[String: Any]] ?? [])
        return out
    }

    @Test func ramIsNeverBelowTheDownloadOnAnyEntryOrVariant() throws {
        for row in try rows() where !row.exempt {
            #expect(row.ram >= row.download, "\(row.id): ramGB \(row.ram) < downloadSizeGB \(row.download)")
            for v in row.variants {
                #expect(v.ram >= v.download, "\(row.id) variant \(v.hf): ramGB \(v.ram) < downloadSizeGB \(v.download)")
            }
        }
    }

    @Test func everyVariantHasADownloadSize() throws {
        for row in try rows() {
            for v in row.variants {
                #expect(v.download > 0, "\(row.id) variant \(v.hf): downloadSizeGB \(v.download)")
            }
        }
    }

    @Test func wanCarriesTheExactExemptMarker() throws {
        let wan = try #require(try rows().first { $0.id == "Wan-AI--Wan2.1-T2V-1.3B-Diffusers" })
        #expect(wan.notes.contains("Floor exempt"))
        #expect(wan.exempt)
    }

    @Test func theGuardCoversTheWholeCatalog() throws {
        #expect(try rows().count >= 56)
    }
}
