import Testing
import Foundation
@testable import MLXUI

/// CAT-SIZE-1 — the corrected download sizes and the RAM floor (owner, 2026-11-16) hold in the bundled catalog.
struct CatalogSizeTests {
    private func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    /// Corrected download sizes (real HF file sizes through `ModelFileSelector`, companions included).
    private let corrected: [String: Double] = [
        "mlx-community--Ministral-3-3B-Instruct-2512-4bit": 2.78, "mlx-community--Devstral-Small-2-24B-Instruct-2512-4bit": 15.14,
        "mlx-community--LFM2-VL-1.6B-4bit": 1.47, "mlx-community--gemma-3-4b-it-4bit": 3.44,
        "mlx-community--Qwen3-VL-4B-Instruct-4bit": 3.11, "mlx-community--gemma-3-12b-it-4bit": 8.07,
        "mlx-community--PaddleOCR-VL-1.5-4bit": 0.71, "mlx-community--dots.ocr-4bit": 3.54,
        "mlx-community--olmOCR-2-7B-1025-mlx-4bit": 5.65, "mlx-community--Voxtral-Mini-4B-Realtime-2602-4bit": 3.15,
        "mlx-community--embeddinggemma-300m-4bit": 0.21, "mlx-community--gemma-3-1b-it-qat-4bit": 0.77,
        "mlx-community--orpheus-3b-0.1-ft-bf16": 6.63, "mlx-community--Kokoro-82M-bf16": 0.36,
        "Wan-AI--Wan2.1-T2V-1.3B-Diffusers": 28.93, "mlx-community--SeedVR2-3B-mlx-int8": 4.72,
        "mlx-community--Qwen3-TTS-12Hz-0.6B-Base-bf16": 2.52,   // CAT-SIZE-2
    ]

    @Test func correctedSizesAreInTheEntryAndItsVariant() throws {
        let entries = try catalog()
        for (id, size) in corrected {
            let entry = try #require(entries.first { $0.id == id }, "\(id) not in catalog")
            #expect(entry.downloadSizeGB == size, "\(id)")
            #expect(entry.variants?.filter { $0.hfModelId == entry.hfModelId }.allSatisfy { $0.downloadSizeGB == size } ?? true, "\(id) variant")
        }
    }

    @Test func ramIsAtLeastTwentyPercentAboveTheDownloadForEveryCorrectedEntryButWan() throws {
        let entries = try catalog()
        for id in corrected.keys where id != "Wan-AI--Wan2.1-T2V-1.3B-Diffusers" {
            let entry = try #require(entries.first { $0.id == id })
            #expect(entry.ramGB >= (entry.downloadSizeGB * 1.2 * 100).rounded() / 100 - 0.005, "\(id) ram \(entry.ramGB)")
        }
        // Ministral's RAM figure on its model page is at least its download size (CAT-S1).
        let ministral = try #require(entries.first { $0.id == "mlx-community--Ministral-3-3B-Instruct-2512-4bit" })
        #expect(ministral.ramGB >= 2.8)
    }

    @Test func seedVR2ProseQuotesItsCatalogRAMFigure() throws {
        let entry = try #require(try catalog().first { $0.id == "mlx-community--SeedVR2-3B-mlx-int8" })
        #expect(entry.ramGB == 5.66)
        for text in [entry.description, entry.summary].compactMap({ $0 }) {
            #expect(!text.contains("4.5 GB"))
        }
        #expect(entry.summary?.contains("~5.7 GB RAM") == true)
    }
}
