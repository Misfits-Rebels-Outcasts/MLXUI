import Foundation
import Testing
@testable import MLXUI

// WW2-3-7 — Kokoro is shown dimmed with "needs macOS 15" on macOS < 15 (owner ruling 2026-10-23, the WW6 "show it, don't
// fail it" rule), because MLXAudioThread needs TaskExecutor (macOS 15+) and Kokoro's `generate` returns an array the
// main thread can't evaluate under mlx-swift 0.32 (journal 2026-557/558). Only Kokoro; the other TTS families don't use
// the pattern. The injectable `osVersion` stands in for macOS 14 on this Mac.

private func macOS(_ major: Int, _ minor: Int = 0) -> OperatingSystemVersion {
    OperatingSystemVersion(majorVersion: major, minorVersion: minor, patchVersion: 0)
}

private func loadCatalog() throws -> [ModelEntry] {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    let url = root.appendingPathComponent("MLXUI/Resources/browser.json")
    return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
}

struct ModelOSRequirementTests {

    @Test func kokoroNeedsMacOS15BelowMacOS15() {
        for major in [13, 14] {
            #expect(ModelSupport.osRequirement(forID: "mlx-community--Kokoro-82M-bf16", osVersion: macOS(major)) == "needs macOS 15")
        }
        #expect(ModelSupport.osRequirement(forID: "mlx-community/Kokoro-82M-bf16", osVersion: macOS(14, 7)) == "needs macOS 15")
    }

    @Test func kokoroIsAvailableFromMacOS15() {
        for (major, minor) in [(15, 0), (15, 5), (26, 0), (27, 0)] {
            #expect(ModelSupport.osRequirement(forID: "mlx-community--Kokoro-82M-bf16", osVersion: macOS(major, minor)) == nil)
        }
    }

    @Test func otherModelsAreNeverGated() {
        for id in ["mlx-community--Qwen3-TTS-12Hz-0.6B-Base-bf16", "mlx-community--chatterbox-fp16",
                   "mlx-community--Qwen3-8B-4bit", "mlx-community--whisper-tiny-asr-fp16", "mlx-community--sam3-4bit"] {
            #expect(ModelSupport.osRequirement(forID: id, osVersion: macOS(14)) == nil, "\(id)")
        }
    }

    @Test func theRealCatalogEntryIsGatedOnMacOS14() throws {
        let catalog = try loadCatalog()
        let kokoro = catalog.filter { ($0.id + $0.hfModelId).lowercased().contains("kokoro") }
        #expect(!kokoro.isEmpty)
        for model in kokoro {
            #expect(ModelSupport.osRequirement(for: model, osVersion: macOS(14)) == "needs macOS 15")
            #expect(ModelSupport.osRequirement(for: model, osVersion: macOS(15)) == nil)
        }
        let others = catalog.filter { !($0.id + $0.hfModelId).lowercased().contains("kokoro") }
        #expect(others.allSatisfy { ModelSupport.osRequirement(for: $0, osVersion: macOS(14)) == nil })
    }

    @Test func titleCapitalisesTheFirstLetter() {
        #expect(ModelSupport.osRequirementTitle("needs macOS 15") == "Needs macOS 15")
    }

    /// The flow Model menu: a gated model is not selectable, an ordinary one still is (on the same fake OS).
    @Test func theFlowModelMenuDisablesTheGatedModelOnMacOS14() throws {
        let catalog = try loadCatalog()
        let kokoro = try #require(catalog.first { ($0.id + $0.hfModelId).lowercased().contains("kokoro") })
        let qwenTTS = try #require(catalog.first { $0.id.lowercased().contains("qwen3-tts") })
        let installed: Set<String> = [kokoro.id, qwenTTS.id]
        func enabled(_ m: ModelEntry, _ os: OperatingSystemVersion) -> Bool {
            FlowRowInspectorView.isModelButtonEnabled(.cataloged(m), installedModelIDs: installed, totalRAMGB: 64, osVersion: os)
        }
        #expect(!enabled(kokoro, macOS(14)))
        #expect(enabled(kokoro, macOS(15)))
        #expect(enabled(qwenTTS, macOS(14)))
    }
}

