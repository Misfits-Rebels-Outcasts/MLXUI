import Testing
import Foundation
@testable import MLXUI

/// SET-1 (`RSI/DelegateDeciderBacklog.md`) — the app now binds an `engines.llm.*` row's
/// `key=value` settings against its serving manifest's schema, fills omitted keys from the
/// declared defaults, and translates via `maps_to`. Ported from
/// `catflow-mlx/src/catflow/catalog/registry.py` (`resolve_settings` + `translate_settings`)
/// and `engines/real.py::_resolve_for_run` (RA-04). `max_tokens` landed first; `temperature`
/// followed (owner-approved 2026-09-10) — a framed row's prose is now less varied, aligned
/// with the reference.
struct CatFlowSettingsResolutionTests {

    private func manifest(_ file: String) throws -> CuratedManifest {
        try #require(CuratedManifest.load(manifestFile: file), "\(file) must load from the app bundle")
    }

    // MARK: - resolveEngineSettings: bind + fill + translate (the port)

    @Test func boundValueOverridesTheDefaultAndOmittedKeysFill() throws {
        let m = try manifest("qwen3-8b-4bit.json")
        let resolved = try m.resolveEngineSettings("max_tokens=1000", rowLabel: "2")
        // max_tokens bound to the row's value; temperature filled from the manifest default;
        // both under their `maps_to` engine names.
        #expect(resolved["max_tokens"] == "1000")
        #expect(resolved["temp"] == "0.3")
        #expect(resolved["temperature"] == nil)   // renamed away by maps_to
    }

    @Test func nothingSetFillsEveryKeyFromTheManifestDefaults() throws {
        let m = try manifest("qwen3-8b-4bit.json")
        let resolved = try m.resolveEngineSettings(nil, rowLabel: "2")
        #expect(resolved == ["max_tokens": "2048", "temp": "0.3"])
    }

    @Test func anEmptySchemaIsAPurePassthrough() throws {
        // Ministral / Llama ship `settings: {}` — SPEC-Q76: no binding, no fill.
        let m = try manifest("ministral-3b-4bit.json")
        #expect(m.settings.isEmpty)
        #expect(try m.resolveEngineSettings("max_tokens=100", rowLabel: "1") == [:])
    }

    // MARK: - resolveEngineSettings: refuse at the seam (MoC-FIX-2), don't clamp

    @Test func anUnknownKeyIsRefused() throws {
        let m = try manifest("qwen3-8b-4bit.json")
        #expect(throws: (any Error).self) {
            try m.resolveEngineSettings("top_p=0.9", rowLabel: "2")
        }
    }

    @Test func anOutOfRangeNumberIsRefusedNotClamped() throws {
        let m = try manifest("qwen3-8b-4bit.json")   // max_tokens min 1, max 32768
        for bad in ["max_tokens=0", "max_tokens=99999", "max_tokens=-5", "temperature=3.0"] {
            #expect(throws: (any Error).self, "\(bad) should be refused") {
                try m.resolveEngineSettings(bad, rowLabel: "2")
            }
        }
    }

    // MARK: - llmRunConfig: applied to the StageConfig, max_tokens only

    private func executor() -> RealExecutor {
        RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "set1",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in SetOneStubStage() },
            installedModelIDs: [],
            catalog: [])
    }

    @Test func aRowSettingReachesTheConfig() throws {
        let config = try executor().llmRunConfig(.default, model: "Qwen3 8B",
                                                 rowSettings: "max_tokens=100; temperature=0.9", path: "1")
        #expect(config.maxTokens == 100)
        #expect(config.temperature == 0.9)
    }

    @Test func noRowSettingGetsTheManifestDefaultsNotStageConfigFallbacks() throws {
        let config = try executor().llmRunConfig(.default, model: "Qwen3 8B",
                                                 rowSettings: nil, path: "1")
        #expect(config.maxTokens == 2048)      // the manifest default, not StageConfig's 512
        #expect(config.temperature == 0.3)     // the manifest default, not StageConfig's 0.7
    }

    @Test func anEmptySchemaModelIsUnchanged() throws {
        let config = try executor().llmRunConfig(.default, model: "Ministral 3B",
                                                 rowSettings: "max_tokens=100", path: "1")
        #expect(config.maxTokens == 512)   // StageConfig's fallback — Ministral declares no schema
        #expect(config.temperature == 0.7)
    }

    @Test func anUnbridgedOrUnknownModelIsUnchanged() throws {
        let config = try executor().llmRunConfig(.default, model: "Not A Real Model",
                                                 rowSettings: "max_tokens=100", path: "1")
        #expect(config.maxTokens == 512)
    }

    @Test func anOutOfRangeRowSettingThrowsFromLlmRunConfig() throws {
        #expect(throws: (any Error).self) {
            try executor().llmRunConfig(.default, model: "Qwen3 8B",
                                        rowSettings: "max_tokens=999999", path: "1")
        }
    }

    // MARK: - the pins are untouched

    /// Extract Structured keeps its 32-token / temp-0 pin regardless of the model's manifest —
    /// `makeExtractionStage`, never `llmRunConfig`.
    @Test func extractStructuredKeepsItsPinWithAManifestModel() async throws {
        let entry = makeEntry(id: "qwen3-8b-set1", modelType: .llm,
                              hfModelId: "mlx-community/Qwen3-8B-4bit")
        let seen = SetOneConfigBox()
        let exec = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory
                .appendingPathComponent("set1-es-\(UUID().uuidString)")),
            flowID: "set1",
            blobDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("set1-es-blob-\(UUID().uuidString)"),
            makeModelStage: { _, config in seen.set(config); return SetOneStubStage() },
            installedModelIDs: [entry.id],
            catalog: [entry])
        _ = try? await exec.execute(
            path: "1",
            row: Row(task: "Extract Structured", model: "Qwen3 8B", settings: "\"name\""),
            inputs: [Asset(items: [Item(kind: .text, value: "Ada.", path: nil, sourceText: nil)])],
            transcript: nil, context: nil, usedFlowContent: nil)
        let config = try #require(seen.value)
        #expect(config.maxTokens == ExtractStructuredStage.maxFieldTokens)   // 32, not 2048
        #expect(config.temperature == 0)
    }
}

private struct SetOneStubStage: PipelineStage {
    let id = "stub.set1"
    let name = "SET-1 stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        progress(1.0)
        return .text("no")
    }
}

private final class SetOneConfigBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: StageConfig?
    func set(_ c: StageConfig) { lock.lock(); _value = c; lock.unlock() }
    var value: StageConfig? { lock.lock(); defer { lock.unlock() }; return _value }
}
