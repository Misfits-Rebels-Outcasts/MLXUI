import Testing
import Foundation
@testable import MLXUI

/// MT1-3 (owner ruling 2026-11-08, MT1-1 review): for every model `CapabilityResolver` can return for
/// `generate_image`, an 8192x8192 request is refused by the stage/SDK **before any weights load**, through
/// `RealExecutor.execute` with the real SDKs and no weights. A stage that does not refuse is a finding for the
/// owner; no cap is added in a stage.
@MainActor @Suite struct ModelToolOversizeTests {

    private func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    /// What one model did with the oversize request.
    struct Outcome { let model: String; let sentence: String? }

    private func oversizeOutcomes() async throws -> [Outcome] {
        let cat = try catalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claim = Set(cat.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        let spec = try #require(ModelToolSpec.spec(named: "generate_image"))
        let candidates = CapabilityResolver.orderedCandidates(task: "Generate Image", kinds: spec.kinds, catalog: cat,
                                                              claimableModelIDs: claim)
        try #require(!candidates.isEmpty)
        var outcomes: [Outcome] = []
        for model in candidates {
            // Safety: if this Mac really has the model installed, an unrefused request would load its weights. Do not.
            let dir = ModelStore.shared.directory(forModelID: model.id)
            guard !FileManager.default.fileExists(atPath: dir.path) else {
                Issue.record("\(model.id) is installed on this Mac; the oversize check is skipped for it so no weights can load.")
                continue
            }
            let blobs = FileManager.default.temporaryDirectory.appendingPathComponent("oversize-\(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: blobs) }
            let make: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
                try await MainActor.run {
                    guard let resolved = registry.bestModule(for: model) else {
                        throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind)
                    }
                    return try resolved.sdk.makeStage(for: model, config: config)
                }
            }
            let executor = RealExecutor(workspace: FlowWorkspace(root: blobs), flowID: "oversize", blobDirectory: blobs,
                                        makeModelStage: make, installedModelIDs: [model.id], catalog: cat)
            let text = ModelToolRunner.rowText(spec: spec, modelDisplay: TaskModels.displayName(for: model),
                                               values: ["width": .int(8192), "height": .int(8192)])
            let doc = try ModelToolRunner.document(rowText: text)
            let row = try #require(doc.rows.first)
            do {
                _ = try await executor.execute(path: "1", row: row,
                                               inputs: [Asset(items: [Item(kind: .text, value: "a red bicycle")])],
                                               transcript: nil, context: nil, usedFlowContent: nil)
                outcomes.append(Outcome(model: model.id, sentence: nil))
            } catch {
                outcomes.append(Outcome(model: model.id, sentence: FlowErrorDisplay.sentence(for: error)))
            }
        }
        return outcomes
    }

    @Test func everyGenerateImageModelRefusesAnOversizeRequestBeforeWeightsLoad() async throws {
        let outcomes = try await oversizeOutcomes()
        for outcome in outcomes { print("MT13|oversize|\(outcome.model)|\(outcome.sentence ?? "NOT REFUSED")") }
        #expect(!outcomes.isEmpty)
        for outcome in outcomes {
            guard let sentence = outcome.sentence else { Issue.record("\(outcome.model): not refused"); continue }
            // A refusal names the size or the setting; a weights-load failure would talk about files or installs.
            let check = {
                let lower = sentence.lowercased()
                let loadFailure = lower.contains("isn't installed") || lower.contains("couldn't find") || lower.contains("no such file")
                    || lower.contains("not found") || lower.contains("missing")
                let namesTheRequest = lower.contains("width") || lower.contains("height") || lower.contains("8192") || lower.contains("size")
                    || lower.contains("too large") || lower.contains("cap")
                #expect(namesTheRequest && !loadFailure, "\(outcome.model): \(sentence)")
            }
            if outcome.model.lowercased().contains("sdxl-turbo") {
                // Owner ruling 2026-11-08: accepted for MT. RSI/backlog.md "SDXLTurboSDK ignores width/height/steps…": the SDK
                // drops the settings and renders at its fixed size, so it does not refuse. No stage change in MT1.
                withKnownIssue("SDXLTurboSDK ignores width/height/steps (RSI/backlog.md)") { check() }
            } else {
                check()
            }
        }
    }
}
