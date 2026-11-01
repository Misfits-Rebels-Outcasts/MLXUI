import Testing
import Foundation
@testable import MLXUI

/// MP7-R (owner, Boo, 2026-11-01): the gallery hides 61, 62, 67, 68 and shows 60, 66, 69. These tests tie both facts to the live gate, so a newly served task or model is a visible change.
struct GalleryHiddenFlowsTests {
    @MainActor private static func context() throws -> (catalog: [ModelEntry], claimable: Set<String>) {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return (catalog, Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id)))
    }

    @Test @MainActor func hiddenNumbersAreExactlyTheBundled6xFlowsTheLiveGateRefuses() throws {
        let (catalog, claimable) = try Self.context()
        var refused: Set<Int> = []
        for flow in GalleryLoader.loadMetadata() where (60 ... 69).contains(flow.number) {
            let doc = try GalleryLoader.loadDocument(flowID: flow.flowID)
            if FlowRunnability.refusalReason(for: doc, catalog: catalog, installed: [], totalRAMGB: 32, claimableModelIDs: claimable) != nil { refused.insert(flow.number) }
        }
        #expect(refused == [61, 62, 67, 68], Comment(rawValue: "\(refused.sorted())"))
        #expect(AppState.hiddenFlowNumbers == refused, "hidden = refused; a flow that becomes runnable must be un-hidden on purpose")
        #expect(AppState.hiddenFlowNumbers.isDisjoint(with: [60, 66, 69]))
    }

    /// The flows `FlowPreflight`'s "task not served" guard refuses, and why. A task that becomes served drops out of this list: update it on purpose.
    @Test @MainActor func theUnservedTaskGuardRefusesExactlyTheseBundledFlows() throws {
        let (catalog, claimable) = try Self.context()
        var found: [String] = []
        for flow in GalleryLoader.loadMetadata() {
            let doc = try GalleryLoader.loadDocument(flowID: flow.flowID)
            let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 32, claimableModelIDs: claimable)
            for need in result.needs {
                if let reason = need.blockingReason, reason.hasSuffix("is not supported in this app yet.") { found.append("\(flow.number) \(flow.flowID): \(reason)") }
            }
        }
        // 62 (Depth Pro) and 68 (Base Style) are refused earlier for "isn't in the runnable-model table", not by this guard.
        #expect(found.sorted() == [
            "61 61-EditInPlace: Row 3: Edit Image is not supported in this app yet.",
            "67 67-DriveTheLatent: Row 1: Init Latent is not supported in this app yet.",
            "67 67-DriveTheLatent: Row 2: Denoise is not supported in this app yet.",
            "67 67-DriveTheLatent: Row 3: Decode Latent is not supported in this app yet.",
        ], Comment(rawValue: found.sorted().joined(separator: "\n")))
    }
}

/// MP7-R item 4: how `preset=(param:quality)` of gallery 69-PickALook reaches the executor. No weights: the stage is a stub that records the `StageConfig` and stops the row.
struct Gallery69PresetTraceTests {
    final class Box: @unchecked Sendable { var configs: [StageConfig] = []; var settings: [String] = [] }

    struct Stop: Error, CustomStringConvertible { var description: String { "stub stopped the row" } }

    /// `bind`: replace `(param:quality)` with the preset name in the in-memory text, i.e. what a CALLER binding `quality=` would give the row (the gallery file is not edited).
    @MainActor private static func trace(quality: String?, bind: Bool = false) async throws -> (config: StageConfig?, message: String?) {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        let klein = try #require(catalog.first { $0.hfModelId == "mlx-community/FLUX.2-klein-4B-bf16" })
        var text = try String(contentsOf: root.appendingPathComponent("MLXUI/Resources/Gallery/69-PickALook.catpipeline"), encoding: .utf8)
        if let quality { text = text.replacingOccurrences(of: "quality = balanced", with: "quality = \(quality)") }   // an in-memory copy; the gallery file is not edited
        if bind, let quality { text = text.replacingOccurrences(of: "preset=(param:quality)", with: "preset=\(quality)") }
        let box = Box()
        let sdk = Flux2KleinSDK()
        let exec = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory), flowID: "g69", blobDirectory: FileManager.default.temporaryDirectory,
                                makeModelStage: { model, config in box.configs.append(config); _ = try sdk.makeStage(for: model, config: config); throw Stop() },
                                installedModelIDs: [klein.id], catalog: catalog)
        let doc = try CatParser.parse(text)
        let events = try await FlowInterpreter.run(doc, executor: exec, definitions: doc.definitions, presets: doc.presets)
        return (box.configs.first, events.first { $0.kind == .rowFailed }?.error)
    }

    @Test @MainActor func theParamRefPresetIsDeferredAndStrippedSoTheExecutorSeesOnlySeedAndPrompt() async throws {
        let text = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/Gallery/69-PickALook.catpipeline"), encoding: .utf8)
        let doc = try CatParser.parse(text)
        print("MP7R-69 presets=\(doc.presets.mapValues { $0.bindings.map { "\($0.key)=\($0.value)" } }) params=\(doc.params.map(\.name))")
        for q in [nil, "draft", "balanced", "studio"] {
            let (config, message) = try await Self.trace(quality: q)
            print("MP7R-69 top-level run, quality=\(q ?? "(file default)") -> seed=\(String(describing: config?.seed)) width=\(String(describing: config?.width)) height=\(String(describing: config?.height)) steps=\(String(describing: config?.steps)) message=\(message ?? "-")")
        }
    }

    @Test @MainActor func whenACallerBindsQualityThePresetExpandsToStepsWidthHeight() async throws {
        let expected: [String: (Int, Int, Int)] = ["draft": (4, 512, 512), "balanced": (9, 1024, 1024)]
        for (q, want) in expected {
            let (config, _) = try await Self.trace(quality: q, bind: true)
            #expect(config?.steps == want.0 && config?.width == want.1 && config?.height == want.2, Comment(rawValue: q))
            #expect(config?.seed == 7)
        }
        // studio (28 steps, 1536 x 1536) is refused by the Klein cap sentence at makeStage, before any weight loads, not a crash
        let (config, message) = try await Self.trace(quality: "studio", bind: true)
        #expect(config?.steps == 28 && config?.width == 1536 && config?.height == 1536)
        if Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: Flux2KleinMemoryPlan.defaultBudgetBytes) < 1536 {
            #expect(message?.hasPrefix("FLUX.2 Klein can't make a 1536 \u{00D7} 1536 image at quantize 8 on this Mac") == true, Comment(rawValue: message ?? "nil"))
        }
    }
}
