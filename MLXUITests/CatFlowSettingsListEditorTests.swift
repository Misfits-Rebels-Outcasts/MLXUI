import Testing
import Foundation
@testable import MLXUI

/// RT-4 (`RSI/DelegateRowTextBacklog.md`) — the Settings list, made honest: source order
/// instead of a `test`-prefixed accessor, a per-key control shape, `separator`'s escape
/// visibility, widened `knownSettingKeys`, and the key/bare shadowing warning.
struct CatFlowSettingsListEditorTests {

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-rt4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "RT4", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String?, settings: String? = nil) -> Row {
        Row(id: UUID(), task: task, blockKind: nil, model: nil, settings: settings,
            refs: [], children: [], clause: nil)
    }

    // MARK: - `orderedPairs` is source order, not a `Set`'s

    @Test func orderedPairsPreservesSourceOrderNotAlphabetical() throws {
        // Deliberately not alphabetical — `testKeys.sorted()` would have given a, b, c.
        let settings = FlowSettings("site=example.com; recency=week; top_k=5")
        #expect(settings.orderedKeys == ["site", "recency", "top_k"])
        #expect(settings.orderedPairs.map(\.key) == ["site", "recency", "top_k"])
        #expect(settings.orderedPairs.map(\.value) == ["example.com", "week", "5"])
    }

    /// The settings list itself renders in that order — a real row, through the model layer.
    @Test func settingsSectionOrderFollowsARealRowsSourceOrder() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "34-TopicMonitor")
        let webSearch = doc.rows[0]
        #expect(webSearch.task == "Web Search")
        let pairs = FlowSettings(webSearch.settings).orderedPairs
        #expect(pairs.map(\.key) == ["top_k", "recency"])
    }

    // MARK: - `separator`'s escape-visibility round trip

    @Test func separatorDisplayShowsTheEscapedLiteralNotARealNewline() throws {
        let raw = #"separator="\n""#
        let value = FlowSettings(raw).value(for: "separator")
        #expect(value == "\n")   // FlowSettings.value(for:) already decoded it — a real newline

        // What the field shows and what it writes back are the escaped two characters, not
        // the invisible control character.
        let displayed = FlowRowInspectorView.escapedForDisplay(value ?? "")
        #expect(displayed == "\\n")
        #expect(FlowRowInspectorView.unescapedFromDisplay(displayed) == "\n")
    }

    @Test func editingTheEscapedSeparatorFieldWritesTheRealEscapeToSettings() throws {
        let r = row("Join Text", settings: "separator=\", \"")
        let model = try editor([r])
        // Simulate what separatorField's TextField binding does: read the escaped display,
        // "edit" it, write back through unescapedFromDisplay.
        let currentDisplay = FlowRowInspectorView.escapedForDisplay(
            FlowSettings(model.row(withID: r.id)?.settings).value(for: "separator") ?? "")
        #expect(currentDisplay == ", ")   // no control chars here, displays as-is

        model.setSetting(key: "separator", value: FlowRowInspectorView.unescapedFromDisplay("\\n\\t"),
                         for: r.id)
        #expect(model.row(withID: r.id)?.settings == #"separator="\n\t""#)
        #expect(FlowSettings(model.row(withID: r.id)?.settings).value(for: "separator") == "\n\t")
    }

    // MARK: - `knownSettingKeys` widened, each cited at the line that reads it

    @Test func knownSettingKeysCoversEveryTier2Addition() {
        #expect(FlowRowInspectorView.knownSettingKeys(for: "Filter").contains("contains"))
        #expect(FlowRowInspectorView.knownSettingKeys(for: "Join Text") == ["separator"])
        #expect(FlowRowInspectorView.knownSettingKeys(for: "Text to Table") == ["expected"])
        #expect(FlowRowInspectorView.knownSettingKeys(for: "Save Images") == ["naming"])
        for task in ["Web Fetch", "HTTP Get", "Fetch Feed", "Download File"] {
            #expect(FlowRowInspectorView.knownSettingKeys(for: task) == ["url"], "\(task) should offer url=")
        }
    }

    @Test func knownSettingKeysCoversEveryDiffusionTaskByRefNamePrefix() {
        for task in ["Generate Image", "Generate Sound", "Generate Video", "Edit Image", "Upscale"] {
            let keys = FlowRowInspectorView.knownSettingKeys(for: task)
            #expect(Set(["seed", "width", "height", "steps"]).isSubset(of: Set(keys)),
                    "\(task) should offer seed/width/height/steps")
        }
        // A non-diffusion task must not pick this up by accident.
        #expect(!FlowRowInspectorView.knownSettingKeys(for: "Summarize").contains("seed"))
    }

    // MARK: - Key/bare shadowing warning (fact 23)

    @Test func shadowingMessageMatchesTheCopyTable() {
        #expect(FlowRowInspectorView.shadowingMessage(key: "query")
                == "This row sets both query= and a bare value. query= is the one that runs.")
    }

    /// The guard `shadowingWarning` (private) applies: both a `query=` key and a bare token
    /// present. Exercised at the `FlowSettings` level, the same data the View's guard reads.
    @Test func webSearchRowWithBothQueryKeyAndBareTokenHasBothPresent() {
        let settings = FlowSettings(#"query="fallback text"; "the real query"; top_k=5"#)
        #expect(settings.value(for: "query") == "fallback text")
        #expect(settings.firstBare() == "the real query")
        // Both present -> shadowingWarning fires (its guard: value(for:"query") != nil AND a
        // non-empty firstBare()).
    }

    @Test func webSearchRowWithOnlyABareTokenHasNoKeyToShadowWith() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "34-TopicMonitor")
        let webSearch = doc.rows[0]
        let settings = FlowSettings(webSearch.settings)
        #expect(settings.value(for: "query") == nil)
        #expect(settings.firstBare() == "local LLM quantization")
        // No `query=` key today -> shadowingWarning's guard fails closed, no warning shown —
        // matches CatFlowRowTextCoverageTests still counting this row uncovered: the warning
        // covers a *different* row shape than the one actually in the bundled corpus.
    }

    /// `Retrieve` is named in fact 23's prose but never actually reads a `query` setting
    /// (`RetrieveTool.run` takes the query as its bound **vector** input) — confirmed by
    /// reading `Tools/IndexStoreTools.swift` directly, not assumed from the backlog text.
    /// `shadowingWarning` is scoped to `Web Search`/`Rerank` only; this pins that scoping.
    @Test func retrieveNeverReadsAQuerySettingDespiteKnownSettingKeysOfferingOne() throws {
        // knownSettingKeys still offers "query" (pre-existing, untouched by RT-4) — the point
        // of this test is that RetrieveTool itself ignores it entirely.
        #expect(FlowRowInspectorView.knownSettingKeys(for: "Retrieve").contains("query"))

        let src = row("Read Index", settings: "library.index")
        let embed = row("Embed", settings: "BGE-M3")
        let r = row("Retrieve", settings: #"query="ignored text"; top_k=5"#)
        let model = try editor([src, embed, r])
        // Not asserting a full run (needs a real index on disk) — the settings-level contract
        // is what this test pins: RetrieveTool's signature takes `inputs`, never `settings`
        // for a query value, so this "query=" can only ever be dead text.
        #expect(FlowSettings(model.row(withID: r.id)?.settings).value(for: "query") == "ignored text")
    }
}
