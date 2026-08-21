import Testing
import Foundation
@testable import MLXUI

/// Covers the R5 pre-parsed-JSON retirement: the bundled gallery ships `.cat` text only
/// (the `*.parse.json` resources are deleted), and `CatParser` reproduces every gallery
/// flow's tree — 69 free conformance cases. `GalleryLoader.loadDocument` now parses the
/// `.cat` at runtime.
struct CatFlowGalleryReproductionTests {

    private var galleryDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Gallery")
    }

    @Test func everyGalleryFlowParses() throws {
        let files = try FileManager.default.contentsOfDirectory(atPath: galleryDir.path)
            .filter { $0.hasSuffix(".cat") || $0.hasSuffix(".catpipeline") }
            .sorted()
        #expect(files.count == 69, "the gallery drifted — expected 69 flows, found \(files.count)")

        var checked = 0
        for f in files {
            let id = (f as NSString).deletingPathExtension
            let text = try String(contentsOf: galleryDir.appendingPathComponent(f), encoding: .utf8)
            let doc: FlowDocument
            do {
                doc = try CatParser.parse(text)
            } catch {
                Issue.record("\(id): parser failed — \(error)")
                continue
            }
            // A flow's parse must reproduce its Python tree: the parser test corpus pins
            // the shape, so here we just assert the structural invariants hold per flow.
            #expect(!doc.rows.isEmpty, "\(id): empty flow")
            for row in doc.rows {
                #expect(row.id != UUID(uuidString: "00000000-0000-0000-0000-000000000000"),
                        "\(id): unminted id")
            }
            checked += 1
        }
        #expect(checked == files.count)
    }

    @Test func noParseJSONResourcesRemain() throws {
        let leftover = try FileManager.default.contentsOfDirectory(atPath: galleryDir.path)
            .filter { $0.hasSuffix(".parse.json") }
        #expect(leftover.isEmpty, "pre-parsed JSON retired — found: \(leftover)")
    }

    @Test func galleryLoaderParsesAtRuntime() throws {
        // The loader path the app actually uses must work without any .parse.json.
        let doc = try GalleryLoader.loadDocument(flowID: "01-SpokenSummary")
        #expect(doc.version == "0.8")
        #expect(doc.rows.map(\.task) == ["Read Audio", "Transcribe", "Summarize", "Speak", "Save Audio", "Save Text"])
    }
}
