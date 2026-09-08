import Testing
import Foundation
@testable import MLXUI

/// Covers the R5 pre-parsed-JSON retirement (CFM-FIX-5 / M11): the bundled gallery ships
/// `.cat` text only (no `*.parse.json` in the app bundle), and `CatParser` reproduces every
/// gallery flow's parse tree **byte-for-byte** against trees regenerated from the Python
/// runtime into `Fixtures/CatFlow/gallery/`. 72 free conformance cases — the retired
/// `*.parse.json` corpus, restored as fixtures instead of deleted. Two shelves ship the flows:
/// `MLXUI/Resources/Gallery` (Advance) and `MLXUI/Resources/BasicGallery` (Basic) — both are
/// flattened into the same app-bundle `Contents/Resources/` root, so every test here reads
/// both directories.
struct CatFlowGalleryReproductionTests {

    private var galleryDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Gallery")
    }

    private var basicGalleryDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/BasicGallery")
    }

    /// Every bundled `.cat`/`.catpipeline` filename across both shelves, sorted.
    private func allGalleryFiles() throws -> [String] {
        let suffixes = [".cat", ".catpipeline"]
        let advance = try FileManager.default.contentsOfDirectory(atPath: galleryDir.path)
            .filter { name in suffixes.contains { name.hasSuffix($0) } }
        let basic = try FileManager.default.contentsOfDirectory(atPath: basicGalleryDir.path)
            .filter { name in suffixes.contains { name.hasSuffix($0) } }
        return (advance + basic).sorted()
    }

    /// Resolve a flow's flat filename to the directory (Gallery or BasicGallery) it lives in.
    private func directory(for filename: String) -> URL {
        FileManager.default.fileExists(atPath: basicGalleryDir.appendingPathComponent(filename).path)
            ? basicGalleryDir
            : galleryDir
    }

    private var fixturesGalleryDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow/gallery")
    }

    private func canonicalJSON(_ data: Data) throws -> String {
        let obj = try JSONSerialization.jsonObject(with: data)
        return String(data: try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
                      encoding: .utf8) ?? "?"
    }

    @Test func everyGalleryFlowParses() throws {
        let files = try allGalleryFiles()
        #expect(files.count == 72, "the gallery drifted — expected 72 flows, found \(files.count)")

        var checked = 0
        for f in files {
            let id = (f as NSString).deletingPathExtension
            let text = try String(contentsOf: directory(for: f).appendingPathComponent(f), encoding: .utf8)
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

    /// CFM-FIX-5 / M11: every gallery flow's parse tree matches the Python-generated golden
    /// in `Fixtures/CatFlow/gallery/`, byte-for-byte on the canonical JSON.
    @Test func everyGalleryFlowMatchesPythonParseTree() throws {
        let files = try allGalleryFiles()
        #expect(files.count == 72)

        var checked = 0
        for f in files {
            let id = (f as NSString).deletingPathExtension
            let catURL = directory(for: f).appendingPathComponent(f)
            let doc = try CatParser.parse(try String(contentsOf: catURL, encoding: .utf8))

            let goldenURL = fixturesGalleryDir.appendingPathComponent("\(id).parse.json")
            guard FileManager.default.fileExists(atPath: goldenURL.path) else {
                Issue.record("\(id): no golden tree at \(goldenURL.path) — run the Python generator")
                continue
            }
            let actual = try canonicalJSON(try doc.toParseTreeJSON())
            let expected = try canonicalJSON(try Data(contentsOf: goldenURL))
            #expect(actual == expected, "\(id): parse tree differs from the Python golden")
            checked += 1
        }
        #expect(checked == files.count)
    }

    @Test func noParseJSONResourcesRemain() throws {
        let leftoverAdvance = try FileManager.default.contentsOfDirectory(atPath: galleryDir.path)
            .filter { $0.hasSuffix(".parse.json") }
        let leftoverBasic = try FileManager.default.contentsOfDirectory(atPath: basicGalleryDir.path)
            .filter { $0.hasSuffix(".parse.json") }
        #expect(leftoverAdvance.isEmpty, "pre-parsed JSON retired — found: \(leftoverAdvance)")
        #expect(leftoverBasic.isEmpty, "pre-parsed JSON retired — found: \(leftoverBasic)")
    }

    @Test func galleryLoaderParsesAtRuntime() throws {
        // The loader path the app actually uses must work without any .parse.json.
        let doc = try GalleryLoader.loadDocument(flowID: "01-SpokenSummary")
        #expect(doc.version == "0.8")
        #expect(doc.rows.map(\.task) == ["Read Audio", "Transcribe", "Summarize", "Speak", "Save Audio", "Save Text"])
    }
}
