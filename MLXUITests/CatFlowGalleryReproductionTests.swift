import Testing
import Foundation
import ImageIO
import CryptoKit
@testable import MLXUI

/// Covers the R5 pre-parsed-JSON retirement (CFM-FIX-5 / M11): the bundled gallery ships
/// `.cat` text only (no `*.parse.json` in the app bundle), and `CatParser` reproduces every
/// gallery flow's parse tree **byte-for-byte** against trees regenerated from the Python
/// runtime into `Fixtures/CatFlow/gallery/`. 76 free conformance cases — the retired
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
        #expect(files.count == 77, "the gallery drifted — expected 77 flows, found \(files.count)")

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

    /// DA-7 (`RSI/DelegateDeciderBacklog.md`, owner ruling 2026-09-09, **SPEC-Q215**) — a
    /// deliberate, temporary corpus divergence. `22-ScreenshotHowTo`, `30-ReceiptsExpense` and
    /// `55-ReceiptsLedger` now name `GLM-OCR` on their previously model-less OCR row (MLXUI's
    /// OCR is model-routed with no tesseract fallback; the reference's SPEC-Q95 path isn't
    /// ported). The Python-generated parse trees still record `"model": null` there.
    /// `catflow-mlx` will be made compatible (the gallery flows **and** an upstream curated
    /// manifest — GLM-OCR's is MLXUI-authored). **Delete this set** and restore the
    /// unconditional assert the moment `scripts/sync-catflow-fixtures.sh` re-syncs.
    private let pendingUpstreamOCRModel: Set<String> = [
        "22-ScreenshotHowTo", "30-ReceiptsExpense", "55-ReceiptsLedger",
    ]

    /// CFM-FIX-5 / M11: every gallery flow's parse tree matches the Python-generated golden
    /// in `Fixtures/CatFlow/gallery/`, byte-for-byte on the canonical JSON.
    @Test func everyGalleryFlowMatchesPythonParseTree() throws {
        let files = try allGalleryFiles()
        #expect(files.count == 77)

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

            if pendingUpstreamOCRModel.contains(id) {
                // SPEC-Q215: normalise ONLY the OCR row's `model` field, then require a
                // byte-for-byte match — any *other* drift in these three files still fails.
                let marker = "\"model\":\"GLM-OCR\""
                if expected.contains(marker) {
                    Issue.record("\(id): the Python golden now names the OCR model — fixtures re-synced; DELETE `pendingUpstreamOCRModel` and restore the unconditional assert.")
                    checked += 1
                    continue
                }
                let hits = actual.components(separatedBy: marker).count - 1
                #expect(hits == 1, "\(id): expected exactly one OCR row naming `GLM-OCR`, found \(hits) — the row edit or the parser changed.")
                #expect(actual.replacingOccurrences(of: marker, with: "\"model\":null") == expected,
                        "\(id): parse tree differs from the Python golden beyond DA-7's OCR-model divergence.")
                checked += 1
                continue
            }

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

    // MARK: - BG-3: the two shelves are one flattened bundle

    /// BG-3 (`RSI/DelegateBasicImageFlowsBacklog.md`, owner rulings R3 + R4, 2026-10-10) moved
    /// 63/64/65 onto the Basic shelf as 6/7/8 **keeping their `number` values**, so the shelf a
    /// flow is on and its badge number are now deliberately different things: Basic shows
    /// positions 1…8 (`FlowGalleryView`'s `displayNumber: index + 1`) while the numbers stay
    /// unique in the global sequence. This pins that arrangement as data — the shelf order is
    /// `_metadata.json` file order, so a re-sort or an inserted entry silently renumbers the
    /// shelf badge and nothing else would notice.
    @Test func theBasicShelfIsEightSourcedPositionsWhileKeepingGlobalNumbers() throws {
        let entries = GalleryLoader.loadMetadata()
        #expect(entries.count == 77, "a move, never an add — the gallery stays 77")
        #expect(entries.count == Set(entries.map(\.number)).count, "gallery numbers must stay unique")

        let basic = entries.filter(\.isBasic)
        #expect(basic.count == 8, "Basic Gallery is 1…8 now: 70,71,72,75,76,63,64,65")
        // Shelf **order** is metadata order, and the three moved entries sit after 76 (owner
        // ruling R4: they read 6, 7, 8 — not 1–3, which is where they would land if they had
        // been left in numeric position or appended before the existing basics).
        #expect(basic.map(\.flowID) == ["1-TranscribeAudio", "2-SummaryFromAudio",
                                        "3-ExtractTableFromImage", "4-WebPageSummary",
                                        "5-ExtractWebPageLinks", "6-CutOutSubject",
                                        "7-UpscaleSmall", "8-VoiceoverBed"])
        // R4: the numbers are kept, not renumbered — 63/64/65 are still 63/64/65.
        #expect(basic.suffix(3).map(\.number) == [63, 64, 65])

        // Advance keeps every other number, with 63–65 simply absent (the gap R4 chose), and
        // 66–69/73/74/77 untouched.
        let advance = entries.filter { !$0.isBasic }.map(\.number)
        #expect(advance.count == 69)
        #expect(!advance.contains(63) && !advance.contains(64) && !advance.contains(65))
        for kept in [62, 66, 67, 68, 69, 73, 74, 77] {
            #expect(advance.contains(kept), "\(kept) must not be renumbered or moved (R4)")
        }
    }

    /// The mechanism R3 depends on: both shelves flatten into one `Contents/Resources/` root,
    /// so a **duplicate flat filename** would be resolved by the bundle in an undefined order —
    /// the failure mode being the *wrong flow's* `.cat` text loading with a green parse test.
    /// `GalleryLoader`'s own doc comment calls the names globally unique; this makes that an
    /// assertion rather than a comment, and it is the check that would have caught a careless
    /// `git mv` into `BasicGallery/` reusing an existing `photo.jpg`-style name.
    @Test func noFlatFilenameCollidesAcrossTheTwoShelves() throws {
        let suffixes = [".cat", ".catpipeline"]
        func names(_ dir: URL) throws -> [String] {
            try FileManager.default.contentsOfDirectory(atPath: dir.path)
        }
        let advanceCats = try names(galleryDir).filter { n in suffixes.contains { n.hasSuffix($0) } }
        // Every bundled resource in both dirs, not just the flows — the assets flatten too.
        let basicAll = try names(basicGalleryDir)
        let duplicates = Set(advanceCats).intersection(Set(basicAll))
        #expect(duplicates.isEmpty, "flattened-bundle name collision: \(duplicates.sorted())")

        // And the three moved flows really are on the Basic shelf, loadable, with Fixtures goldens.
        for id in ["6-CutOutSubject", "7-UpscaleSmall", "8-VoiceoverBed"] {
            #expect(basicAll.contains("\(id).cat"), "\(id).cat should live in BasicGallery/")
            #expect(try !GalleryLoader.rawCatText(flowID: id).isEmpty)
            let golden = fixturesGalleryDir.appendingPathComponent("\(id).parse.json")
            #expect(FileManager.default.fileExists(atPath: golden.path),
                    "\(id) needs its renamed fixture — 63/64/65 were `git mv`d, never edited")
        }
    }

    // MARK: - BG-4: every Basic flow's sample input is actually bundled

    /// BG-4 (`RSI/DelegateBasicImageFlowsBacklog.md`, owner ruling R1, 2026-10-10). The reason
    /// 63/64 never ran is that nobody had ever checked the *other* direction: the `.cat` names a
    /// file, `bundledAssets` is the only thing that puts it in the flow's folder, and nothing
    /// asserted the two agreed. `FlowInputAdvisory` said so at Run time and only for whichever
    /// flow a human happened to open.
    ///
    /// This closes the loop for **every** Basic Gallery flow: each `Read *` row's own path
    /// resolves to a **destination** in that flow's `bundledAssets`, that entry's **source** is
    /// a file that really exists under `Resources/BasicGallery/**` (the flattened bundle's flat
    /// name), and — for images — it is a real decodable image. A flow that reads a file with no
    /// asset entry is exactly the "Read Image needs 'photo.jpg'" bug; the source-existence half
    /// is the neglected one that a directory rename would silently break.
    @Test func everyBasicGalleryReadRowHasABundledSourceFile() throws {
        // Walk the whole BasicGallery tree: the resources are flat in the bundle but nested in
        // the repo (`<flowID>/<flat name>`), the `3-ExtractTableFromImage-budget.png` layout.
        var sources: [String: URL] = [:]
        if let walker = FileManager.default.enumerator(at: basicGalleryDir,
                                                      includingPropertiesForKeys: [.isRegularFileKey]) {
            for case let url as URL in walker {
                guard (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
                else { continue }
                sources[url.lastPathComponent] = url
            }
        }

        let basics = GalleryLoader.loadMetadata().filter(\.isBasic)
        #expect(basics.count == 8, "the Basic shelf is 1…8 after BG-3")

        var checked = 0
        for flow in basics {
            let doc = try GalleryLoader.loadDocument(flowID: flow.flowID)
            let assets = GalleryLoader.bundledAssets(flowID: flow.flowID)
            let destinations = Set(assets.map(\.destination))

            for row in doc.rows {
                guard let task = row.task, SampleSeed.readTaskNames.contains(task) else { continue }
                // A row fed by a matching upstream, or one with no path, has nothing to bundle
                // — the same precedence `FlowInputFile` uses, so this never over-warns.
                guard row.refs.isEmpty,
                      let raw = FlowSettings(row.settings).pathValue() else { continue }
                #expect(destinations.contains(raw),
                        "\(flow.flowID): \(task) reads '\(raw)' but `bundledAssets` has no entry for it — Run would say it isn't in the flow's folder")
                checked += 1
            }

            // Every declared asset must have a real source file, or the copy silently no-ops.
            for asset in assets {
                let url = sources[asset.source]
                #expect(url != nil,
                        "\(flow.flowID): asset source '\(asset.source)' is not under MLXUI/Resources/BasicGallery/**")
                guard let url else { continue }
                if asset.destination.hasSuffix(".jpg") || asset.destination.hasSuffix(".png") {
                    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                        Issue.record("\(flow.flowID): '\(asset.source)' is not a decodable image")
                        continue
                    }
                    #expect(img.width > 0 && img.height > 0)
                }
            }
        }
        // The two flows BG-4 fixed must be among them, or this sweep proves nothing about them.
        #expect(checked >= 2, "expected at least 6- and 7-'s Read rows to be checked, got \(checked)")
    }

    /// BG-7 (`RSI/DelegateBasicImageFlowsBacklog.md`, owner ruling R6, 2026-10-12): the shipped
    /// sample inputs are the **owner's own kitten photo**, not BG-4's placeholders — full size
    /// for `6`, exact half size for `7` so a 2× upscale lands back at roughly the original. This
    /// pins both the geometry and the **bytes**, so any future swap is a deliberate edit with a
    /// failing test to explain, never a silent change of what the smoke rows exercise. The hash
    /// half matters: the dimensions alone would not notice a re-encode, a colour-profile change,
    /// or a different crop of the same size.
    ///
    /// **BG-9** (2026-10-12) stripped the source's EXIF/Photoshop metadata from both files, so
    /// the hashes below are the **post-strip** bytes. See `theSampleInputsCarryNoMetadata` for
    /// the other half of that change.
    @Test func theKittenInputsMatchTheShapeAndBytesTheSmokeRowsAssume() throws {
        func fixture(_ flat: String) throws -> (size: (Int, Int), sha: String) {
            let url = try fixtureURL(flat)
            let src = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
            let img = try #require(CGImageSourceCreateImageAtIndex(src, 0, nil))
            let sha = SHA256.hash(data: try Data(contentsOf: url)).hex
            return ((img.width, img.height), sha)
        }

        // 6 — the owner's photo at **full size**, 740×503, so SAM's centre point has a real
        // subject under it (the right-hand kitten's chest/paw; row 103 records what it picks).
        let six = try fixture("6-CutOutSubject-photo.jpg")
        #expect(six.size == (740, 503))
        #expect(six.sha == "36b9cb9e3eb45df2be4cdf642d633fcdbd2312942ac5cb8a4cb623c9d62d3043")

        // 7 — the same photo at exactly **half** size, 370×252, so 2× returns ~740×504.
        let seven = try fixture("7-UpscaleSmall-small-photo.jpg")
        #expect(seven.size == (370, 252))
        #expect(seven.sha == "a0d3183516de9a431232b6d1b90d33bb9fde1f4dc69f41ce8c74c0865e903893")
    }

    /// BG-9 (`RSI/DelegateBasicImageFlowsBacklog.md`, 2026-10-12): the shipped sample JPEGs must
    /// carry **no metadata** — the source had `dc:creator` / `UserComment`
    /// `"1.90.5-PYCZW7L356EJMUH44D4W726LBE.0.1-9"` in an EXIF block plus a Photoshop IRB block
    /// (both visible in the JPEG's APP1/APP13 segments), which a shipped sample has no business
    /// carrying.
    ///
    /// Asserts on **every** APP1…APP15 and COM segment, not just the two names — "no metadata at
    /// all" is the requirement, and a future re-encode would otherwise be free to reintroduce ICC
    /// or IPTC blocks that this test would not see. APP0 (JFIF) is deliberately allowed: it is
    /// structural (colour space + density), not an annotation, and every decoder reads it.
    ///
    /// This is also the test that would have caught the *first* attempt at this change: ImageIO's
    /// own encoder and its lossless copy path both **synthesise** a minimal EXIF block and an
    /// empty "Photoshop 3.0" IPTC/digest block, so a re-encode cannot satisfy it. The files ship
    /// with those segments removed from the container instead (journal `2026-467`).
    @Test func theSampleInputsCarryNoMetadata() throws {
        for flat in ["6-CutOutSubject-photo.jpg", "7-UpscaleSmall-small-photo.jpg"] {
            let data = try Data(contentsOf: try fixtureURL(flat))
            let bytes = [UInt8](data)
            #expect(bytes.count > 4 && bytes[0] == 0xFF && bytes[1] == 0xD8, "\(flat) is not a JPEG")

            var offset = 2
            var found: [String] = []
            while offset + 3 < bytes.count {
                guard bytes[offset] == 0xFF else { break }
                let marker = bytes[offset + 1]
                if marker == 0xDA { break }                       // SOS: entropy data follows
                if marker == 0xD8 || (0xD0 ... 0xD7).contains(marker) {
                    offset += 2
                    continue
                }
                let length = Int(bytes[offset + 2]) << 8 | Int(bytes[offset + 3])
                let start = offset + 4
                let end = min(offset + 2 + length, bytes.count)
                let payload = bytes[start ..< end]
                // APP1…APP15 and COM are the annotation segments; APP0 is JFIF and stays.
                if (0xE1 ... 0xEF).contains(marker) || marker == 0xFE {
                    let name: String
                    if payload.starts(with: Array("Exif".utf8)) { name = "EXIF" }
                    else if payload.starts(with: Array("http://ns.adobe.com/xap/1.0/".utf8)) { name = "XMP" }
                    else if payload.starts(with: Array("http://ns.adobe.com/xmp/extension/".utf8)) { name = "XMP (extension)" }
                    else if payload.starts(with: Array("Photoshop 3.0".utf8)) { name = "Photoshop IRB" }
                    else if payload.starts(with: Array("ICC_PROFILE".utf8)) { name = "ICC profile" }
                    else { name = "unclassified" }
                    found.append("APP\(marker - 0xE0) \(name)")
                }
                offset += 2 + length
            }
            #expect(found.isEmpty, "\(flat) still carries metadata segments: \(found)")
            // And the walk really did reach the image data — otherwise an early `break` would make
            // the assertion above vacuously true.
            #expect(offset + 1 < bytes.count && bytes[offset + 1] == 0xDA,
                    "\(flat): the segment walk should stop at SOS")
        }
    }

    /// Locate a flat fixture name anywhere under `Resources/BasicGallery/**` (the repo keeps them
    /// nested by flow; the bundle flattens them).
    private func fixtureURL(_ flat: String) throws -> URL {
        var found: URL?
        let walker = FileManager.default.enumerator(at: basicGalleryDir,
                                                   includingPropertiesForKeys: nil)
        while let url = walker?.nextObject() as? URL {
            if url.lastPathComponent == flat { found = url; break }
        }
        return try #require(found, "\(flat) not found under BasicGallery/**")
    }

    /// The end-to-end half of BG-4: prepare a real flow folder from the **built bundle** the way
    /// a Run does, and assert the sentence the owner originally reported is gone. The
    /// metadata-level test above proves the table and the files agree; this proves the copy
    /// actually happens and `FlowInputFile` is then satisfied — the exact function
    /// `FlowInputAdvisory` calls to produce "…which isn't in the flow's folder yet."
    @Test @MainActor func flowsSixAndSevenHaveNoMissingInputBannerAfterPrepare() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("bg4-prepare-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let ws = try #require(FlowWorkspace(root: root))
        let sourceDir = try #require(GalleryLoader.resourcesDirectory,
                                     "the test host's bundle must have its Resources/")

        for flowID in ["6-CutOutSubject", "7-UpscaleSmall"] {
            try ws.prepare(flowID: flowID, sourceDir: sourceDir,
                           bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
            let doc = try GalleryLoader.loadDocument(flowID: flowID)
            let scope = FlowScope(identity: flowID, workspace: ws, locationID: flowID)
            let advisory = FlowInputAdvisory.advisory(for: doc, scope: scope)
            #expect(advisory == nil,
                    "\(flowID) still shows the missing-input banner: \(advisory?.reason ?? "")")
        }
    }
}
