import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R6-1: `CatSerializer` — a faithful port of `core/fmt.py::render`. The 8
/// `goldens/fmt/*.fmt.cat` fixtures reproduce byte-for-byte from `parse → serialize`; the
/// whole gallery round-trips (`serialize(parse(text)) == the file's own bytes`); and
/// `serialize(parse(x))` is idempotent.
struct CatFlowSerializerTests {

    private var fixturesDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow")
    }

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

    /// Every bundled `.cat`/`.catpipeline` across both shelves (Gallery + BasicGallery,
    /// both flattened into the same app-bundle `Contents/Resources/` root), as
    /// `(filename, directory)` pairs sorted by filename.
    private func allGalleryFiles() throws -> [(name: String, dir: URL)] {
        let suffixes = [".cat", ".catpipeline"]
        let advance = try FileManager.default.contentsOfDirectory(atPath: galleryDir.path)
            .filter { name in suffixes.contains { name.hasSuffix($0) } }
            .map { (name: $0, dir: galleryDir) }
        let basic = try FileManager.default.contentsOfDirectory(atPath: basicGalleryDir.path)
            .filter { name in suffixes.contains { name.hasSuffix($0) } }
            .map { (name: $0, dir: basicGalleryDir) }
        return (advance + basic).sorted { $0.name < $1.name }
    }

    // MARK: - The 8 exact-output goldens (parse → serialize == .fmt.cat)

    @Test func eightFmtGoldensReproduceByteForByte() throws {
        let fmtDir = fixturesDir.appendingPathComponent("goldens/fmt")
        let parseDir = fixturesDir.appendingPathComponent("goldens/parse")
        let goldens = try FileManager.default.contentsOfDirectory(atPath: fmtDir.path)
            .filter { $0.hasSuffix(".fmt.cat") }
            .sorted()
        #expect(goldens.count == 8)

        var checked = 0
        for golden in goldens {
            let name = String(golden.dropLast(".fmt.cat".count))
            let source = try String(contentsOf: parseDir.appendingPathComponent("\(name).cat"), encoding: .utf8)
            let expected = try String(contentsOf: fmtDir.appendingPathComponent(golden), encoding: .utf8)
            let doc = try CatParser.parse(source)
            let actual = CatSerializer.serialize(doc)
            if actual != expected {
                let dest = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-actual.cat")
                try? actual.write(to: dest, atomically: true, encoding: .utf8)
                Issue.record("\(name): serialize mismatch — actual written to \(dest.path)")
            }
            checked += 1
        }
        #expect(checked == 8)
    }

    // MARK: - Whole-gallery round-trip (serialize(parse(x)) == x)

    @Test func wholeGalleryRoundTripsByteForByte() throws {
        let files = try allGalleryFiles()
        #expect(files.count == 72)

        var checked = 0
        var failing: [String] = []
        for (f, dir) in files {
            let id = (f as NSString).deletingPathExtension
            let source = try String(contentsOf: dir.appendingPathComponent(f), encoding: .utf8)
            let doc = try CatParser.parse(source)
            let actual = CatSerializer.serialize(doc)
            if actual != source {
                failing.append(id)
            }
            checked += 1
        }
        #expect(checked == 72)
        #expect(failing.isEmpty,
                "non-canonical gallery files (serialize(parse(x)) != x): \(failing.joined(separator: ", "))")
        if !failing.isEmpty {
            try? failing.joined(separator: "\n").write(
                to: FileManager.default.temporaryDirectory.appendingPathComponent("r6-failures.txt"),
                atomically: true, encoding: .utf8)
        }
    }

    // MARK: - Idempotence

    @Test func serializeIsIdempotent() throws {
        let files = try allGalleryFiles()
        for (f, dir) in files {
            let source = try String(contentsOf: dir.appendingPathComponent(f), encoding: .utf8)
            let once = CatSerializer.serialize(try CatParser.parse(source))
            let twice = CatSerializer.serialize(try CatParser.parse(once))
            #expect(once == twice, "\(f): not idempotent")
        }
    }

    // MARK: - Canonicalization pairs (CFM-R6-FIX-4)

    /// The 8 `.fmt.cat` goldens are canonical-in→canonical-out, so they don't exercise
    /// canonicalization. These `.in.cat`→`.out.cat` pairs are the gallery files the phase
    /// canonicalized (`.out` generated by Python `format_flow`): they prove the serializer
    /// reorders source-order fields, relifts tags, repositions comments, moves refs to their
    /// column, relocates block markers, and re-wraps — the paths the goldens never hit.
    @Test func canonicalizationPairsMatchPython() throws {
        let dir = fixturesDir.appendingPathComponent("goldens/canonicalize")
        let ins = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.hasSuffix(".in.cat") }
            .sorted()
        #expect(ins.count >= 12, "expected ≥12 canonicalization pairs, found \(ins.count)")

        var checked = 0
        for name in ins {
            let base = String(name.dropLast(".in.cat".count))
            let input = try String(contentsOf: dir.appendingPathComponent("\(base).in.cat"), encoding: .utf8)
            let expected = try String(contentsOf: dir.appendingPathComponent("\(base).out.cat"), encoding: .utf8)
            let actual = CatSerializer.serialize(try CatParser.parse(input))
            #expect(actual == expected, "\(base): serialize(parse(in)) != Python format_flow(in)")
            checked += 1
        }
        #expect(checked == ins.count)

        // The reviewer's two named cases must be present.
        #expect(ins.contains("50-NightlyDrift.in.cat"))   // <list nightly_suite>; ctx+ → <list nightly_suite; ctx+>
        #expect(ins.contains("42-PlanExecute.in.cat"))    // re-wrap case
    }

    // MARK: - Code-point counting (CFM-R6-FIX-2/3)

    /// A Python-canonical non-ASCII flow (NFD "Café" + an emoji in `models:`) must round-trip
    /// byte-for-byte — proving the Swift measures widths in code points, not graphemes.
    @Test func nonAsciiCanonicalRoundTrips() throws {
        let url = fixturesDir.appendingPathComponent("nonascii_canonical.cat")
        let source = try String(contentsOf: url, encoding: .utf8)
        let actual = CatSerializer.serialize(try CatParser.parse(source))
        #expect(actual == source, "non-ASCII canonical flow didn't round-trip")
    }

    /// `ljust` must never truncate (the old `padding(toLength:)` dropped an NFD name's last
    /// UTF-16 unit — "Café Mode" for "Café Model").
    @Test func ljustNeverTruncatesAnOverlongName() {
        let doc = FlowDocument(
            version: "0.8",
            rows: [Row(task: "Read Text", settings: "a.txt")],
            models: ["Cafe\u{301} Model": "mlx-community/cafe", "X": "mlx-community/x"],
            modelsOrder: ["Cafe\u{301} Model", "X"])
        let out = CatSerializer.serialize(doc)
        #expect(out.contains("Cafe\u{301} Model = mlx-community/cafe"))
        #expect(out.contains("X           = mlx-community/x"))
        // The emoji (1 grapheme, 1 code point) behaves identically to the code-point ruler.
        let emoji = FlowDocument(
            version: "0.8",
            rows: [Row(task: "Read Text", settings: "a.txt")],
            models: ["🔥 Hot": "mlx-community/hot", "X": "mlx-community/x"],
            modelsOrder: ["🔥 Hot", "X"])
        let emojiOut = CatSerializer.serialize(emoji)
        #expect(emojiOut.contains("🔥 Hot = mlx-community/hot"))
        #expect(emojiOut.contains("X     = mlx-community/x"))
    }
}
