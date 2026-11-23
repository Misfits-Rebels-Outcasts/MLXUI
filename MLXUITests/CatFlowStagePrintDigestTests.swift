import Testing
import Foundation
import CryptoKit
@testable import MLXUI

/// ER-FIX-6 (SPEC-Q259; owner ruling F1, Boo, 2026-11-22): the `Stage Print` id of a PDF ignores `/CreationDate`, `/ModDate` and the
/// trailer `/ID`, identically to the Python. The byte rule is the synced golden `goldens/check/q259_pdf_digest.json`.
struct CatFlowStagePrintDigestTests {
    private struct DigestCase: Decodable { var name: String; var input: String; var expected: String }
    private struct Golden: Decodable { var cases: [DigestCase] }

    private var goldenURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow/goldens/check/q259_pdf_digest.json")
    }

    @Test func theByteRuleMatchesThePythonGolden() throws {
        let cases = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: goldenURL)).cases
        #expect(cases.count == 8)
        for c in cases {
            let input = try #require(c.input.data(using: .isoLatin1), "\(c.name)")
            let expected = try #require(c.expected.data(using: .isoLatin1), "\(c.name)")
            #expect(StagePrintTool.pdfDigestBytes(input) == expected, "\(c.name)")
        }
    }

    private func scratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("q259-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func makePDF(_ text: String, in dir: URL) throws -> Asset {
        let input = [Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])]
        return try MakePDFTool.make(inputs: input, settings: nil, outDirectory: dir)
    }

    private func id(of asset: Asset) throws -> String {
        try StagePrintTool.prepare(settings: nil, inputs: [asset], path: "3").id
    }

    @Test func theSameTextThroughMakePDFTwiceHasOneId() async throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let first = try makePDF("Ship to: A. Example\n12 Example Road", in: dir.appendingPathComponent("a"))
        try await Task.sleep(nanoseconds: 1_200_000_000)  // the creation date has one-second resolution
        let second = try makePDF("Ship to: A. Example\n12 Example Road", in: dir.appendingPathComponent("b"))
        let a = try Data(contentsOf: try #require(first.items.first?.path))
        let b = try Data(contentsOf: try #require(second.items.first?.path))
        #expect(a != b, "premise: the raw bytes differ run to run")
        #expect(try id(of: first) == id(of: second))
    }

    @Test func aDifferentTextIsADifferentId() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let a = try makePDF("Ship to: A. Example", in: dir.appendingPathComponent("a"))
        let b = try makePDF("Ship to: B. Example", in: dir.appendingPathComponent("b"))
        #expect(try id(of: a) != id(of: b))
    }

    @Test func theStoredFileKeepsItsBytesAndTheEntryItsRawDigest() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let asset = try makePDF("keep my bytes", in: dir.appendingPathComponent("a"))
        let raw = try Data(contentsOf: try #require(asset.items.first?.path))
        let prepared = try StagePrintTool.prepare(settings: nil, inputs: [asset], path: "3")
        #expect(prepared.data == raw)
        #expect(prepared.digest == SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined())
    }
}
