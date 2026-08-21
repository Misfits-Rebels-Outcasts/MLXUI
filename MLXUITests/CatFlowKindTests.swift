import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R1-2: the ported `Kind`/`Shape`/`Item`/`Asset` type system from
/// `catflow-mlx/src/catflow/core/{kinds,shapes,assets}.py`. The `singleCompatible` /
/// `bundleCompatible` truth tables are ported directly from `shapes.py`, so "check and run
/// can't drift". See `RSI/DelegateMergeBacklog.md` CFM-R1-2.
struct CatFlowKindTests {

    // MARK: - Kind enum (test_kinds.py)

    @Test func allFourteenKindsPresent() {
        let names = Set(Kind.allCases.map(\.rawValue))
        #expect(names == [
            "text", "audio", "image", "video", "file", "folder",
            "index", "vector", "table", "status", "context", "occurrence",
            "model", "latent",
        ])
    }

    @Test func shapeSignatureTexts() {
        #expect(Shape.single(.text).signatureText == "text")
        #expect(Shape.listOf(.image).signatureText == "[image]")
        #expect(Shape.tupleOf([.text, .vector]).signatureText == "[text, vector]")
        #expect(Shape.unionOf([.audio, .video]).signatureText == "audio|video")
        #expect(Shape.anyKind.signatureText == "anything")
        #expect(Shape.sameAsInput.signatureText == "same")
    }

    @Test func shapesAreEquatable() {
        #expect(Shape.single(.text) == Shape.single(.text))
        #expect(Shape.listOf(.text) != Shape.single(.text))
        #expect(Shape.tupleOf([.text, .vector]) == Shape.tupleOf([.text, .vector]))
    }

    // MARK: - singleCompatible truth table (shapes.py)

    @Test func frameRefKindRequiresText() {
        // rk == FRAME: baseKind(given) == text, before any accepts branch.
        #expect(Shape.singleCompatible(accepts: .anyKind, given: .single(.text), rk: .frame) == true)
        #expect(Shape.singleCompatible(accepts: .anyKind, given: .single(.audio), rk: .frame) == false)
        // base_kind of ListOf(TEXT) is TEXT — still compatible under FRAME.
        #expect(Shape.singleCompatible(accepts: .anyKind, given: .listOf(.text), rk: .frame) == true)
        // FRAME overrides the accepts branch: given is text, so it's compatible.
        #expect(Shape.singleCompatible(accepts: .single(.audio), given: .single(.text), rk: .frame) == true)
    }

    @Test func anyKindAcceptsEverything() {
        #expect(Shape.singleCompatible(accepts: .anyKind, given: .single(.text), rk: nil))
        #expect(Shape.singleCompatible(accepts: .anyKind, given: .single(.audio), rk: nil))
        #expect(Shape.singleCompatible(accepts: .anyKind, given: .listOf(.image), rk: nil))
    }

    @Test func unionAcceptsSingleMember() {
        let accepts = Shape.unionOf([.audio, .video])
        #expect(Shape.singleCompatible(accepts: accepts, given: .single(.audio), rk: nil))
        #expect(Shape.singleCompatible(accepts: accepts, given: .single(.video), rk: nil))
        #expect(!Shape.singleCompatible(accepts: accepts, given: .single(.text), rk: nil))
        #expect(!Shape.singleCompatible(accepts: accepts, given: .listOf(.audio), rk: nil))
    }

    @Test func singleAcceptsSameKindSingle() {
        #expect(Shape.singleCompatible(accepts: .single(.text), given: .single(.text), rk: nil))
        #expect(!Shape.singleCompatible(accepts: .single(.text), given: .single(.audio), rk: nil))
        #expect(!Shape.singleCompatible(accepts: .single(.text), given: .listOf(.text), rk: nil))
        #expect(!Shape.singleCompatible(accepts: .single(.text), given: .anyKind, rk: nil))
    }

    @Test func listOfAcceptsLiftsSingleAndTuple() {
        let accepts = Shape.listOf(.text)
        // SPEC-Q71(a): a Single(K) given lifts to a one-item [K].
        #expect(Shape.singleCompatible(accepts: accepts, given: .single(.text), rk: nil))
        #expect(Shape.singleCompatible(accepts: accepts, given: .listOf(.text), rk: nil))
        // SPEC-Q71(b): a homogeneous TupleOf given fills a ListOf accepts.
        #expect(Shape.singleCompatible(accepts: accepts, given: .tupleOf([.text, .text, .text]), rk: nil))
        // Wrong kind fails.
        #expect(!Shape.singleCompatible(accepts: accepts, given: .single(.audio), rk: nil))
        #expect(!Shape.singleCompatible(accepts: accepts, given: .listOf(.audio), rk: nil))
        #expect(!Shape.singleCompatible(accepts: accepts, given: .tupleOf([.text, .audio]), rk: nil))
    }

    @Test func tupleAcceptsNeedsExplicitBundle() {
        // TupleOf accepts needs a multi-ref bundle — single_compatible returns false.
        let accepts = Shape.tupleOf([.text, .vector])
        #expect(!Shape.singleCompatible(accepts: accepts, given: .single(.text), rk: nil))
        #expect(Shape.singleCompatible(accepts: accepts, given: .anyKind, rk: nil) == false)
    }

    @Test func sameAsInputIsNotAnAccepts() {
        #expect(!Shape.singleCompatible(accepts: .sameAsInput, given: .single(.text), rk: nil))
    }

    // MARK: - bundleCompatible truth table (shapes.py)

    @Test func bundleFrameRequiresAllText() {
        #expect(Shape.bundleCompatible(
            accepts: .tupleOf([.text, .text]),
            given: [.single(.text), .single(.text)], rk: .frame) == true)
        #expect(Shape.bundleCompatible(
            accepts: .tupleOf([.text, .text]),
            given: [.single(.text), .single(.audio)], rk: .frame) == false)
    }

    @Test func bundleAnyKindAcceptsAnything() {
        #expect(Shape.bundleCompatible(accepts: .anyKind, given: [.single(.text)], rk: nil))
        #expect(Shape.bundleCompatible(accepts: .anyKind, given: [.single(.text), .single(.audio)], rk: nil))
    }

    @Test func bundleSingleRefDelegatesToSingleCompatible() {
        #expect(Shape.bundleCompatible(accepts: .single(.text), given: [.single(.text)], rk: nil))
        #expect(!Shape.bundleCompatible(accepts: .single(.text), given: [.single(.audio)], rk: nil))
        #expect(Shape.bundleCompatible(accepts: .anyKind, given: [.single(.text)], rk: nil))
    }

    @Test func bundleTupleChecksArityAndKinds() {
        let accepts = Shape.tupleOf([.text, .vector])
        #expect(Shape.bundleCompatible(
            accepts: accepts, given: [.single(.text), .single(.vector)], rk: nil))
        #expect(!Shape.bundleCompatible(
            accepts: accepts, given: [.single(.text)], rk: nil))
        #expect(!Shape.bundleCompatible(
            accepts: accepts, given: [.single(.text), .single(.image)], rk: nil))
        // A ListOf ref fills exactly one position via base_kind.
        #expect(Shape.bundleCompatible(
            accepts: accepts, given: [.listOf(.text), .single(.vector)], rk: nil))
    }

    @Test func bundleListChecksEachBaseKind() {
        // SPEC-Q117: each ref in a multi-ref bundle fills one position.
        let accepts = Shape.listOf(.text)
        #expect(Shape.bundleCompatible(
            accepts: accepts, given: [.single(.text), .listOf(.text)], rk: nil))
        #expect(!Shape.bundleCompatible(
            accepts: accepts, given: [.single(.text), .single(.audio)], rk: nil))
        #expect(!Shape.bundleCompatible(
            accepts: accepts, given: [.single(.text), .tupleOf([.text, .audio])], rk: nil))
    }

    // MARK: - signature (shapes.py)

    /// A minimal RowShape for testing `Shape.signature`.
    private struct TestRow: RowShape {
        var task: String?
        var blockKind: BlockKind?
        var children: [any RowShape]
    }

    @Test func signatureResolvesTaskViaLookup() {
        let row = TestRow(task: "Summarize", blockKind: nil, children: [])
        let sig = Shape.signature(row) { task in
            task == "Summarize" ? (.single(.text), .single(.text)) : nil
        }
        #expect(sig?.0 == .single(.text))
        #expect(sig?.1 == .single(.text))
    }

    @Test func signatureUnknownTaskIsNil() {
        let row = TestRow(task: "Nope", blockKind: nil, children: [])
        #expect(Shape.signature(row, lookup: { _ in nil }) == nil)
    }

    @Test func signatureEachLiftsSingleToList() {
        let leaf = TestRow(task: "Read Image", blockKind: nil, children: [])
        let each = TestRow(task: nil, blockKind: .each, children: [leaf])
        let sig = Shape.signature(each) { task in
            task == "Read Image" ? (.single(.image), .single(.image)) : nil
        }
        #expect(sig?.0 == .listOf(.image))
        #expect(sig?.1 == .listOf(.image))
    }

    @Test func signatureBlockUsesFirstAndLastChild() {
        let first = TestRow(task: "Split", blockKind: nil, children: [])
        let last = TestRow(task: "Join Text", blockKind: nil, children: [])
        let list = TestRow(task: nil, blockKind: .list, children: [first, last])
        let sig = Shape.signature(list) { task in
            switch task {
            case "Split":      return (.single(.text), .listOf(.text))
            case "Join Text":  return (.listOf(.text), .single(.text))
            default:           return nil
            }
        }
        #expect(sig?.0 == .single(.text))     // first child's accepts
        #expect(sig?.1 == .single(.text))     // last child's gives
    }

    @Test func signatureEmptyBlockIsNil() {
        let empty = TestRow(task: nil, blockKind: .list, children: [])
        #expect(Shape.signature(empty, lookup: { _ in nil }) == nil)
    }

    // MARK: - Item/Asset content hash (assets.py, cache-key right half)

    @Test func itemHashIsStableAcrossConstructions() throws {
        let a = Item(kind: .text, value: "hello", path: nil, sourceText: nil)
        let b = Item(kind: .text, value: "hello", path: nil, sourceText: nil)
        #expect(try a.contentHash() == b.contentHash())
    }

    @Test func itemHashDiffersForDifferentContent() throws {
        let a = Item(kind: .text, value: "hello", path: nil, sourceText: nil)
        let b = Item(kind: .text, value: "world", path: nil, sourceText: nil)
        #expect(try a.contentHash() != b.contentHash())
    }

    @Test func itemHashDistinguishesKind() throws {
        let text = Item(kind: .text, value: "x", path: nil, sourceText: nil)
        let status = Item(kind: .status, value: "x", path: nil, sourceText: nil)
        #expect(try text.contentHash() != status.contentHash())
    }

    @Test func fileBackedItemHashesFileContents() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-hash-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let f1 = dir.appendingPathComponent("a.bin")
        let f2 = dir.appendingPathComponent("b.bin")
        try Data("payload".utf8).write(to: f1)
        try Data("payload".utf8).write(to: f2)
        let item1 = Item(kind: .audio, value: nil, path: f1, sourceText: nil)
        let item2 = Item(kind: .audio, value: nil, path: f2, sourceText: nil)
        #expect(try item1.contentHash() == item2.contentHash())

        try Data("different".utf8).write(to: f2)
        #expect(try item1.contentHash() != item2.contentHash())
    }

    @Test func assetShapeIsItemKinds() {
        let asset = Asset(items: [
            Item(kind: .text, value: "a", path: nil, sourceText: nil),
            Item(kind: .text, value: "b", path: nil, sourceText: nil),
        ])
        #expect(asset.shape == [.text, .text])
    }
}
