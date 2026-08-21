import Testing
import Foundation
import CryptoKit
@testable import MLXUI

/// Covers CFM-R2-5: the 16 bundled frame files' byte-identity (SHA-256 pinned against the
/// Python's `src/catflow/frames/`, the Swift mirror of `test_frames_doc_sync.py`) and the
/// `FrameRenderer` port of `engines/llm.py::render_frame`. When a frame legitimately changes
/// on the Python side, update the hash **in the same commit as the file** — never the other
/// way round. See `RSI/DelegateMergeBacklog.md` CFM-R2-5.
struct CatFlowFrameTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func frameURL(_ name: String) -> URL {
        repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/frames/\(name).frame.txt")
    }

    // MARK: - Byte-identity (hashes recorded 2026-08-21 from catflow-mlx @ HEAD)

    @Test func allSixteenFramesShipWithPinnedHashes() throws {
        let expected: [String: String] = [
            "Answer":    "3b065bd66df671ca328b64cdbbf8b941ac0755390ec0c40397cad703608cc83d",
            "Ask":       "a80700f1f99e71588266c96e384334fe644eb42e3f84fd317dfb1718f5c537a3",
            "Classify":  "7abf0c1f4651ec8f8f1610cb104a67d969e379d17a7ce2439264ac9074ae3cca",
            "Critique":  "355b761c0fdfa2ba245ccbeafccfd3c7649ec40b8f9080c87fd8fdc85acdcc9b",
            "Draft":     "d646c22cbbeba4342bd0ea5117492e48fa458d6d2f086efee4a66c52d6dfd0b1",
            "Gate":      "551ea11e913ea343ca5a6d79c9da0a44daee1c20189505fa40f2e79aeb2dfe8b",
            "Judge":     "3e79023f608486cac5581978cf3cc4e3acf91806a6a1390f4f2b8771d7e66743",
            "Merge":     "ad6900c68814c6d40619a3771c5f83ac9ecb9e3a4fb4b81dab91b5eb6b3a2aec",
            "Revise":    "9e3a27a5b21bf1ccfbd9e5e27f0eca9001c0b32f6a6f9b21f7aa6d47220a52f8",
            "Rewrite":   "2556ebef496c67260cf24ec303f57893c82a5b266a0feba2ab7fbedcf638a444",
            "Score":     "4ccc5d23b4a45a12c89fad573d249df6c220810eeee534e378b34c3330a62fae",
            "Summarize": "3829758c396fd157086b7927addf670041a76f6835f6dffbe9cbb9d465a0b820",
            "Think":     "d713291381201466956d2a438473f36f6f619244630ba93279412f94b4d2f730",
            "Title":     "ccb6acf9b8735778cf5860d4777fb343083437d1bff5fbde5cf76a57ee63c265",
            "Translate": "db2b44e70957b20360fd09378b31e69c1414abf6a8844533645a4db68e082991",
            "Verify":    "76946e7e2bce018fcf80500d4752c7aa0089ac5e74d23e646497f528bdcac24d",
        ]
        #expect(expected.count == 16)
        for (name, hash) in expected {
            let data = try Data(contentsOf: frameURL(name))
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            #expect(digest == hash, "\(name).frame.txt changed — update hash and file in the same commit")
        }
    }

    // MARK: - Exact render (expectation written from the Python renderer)

    @Test func summarizeFrameRendersExactExpectedString() throws {
        let frame = try FrameRenderer.loadFrame(named: "Summarize")
        let asset = Asset(items: [
            Item(kind: .text, value: "The meeting covered Q3 revenue, which beat guidance.",
                 path: nil, sourceText: nil),
        ])
        let rendered = try FrameRenderer.render(frameText: frame, settings: "TL;DR in 3 bullets", asset: asset)
        let expected = "You are a summarizer. Summarize the text below, following the\n"
            + "instruction exactly. Do not add information that isn't in the text.\n\n"
            + "Instruction: TL;DR in 3 bullets\n\n"
            + "Text:\n"
            + "The meeting covered Q3 revenue, which beat guidance.\n\n"
            + "Summary:\n"
        if rendered != expected {
            Issue.record("RENDER MISMATCH\n--- actual ---\n\(rendered)\n--- expected ---\n\(expected)")
        }
        #expect(rendered == expected)
    }

    @Test func assetListRendersNumbered() throws {
        let asset = Asset(items: [
            Item(kind: .text, value: "first", path: nil, sourceText: nil),
            Item(kind: .text, value: "second", path: nil, sourceText: nil),
        ])
        let frame = "Inputs:\n{asset list}\n"
        let rendered = try FrameRenderer.render(frameText: frame, settings: nil, asset: asset)
        #expect(rendered == "Inputs:\n1. first\n2. second\n")
    }

    @Test func settingsKeyAndTagsSubstitute() throws {
        let asset = Asset(items: [Item(kind: .text, value: "x", path: nil, sourceText: nil)])
        let frame = "{settings.lang} {settings.voice}|{tags}|{settings}"
        let rendered = try FrameRenderer.render(frameText: frame, settings: "lang=en; voice=af_heart",
                                                asset: asset, tags: ["a", "b"])
        #expect(rendered == "en af_heart|a, b|lang=en; voice=af_heart")
    }

    @Test func missingSettingsKeyBecomesEmpty() throws {
        let asset = Asset(items: [Item(kind: .text, value: "x", path: nil, sourceText: nil)])
        let rendered = try FrameRenderer.render(frameText: "[{settings.nope}]", settings: "lang=en",
                                                asset: asset)
        #expect(rendered == "[]")
    }
}
