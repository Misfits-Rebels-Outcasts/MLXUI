import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R3-1: the `CacheKey` port of `core/cache.py::cache_key` (same ingredients,
/// field order, `\x1f` join, canonical settings), the `FlowSeed` port of `core/seeds.py`
/// (the `_with_seed` substitution), and `NEVER_CACHE`. Keys are pinned against values
/// produced by the Python runtime. See `RSI/DelegateMergeBacklog.md` CFM-R3-1.
struct CatFlowCacheKeyTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func textItem(_ value: String) -> Item {
        Item(kind: .text, value: value, path: nil, sourceText: nil)
    }

    private func audioItem(_ bytes: Data) throws -> Item {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-key-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("audio.bin")
        try bytes.write(to: url)
        return Item(kind: .audio, value: nil, path: url, sourceText: nil)
    }

    // MARK: - Canonical settings

    @Test func canonicalSettingsCollapseWhitespace() {
        #expect(CacheKey.canonicalSettings("  a   b\t\nc  ") == "a b c")
        #expect(CacheKey.canonicalSettings("") == "")
        #expect(CacheKey.canonicalSettings(nil) == "")
    }

    // MARK: - Golden keys (produced by the Python runtime)

    @Test func summarizeRealKeyMatchesPython() throws {
        let input = Asset(items: [textItem("The meeting covered Q3 revenue.")])
        let key = try CacheKey.cacheKey(
            task: "Summarize",
            model: "mlx-community/Qwen3-8B-4bit",
            settings: "\"TL;DR in 3 bullets\"",
            inputs: [input],
            realism: "real",
            frameVersion: CacheKey.frameVersion(task: "Summarize"))
        #expect(key == "452041c26527677c8886340036e1baeca447c614949beadb63c60604e53ca32a")
    }

    @Test func summarizeMockKeyDiffersByRealism() throws {
        let input = Asset(items: [textItem("The meeting covered Q3 revenue.")])
        let key = try CacheKey.cacheKey(
            task: "Summarize",
            model: "mlx-community/Qwen3-8B-4bit",
            settings: "\"TL;DR in 3 bullets\"",
            inputs: [input],
            realism: "mock",
            frameVersion: CacheKey.frameVersion(task: "Summarize"))
        #expect(key == "0aa9cf055419ac898d66aed0bd322719ab0922b6eb1a3726793154448578beb9")
    }

    @Test func transcribeRealKeyMatchesPython() throws {
        let input = Asset(items: [try audioItem(Data("MOCKAUDIOBYTES".utf8))])
        let key = try CacheKey.cacheKey(
            task: "Transcribe",
            model: "mlx-community/whisper-large-v3-asr-fp16",
            settings: "lang=en",
            inputs: [input],
            realism: "real")
        #expect(key == "4a32b5c7f5350abec0a60449d72926dfe0df78dea676cf4b0ffa942394353815")
    }

    @Test func transcribeSettingsChangeTheKey() throws {
        let input = Asset(items: [try audioItem(Data("MOCKAUDIOBYTES".utf8))])
        let key = try CacheKey.cacheKey(
            task: "Transcribe",
            model: "mlx-community/whisper-large-v3-asr-fp16",
            settings: "lang=fr",
            inputs: [input],
            realism: "real")
        #expect(key == "936924e22eadfff71b232aaee2bb9dfcb02df6009892532f4014a5b63735fe14")
    }

    @Test func settingsWhitespaceRunsDoNotChangeKey() throws {
        // Whitespace *runs* collapse to one space — but a space around `=` is a different
        // canonical string (`"lang = en"` vs `"lang=en"`), exactly as the Python behaves.
        let input = Asset(items: [textItem("same")])
        let a = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: "lang=en",
                                      inputs: [input], realism: "real")
        let b = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: "lang = en",
                                      inputs: [input], realism: "real")
        #expect(a != b)
        // Extra whitespace runs (spaces/tabs/newlines between tokens) collapse identically.
        let c = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: "lang=en",
                                      inputs: [input], realism: "real")
        let d = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: "  lang=en\t ",
                                      inputs: [input], realism: "real")
        #expect(c == d)
    }

    // MARK: - Frame version (folded for frame tasks)

    @Test func frameVersionFoldedForFrameTask() throws {
        // Summarize is RefKind.FRAME → frame_version is the frame file's content hash.
        let input = Asset(items: [textItem("x")])
        let with = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: nil,
                                         inputs: [input], realism: "real",
                                         frameVersion: CacheKey.frameVersion(task: "Summarize"))
        let without = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: nil,
                                            inputs: [input], realism: "real")
        #expect(with != without)
    }

    @Test func frameVersionIsNilForEngineTask() {
        #expect(CacheKey.frameVersion(task: "Transcribe") == nil)
    }

    // MARK: - NEVER_CACHE

    @Test func neverCacheMatchesPythonSet() {
        #expect(CacheKey.neverCache == [
            "Ask Human", "Download File", "Human Input", "Improvise",
            "Save Audio", "Save Context", "Save Image", "Save Images", "Save Text", "Save Video",
            "Stage Post", "Stage Send", "Store Index", "Store Write",
        ])
    }

    // MARK: - FlowSeed (core/seeds.py goldens)

    @Test func derivePinnedMatchesPython() {
        #expect(FlowSeed.derivePinned(identity: "3|Speak|mlx-community/Kokoro-82M-4bit") == 3_079_325_842)
    }

    @Test func deriveAutoMatchesPython() {
        #expect(FlowSeed.deriveAuto(runSeed: 42, activationIndex: 2) == 44)
    }

    @Test func activationIndexMatchesPython() {
        #expect(FlowSeed.activationIndex(execPath: "4") == 1)
        #expect(FlowSeed.activationIndex(execPath: "4@2") == 2)
    }

    @Test func rowIdentityStripsActivationSuffix() {
        #expect(FlowSeed.rowIdentity(path: "4@2", row: Row(task: "Speak"),
                                     modelID: "mlx-community/Kokoro-82M-4bit")
                == "4|Speak|mlx-community/Kokoro-82M-4bit")
    }

    @Test func resolveSeedSettingsMatchesPython() {
        // Omitted seed → deterministic pin appended.
        #expect(FlowSeed.resolveSeedSettings(settings: "af_heart", runSeed: 99, activationIndex: 1,
                                             identity: "3|Speak|mlx-community/Kokoro-82M-4bit")
                == "af_heart; seed=3079325842")
        // seed=auto → runSeed + k.
        #expect(FlowSeed.resolveSeedSettings(settings: "af_heart; seed=auto", runSeed: 99,
                                             activationIndex: 3, identity: "x")
                == "af_heart; seed=102")
        // Author-pinned → unchanged (nil).
        #expect(FlowSeed.resolveSeedSettings(settings: "af_heart; seed=7", runSeed: 99,
                                             activationIndex: 1, identity: "x") == nil)
    }

    @Test func runSeedIsDeterministicFromFlowText() {
        let a = FlowSeed.runSeed(for: "catflow 0.8\n1. Read Audio memo.m4a\n")
        let b = FlowSeed.runSeed(for: "catflow 0.8\n1. Read Audio memo.m4a\n")
        #expect(a == b)
        let c = FlowSeed.runSeed(for: "catflow 0.8\n1. Read Audio other.m4a\n")
        #expect(a != c)
    }
}
