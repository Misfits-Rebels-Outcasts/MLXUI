import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3d: the real text side against the lead's Python oracle (`<ORACLE>/<prompt>/stage_a.safetensors`). Owner-run, skipped by
/// default: needs `MLXUI_TEST_ACESTEP=1` (host: `TEST_RUNNER_MLXUI_TEST_ACESTEP=1`), the model folder (default the oracle copy,
/// or `MLXUI_TEST_ACESTEP_DIR`) and at least one stage-a dump; `MLXUI_TEST_ACESTEP_ORACLE` overrides the dump root.
/// Compares, per prompt: token ids exact; `text.hidden` cosine >= 0.995 (per-token mean) and norm ratio within 2%;
/// `text.lyric_embed` max abs diff <= 1e-3. Prints `ACESTEP-TEXT prompt=<id> cosMean=<x> cosMin=<x> normRatio=<x> lyricMaxAbs=<x>`.
enum AceStepOracle {
    nonisolated static let defaultRoot = "/Volumes/Ovid/Users/chinpinboo/Documents/PipelineStudio/RSI/notes/modelcandidates/mc5/oracle"
    nonisolated static let promptIDs = ["instrumental", "english_vocal", "chinese_vocal"]

    nonisolated static var root: URL {
        let env = ProcessInfo.processInfo.environment
        return URL(fileURLWithPath: env["MLXUI_TEST_ACESTEP_ORACLE"] ?? env["TEST_RUNNER_MLXUI_TEST_ACESTEP_ORACLE"] ?? defaultRoot)
    }

    nonisolated static func dump(_ id: String) -> URL { root.appendingPathComponent(id).appendingPathComponent("stage_a.safetensors") }

    nonisolated static var haveDumps: Bool {
        promptIDs.contains { FileManager.default.fileExists(atPath: dump($0).path) }
    }

    nonisolated static var canRun: Bool { AceStepRealLoad.canRun && haveDumps }
}

@Suite(.serialized, .enabled(if: AceStepOracle.canRun, "needs MLXUI_TEST_ACESTEP=1, the model folder and a stage_a.safetensors oracle dump"))
struct AceStepTextSideRealTests {

    @Test func textSideMatchesTheOracle() async throws {
        let dir = try #require(AceStepRealLoad.directory)
        let tokenizer = try await AceStepTokenizer(directory: dir)
        let encoder = try await AceStepTextEncoder.load(directory: dir)
        var ran = 0
        for id in AceStepOracle.promptIDs where FileManager.default.fileExists(atPath: AceStepOracle.dump(id).path) {
            let p = try #require(AceStepTextFixtures.prompts[id])
            let index = try SafetensorsIndex(url: AceStepOracle.dump(id))
            let ref = try index.read(index.names)
            let textIds = tokenizer.encodeText(caption: p.caption, duration: 10)
            let lyricIds = tokenizer.encodeLyrics(lyrics: p.lyrics, language: p.language, instrumental: p.instrumental)
            let wantText = try #require(ref["tokens.text_ids"]).asType(.int32).asArray(Int32.self).map(Int.init)
            let wantLyric = try #require(ref["tokens.lyric_ids"]).asType(.int32).asArray(Int32.self).map(Int.init)
            #expect(textIds == wantText, "\(id) text ids")
            #expect(lyricIds == wantLyric, "\(id) lyric ids")

            let out = try encoder.inputs(textIds: textIds, lyricIds: lyricIds)
            let want = try #require(ref["text.hidden"]).asType(.float32)        // [1, Lt, 1024]
            let got = out.textHidden.asType(.float32)
            #expect(got.shape == want.shape, "\(id) hidden shape")
            let dot = (got * want).sum(axis: -1)
            let cos = dot / (sqrt(got.square().sum(axis: -1)) * sqrt(want.square().sum(axis: -1)) + 1e-12)
            let cosMean = cos.mean().item(Float.self), cosMin = cos.min().item(Float.self)
            let normRatio = (sqrt(got.square().sum()) / sqrt(want.square().sum())).item(Float.self)
            #expect(cosMean >= 0.995, "\(id) hidden cosine \(cosMean)")
            #expect(abs(normRatio - 1) <= 0.02, "\(id) norm ratio \(normRatio)")

            let wantLyricEmbed = try #require(ref["text.lyric_embed"]).asType(.float32)
            #expect(out.lyricEmbeds.shape == wantLyricEmbed.shape, "\(id) lyric embed shape")
            let lyricMax = abs(out.lyricEmbeds.asType(.float32) - wantLyricEmbed).max().item(Float.self)
            #expect(lyricMax <= 1e-3, "\(id) lyric embed max abs diff \(lyricMax)")
            print(String(format: "ACESTEP-TEXT prompt=%@ cosMean=%.5f cosMin=%.5f normRatio=%.4f lyricMaxAbs=%.6f", id, cosMean, cosMin, normRatio, lyricMax))
            ran += 1
        }
        #expect(ran > 0)
    }
}
