import Foundation
import Testing
@testable import MLXUI

/// CW-3 — the C11 order for the context window: the catalog's `contextWindow`, then the installed
/// `config.json`, then 8 192. The owner's own test model is the reason the config fallback exists
/// (`mlx-community--Qwen3-8B-4bit` has `contextWindow: nil` in the catalog).
struct ContextWindowTests {

    private func makeModelDirectory(config: String?) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ctx-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let config {
            try config.write(
                to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        }
        return dir
    }

    @Test func theCatalogWinsWhenItHasAValue() throws {
        let dir = try makeModelDirectory(config: #"{"max_position_embeddings": 4096}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        let entry = makeEntry(contextWindow: 32_768)
        #expect(ContextWindow.tokens(for: entry, modelDirectory: dir) == 32_768)
    }

    @Test func theConfigIsTheFallbackWhenTheCatalogIsNil() throws {
        let dir = try makeModelDirectory(config: #"{"max_position_embeddings": 40960}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        // Explicit `nil`: the fixture's own default is 4096 (CW-3-FIX-1 B4).
        #expect(ContextWindow.tokens(for: makeEntry(contextWindow: nil), modelDirectory: dir) == 40_960)
    }

    @Test func aNestedTextConfigIsReadToo() throws {
        let dir = try makeModelDirectory(
            config: #"{"text_config": {"max_position_embeddings": 12288}}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ContextWindow.tokens(for: nil, modelDirectory: dir) == 12_288)
    }

    @Test func theFallbackIs8192WhenNothingSays() throws {
        let dir = try makeModelDirectory(config: nil)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ContextWindow.tokens(for: nil, modelDirectory: dir) == 8_192)
        #expect(ContextWindow.fallback == 8_192)
    }

    @Test func aMalformedOrZeroConfigFallsBack() throws {
        let malformed = try makeModelDirectory(config: "{ not json")
        defer { try? FileManager.default.removeItem(at: malformed) }
        #expect(ContextWindow.tokens(for: nil, modelDirectory: malformed) == 8_192)

        let zero = try makeModelDirectory(config: #"{"max_position_embeddings": 0}"#)
        defer { try? FileManager.default.removeItem(at: zero) }
        #expect(ContextWindow.tokens(for: nil, modelDirectory: zero) == 8_192)
    }

    /// The fixture default is the pre-CW-3 value again (B4), so no existing caller changed.
    @Test func theEntryFixtureDefaultIs4096() {
        #expect(makeEntry().contextWindow == 4096)
    }
}
