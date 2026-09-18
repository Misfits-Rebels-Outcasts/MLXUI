import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R2-3: the `AssetStage` protocol and `SingleMediaStage` adapter. The adapter
/// unwraps a one-`Item` `Asset` into a `Media`, runs the inner `PipelineStage`, wraps back.
/// See `RSI/DelegateMergeBacklog.md` CFM-R2-3.
struct CatFlowAssetStageTests {

    /// A stub `PipelineStage` that echoes audio at a fixed rate (like ASR/TTS stages).
    private nonisolated struct EchoAudioStage: PipelineStage {
        let id = "stub.echo-audio"
        let name = "Stub Audio"
        var accepts: MediaKind { .audio }
        var produces: MediaKind { .audio }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            try require(input, .audio)
            guard case let .audio(buffer) = input else {
                throw StageError.kindMismatch(expected: .audio, got: input.kind)
            }
            progress(1.0)
            return .audio(buffer)
        }
    }

    private func tempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-asset-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func tempFile(_ name: String, _ bytes: Data) throws -> URL {
        let dir = try tempDir()
        let url = dir.appendingPathComponent(name)
        try bytes.write(to: url)
        return url
    }

    // MARK: - Round-trip an audio item through a stub stage

    @Test func audioItemRoundTripsThroughSingleMediaStage() async throws {
        // Write a tiny valid WAV and read it back via AudioFileReader so the stage sees
        // a real AudioBuffer with a known rate.
        let buffer = AudioBuffer(samples: [0.0, 0.1, -0.1, 0.5], sampleRate: 16_000)
        let wavURL = try tempFile("tone.wav", AudioWriter.wavData(buffer))
        let blobDir = try tempDir()
        defer { try? FileManager.default.removeItem(at: wavURL.deletingLastPathComponent()) }
        defer { try? FileManager.default.removeItem(at: blobDir) }

        let adapter = SingleMediaStage(id: "stub", name: "Echo", inner: EchoAudioStage(), rowLabel: "Row 1", blobDirectory: blobDir)
        let input = Asset(items: [Item(kind: .audio, value: nil, path: wavURL, sourceText: nil)])
        let output = try await adapter.run(input) { _ in }

        #expect(output.items.count == 1)
        let item = try #require(output.items.first)
        #expect(item.kind == .audio)
        // Round-trip the produced asset's file back through AudioFileReader.
        let fileURL = try #require(item.path)
        let readBack = try AudioFileReader.read(fileURL)
        #expect(readBack.sampleRate == 16_000)
        // WAV is 16-bit PCM — the values survive within quantization tolerance.
        #expect(readBack.samples.count == buffer.samples.count)
        for (a, b) in zip(readBack.samples, buffer.samples) {
            #expect(abs(a - b) < 0.001)
        }
    }

    // MARK: - A two-item Asset throws with the row named

    @Test func twoItemAssetThrowsNamingTheRow() async throws {
        let blobDir = try tempDir()
        defer { try? FileManager.default.removeItem(at: blobDir) }
        let adapter = SingleMediaStage(id: "stub", name: "Echo", inner: EchoAudioStage(), rowLabel: "Row 3", blobDirectory: blobDir)
        let input = Asset(items: [
            Item(kind: .audio, value: nil, path: nil, sourceText: nil),
            Item(kind: .audio, value: nil, path: nil, sourceText: nil),
        ])
        await #expect(throws: FlowError.self) {
            _ = try await adapter.run(input) { _ in }
        }
        // The error names the row.
        do {
            _ = try await adapter.run(input) { _ in }
            Issue.record("expected throw")
        } catch let error as FlowError {
            guard case .badInputCardinality(let row, _, _) = error else {
                Issue.record("wrong FlowError case: \(error)")
                return
            }
            #expect(row == "Row 3")
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    @Test func emptyAssetThrows() async throws {
        let blobDir = try tempDir()
        defer { try? FileManager.default.removeItem(at: blobDir) }
        let adapter = SingleMediaStage(id: "stub", name: "Echo", inner: EchoAudioStage(), rowLabel: "Row 1", blobDirectory: blobDir)
        await #expect(throws: FlowError.self) {
            _ = try await adapter.run(Asset(items: [])) { _ in }
        }
    }

    // MARK: - An index-kind item throws the refusal, not a crash

    @Test func indexKindItemThrowsRefusal() async throws {
        let blobDir = try tempDir()
        defer { try? FileManager.default.removeItem(at: blobDir) }
        let adapter = SingleMediaStage(id: "stub", name: "Echo", inner: EchoAudioStage(), rowLabel: "Row 2", blobDirectory: blobDir)
        let input = Asset(items: [Item(kind: .index, value: nil, path: nil, sourceText: nil)])
        do {
            _ = try await adapter.run(input) { _ in }
            Issue.record("expected throw")
        } catch let error as FlowError {
            guard case .unsupportedKind(let row, let kind) = error else {
                Issue.record("wrong FlowError case: \(error)")
                return
            }
            #expect(row == "Row 2")
            #expect(kind == .index)
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    // MARK: - Kind ↔ MediaKind mapping is total in one direction

    @Test func kindMappingCoversFourKinds() {
        #expect(MediaKindMapping.mediaKind(for: .text) == .text)
        #expect(MediaKindMapping.mediaKind(for: .audio) == .audio)
        #expect(MediaKindMapping.mediaKind(for: .image) == .image)
        #expect(MediaKindMapping.mediaKind(for: .vector) == .embedding)
    }

    @Test func kindMappingRejectsOtherKinds() {
        for kind in Kind.allCases where ![.text, .audio, .image, .vector].contains(kind) {
            #expect(MediaKindMapping.mediaKind(for: kind) == nil, "\(kind) must not map")
        }
    }

    // MARK: - Inline text item

    @Test func textItemRoundTrips() async throws {
        let stage = StubTextPipelineStage()
        let blobDir = try tempDir()
        defer { try? FileManager.default.removeItem(at: blobDir) }
        let adapter = SingleMediaStage(id: "stub", name: "Text", inner: stage, rowLabel: "Row 1", blobDirectory: blobDir)
        let input = Asset(items: [Item(kind: .text, value: "hello", path: nil, sourceText: nil)])
        let output = try await adapter.run(input) { _ in }
        let item = try #require(output.items.first)
        #expect(item.kind == .text)
        #expect(item.value == "hello")
    }
}

/// A stub `text → text` PipelineStage.
private nonisolated struct StubTextPipelineStage: PipelineStage {
    let id = "stub.text"
    let name = "Stub Text"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        progress(1.0)
        return input
    }
}
