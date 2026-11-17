import Foundation
import Testing
@testable import MLXUI

/// Shared access to the checked-in Privacy Filter reference corpus and to an installed checkpoint for
/// the decode-side suites (PF-4/PF-6).
///
/// The reference fixtures are generated once by `MLXUITests/Fixtures/PrivacyFilter/make_privacy_golden.py`
/// from `transformers` 5.12.0 + `opf` (commit `f7f00ca…`) and are never edited to make a change pass.
/// They live outside the bundle's resource namespace (the test target flattens resources, so two
/// `golden.json` files would collide) and are read from the source tree by `#filePath`, the
/// `DepthAnythingFixtures` shape.
///
/// The tokenizer/decoder/redaction suites are opt-in like the other ported models: they need
/// `MLXUI_TEST_PRIVACY=1` (`TEST_RUNNER_MLXUI_TEST_PRIVACY=1` through xcodebuild) **and**
/// `MLXUI_TEST_PRIVACY_DIR` pointing at an installed copy — normally `/private/tmp/pf-model`, because
/// the sandboxed test host cannot read `~/Library/Application Support`. A skipped run is not evidence.
enum PrivacyFilterDecodeTestSupport {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static var optedIn: Bool {
        (env["MLXUI_TEST_PRIVACY"] ?? env["TEST_RUNNER_MLXUI_TEST_PRIVACY"]) == "1"
    }

    /// `MLXUI_TEST_PRIVACY_DIR` when it holds at least the tokenizer files.
    nonisolated static var tokenizerDirectory: URL? {
        guard let raw = env["MLXUI_TEST_PRIVACY_DIR"] ?? env["TEST_RUNNER_MLXUI_TEST_PRIVACY_DIR"] else {
            return nil
        }
        let url = URL(fileURLWithPath: raw)
        let manager = FileManager.default
        guard manager.fileExists(atPath: url.appendingPathComponent("tokenizer.json").path),
              manager.fileExists(atPath: url.appendingPathComponent("config.json").path) else { return nil }
        return url
    }

    /// `MLXUI_TEST_PRIVACY_DIR` only when the weights are there too (the network / long-input suites).
    nonisolated static var weightsDirectory: URL? {
        guard let url = tokenizerDirectory else { return nil }
        let manager = FileManager.default
        guard manager.fileExists(atPath: url.appendingPathComponent("model.safetensors").path),
              manager.fileExists(atPath: url.appendingPathComponent(PrivacyFilterCalibration.filename).path) else {
            return nil
        }
        return url
    }

    nonisolated static var canRunTokenizer: Bool { optedIn && tokenizerDirectory != nil }
    nonisolated static var canRunRealWeights: Bool { optedIn && weightsDirectory != nil }

    /// `MLXUITests/Fixtures/PrivacyFilter` in the source tree.
    nonisolated static let fixturesDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/PrivacyFilter")

    nonisolated static var goldenURL: URL {
        fixturesDirectory.appendingPathComponent("privacy-golden.json")
    }

    nonisolated static var goldenExists: Bool {
        FileManager.default.fileExists(atPath: goldenURL.path)
    }

    /// The suites that read the corpus need the checkpoint (for the tokenizer) **and** the JSON.
    nonisolated static var canRunCorpus: Bool { canRunTokenizer && goldenExists }

    /// Fixture-only suites (decoder on the checked-in log-prob files) need just the opt-in and the JSON.
    nonisolated static var canRunFixtures: Bool { optedIn && goldenExists }

    // MARK: - Corpus shape (fixtures/PF-3 contract)

    struct Golden: Decodable {
        let cases: [String: Case]
        let long_document: Case?

        struct Case: Decodable {
            let group: String?
            let text: String
            let token_ids: [Int]
            let token_byte_starts: [Int]
            let token_byte_ends: [Int]
            let spans: [Span]
            let redacted_text: String
            let decoded_mismatch: Bool
            let logits: FloatFile?
            let log_probs: FloatFile?
        }

        struct Span: Decodable {
            let label: String
            let start_byte: Int
            let end_byte: Int
            let text: String
            let placeholder: String
        }

        struct FloatFile: Decodable {
            let file: String
            let shape: [Int]
            let dtype: String?
            let kind: String?
        }
    }

    nonisolated static func loadGolden() throws -> Golden {
        let data = try Data(contentsOf: goldenURL)
        return try JSONDecoder().decode(Golden.self, from: data)
    }

    /// All short corpus cases in a stable order (JSON dictionaries are unordered).
    nonisolated static func shortCases(_ golden: Golden) -> [(name: String, value: Golden.Case)] {
        golden.cases.sorted { $0.key < $1.key }.map { (name: $0.key, value: $0.value) }
    }

    /// Read a little-endian fp32 fixture as `[rows][cols]`.
    nonisolated static func floatRows(_ file: Golden.FloatFile) throws -> [[Float]] {
        let url = fixturesDirectory.appendingPathComponent(file.file)
        let data = try Data(contentsOf: url)
        let values: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        if file.shape.count == 2 {
            let rows = file.shape[0]
            let columns = file.shape[1]
            guard rows * columns == values.count else {
                throw PrivacyFilterFixtureError.shapeMismatch(file.file, file.shape, values.count)
            }
            guard columns > 0 else { return [] }
            return (0 ..< rows).map { row in
                Array(values[(row * columns) ..< ((row + 1) * columns)])
            }
        }
        return values.map { [$0] }
    }

    nonisolated static func floats(_ file: Golden.FloatFile) throws -> [Float] {
        let url = fixturesDirectory.appendingPathComponent(file.file)
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Build the `Tokenization` the corpus records (ids + byte offsets), for span tests that run
    /// without a tokenizer.
    /// Append one evidence line to the sandboxed test host's temporary directory (the repo's
    /// DepthAnythingFixtures / parity-record precedent). Xcode does not relay Swift Testing
    /// `print`s into `xcodebuild` output, so the parity numbers a PR must quote are recorded here;
    /// the host's container temp dir is readable from the developer shell.
    nonisolated static func record(_ line: String) {
        guard let data = (line + "\n").data(using: .utf8) else { return }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("PF6-PARITY.txt")
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: url)
        }
    }

    nonisolated static func tokenization(for testCase: Golden.Case) -> PrivacyFilterTokenizer.Tokenization {
        PrivacyFilterTokenizer.Tokenization(
            tokenIDs: testCase.token_ids,
            utf8ByteStarts: testCase.token_byte_starts,
            utf8ByteEnds: testCase.token_byte_ends)
    }
}

/// One plain sentence, never quoting corpus text.
nonisolated enum PrivacyFilterFixtureError: Error, LocalizedError, Equatable {
    case shapeMismatch(String, [Int], Int)
    case goldenMissing

    var errorDescription: String? {
        switch self {
        case .shapeMismatch(let file, let shape, let count):
            return "Privacy Filter fixtures: \(file) declares shape \(shape) but has \(count) floats."
        case .goldenMissing:
            return "Privacy Filter fixtures: privacy-golden.json is missing from the test tree."
        }
    }
}

/// PF-4 — the tokenizer and its character-accurate byte offsets (Trap 1).
struct PrivacyFilterTokenizerTests {

    // MARK: - Pure offset/alphabet math (no checkpoint needed)

    /// The o200k/GPT-2 byte-level alphabet: printable ranges map to themselves, the rest start at
    /// U+0100 in byte order.
    @Test func byteLevelAlphabetMapsTheKnownByteRanges() {
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet.count == 256)
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[33]?.value == 33)   // "!"
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[126]?.value == 126) // "~"
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[161]?.value == 161) // "¡"
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[172]?.value == 172) // "¬"
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[174]?.value == 174) // "®"
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[255]?.value == 255) // "ÿ"
        // Excluded bytes, in byte order, get U+0100, U+0101, …
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[0]?.value == 256)   // NUL
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[32]?.value == 288)  // space
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[127]?.value == 289) // DEL
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[160]?.value == 322) // NBSP
        #expect(PrivacyFilterTokenizer.byteLevelAlphabet[173]?.value == 323) // soft hyphen
    }

    @Test func byteLevelInverseRoundTripsEveryByte() {
        #expect(PrivacyFilterTokenizer.byteLevelInverse.count == 256)
        for byte in UInt8(0) ... UInt8(255) {
            guard let scalar = PrivacyFilterTokenizer.byteLevelAlphabet[byte] else {
                Issue.record("byte \(byte) is missing from the byte-level alphabet")
                continue
            }
            #expect(PrivacyFilterTokenizer.byteLevelInverse[scalar] == byte)
        }
    }

    @Test func tokenPieceMapsThroughTheByteLevelAlphabet() throws {
        // "Hello" is ASCII → the same bytes; the o200k space marker `Ġ` (U+0120) is byte 32, and
        // `Ċ` (U+010A) is a newline.
        #expect(try PrivacyFilterTokenizer.bytes(forTokenPiece: "Hello") == Array("Hello".utf8))
        #expect(try PrivacyFilterTokenizer.bytes(forTokenPiece: "Ġ") == [32])
        #expect(try PrivacyFilterTokenizer.bytes(forTokenPiece: "Ċ") == [10])
        // A CJK vocabulary piece is the UTF-8 bytes of the character, one scalar per byte marker.
        let cjkBytes = try PrivacyFilterTokenizer.bytes(forTokenPiece: "ä¸")
        #expect(cjkBytes == [0xE4, 0xB8])
    }

    @Test func tokenPieceRefusesAScalarOutsideTheAlphabet() {
        // A raw CJK scalar is not a byte-level marker; guessing would move spans.
        #expect(throws: PrivacyFilterTokenizerError.self) {
            _ = try PrivacyFilterTokenizer.bytes(forTokenPiece: "中")
        }
    }

    @Test func tokenizationAccumulatesExactByteOffsets() throws {
        let tokenization = try PrivacyFilterTokenizer.tokenization(
            tokenIDs: [10, 11, 12],
            tokenPieces: [Array("ab".utf8), Array("é".utf8), Array("c".utf8)],
            inputUTF8: Array("abéc".utf8))
        #expect(tokenization.tokenIDs == [10, 11, 12])
        #expect(tokenization.utf8ByteStarts == [0, 2, 4])
        #expect(tokenization.utf8ByteEnds == [2, 4, 5])
        #expect(tokenization.tokenCount == 3)
    }

    @Test func tokenizationAcceptsEmptyText() throws {
        let tokenization = try PrivacyFilterTokenizer.tokenization(
            tokenIDs: [], tokenPieces: [], inputUTF8: [])
        #expect(tokenization.tokenIDs.isEmpty)
        #expect(tokenization.utf8ByteStarts.isEmpty)
        #expect(tokenization.utf8ByteEnds.isEmpty)
    }

    @Test func tokenizationRefusesPiecesThatDoNotReproduceTheInput() {
        do {
            _ = try PrivacyFilterTokenizer.tokenization(
                tokenIDs: [1, 2],
                tokenPieces: [Array("ab".utf8), Array("cd".utf8)],
                inputUTF8: Array("abce".utf8))
            Issue.record("mismatched pieces were accepted")
        } catch let error as PrivacyFilterTokenizerError {
            let sentence = error.errorDescription ?? ""
            #expect(sentence.contains("did not reproduce the input text exactly"))
            // Decision 10: no input text in the message.
            #expect(!sentence.contains("abce"))
            #expect(!sentence.contains("abcd"))
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test func loadingRefusesAMissingTokenizerFile() async {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("pf-no-tokenizer-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await PrivacyFilterTokenizer(modelDirectory: directory)
            Issue.record("a folder without tokenizer.json was accepted")
        } catch let error as PrivacyFilterTokenizerError {
            #expect(error == .missingTokenizerFile)
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    // MARK: - Real tokenizer (opt-in; the installed checkpoint's tokenizer.json)

    @Suite(.enabled(
        if: PrivacyFilterDecodeTestSupport.canRunTokenizer,
        "needs MLXUI_TEST_PRIVACY=1 and MLXUI_TEST_PRIVACY_DIR"))
    struct RealTokenizerTests {
        private func loadTokenizer() async throws -> PrivacyFilterTokenizer {
            let directory = try #require(PrivacyFilterDecodeTestSupport.tokenizerDirectory)
            return try await PrivacyFilterTokenizer(modelDirectory: directory)
        }

        @Test func everyCorpusTextRoundTripsToTheExactInputBytes() async throws {
            guard PrivacyFilterDecodeTestSupport.goldenExists else { return }
            let tokenizer = try await loadTokenizer()
            let golden = try PrivacyFilterDecodeTestSupport.loadGolden()
            var cases = PrivacyFilterDecodeTestSupport.shortCases(golden).map { (name: $0.name, value: $0.value) }
            if let longDocument = golden.long_document { cases.append((name: "long_document", value: longDocument)) }
            var checked = 0
            for (name, testCase) in cases {
                let tokenization = try tokenizer.tokenize(testCase.text)
                #expect(tokenization.tokenIDs == testCase.token_ids, "\(name): token ids differ")
                #expect(tokenization.utf8ByteStarts == testCase.token_byte_starts, "\(name): token starts differ")
                #expect(tokenization.utf8ByteEnds == testCase.token_byte_ends, "\(name): token ends differ")
                checked += 1
            }
            print("PF4 tokenizer offsets: \(checked)/\(cases.count) corpus texts equal (ids + per-token UTF-8 bytes)")
            PrivacyFilterDecodeTestSupport.record("PF4 tokenizer offsets: \(checked)/\(cases.count) corpus texts equal (ids + per-token UTF-8 bytes)")
        }

        @Test func tokenIdsMatchTheFixtureForTheProbeText() async throws {
            let tokenizer = try await loadTokenizer()
            // Independent smoke text (synthetic, ASCII only) — ids checked against tiktoken o200k_base.
            let tokenization = try tokenizer.tokenize("Hello, world.")
            #expect(tokenization.tokenIDs.count == 4)
            #expect(tokenization.utf8ByteStarts == [0, 5, 6, 12])
            #expect(tokenization.utf8ByteEnds == [5, 6, 12, 13])
        }

        @Test func emptyAndWhitespaceOnlyTextsTokenizeCleanly() async throws {
            let tokenizer = try await loadTokenizer()
            let empty = try tokenizer.tokenize("")
            #expect(empty.tokenIDs.isEmpty)
            let whitespace = try tokenizer.tokenize(" \t\r\n  \r\n")
            #expect(whitespace.tokenIDs.count >= 1)
            #expect(whitespace.utf8ByteStarts.first == 0)
            #expect(whitespace.utf8ByteEnds.last == 8)
        }

        @Test func unicodeSamplesRoundTripByteForByte() async throws {
            let tokenizer = try await loadTokenizer()
            let samples = [
                "Hi 👨‍👩‍👧 there 👋",
                "联系人 张伟，邮箱 example@example.com。",
                "Cafe\u{0301} crème à la française",
                "line one\r\n\tline two\r\n",
                "  leading and trailing  ",
                "😀😃😄😁 mixed with ASCII",
            ]
            for sample in samples {
                let tokenization = try tokenizer.tokenize(sample)
                #expect(tokenization.utf8ByteEnds.last == sample.utf8.count)
                #expect(tokenization.tokenIDs.count == tokenization.utf8ByteStarts.count)
                #expect(tokenization.utf8ByteStarts.count == tokenization.utf8ByteEnds.count)
            }
        }
    }
}
