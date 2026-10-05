import Foundation
import Testing
@testable import MLXUI

/// EF-1 — the shared exact-string edit engine (`TextEdit`). Pure `replace` outcomes plus
/// the real file round-trip against a throwaway temp directory. No `PathPolicy`, no tool,
/// no UI: those are EF-2's job.
struct TextEditTests {

    // MARK: - Helpers

    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("text-edit-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func write(_ data: Data, to url: URL) throws {
        try data.write(to: url)
    }

    private func write(_ text: String, to url: URL) throws {
        try Data(text.utf8).write(to: url)
    }

    private func bytes(of url: URL) throws -> Data {
        try Data(contentsOf: url)
    }

    private func modificationDate(of url: URL) throws -> Date? {
        try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date
    }

    private func posixMode(of url: URL) throws -> Int {
        let value = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions]
        return (value as? NSNumber)?.intValue ?? -1
    }

    /// The numbered context block of a D8 result, with the success header removed.
    private func snippet(of result: String) -> String {
        result.components(separatedBy: "\n").dropFirst().joined(separator: "\n")
    }

    /// The path a success message reports: `editFile` resolves symlinks at the top (N1), and
    /// the temp directory is reached through `/var` -> `/private/var` on macOS.
    private func reportedPath(of url: URL) -> String {
        url.resolvingSymlinksInPath().path
    }

    // MARK: - Layer 1: pure replacement

    @Test func replaceEditsOneMatchAndReportsItsLine() {
        #expect(TextEdit.replace(in: "alpha\nbeta\ngamma\n", old: "beta", new: "BETA", replaceAll: false)
            == .edited(newText: "alpha\nBETA\ngamma\n", count: 1, lines: [2]))
    }

    @Test func replaceIsPureAndNonOverlapping() {
        let first = TextEdit.replace(in: "aaa", old: "aa", new: "b", replaceAll: true)
        let second = TextEdit.replace(in: "aaa", old: "aa", new: "b", replaceAll: true)
        // "aa" in "aaa" is ONE non-overlapping match, not two.
        #expect(first == second)
        #expect(first == .edited(newText: "ba", count: 1, lines: [1]))
    }

    @Test func replaceCountsEveryMatchWithLinesInTheOriginalText() {
        #expect(TextEdit.replace(in: "x\ny\nx\n", old: "x", new: "X", replaceAll: true)
            == .edited(newText: "X\ny\nX\n", count: 2, lines: [1, 3]))
        #expect(TextEdit.replace(in: "same\nsame\nsame\n", old: "same", new: "SAME", replaceAll: false)
            == .ambiguous(count: 3, lines: [1, 2, 3]))
    }

    @Test func replaceReportsNoMatchWithAndWithoutAnIndentHint() {
        // Tab in the file, spaces in `old` — the only difference is indentation, so the
        // D3 hint must name line 2. (Two spaces would NOT work as a fixture here: they
        // are a substring of a four-space indent and match exactly.)
        let hinted = TextEdit.replace(
            in: "func a() {\n\treturn 1\n}\n",
            old: "    return 1",
            new: "    return 2",
            replaceAll: false)
        #expect(hinted == .notFound(indentTolerantLine: 2))
        #expect(TextEdit.replace(in: "a\n", old: "zz", new: "c", replaceAll: false)
            == .notFound(indentTolerantLine: nil))
    }

    @Test func anOldStringInsideDeeperIndentationStillMatchesExactly() {
        // Plain substring replacement, like Python's `str.replace`: the last two spaces
        // of the four-space indent plus "return 1" are the match. The D3 hint is only
        // consulted when nothing matches at all.
        #expect(TextEdit.replace(
            in: "    return 1\n",
            old: "  return 1",
            new: "  return 2",
            replaceAll: false) == .edited(newText: "    return 2\n", count: 1, lines: [1]))
    }

    @Test func replaceRejectsDegenerateInput() {
        let empty = TextEdit.replace(in: "a\n", old: "", new: "b", replaceAll: false)
        guard case .invalid(let emptyMessage) = empty else {
            Issue.record("expected .invalid for an empty old_string, got \(empty)")
            return
        }
        #expect(emptyMessage.contains("write_file"))

        let identical = TextEdit.replace(in: "a\n", old: "a", new: "a", replaceAll: false)
        guard case .invalid(let identicalMessage) = identical else {
            Issue.record("expected .invalid for old == new, got \(identical)")
            return
        }
        #expect(identicalMessage != emptyMessage)
    }

    @Test func canonicallyEquivalentOldAndNewAreStillDistinctBytes() {
        // Swift's `String ==` calls these two spellings equal; the bytes are not, so an
        // edit between them is real and must not be refused as "identical".
        let precomposed = "caf\u{00E9}"
        let decomposed = "cafe\u{301}"
        let outcome = TextEdit.replace(
            in: "\(precomposed)\n",
            old: precomposed,
            new: decomposed,
            replaceAll: false)
        guard case .edited(let newText, let count, let lines) = outcome else {
            Issue.record("expected an edit, got \(outcome)")
            return
        }
        #expect(Array(newText.utf8) == Array("\(decomposed)\n".utf8))
        #expect(count == 1)
        #expect(lines == [1])
    }

    // MARK: - Layer 2: the file round-trip

    @Test func singleMatchReplacesOnlyThatMatch() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("notes.txt")
        try write("alpha\nbeta\ngamma\n", to: file)

        let result = TextEdit.editFile(at: file, old: "beta", new: "BETA", replaceAll: false)
        let after = try bytes(of: file)

        #expect(result.hasPrefix(
            "Edited \(reportedPath(of: file)): replaced 1 occurrence (line 2)."))
        #expect(after == Data("alpha\nBETA\ngamma\n".utf8))
    }

    @Test func zeroMatchesLeavesBytesAndModificationDateUntouched() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("notes.txt")
        try write("alpha\nbeta\n", to: file)
        let before = try bytes(of: file)
        let beforeDate = try modificationDate(of: file)

        let result = TextEdit.editFile(at: file, old: "delta", new: "DELTA", replaceAll: false)
        let after = try bytes(of: file)
        let afterDate = try modificationDate(of: file)

        #expect(result.hasPrefix("Error:"))
        #expect(result.contains("old_string not found"))
        #expect(result.contains("read_file"))
        #expect(after == before)
        #expect(afterDate == beforeDate)
    }

    @Test func indentationHintNamesTheRightLineAndLeavesTheFileAlone() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("code.swift")
        try write("func a() {\n\treturn 1\n}\n", to: file)
        let before = try bytes(of: file)

        let result = TextEdit.editFile(at: file, old: "    return 1", new: "    return 2", replaceAll: false)
        let after = try bytes(of: file)

        #expect(result.hasPrefix("Error:"))
        #expect(result.contains("A match ignoring indentation exists at line 2"))
        #expect(result.contains("read_file"))
        #expect(after == before)
    }

    @Test func severalMatchesRefuseWithoutReplaceAllAndAllLandWithIt() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("notes.txt")
        try write("same\nsame\nsame\n", to: file)
        let before = try bytes(of: file)
        let beforeDate = try modificationDate(of: file)

        let refused = TextEdit.editFile(at: file, old: "same", new: "SAME", replaceAll: false)
        let afterRefusal = try bytes(of: file)
        let afterRefusalDate = try modificationDate(of: file)

        #expect(refused.hasPrefix("Error:"))
        #expect(refused.contains("old_string matches 3 places (lines 1, 2, 3)"))
        #expect(refused.contains("replace_all"))
        #expect(afterRefusal == before)
        #expect(afterRefusalDate == beforeDate)

        let applied = TextEdit.editFile(at: file, old: "same", new: "SAME", replaceAll: true)
        let after = try bytes(of: file)

        #expect(applied.hasPrefix(
            "Edited \(reportedPath(of: file)): replaced 3 occurrences (lines 1, 2, 3)."))
        #expect(after == Data("SAME\nSAME\nSAME\n".utf8))
    }

    @Test func multiLineOldStringSpansThreeLines() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("notes.txt")
        try write("one\ntwo\nthree\nfour\n", to: file)

        let result = TextEdit.editFile(
            at: file,
            old: "two\nthree\nfour",
            new: "TWO\nTHREE\nFOUR",
            replaceAll: false)
        let after = try bytes(of: file)

        #expect(result.contains("replaced 1 occurrence (line 2)"))
        #expect(after == Data("one\nTWO\nTHREE\nFOUR\n".utf8))
    }

    @Test func degenerateFilesAndInputsReturnDistinctErrors() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let normal = directory.appendingPathComponent("normal.txt")
        try write("hello\n", to: normal)
        let subdirectory = directory.appendingPathComponent("sub", isDirectory: true)
        try FileManager.default.createDirectory(at: subdirectory, withIntermediateDirectories: true)
        let missing = directory.appendingPathComponent("missing.txt")
        let oversized = directory.appendingPathComponent("oversized.txt")
        try write(Data(repeating: 0x61, count: TextEdit.maxFileBytes + 1), to: oversized)
        let binary = directory.appendingPathComponent("binary.bin")
        try write(Data([0xFF, 0xFE, 0x00, 0x01]), to: binary)

        let errors = [
            TextEdit.editFile(at: normal, old: "", new: "x", replaceAll: false),
            TextEdit.editFile(at: normal, old: "hello", new: "hello", replaceAll: false),
            TextEdit.editFile(at: missing, old: "hello", new: "x", replaceAll: false),
            TextEdit.editFile(at: subdirectory, old: "hello", new: "x", replaceAll: false),
            TextEdit.editFile(at: oversized, old: "a", new: "b", replaceAll: false),
            TextEdit.editFile(at: binary, old: "x", new: "y", replaceAll: false),
        ]

        for error in errors { #expect(error.hasPrefix("Error:")) }
        #expect(Set(errors).count == errors.count)                      // each distinct
        #expect(errors[0].contains("write_file"))                       // empty old_string
        #expect(errors[1].contains("identical"))                        // old == new
        #expect(errors[2].contains("does not exist"))
        #expect(errors[2].contains("write_file"))                       // missing file
        #expect(errors[3].contains("directory"))
        #expect(errors[4].contains("1 MB"))
        #expect(errors[5].contains("not UTF-8"))

        let normalAfter = try bytes(of: normal)
        #expect(!FileManager.default.fileExists(atPath: missing.path))  // nothing created
        #expect(normalAfter == Data("hello\n".utf8))                    // nothing changed
    }

    @Test func crlfFileAdaptsLfEditAndStaysAllCrlf() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("windows.txt")
        try write("alpha\r\nbeta\r\ngamma\r\n", to: file)

        let result = TextEdit.editFile(at: file, old: "beta\ngamma", new: "BETA\nGAMMA", replaceAll: false)
        let after = try bytes(of: file)
        let text = String(decoding: after, as: UTF8.self)

        #expect(result.contains("replaced 1 occurrence (line 2)"))
        #expect(after == Data("alpha\r\nBETA\r\nGAMMA\r\n".utf8))
        #expect(!text.replacingOccurrences(of: "\r\n", with: "").contains("\n"))
    }

    @Test func utf8BomIsPreservedAsTheFirstThreeBytes() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("bom.txt")
        let bom: [UInt8] = [0xEF, 0xBB, 0xBF]
        try write(Data(bom + Array("alpha\nbeta\n".utf8)), to: file)

        _ = TextEdit.editFile(at: file, old: "beta", new: "BETA", replaceAll: false)
        let after = try bytes(of: file)

        #expect(Array(after.prefix(3)) == bom)
        #expect(after == Data(bom + Array("alpha\nBETA\n".utf8)))
    }

    @Test func posixPermissionsSurviveTheAtomicWrite() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("script.sh")
        try write("alpha\nbeta\n", to: file)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: file.path)
        let beforeMode = try posixMode(of: file)

        _ = TextEdit.editFile(at: file, old: "beta", new: "BETA", replaceAll: false)
        let afterMode = try posixMode(of: file)
        let after = try bytes(of: file)

        #expect(beforeMode == 0o755)
        #expect(afterMode == 0o755)
        #expect(after == Data("alpha\nBETA\n".utf8))
    }

    @Test func decomposedOldStringDoesNotMatchAPrecomposedFile() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("accents.txt")
        try write("caf\u{00E9}\n", to: file)
        let before = try bytes(of: file)

        let result = TextEdit.editFile(at: file, old: "cafe\u{301}", new: "tea", replaceAll: false)
        let after = try bytes(of: file)

        #expect(result.hasPrefix("Error:"))
        #expect(result.contains("not found"))
        #expect(after == before)
        // The same comparison at the pure layer.
        #expect(TextEdit.replace(in: "caf\u{00E9}\n", old: "cafe\u{301}", new: "tea", replaceAll: false)
            == .notFound(indentTolerantLine: nil))
    }

    // MARK: - D8 result snippet

    @Test func manyReplaceAllMatchesSummariseTheLineList() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("many.txt")
        try write(Array(repeating: "x", count: 25).joined(separator: "\n") + "\n", to: file)

        let result = TextEdit.editFile(at: file, old: "x", new: "y", replaceAll: true)
        let after = try bytes(of: file)

        #expect(result.contains("replaced 25 occurrences"))
        #expect(result.contains("(lines 1, 2, 3, 4, 5, 6, 7, 8, 9, 10; 15 more)"))
        #expect(after == Data((Array(repeating: "y", count: 25).joined(separator: "\n") + "\n").utf8))
    }

    @Test func snippetIsNumberedShowsTheNewTextAndStaysWithinSevenLines() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("numbered.txt")
        try write((1...12).map { "line \($0)" }.joined(separator: "\n") + "\n", to: file)

        let result = TextEdit.editFile(at: file, old: "line 6", new: "LINE 6", replaceAll: false)
        let block = snippet(of: result)

        #expect(result.contains("replaced 1 occurrence (line 6)"))
        #expect(block.contains("6: LINE 6"))
        #expect(block.components(separatedBy: "\n").count == TextEdit.maxSnippetLines)
        #expect(block.utf8.count <= TextEdit.maxSnippetBytes)
    }

    @Test func snippetIsCappedAtOneKilobyteWithAnExplicitNotice() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("long.txt")
        let long = String(repeating: "x", count: 4_000)
        try write("head\n\(long)\ntail\n", to: file)

        let result = TextEdit.editFile(at: file, old: "head", new: "HEAD", replaceAll: false)
        let block = snippet(of: result)

        #expect(block.utf8.count <= TextEdit.maxSnippetBytes)
        #expect(block.contains("snippet clipped"))
        #expect(block.contains("HEAD"))
    }

    // MARK: - EF-A fixes (F1, F2, N1, N2)

    @Test func replaceAllOnALargeFileFinishesWellUnderTwoSeconds() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("large.txt")
        let lineCount = 450_000                        // "x\n" x 450_000 = 900 KB
        try write(String(repeating: "x\n", count: lineCount), to: file)

        let start = ContinuousClock.now
        let result = TextEdit.editFile(at: file, old: "x", new: "y", replaceAll: true)
        let elapsed = ContinuousClock.now - start
        print("EF-A F1: 900 KB replace_all took \(elapsed)")

        #expect(elapsed < .seconds(2))
        #expect(result.contains("replaced \(lineCount) occurrences"))
        let after = try bytes(of: file)
        #expect(after.count == lineCount * 2)
        #expect(Array(after.prefix(4)) == Array("y\ny\n".utf8))
    }

    @Test func mixedLineEndingsTakeTheExactLfTextAsSent() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("mixed.txt")
        try write("a\r\nb\nc\n", to: file)

        let result = TextEdit.editFile(at: file, old: "b\nc", new: "B\nC", replaceAll: false)
        let after = try bytes(of: file)

        #expect(result.contains("replaced 1 occurrence (line 2)"))
        #expect(after == Data("a\r\nB\nC\n".utf8))
    }

    @Test func mixedLineEndingsUseTheCrlfFallbackOnlyWhenTheExactTextIsAbsent() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("mixed.txt")
        // "b\nc" is not in this file; "b\r\nc" is, so only the CRLF fallback can match.
        try write("a\r\nb\r\nc\n", to: file)

        let result = TextEdit.editFile(at: file, old: "b\nc", new: "B\nC", replaceAll: false)
        let after = try bytes(of: file)

        #expect(result.contains("replaced 1 occurrence (line 2)"))
        #expect(after == Data("a\r\nB\r\nC\n".utf8))
    }

    @Test func editingThroughASymlinkChangesTheTargetAndKeepsTheLink() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("target.txt")
        let link = directory.appendingPathComponent("link.txt")
        try write("alpha\nbeta\n", to: target)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

        let result = TextEdit.editFile(at: link, old: "beta", new: "BETA", replaceAll: false)

        #expect(result.contains("replaced 1 occurrence (line 2)"))
        let targetAfter = try bytes(of: target)
        #expect(targetAfter == Data("alpha\nBETA\n".utf8))
        // The link survived the atomic write and still points at the target.
        let type = try FileManager.default.attributesOfItem(atPath: link.path)[.type]
            as? FileAttributeType
        #expect(type == .typeSymbolicLink)
        let destination = try FileManager.default.destinationOfSymbolicLink(atPath: link.path)
        #expect(destination == target.path)
    }

    @Test func indentHintDropsATrailingNewlineFromOldString() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("code.swift")
        try write("func a() {\n\treturn 1\n}\n", to: file)
        let before = try bytes(of: file)

        // A tab indents line 2 and `old_string` ends with its newline, as models often send.
        let result = TextEdit.editFile(
            at: file, old: "    return 1\n", new: "    return 2\n", replaceAll: false)
        let after = try bytes(of: file)

        #expect(result.hasPrefix("Error:"))
        #expect(result.contains("A match ignoring indentation exists at line 2"))
        #expect(after == before)
    }
}
