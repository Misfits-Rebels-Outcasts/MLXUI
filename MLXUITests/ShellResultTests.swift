import Foundation
import Testing
@testable import MLXUI

/// `ShellResult` — the model-facing rendering of a finished command. The clipping
/// here is not cosmetic: tool output goes straight into a local model's context
/// window, and an unclipped `yes | head -c 100000` made an 8B model echo the result
/// back a token at a time instead of answering.
struct ShellResultTests {

    private func result(stdout: String = "", stderr: String = "", exit: Int32 = 0,
                        truncated: Bool = false) -> ShellResult {
        ShellResult(standardOutput: stdout, standardError: stderr,
                    exitCode: exit, outputWasTruncated: truncated)
    }

    // MARK: clip

    @Test func shortOutputPassesThroughUnchanged() {
        let text = (1...10).map { "line \($0)" }.joined(separator: "\n")
        #expect(ShellResult.clip(text) == text)
    }

    @Test func outputAtTheLimitIsNotClipped() {
        let text = (1...ShellResult.maxTranscriptLines).map(String.init).joined(separator: "\n")
        #expect(ShellResult.clip(text) == text)
    }

    @Test func longOutputKeepsHeadAndTail() {
        let text = (1...1000).map { "line \($0)" }.joined(separator: "\n")
        let clipped = ShellResult.clip(text)

        #expect(clipped.contains("line 1\n"))
        #expect(clipped.contains("line \(ShellResult.headLines)"))
        #expect(clipped.contains("line 1000"))
        #expect(!clipped.contains("line 500"))
        #expect(clipped.contains("more lines"))
    }

    @Test func clipReportsHowMuchWasDropped() {
        let text = (1...1000).map(String.init).joined(separator: "\n")
        let dropped = 1000 - ShellResult.maxTranscriptLines
        #expect(ShellResult.clip(text).contains("… \(dropped) more lines …"))
    }

    /// The pathological case that prompted the clip: many short identical lines.
    @Test func clipBoundsRepetitiveOutput() {
        let text = String(repeating: "y\n", count: 4000)
        let clipped = ShellResult.clip(text)
        #expect(clipped.split(separator: "\n").count <= ShellResult.maxTranscriptLines + 1)
    }

    // MARK: transcript

    @Test func transcriptLeadsWithExitCode() {
        #expect(result(stdout: "hi", exit: 3).transcript.hasPrefix("exit code: 3"))
    }

    /// A command can fail while printing nothing at all — the exit code must still
    /// reach the model, and "(no output)" must not read as success.
    @Test func transcriptReportsSilentFailure() {
        let transcript = result(exit: 1).transcript
        #expect(transcript.contains("exit code: 1"))
        #expect(transcript.contains("(no output)"))
    }

    @Test func transcriptIncludesBothStreams() {
        let transcript = result(stdout: "out", stderr: "err").transcript
        #expect(transcript.contains("stdout:\nout"))
        #expect(transcript.contains("stderr:\nerr"))
    }

    @Test func transcriptFlagsByteTruncation() {
        #expect(result(stdout: "partial", truncated: true).transcript.contains("truncated"))
    }

    @Test func succeededTracksExitCode() {
        #expect(result(exit: 0).succeeded)
        #expect(!result(exit: 1).succeeded)
    }
}
