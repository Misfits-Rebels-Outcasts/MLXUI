import Foundation
import Testing
@testable import MLXUI

/// Shared-code shell seam (`ShellCapability.swift`). `ProcessShell`, `ShellTool`, and
/// `FileTools` live in `Apps/Direct/` (MLXUI-Direct only — no Direct test target); these
/// cases cover the shared side: the `UnavailableShell` App Store stub, `ShellInvocation`
/// builder, and `ShellError` descriptions that appear in model transcripts.
struct ShellCapabilityTests {

    // MARK: UnavailableShell — App Store seam

    @Test func unavailableShellReportsNotAvailable() {
        #expect(!UnavailableShell().isAvailable)
    }

    @Test func unavailableShellThrowsOnRun() async {
        await #expect(throws: ShellError.unavailable) {
            _ = try await UnavailableShell().run(.commandLine("echo hi"))
        }
    }

    // MARK: ShellInvocation.commandLine

    /// The login-shell flag `-lc` is the security-relevant detail: it expands the user's
    /// PATH and shell functions, which is what makes pipes, globbing, and `brew` work.
    @Test func commandLineBuildsZshInvocation() {
        let inv = ShellInvocation.commandLine("ls -la")
        #expect(inv.executable == "/bin/zsh")
        #expect(inv.arguments == ["-lc", "ls -la"])
    }

    @Test func commandLinePassesThroughTimeout() {
        let inv = ShellInvocation.commandLine("sleep 1", timeout: 120)
        #expect(inv.timeout == 120)
    }

    @Test func commandLinePassesThroughWorkingDirectory() {
        let dir = URL(fileURLWithPath: "/tmp")
        let inv = ShellInvocation.commandLine("ls", workingDirectory: dir)
        #expect(inv.workingDirectory == dir)
    }

    // MARK: ShellError descriptions (shown verbatim in model transcripts)

    @Test func errorDescriptionsContainKeyTerms() {
        #expect(ShellError.unavailable.errorDescription?.contains("not available") == true)
        #expect(ShellError.executableNotFound("mybin").errorDescription?.contains("mybin") == true)
        #expect(ShellError.notExecutable("/usr/bin/notx").errorDescription?.contains("/usr/bin/notx") == true)
        #expect(ShellError.timedOut(after: 30).errorDescription?.contains("30") == true)
        #expect(ShellError.blockedByPolicy(reason: "no rel paths").errorDescription?.contains("no rel paths") == true)
    }
}
