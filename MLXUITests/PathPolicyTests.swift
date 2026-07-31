import Foundation
import Testing
@testable import MLXUI

/// `PathPolicy` — the allowlist/denylist gate every filesystem-touching agent tool
/// passes through. The tools themselves ship only in the direct-download edition
/// (`Apps/Direct/`), but the policy is pure logic and lives in shared code so it can
/// be tested here. Getting it wrong is the difference between "read the file I named"
/// and "read my SSH key", so the escape cases matter more than the happy path.
struct PathPolicyTests {

    private var home: URL {
        FileManager.default.homeDirectoryForCurrentUser
    }

    /// A fresh allowlist root, a sibling "outside" directory, and a `root-evil`
    /// lookalike — with every file actually created on disk.
    ///
    /// The files have to exist: `resolvingSymlinksInPath()` only rewrites
    /// `/var` → `/private/var` for path components that exist, so comparing a
    /// real root against a made-up leaf would compare resolved against
    /// unresolved and fail for the wrong reason.
    private struct Sandbox {
        let root: URL       // …/root
        let inside: URL     // …/root/notes.txt
        let outside: URL    // …/outside
        let secret: URL     // …/outside/secret.txt
        let lookalike: URL  // …/root-evil/x.txt
    }

    private func makeSandbox() throws -> Sandbox {
        let fm = FileManager.default
        let base = fm.temporaryDirectory
            .appendingPathComponent("PathPolicyTests-\(UUID().uuidString)", isDirectory: true)
        let root = base.appendingPathComponent("root", isDirectory: true)
        let outside = base.appendingPathComponent("outside", isDirectory: true)
        let evil = base.appendingPathComponent("root-evil", isDirectory: true)
        for dir in [root, outside, evil] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        let inside = root.appendingPathComponent("notes.txt")
        let secret = outside.appendingPathComponent("secret.txt")
        let lookalike = evil.appendingPathComponent("x.txt")
        for file in [inside, secret, lookalike] {
            try "contents".write(to: file, atomically: true, encoding: .utf8)
        }
        return Sandbox(root: root, inside: inside, outside: outside,
                       secret: secret, lookalike: lookalike)
    }

    // MARK: allowlist

    @Test func allowsPathsInsideARoot() throws {
        let s = try makeSandbox()
        #expect(PathPolicy.isAllowed(s.inside, roots: [s.root]))
    }

    @Test func allowsTheRootItself() throws {
        let s = try makeSandbox()
        #expect(PathPolicy.isAllowed(s.root, roots: [s.root]))
    }

    @Test func rejectsPathsOutsideEveryRoot() throws {
        let s = try makeSandbox()
        #expect(!PathPolicy.isAllowed(s.secret, roots: [s.root]))
    }

    /// A sibling whose name merely starts with the root's name must not pass —
    /// the prefix check has to include the trailing separator.
    @Test func rejectsSiblingWithRootAsNamePrefix() throws {
        let s = try makeSandbox()
        #expect(!PathPolicy.isAllowed(s.lookalike, roots: [s.root]))
    }

    /// `..` traversal is collapsed by `standardizedFileURL` before the check.
    @Test func rejectsDotDotTraversalOutOfRoot() throws {
        let s = try makeSandbox()
        let escaped = s.root.appendingPathComponent("../outside/secret.txt")
        #expect(!PathPolicy.isAllowed(escaped, roots: [s.root]))
    }

    /// The case the whole design turns on: a symlink *inside* an allowed root that
    /// points outside it. Without resolution this reads as allowed.
    @Test func rejectsSymlinkEscapingRoot() throws {
        let s = try makeSandbox()
        let link = s.root.appendingPathComponent("innocent.txt")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: s.secret)
        #expect(!PathPolicy.isAllowed(link, roots: [s.root]))
    }

    // MARK: denylist

    @Test func rejectsDeniedDirectory() {
        #expect(!PathPolicy.isAllowed(home.appendingPathComponent(".ssh")))
    }

    @Test func rejectsFilesInsideADeniedDirectory() {
        #expect(!PathPolicy.isAllowed(home.appendingPathComponent(".ssh/id_rsa")))
        #expect(!PathPolicy.isAllowed(home.appendingPathComponent("Library/Keychains/login.keychain-db")))
    }

    @Test func rejectsDeniedFile() {
        #expect(!PathPolicy.isAllowed(home.appendingPathComponent(".netrc")))
    }

    /// The denylist matches whole path components: `.ssh-notes` is not `.ssh`.
    @Test func allowsNeighbourOfDeniedPath() {
        #expect(PathPolicy.isAllowed(home.appendingPathComponent(".ssh-notes")))
    }

    @Test func allowsOrdinaryHomePaths() {
        #expect(PathPolicy.isAllowed(home.appendingPathComponent("Documents/report.md")))
    }

    // MARK: resolve()

    @Test func resolveRejectsMissingArgument() {
        #expect(PathPolicy.resolve(nil) == .rejected("Error: 'path' is required."))
        #expect(PathPolicy.resolve("") == .rejected("Error: 'path' is required."))
    }

    @Test func resolveRejectsRelativePaths() {
        guard case .rejected(let message) = PathPolicy.resolve("Documents/report.md") else {
            Issue.record("expected a relative path to be rejected")
            return
        }
        #expect(message.contains("must be absolute"))
    }

    @Test func resolveExpandsTilde() {
        guard case .allowed(let url) = PathPolicy.resolve("~/Documents/report.md") else {
            Issue.record("expected ~ to expand into the home directory")
            return
        }
        #expect(url.path == home.appendingPathComponent("Documents/report.md").path)
    }

    @Test func resolveAppliesTheDenylist() {
        guard case .rejected(let message) = PathPolicy.resolve("~/.ssh/id_rsa") else {
            Issue.record("expected ~/.ssh/id_rsa to be refused")
            return
        }
        #expect(message.contains("not permitted"))
    }

    @Test func resolveHonoursCustomRoots() throws {
        let s = try makeSandbox()
        guard case .allowed = PathPolicy.resolve(s.inside.path, roots: [s.root]) else {
            Issue.record("expected a path inside the root to resolve")
            return
        }
        guard case .rejected = PathPolicy.resolve(s.secret.path, roots: [s.root]) else {
            Issue.record("expected a path outside the root to be refused")
            return
        }
    }
}
