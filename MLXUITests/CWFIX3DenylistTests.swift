//
//  CWFIX3DenylistTests.swift
//  CW-FIX-3: the Direct file tools' denial is deny-list only, and this conversation's attachments
//  folder is exempt. `DirectFileScope.denial(for:)` (Apps/Direct, outside this target) is
//  `PathPolicy.isDenied(url, exempt: [scope.attachmentsFolder])`, so the decision is tested on the
//  shared helper it calls.
//

import Testing
import Foundation
@testable import MLXUI

@Suite("CWFIX3 denylist-only policy")
struct CWFIX3DenylistTests {
    private let home = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("cwfix3-home-\(UUID().uuidString)", isDirectory: true)

    private func make(_ relative: String) throws -> URL {
        let url = home.appendingPathComponent(relative, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func ordinaryPathsAreNotDenied() throws {
        defer { try? FileManager.default.removeItem(at: home) }
        let docs = try make("Documents")
        let external = URL(fileURLWithPath: "/Volumes/SomeExternal/Project")
        #expect(!PathPolicy.isDenied(docs, home: home))
        #expect(!PathPolicy.isDenied(home, home: home))
        #expect(!PathPolicy.isDenied(external, home: home))
    }

    @Test func credentialPathsAreDenied() throws {
        defer { try? FileManager.default.removeItem(at: home) }
        let ssh = try make(".ssh")
        let keychains = try make("Library/Keychains")
        #expect(PathPolicy.isDenied(ssh, home: home))
        #expect(PathPolicy.isDenied(ssh.appendingPathComponent("id_ed25519"), home: home))
        #expect(PathPolicy.isDenied(keychains, home: home))
    }

    @Test func aSymlinkIntoSSHIsDenied() throws {
        defer { try? FileManager.default.removeItem(at: home) }
        let ssh = try make(".ssh")
        let link = home.appendingPathComponent("innocent")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: ssh)
        #expect(PathPolicy.isDenied(link, home: home))
    }

    @Test func theExemptFolderIsNotDeniedEvenUnderADeniedEntry() throws {
        defer { try? FileManager.default.removeItem(at: home) }
        // A conversation's attachments folder placed under a deny entry (Library/Mail stands in
        // for any deny entry that covers ~/Library).
        let mine = try make("Library/Mail/conversations/AAA/attachments")
        let file = mine.appendingPathComponent("0-big.txt")
        #expect(PathPolicy.isDenied(file, home: home))                      // without the exemption
        #expect(!PathPolicy.isDenied(file, exempt: [mine], home: home))
        #expect(!PathPolicy.isDenied(mine, exempt: [mine], home: home))
    }

    @Test func anotherConversationsFolderKeepsTheNormalRules() throws {
        defer { try? FileManager.default.removeItem(at: home) }
        let mine = try make("Library/Mail/conversations/AAA/attachments")
        let other = try make("Library/Mail/conversations/BBB/attachments")
        #expect(PathPolicy.isDenied(other.appendingPathComponent("x.txt"), exempt: [mine], home: home))
        // A sibling that merely shares a name prefix is not inside the exempt folder either.
        let lookalike = try make("Library/Mail/conversations/AAA/attachments-extra")
        #expect(PathPolicy.isDenied(lookalike, exempt: [mine], home: home))
    }

    @Test func isAllowedIsUnchanged() throws {
        let realHome = FileManager.default.homeDirectoryForCurrentUser
        #expect(PathPolicy.isAllowed(realHome.appendingPathComponent("Documents")))
        #expect(!PathPolicy.isAllowed(realHome.appendingPathComponent(".ssh/id_rsa")))
        #expect(!PathPolicy.isAllowed(URL(fileURLWithPath: "/etc/passwd")))
        #expect(!PathPolicy.isAllowed(realHome.appendingPathComponent("Documents"), roots: []))
    }

    @Test func theScopeCarriesTheAttachmentsFolderAndOldInitsStillWork() {
        let plain = FileScope(folders: ["/tmp/a"], workingFolder: "/tmp/a")
        #expect(plain.attachmentsFolder == nil)
        let direct = FileScope.forConversation(
            attached: ["/tmp/a"], workingFolder: "/tmp/a", grants: [], edition: .direct,
            attachmentsFolder: "/tmp/att")
        #expect(direct.attachmentsFolder == "/tmp/att")
        #expect(direct.folders == ["/tmp/a"])
        let store = FileScope.forConversation(
            attached: [], workingFolder: nil, grants: [], edition: .appStore)
        #expect(store.attachmentsFolder == nil)
    }
}
