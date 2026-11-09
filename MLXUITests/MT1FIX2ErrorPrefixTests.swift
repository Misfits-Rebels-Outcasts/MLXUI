//
//  MT1FIX2ErrorPrefixTests.swift
//  MT1-FIX-2: the "Error: " prefix is stripped only when present.
//

import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

@Suite("MT1FIX2 error prefix")
struct MT1FIX2ErrorPrefixTests {
    @Test func stripsThePrefixWhenPresent() {
        #expect(ModelToolRunner.withoutErrorPrefix("Error: access to /x is not permitted.") == "access to /x is not permitted.")
        #expect(ModelToolRunner.withoutErrorPrefix("Error: ") == "")
    }

    @Test func leavesTextWithoutThePrefixUntouched() {
        #expect(ModelToolRunner.withoutErrorPrefix("No folder is attached to this chat.") == "No folder is attached to this chat.")
        #expect(ModelToolRunner.withoutErrorPrefix("error: lowercase stays") == "error: lowercase stays")
        #expect(ModelToolRunner.withoutErrorPrefix("") == "")
    }

    @Test func everyPolicyRejectionThroughTheHelperReadsCleanly() {
        for path in [nil, "", "relative/no/base", "/etc/passwd"] as [String?] {
            guard case .rejected(let message) = PathPolicy.resolve(path, roots: [], base: nil, folders: []) else {
                Issue.record("expected a rejection for \(path ?? "nil")"); continue
            }
            #expect(message.hasPrefix("Error: "))
            #expect(!ModelToolRunner.withoutErrorPrefix(message).hasPrefix("Error"))
        }
    }

    #if !DIRECT_BUILD
    private func sandbox() throws -> (container: URL, outside: URL) {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("mt1fix2-\(UUID().uuidString)", isDirectory: true)
        let container = base.appendingPathComponent("container", isDirectory: true)
        let outside = base.appendingPathComponent("elsewhere", isDirectory: true)
        try FileManager.default.createDirectory(at: container, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        return (container, outside)
    }

    @Test func noFolderRejectionReadsAsNoFolderThroughTheHelper() throws {
        let s = try sandbox()
        defer { try? FileManager.default.removeItem(at: s.container.deletingLastPathComponent()) }
        let file = s.outside.appendingPathComponent("a.png")
        try Data([1]).write(to: file)
        let empty = FileScope(folders: [], workingFolder: nil)
        let raw = FileScope.$current.withValue(empty) {
            ScopedFileTools.rejection(for: file.path, container: s.container, roots: [s.container])
        }
        #expect(!raw.hasPrefix("Error: "))   // the case the old dropFirst corrupted
        #expect(ModelToolRunner.withoutErrorPrefix(raw).hasPrefix("No folder"))
    }

    @Test func theModelToolPathReportsNoFolderIntactNotACutSentence() async throws {
        let s = try sandbox()
        defer { try? FileManager.default.removeItem(at: s.container.deletingLastPathComponent()) }
        let file = s.outside.appendingPathComponent("a.png")
        try Data([1]).write(to: file)
        let spec = try #require(ModelToolSpec.spec(named: "ocr_image"))
        let runtime = ModelToolRuntime(
            catalog: [], installedModelIDs: [], claimableModelIDs: [], totalRAMGB: 16,
            makeExecutor: { _, _ in MT13Executor.text("x") }, pathRoots: [s.container])
        let runner = ModelToolRunner(spec: spec, runtime: runtime)
        let result = await FileScope.$current.withValue(FileScope(folders: [], workingFolder: nil)) {
            await runner.resolveInput(arguments: ["path": .string(file.path)], host: nil,
                                      scratch: s.container.appendingPathComponent("scratch"))
        }
        guard case .failure(let failure) = result else { Issue.record("expected a refusal"); return }
        #expect(failure.message.hasPrefix("No folder is attached"))   // not "er is attached ..."
    }
    #endif
}
