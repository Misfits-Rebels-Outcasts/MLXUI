import Testing
import Foundation
@testable import MLXUI

/// MT4-7-FIX-4 (D): after a cancel, a download's queued progress updates must not put the model back to "downloading"
/// (Browse showed it frozen at the last byte count until the user clicked Cancel there). Stub network, no real traffic.
@Suite(.serialized)
struct InstallManagerCancelTests {

    /// Serves the file listing at once and the one file slowly, in chunks, until `stopLoading`.
    private final class SlowNet: URLProtocol, @unchecked Sendable {
        private let lock = NSLock()
        private var stopped = false
        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() {
            let url = request.url ?? URL(fileURLWithPath: "/")
            if url.path.hasPrefix("/api/models/") {
                let body = try! JSONSerialization.data(withJSONObject: ["siblings": [["rfilename": "model.safetensors", "size": 4000]]])
                let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: body)
                client?.urlProtocolDidFinishLoading(self)
                return
            }
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: ["Content-Length": "4000"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            DispatchQueue.global().async { [self] in
                for _ in 0..<40 {
                    if lock.withLock({ stopped }) { return }
                    client?.urlProtocol(self, didLoad: Data(repeating: 0x41, count: 100))
                    Thread.sleep(forTimeInterval: 0.03)
                }
                if !lock.withLock({ stopped }) { client?.urlProtocolDidFinishLoading(self) }
            }
        }
        override func stopLoading() { lock.withLock { stopped = true } }
    }

    private static func session() -> URLSession {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [SlowNet.self]
        return URLSession(configuration: c)
    }

    @Test @MainActor func aCancelledDownloadStaysIdleAndKeepsNoProgress() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("fix4d-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manager = InstallManager(store: ModelStore(baseDirectory: dir), session: Self.session(), apiSession: Self.session())
        let entry = makeEntry(id: "mlx-community--Cancel-Me", downloadSizeGB: 0.00001, hfModelId: "mlx-community/Cancel-Me")
        manager.install(entry)
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            if case .downloading(_, let downloaded, _)? = manager.modelStates[entry.id], downloaded > 0 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        guard case .downloading? = manager.modelStates[entry.id] else {
            Issue.record("never reached downloading: \(String(describing: manager.modelStates[entry.id]))"); manager.cancel(entry); return
        }
        manager.cancel(entry)
        #expect(manager.modelStates[entry.id] == .idle)
        // Queued progress updates and the stopped transfer's callbacks arrive after this point; none may change the state.
        for _ in 0..<20 {
            try await Task.sleep(for: .milliseconds(25))
            #expect(manager.modelStates[entry.id] == .idle)
        }
    }

    @Test @MainActor func aNewDownloadAfterACancelIsNotDisturbedByTheOldOne() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("fix4d-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manager = InstallManager(store: ModelStore(baseDirectory: dir), session: Self.session(), apiSession: Self.session())
        let entry = makeEntry(id: "mlx-community--Again", downloadSizeGB: 0.00001, hfModelId: "mlx-community/Again")
        manager.install(entry)
        try await Task.sleep(for: .milliseconds(150))
        manager.cancel(entry)
        manager.install(entry)
        #expect(manager.modelStates[entry.id] == .resolving)
        try await Task.sleep(for: .milliseconds(300))
        #expect(manager.modelStates[entry.id] != .idle)          // the second download is alive and reporting
        manager.cancel(entry)
        #expect(manager.modelStates[entry.id] == .idle)
    }
}
