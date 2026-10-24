import Foundation
import Testing

@testable import MLXUI

/// S-FIX-2 — the reproduction (ruling AZ). Opt-in: `TEST_RUNNER_MLXUI_REPRO_DEEP_BODY=<levels>` (and optionally
/// `TEST_RUNNER_MLXUI_REPRO_PATH=/v1/messages`), one depth per process because a stack overflow kills the host.
///
/// The body goes through `ServeAPI.handle` — the function `LocalServerHandler` calls for every request — on a **detached task**,
/// the thread kind FlyingFox uses: it runs each connection in a `withThrowingTaskGroup` child task, i.e. a Swift-concurrency
/// cooperative-pool thread (a secondary pthread with the default 512 KB stack, which this test prints). On unfixed code a deep
/// enough body overflows that stack inside `JSONSerialization.jsonObject` (`newJSONObject` recursion), before any route logic.
/// With the guard in place the same run answers 400. Prints `REPRO …`.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXUI_REPRO_DEEP_BODY"] != nil))
struct ServeDeepBodyReproTests {
    static func body(levels: Int, path: String) -> Data {
        // `MLXUI_REPRO_SHAPE=array|object` (default object): the shape that recurses differs between parsers.
        let object = ProcessInfo.processInfo.environment["MLXUI_REPRO_SHAPE"] != "array"
        let deep = object ? String(repeating: #"{"a":"#, count: levels) + "0" + String(repeating: "}", count: levels)
            : String(repeating: "[", count: levels) + String(repeating: "]", count: levels)
        let text = path == "/v1/responses" ? #"{"model":"m","input":"hi","x":\#(deep)}"#
            : path == "/v1/messages" || path == "/v1/messages/count_tokens"
            ? #"{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"x":\#(deep)}"#
            : #"{"model":"m","messages":[{"role":"user","content":"hi"}],"x":\#(deep)}"#
        return Data(text.utf8)
    }

    @Test func aDeepBodyThroughTheServersEntryPoint() async throws {
        let levels = try #require(Int(ProcessInfo.processInfo.environment["MLXUI_REPRO_DEEP_BODY"] ?? ""))
        let paths = ProcessInfo.processInfo.environment["MLXUI_REPRO_PATH"].map { [$0] }
            ?? ["/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses"]
        for path in paths {
            let data = Self.body(levels: levels, path: path)
            let head = ServeRequestHead(method: "POST", path: path, host: "127.0.0.1:1212", origin: nil, transferEncoding: nil,
                                        contentLength: String(data.count), userAgent: nil)
            let environment = ServeEnvironment(servedModelIDs: { [] }, installed: { InstalledModelIndex(entries: []) },
                                               appleFoundationReadiness: { nil }, created: 0)
            print("REPRO \(path) levels=\(levels) starting")
            let (status, stack, onMain) = await Task.detached { () -> (Int, Int, Bool) in
                let reply = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: environment, body: { data })
                var status = -1
                if case .full(let response) = reply { status = response.status }
                return (status, Int(pthread_get_stacksize_np(pthread_self())), Thread.isMainThread)
            }.value
            print("REPRO \(path) levels=\(levels) survived status=\(status) threadStackBytes=\(stack) onMainThread=\(onMain)")
        }
    }
}
