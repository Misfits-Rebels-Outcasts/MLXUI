import Testing
import SwiftUI
import AppKit
@testable import MLXUI

// SC-4d — the polished Local Server screen renders for running / stopped / failed without crashing, in both
// colour schemes and at a narrow (~700 pt) width. Set SC4D_SHOTS to a directory to also write the PNGs.

@Suite(.serialized)
@MainActor
struct ServerScreenRenderTests {
    private let gib = Int64(1_073_741_824)

    private func controller() -> LocalServerController {
        let server = LocalServer(settings: ServeSettings(servedModelIDs: []), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [] }, installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        return LocalServerController(server: server, defaults: UserDefaults(suiteName: "render-\(UUID().uuidString)")!,
                                     installed: { InstalledModelIndex(entries: []) }, appleFoundation: { nil },
                                     catalogLoaded: { true }, evict: { _ in }, evictPromptCache: { _ in },
                                     budgetBytes: 20 * gib, preloadPool: nil,
                                     ledger: { .init(used: 0, poolWeights: 0, snapshots: 0) })
    }

    private func model(_ state: ServerScreenModel.RunState, lowDisk: Bool = false) -> ServerScreenModel {
        func row(_ name: String, _ line: String?, served: Bool = true) -> ServerScreenModel.Row {
            .init(id: name, name: name, isAppleFoundation: false, ramGB: 4, sizeText: "2.3 GB",
                  contextText: "32,768 token context", isServed: served, stateLine: line)
        }
        let on = state == .running
        return ServerScreenModel(
            runState: state,
            statusSentence: state == .failed ? "Couldn't start — port 1212 is in use." : (on ? "Serving" : "Not serving"),
            address: on ? "127.0.0.1:1212" : nil, switchOn: state != .stopped,
            rows: [row("Qwen3-4B-4bit", on ? "Ready" : "Server is off"),
                   row("Llama-3.2-3B-4bit", on ? "Loading…" : nil, served: on),
                   row("Phi-4-mini-4bit", on ? "Not loaded · loads on next request · pushed out by a flow" : nil, served: on),
                   row("Gemma-3-4b-4bit", on ? "Couldn't load · tries again on the next request" : nil, served: on)],
            stats: on ? [.init(id: "a", name: "Qwen3-4B-4bit", requests: 42, unavailable: 1, ttft: "0.4 s",
                               decode: "58.2 tok/s", reused: "61%", lastRequest: Date())] : [],
            memory: ServeMemoryReading(capacity: 20 * gib, servedKeptReady: 6 * gib, otherModels: 2 * gib,
                                       engines: gib, snapshots: gib / 2),
            diskFact: "Free disk space: 120 GB",
            lowDiskWarning: lowDisk ? ServeDisk.lowSpaceWarning : nil)
    }

    private func render(_ screen: ServerScreenModel, width: CGFloat, scheme: ColorScheme) -> NSImage? {
        let view = ScrollView {
            ServerScreenContent(screen: screen, controller: controller(), systemRAMGB: 16, refresh: {})
                .frame(maxWidth: ServerStyle.maxContentWidth)
                .padding(ServerStyle.pagePadding)
                .frame(maxWidth: .infinity)
        }
        .frame(width: width, height: 1100)
        .background(Color(nsColor: .windowBackgroundColor))
        .environment(\.colorScheme, scheme)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        return renderer.nsImage
    }

    private func save(_ image: NSImage, _ name: String) {
        guard let dir = ProcessInfo.processInfo.environment["SC4D_SHOTS"],
              let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return }
        try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
    }

    @Test(arguments: [ServerScreenModel.RunState.running, .stopped, .failed])
    func rendersEachState(state: ServerScreenModel.RunState) {
        for (scheme, label) in [(ColorScheme.light, "light"), (.dark, "dark")] {
            let image = render(model(state, lowDisk: state == .failed), width: 1000, scheme: scheme)
            #expect(image != nil)
            if let image { save(image, "\(state)-\(label)") }
        }
    }

    @Test func rendersAtNarrowWidth() {
        let image = render(model(.running, lowDisk: true), width: 700, scheme: .light)
        #expect(image != nil)
        if let image { save(image, "running-narrow") }
    }

    @Test func stateChipSplitsLabelFromReason() {
        let ready = ServerScreenModel.stateLine(served: true, serverOn: true, state: .ready) ?? ""
        #expect(ready == "Ready")
        let line = ServerScreenModel.stateLine(served: true, serverOn: true, state: .notLoaded(reason: "pushed out")) ?? ""
        #expect(line.components(separatedBy: " · ")[0] == "Not loaded")
    }
}
