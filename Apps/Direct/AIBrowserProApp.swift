//
//  AIBrowserProApp.swift
//  MLXUI-Direct target (direct download, notarized) — @main shim.
//
//  Identical to the App Store shim except for what it injects: the shell,
//  filesystem and subprocess tools defined alongside it in this target.
//
//  NOT YET IN ANY TARGET — added in Phase 2 of Design/dual-distribution.md.
//

import SwiftUI
import MLXUIKit

@main
struct AIBrowserProApp: App {
    final class AppDelegate: NSObject, NSApplicationDelegate {
        func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
            true
        }
    }
    @NSApplicationDelegateAdaptor private var appDelegate: AppDelegate

    var body: some Scene {
        // The one line that distinguishes the two editions.
        RootScene(agentTools: DirectTools.all())
    }
}
