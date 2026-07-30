//
//  AIBrowserApp.swift
//  MLXUI target (Mac App Store edition) — the entire target, apart from assets.
//
//  Replaces MLXUIApp.swift, whose body moves into MLXUIKit as `RootScene`.
//  Keeping the app target down to an @main shim is what makes two editions cheap:
//  everything real lives in the package, and the only public symbol crossing the
//  boundary is `RootScene`.
//
//  NOT YET IN ANY TARGET — add it in Phase 1 step 6 of
//  Design/dual-distribution.md, at the same time MLXUIApp.swift is removed
//  (two @main types in one module will not compile).
//

import SwiftUI
import MLXUIKit

@main
struct AIBrowserApp: App {
    // Terminate the app when its window closes.
    // https://developer.apple.com/forums/thread/710376
    final class AppDelegate: NSObject, NSApplicationDelegate {
        func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
            true
        }
    }
    @NSApplicationDelegateAdaptor private var appDelegate: AppDelegate

    var body: some Scene {
        // No agent tools: the sandbox blocks shell and arbitrary-path access, and
        // the implementations aren't compiled into this target at all.
        RootScene(agentTools: [])
    }
}
