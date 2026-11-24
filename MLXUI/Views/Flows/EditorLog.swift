import Foundation
import os
import AppKit

/// R7(b) (owner ruling, Boo, 2026-11-24) — a trail of the flow editor's Raw-settings / Run / Save path, so a real click in the real
/// app shows which step is missing: `/usr/bin/log show --start "<time>" --end "<time>" --predicate 'subsystem == "net.connectcode.mlxui"
/// && category == "editor"'`. **Never row content** — only step names, booleans and counts.
///
/// `.notice`, not `.debug`: debug-level entries are never written to the persistent log store, so `log show` after the fact finds
/// nothing (the first R7 run, 2026-11-24 18:21–18:24, returned an empty window for exactly that reason).
nonisolated enum EditorLog {
    static let logger = Logger(subsystem: "net.connectcode.mlxui", category: "editor")
    static func step(_ name: StaticString) { logger.notice("\(name, privacy: .public)") }
    static func step(_ name: StaticString, _ flag: Bool) { logger.notice("\(name, privacy: .public) \(flag, privacy: .public)") }
    static func step(_ name: StaticString, _ a: Bool, _ b: Bool) {
        logger.notice("\(name, privacy: .public) \(a, privacy: .public) \(b, privacy: .public)")
    }
    static func step(_ name: StaticString, count: Int) { logger.notice("\(name, privacy: .public) \(count, privacy: .public)") }

    /// Left-mouse down/up reaching the app, with whether a text view was first responder — separates "the click never arrived" from
    /// "it arrived and the action did not run". Installed once, by the editor.
    @MainActor private static var mouseMonitor: Any?
    @MainActor static func installMouseTrail() {
        guard mouseMonitor == nil else { return }
        mouseMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .leftMouseUp]) { event in
            let typing = event.window?.firstResponder is NSTextView
            if event.type == .leftMouseDown, let content = event.window?.contentView {
                // What the click actually lands on (class names only) — for a Run click whose action never arrives.
                let hit = content.hitTest(content.convert(event.locationInWindow, from: nil))
                var chain: [String] = []; var v = hit
                while let x = v, chain.count < 6 { chain.append(String(describing: type(of: x))); v = x.superview }
                logger.notice("mouse down lands on \(chain.joined(separator: " < "), privacy: .public)")
            }
            logger.notice("mouse \(event.type == .leftMouseDown ? "down" : "up", privacy: .public); a text view is first responder \(typing, privacy: .public)")
            return event
        }
    }
}
