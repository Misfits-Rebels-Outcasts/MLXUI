import Foundation
import os

/// R7(b) (owner ruling, Boo, 2026-11-24) — a trail of the flow editor's Raw-settings / Run / Save path, so a real click in the real
/// app shows which step is missing: `/usr/bin/log show --start "<time>" --end "<time>" --predicate 'subsystem == "net.connectcode.mlxui"
/// && category == "editor"'`. **Never row content** — only step names, booleans and counts.
///
/// `.debug` (R15, owner ruling, Boo, 2026-11-24): not persisted by default, so nothing accumulates on a user's Mac. To read it, run
/// `sudo log config --subsystem net.connectcode.mlxui --mode "level:debug,persist:debug"` first (the 2026-11-24 hunt found that a
/// plain `log show` of `.debug` entries returns an empty window), or stream it live: `log stream --level debug --predicate ...`.
nonisolated enum EditorLog {
    static let logger = Logger(subsystem: "net.connectcode.mlxui", category: "editor")
    static func step(_ name: StaticString) { logger.debug("\(name, privacy: .public)") }
    static func step(_ name: StaticString, _ flag: Bool) { logger.debug("\(name, privacy: .public) \(flag, privacy: .public)") }
    static func step(_ name: StaticString, _ a: Bool, _ b: Bool) {
        logger.debug("\(name, privacy: .public) \(a, privacy: .public) \(b, privacy: .public)")
    }
    static func step(_ name: StaticString, count: Int) { logger.debug("\(name, privacy: .public) \(count, privacy: .public)") }
}
