import Foundation
import os

/// R7(b) (owner ruling, Boo, 2026-11-24) — a debug-level trail of the flow editor's Raw-settings / Run / Save path, so a real click in
/// the real app shows which step is missing: `log show --predicate 'subsystem == "net.connectcode.mlxui" && category == "editor"'
/// --info --debug`. **Never row content** — only step names, booleans and counts.
nonisolated enum EditorLog {
    static let logger = Logger(subsystem: "net.connectcode.mlxui", category: "editor")
    static func step(_ name: StaticString) { logger.debug("\(name, privacy: .public)") }
    static func step(_ name: StaticString, _ flag: Bool) { logger.debug("\(name, privacy: .public) \(flag, privacy: .public)") }
    static func step(_ name: StaticString, _ a: Bool, _ b: Bool) {
        logger.debug("\(name, privacy: .public) \(a, privacy: .public) \(b, privacy: .public)")
    }
    static func step(_ name: StaticString, count: Int) { logger.debug("\(name, privacy: .public) \(count, privacy: .public)") }
}
