import Foundation

/// The KEY review's one non-defect request: "keep the `XCTestConfigurationFilePath`
/// check in one named place, so shipping code has exactly one line that knows tests
/// exist." `XCTestConfigurationFilePath` is the standard, tool-agnostic signal Xcode's
/// test runner sets in the environment — both XCTest and swift-testing launch through
/// the same `xctest` host mechanism, so it covers both, and nothing else sets it.
///
/// Needed because a macOS app's test bundle runs **inside the real app process** as its
/// test host: `@main`/`init()` fire for real under `xcodebuild test`, not only on a
/// genuine launch (confirmed the hard way in AFM-FOLLOWUP-1, journal `2026-259` — the
/// app's own startup logging appeared during a plain test run before this check
/// existed). Anywhere in the app that must behave differently under test than under a
/// real launch reads `TestEnvironment.isRunningTests`, never the raw environment lookup
/// — one name, one place, so a future reader searches for this instead of re-deriving
/// the same fact a second way.
nonisolated enum TestEnvironment {
    static var isRunningTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }
}
