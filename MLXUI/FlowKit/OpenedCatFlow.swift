import Foundation

/// A `.cat` file the user opened from disk (CFM-R5-5). Read-only: the file is parsed and
/// validated but never written. `bookmarkData` is the security-scoped bookmark so a later
/// launch can re-grant access; the file's contents are captured at open time.
nonisolated struct OpenedCatFlow: Sendable {
    var url: URL
    var displayName: String
    var rawText: String
    var parsed: ParsedFlow
    var issues: [FlowIssue]
    var bookmarkData: Data?
}

/// Errors opening an external `.cat` file. Voice names the problem and implies the fix.
nonisolated enum OpenCatFlowError: Error, CustomStringConvertible, Equatable {
    case unreadable(path: String, reason: String)
    case accessDenied(path: String)
    case parseFailed(path: String, message: String)

    var description: String {
        switch self {
        case .unreadable(let path, let reason):
            return "Couldn't read '\(path)': \(reason). Make sure the file is readable and isn't a directory."
        case .accessDenied(let path):
            return "This Mac refused access to '\(path)'. Choose the file again from the Open panel to re-grant permission."
        case .parseFailed(let path, let message):
            return "'\(path)' isn't a valid CAT Flow: \(message)"
        }
    }
}
