//
//  AskInChatResult.swift
//  MLXUI — Model Tools, MT4-5 (`RSI/DelegateModelToolsBacklog.md`: "each Run UI result → new conversation with the
//  result as `a1`").
//
//  What a Run UI holds that a user can send to the Chat page: text (a transcript, an answer, a translation), an image,
//  or audio. It is written to one file in a private temporary folder; the Chat page attaches that file to a new
//  conversation's composer and the ordinary send path registers it as an asset (`image` / `audio` / `text`).
//  A result that cannot become one of those cleanly (a video, a structured answer) has no case here.
//

import Foundation
import CoreGraphics

nonisolated enum AskInChatResult: @unchecked Sendable {
    /// Plain text; `name` is the attachment's file name stem (`transcript`).
    case text(String, name: String)
    case image(CGImage, name: String)
    case audio(AudioBuffer, name: String)

    /// The file name (with extension) the result is written under.
    var fileName: String {
        switch self {
        case .text(_, let name): Self.stem(name) + ".txt"
        case .image(_, let name): Self.stem(name) + ".png"
        case .audio(_, let name): Self.stem(name) + ".wav"
        }
    }

    /// Whether there is anything to send (an empty transcript is not a result).
    var hasContent: Bool {
        switch self {
        case .text(let text, _): !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .image: true
        case .audio(let buffer, _): !buffer.samples.isEmpty
        }
    }

    enum WriteError: Error, Equatable { case empty, encodingFailed }

    /// The temporary folder every Ask in Chat file goes under (`<tmp>/AskInChat/<uuid>/`).
    static func temporaryRoot(in base: URL = FileManager.default.temporaryDirectory) -> URL {
        base.appendingPathComponent("AskInChat", isDirectory: true)
    }

    /// Write the result to `<base>/AskInChat/<uuid>/<fileName>` and return that file. The Chat page copies it into the
    /// conversation on send, so this one is disposable (the OS clears the temporary folder).
    func writeTemporaryFile(in base: URL = FileManager.default.temporaryDirectory) throws -> URL {
        guard hasContent else { throw WriteError.empty }
        let folder = Self.temporaryRoot(in: base).appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent(fileName)
        switch self {
        case .text(let text, _):
            try Data(text.utf8).write(to: url, options: .atomic)
        case .image(let image, _):
            guard let data = PNGEncoder.pngData(from: image) else { throw WriteError.encodingFailed }
            try data.write(to: url, options: .atomic)
        case .audio(let buffer, _):
            try AudioWriter.writeWAV(buffer, to: url)
        }
        return url
    }

    /// A safe, short file name stem: letters, digits, `-` and `_`; anything else becomes `-`; never empty.
    static func stem(_ name: String) -> String {
        let cleaned = String(name.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" ? $0 : "-" }.prefix(40))
        return cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "-")).isEmpty ? "result" : cleaned
    }
}
