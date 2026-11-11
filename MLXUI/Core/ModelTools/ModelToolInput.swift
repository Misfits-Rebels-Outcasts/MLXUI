//
//  ModelToolInput.swift
//  MLXUI — Model Tools, MT1-3, plan §3.7 step 1: resolve the input.
//
//  An asset handle (`a1`), a path inside the conversation's granted scope, or an http(s) URL
//  (`MediaFetch`, kept for back-compat). Text tools also take the text itself. Anything else
//  names what is valid.
//

import Foundation
import MLXLMCommon

nonisolated struct ResolvedModelToolInput: Sendable {
    var asset: Asset?
    /// The handles the input came from (provenance).
    var handles: [String] = []
    /// Files made for this call (a fetched URL, a copy of a granted file), removed afterwards.
    var temporaryFiles: [URL] = []
}

extension ModelToolRunner {

    func resolveInput(arguments: [String: JSONValue], host: ModelToolHost?, scratch: URL) async
        -> Result<ResolvedModelToolInput, ToolFailure>
    {
        guard let kind = spec.input else { return .success(ResolvedModelToolInput()) }
        let asset = arguments.string("asset")?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
        let path = arguments.string("path")?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
        let url = arguments.string("url")?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
        let text = arguments.string("text")?.nonEmptyKeepingWhitespace
        // MT4-7-FIX-1 (owner, 2026-11-12): a text tool given both `text` and `asset` (Apple Intelligence sends both) uses the
        // text and ignores the asset. Only that pair; any other mix still refuses.
        let ignoredAsset = kind == .text && text != nil && asset != nil && path == nil && url == nil
        let given = [ignoredAsset ? nil : asset, path, url, text].compactMap { $0 }
        let names = Self.inputArgumentNames(for: spec)
        guard given.count == 1 else {
            let list = names.joined(separator: ", ")
            return .failure(ToolFailure(given.isEmpty
                ? "give the input for \(spec.name) as exactly one of: \(list)."
                : "give only one of \(list) for \(spec.name)."))
        }

        if let text {
            return .success(ResolvedModelToolInput(asset: Asset(items: [Item(kind: .text, value: text)])))
        }
        if let asset { return resolveHandle(asset, kind: kind, host: host) }
        if let path { return resolvePath(path, kind: kind, scratch: scratch, host: host) }
        if let url { return await resolveURL(url, kind: kind, scratch: scratch) }
        return .failure(ToolFailure("give the input for \(spec.name)."))
    }

    /// MT1-FIX-2: the file tools' rejections lead with "Error: " but not all do ("No folder is attached
    /// to this chat..."), so strip it only when it is there. A blind `dropFirst(7)` cut real text.
    static func withoutErrorPrefix(_ message: String) -> String {
        let prefix = "Error: "
        return message.hasPrefix(prefix) ? String(message.dropFirst(prefix.count)) : message
    }

    private func resolveHandle(_ handle: String, kind: Kind, host: ModelToolHost?) -> Result<ResolvedModelToolInput, ToolFailure> {
        guard let host else {
            return .failure(ToolFailure("asset handles are only available in a chat. Give a path or url instead."))
        }
        guard let record = host.assets.record(handle) else {
            let known = host.assets.all.map { "\($0.handle) (\($0.kind))" }.joined(separator: ", ")
            return .failure(ToolFailure("no asset \"\(handle)\". " + (known.isEmpty ? "This chat has no assets yet." : "Known assets: \(known).")))
        }
        guard record.kind == kind.rawValue else {
            return .failure(ToolFailure("asset \(record.handle) is \(record.kind), but \(spec.name) needs \(kind.rawValue)."))
        }
        if kind == .text {
            guard let text = host.assets.text(for: record.handle) else {
                return .failure(ToolFailure("couldn't read asset \(record.handle)."))
            }
            return .success(ResolvedModelToolInput(asset: Asset(items: [Item(kind: .text, value: text)]), handles: [record.handle]))
        }
        let file = host.assets.url(for: record)
        return .success(ResolvedModelToolInput(asset: Asset(items: [Item(kind: kind, path: file)]), handles: [record.handle]))
    }

    private func resolvePath(_ raw: String, kind: Kind, scratch: URL, host: ModelToolHost?) -> Result<ResolvedModelToolInput, ToolFailure> {
        try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let ext = URL(fileURLWithPath: raw).pathExtension
        let copy = scratch.appendingPathComponent("input-\(UUID().uuidString)").appendingPathExtension(ext.isEmpty ? "bin" : ext)
        #if DIRECT_BUILD
        // MT4-2 (owner ruling 2026-11-09): only the folders attached to this chat, not all of home as `read_file` may.
        let scope = FileScope.current
        let roots = runtime.pathRoots ?? ModelToolPathScope.directRoots(scope: scope, assets: host?.assets.directory)
        switch PathPolicy.resolve(raw, roots: roots, base: ScopedFileTools.workingFolderBase()) {
        case .rejected(let message): return .failure(ToolFailure(ModelToolPathScope.refusal(message, scope: scope)))
        case .allowed(let file):
            do { try FileManager.default.copyItem(at: file, to: copy) } catch {
                return .failure(ToolFailure("couldn't read \(file.path)."))
            }
        }
        #else
        let container = runtime.pathContainer ?? runtime.pathRoots?.first ?? ModelStore.shared.baseDirectory
        // MT4-2 (owner ruling 2026-11-09): attached/granted folders, and from the app container only this chat's own attachments
        // and assets folders (not the models, installed.json or another chat's files, which `read_file` can reach). Production
        // (no `pathRoots`) always narrows; a test seam with `pathRoots` alone keeps its roots literal.
        let roots: [URL]
        if runtime.pathRoots == nil || runtime.pathContainer != nil {
            roots = ModelToolPathScope.appStoreRoots(
                base: runtime.pathRoots ?? FileScope.roots(container: container), container: container,
                scope: FileScope.current, assets: host?.assets.directory)
        } else {
            roots = runtime.pathRoots ?? []
        }
        guard let access = ScopedFileTools.access(raw, container: container, roots: roots) else {
            return .failure(ToolFailure(Self.withoutErrorPrefix(
                ScopedFileTools.rejection(for: raw, container: container, roots: roots))))
        }
        // A granted file is readable only inside its security scope, so the copy is made there.
        let copied: Bool? = FolderGrants.withScopedRead(to: access.url, grants: access.grants,
                                                        container: container) { scoped in
            (try? FileManager.default.copyItem(at: scoped, to: copy)) != nil
        }
        guard copied == true else { return .failure(ToolFailure("couldn't read \(access.url.path).")) }
        #endif
        return .success(ResolvedModelToolInput(asset: Asset(items: [Item(kind: kind, path: copy)]), temporaryFiles: [copy]))
    }

    private func resolveURL(_ raw: String, kind: Kind, scratch: URL) async -> Result<ResolvedModelToolInput, ToolFailure> {
        guard kind == .audio || kind == .image else {
            return .failure(ToolFailure("\(spec.name) does not take a url."))
        }
        switch await MediaFetch.fetchData(raw) {
        case .failure(let message): return .failure(ToolFailure(message))
        case .success(let data, let ext):
            try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            // AVFoundation sniffs by file extension, so keep the URL's extension on the temp file.
            let file = scratch.appendingPathComponent("input-\(UUID().uuidString)")
                .appendingPathExtension(ext.isEmpty ? (kind == .audio ? "audio" : "img") : ext)
            do { try data.write(to: file) } catch { return .failure(ToolFailure("couldn't keep the downloaded file.")) }
            return .success(ResolvedModelToolInput(asset: Asset(items: [Item(kind: kind, path: file)]), temporaryFiles: [file]))
        }
    }
}

private extension String {
    var nonEmpty: String? { isEmpty ? nil : self }
    /// A text argument keeps its inner whitespace; only an all-blank one counts as absent.
    var nonEmptyKeepingWhitespace: String? { trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : self }
}
