//
//  FolderLabels.swift
//  MLXUI — CW-FIX-3. The names a folder list shows.
//
//  A folder's bare name is its last path component. Two folders in one list with the same name
//  would be indistinguishable ("workfolder" twice), so those get their parent folder appended:
//  "workfolder — Desktop", and a second parent only if that is still ambiguous. A unique name
//  stays bare. Tooltips keep the full path.
//

import Foundation

nonisolated enum FolderLabels {

    /// Label for each path in `paths`, keyed by path. Pure; order-independent.
    static func labels(for paths: [String]) -> [String: String] {
        let unique = Array(Set(paths))
        let byName = Dictionary(grouping: unique, by: { FileScope.displayName(for: $0) })
        var result: [String: String] = [:]
        for (name, group) in byName {
            guard group.count > 1 else { result[group[0]] = name; continue }
            let parents = group.map { Array(parentNames(of: $0).reversed()) }   // nearest first
            var depth = 1
            while depth < (parents.map(\.count).max() ?? 0),
                  Set(parents.map { Array($0.prefix(depth)) }).count < group.count {
                depth += 1
            }
            for (path, chain) in zip(group, parents) {
                let shown = chain.prefix(depth).reversed().joined(separator: "/")
                result[path] = shown.isEmpty ? name : "\(name) — \(shown)"
            }
        }
        return result
    }

    /// The label for one path among the others it is listed with.
    static func label(_ path: String, among paths: [String]) -> String {
        labels(for: paths + [path])[path] ?? FileScope.displayName(for: path)
    }

    /// Folder names from the root down to (not including) the folder itself.
    private static func parentNames(of path: String) -> [String] {
        URL(fileURLWithPath: path).deletingLastPathComponent().pathComponents.filter { $0 != "/" }
    }
}
