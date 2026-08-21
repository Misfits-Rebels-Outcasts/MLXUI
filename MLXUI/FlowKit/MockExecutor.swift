import Foundation
import CryptoKit

/// The deterministic, kind-driven executor for CI and tests — mirrors
/// `catflow-mlx/src/catflow/engines/mock.py`'s content contract **without** the decider/
/// improvise/transform branches (out of the linear subset):
///
/// - Every output item's **kind** comes from the task's catalog `gives` shape
///   (`SameAsInput` resolves to whichever kind the input actually was).
/// - `text`/`status`/`context` are inline: `[mock:<kind>] <fingerprint>#<i>`.
/// - Every other kind is **file-backed**: a deterministic blob under `blobDirectory`, named
///   `<path>.<index>.<kind>.<sha256-prefix>.bin`, whose bytes are the digest itself.
/// - **Cardinality**: a `ListOf → ListOf` task with exactly one input mirrors that input's
///   item count; any other list-producing task uses `DEFAULT_LIST_LEN` (3).
/// - **Content** is the SHA-256 hex of a fingerprint string — `path|task|model|settings|
///   input values or blob filenames` — deterministic for a given flow, and changed by any of
///   those. This foreshadows (but does not implement) R3's real cache key.
///
/// No models, no downloads. `FlowRunner` runs every flow test through this, exactly as the
/// Python's default test run does. See `RSI/DelegateMergeBacklog.md` CFM-R2-6.
nonisolated struct MockExecutor: FlowExecutor {
    static let defaultListLen = 3

    let blobDirectory: URL

    init(blobDirectory: URL) {
        self.blobDirectory = blobDirectory
    }

    func execute(path: String, row: Row, inputs: [Asset]) async throws -> Asset {
        guard let desc = TaskCatalog.get(row.task ?? "") else {
            throw FlowError.unknownTask(row: row.task ?? "?")
        }
        let gives = resolveGives(desc.gives, inputs: inputs)
        let count = outputCount(desc, inputs: inputs)
        let kind = itemKind(gives)
        let basis = fingerprint(path: path, row: row, inputs: inputs)
        return Asset(items: (0..<count).map { i in
            mockItem(kind: kind, basis: "\(basis)#\(i)", path: path, index: i)
        })
    }

    // MARK: - Shape resolution

    /// `_resolve_gives` — `SameAsInput` becomes the input's kind (fallback: `.file`).
    private func resolveGives(_ gives: Shape, inputs: [Asset]) -> Shape {
        if gives == .sameAsInput {
            if let kind = inputs.first?.items.first?.kind {
                return .single(kind)
            }
            return .single(.file)
        }
        return gives
    }

    /// `_output_count` — a `ListOf`-giving task mirrors a single `ListOf` input's cardinality
    /// (element-wise map), else `DEFAULT_LIST_LEN`.
    private func outputCount(_ desc: TaskDescriptor, inputs: [Asset]) -> Int {
        let gives = resolveGives(desc.gives, inputs: inputs)
        guard case .listOf = gives else { return 1 }
        if case .listOf = desc.accepts, inputs.count == 1 {
            return inputs[0].items.count
        }
        return Self.defaultListLen
    }

    /// `_item_kind` — the `Single`/`ListOf` element kind of a gives shape.
    private func itemKind(_ gives: Shape) -> Kind {
        if case .single(let k) = gives { return k }
        if case .listOf(let k) = gives { return k }
        return .text   // unreachable on a valid catalog; keep it deterministic
    }

    // MARK: - Content

    /// `_fingerprint` — deterministic and location-independent: file-backed inputs
    /// contribute their blob's filename (stable), never the absolute path.
    private func fingerprint(path: String, row: Row, inputs: [Asset]) -> String {
        var parts = [path, row.task ?? "", row.model ?? "", row.settings ?? ""]
        for asset in inputs {
            for item in asset.items {
                if let value = item.value {
                    parts.append(value)
                } else {
                    parts.append(item.path?.lastPathComponent ?? "")
                }
            }
        }
        return parts.joined(separator: "|")
    }

    private func mockItem(kind: Kind, basis: String, path: String, index: Int) -> Item {
        switch kind {
        case .text, .status, .context:
            return Item(kind: kind, value: "[mock:\(kind.rawValue)] \(basis)", path: nil, sourceText: nil)
        default:
            let digest = SHA256.hash(data: Data(basis.utf8)).map { String(format: "%02x", $0) }.joined()
            let fm = FileManager.default
            try? fm.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
            let blob = blobDirectory
                .appendingPathComponent("\(path).\(index).\(kind.rawValue).\(String(digest.prefix(16))).bin")
            // Deterministic bytes: the digest itself.
            try? Data(digest.utf8).write(to: blob)
            return Item(kind: kind, value: nil, path: blob, sourceText: nil)
        }
    }
}
