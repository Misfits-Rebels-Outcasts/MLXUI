import Foundation
import CryptoKit

/// The deterministic, kind-driven executor for CI and tests — mirrors
/// `catflow-mlx/src/catflow/engines/mock.py`'s content contract, including the decider
/// branches (`last_tag` + R1/R2/R3 payload):
///
/// - Every output item's **kind** comes from the task's catalog `gives` shape
///   (`SameAsInput` resolves to whichever kind the input actually was).
/// - `text`/`status`/`context` are inline: `[mock:<kind>] <fingerprint>#<i>`.
/// - Every other kind is **file-backed**: a deterministic blob under `blobDirectory`, named
///   `<path>.<index>.<kind>.<sha256-prefix>.bin`, whose bytes are the digest itself.
/// - **Cardinality**: a `ListOf → ListOf` task with exactly one input mirrors that input's
///   item count; any other list-producing task uses `DEFAULT_LIST_LEN` (3).
/// - **Deciders** (a task in `TaskCatalog.deciderTasks`) never "produce content": the tag is
///   mock scaffolding (deterministically the first declared tag, or a scripted sequence) and
///   reported via `lastTag`; the payload follows Spec §7.4 R1–R3 (passthrough / Judge
///   candidate / Think synthesized fingerprint keyed by the fired tag).
///
/// A reference type so `lastTag` (and the cache-hit flag) survive across the `FlowExecutor`
/// existential. No models, no downloads. `FlowRunner`/`FlowInterpreter` run flow tests
/// through this, exactly as the Python's default test run does.
final class MockExecutor: FlowExecutor, @unchecked Sendable {
    static let defaultListLen = 3

    let blobDirectory: URL
    /// `decider_script` — `{bare_row_path: [tag, ...]}` — lets a trace pin the exact tag
    /// sequence a decider fires per visit (the Python's `MockExecutor(decider_script=...)`,
    /// P2-M11-01). Visit counting is per bare path, so a scripted row's `@k` activations all
    /// draw from the same list; a `nil` entry means "declared-tag default", not "no tags".
    let deciderScript: [String: [String]]
    private let lock = NSLock()
    private var _lastTag: String?
    private var _lastTimeoutFlag: (code: String, message: String)?
    private var _lastStaged: (id: String, kind: String, summary: String)?
    private var _deciderVisits: [String: Int] = [:]
    var lastTag: String? { lock.withLock { _lastTag } }
    var lastTimeoutFlag: (code: String, message: String)? { lock.withLock { _lastTimeoutFlag } }
    var lastStaged: (id: String, kind: String, summary: String)? { lock.withLock { _lastStaged } }

    init(blobDirectory: URL, deciderScript: [String: [String]] = [:]) {
        self.blobDirectory = blobDirectory
        self.deciderScript = deciderScript
    }

    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        lock.withLock {
            _lastTag = nil
            _lastTimeoutFlag = nil
            _lastStaged = nil
        }
        if TaskCatalog.deciderTasks[row.task ?? ""] != nil {
            let tag = pickTag(path: path, row: row)
            lock.withLock { _lastTag = tag }
            return deciderPayload(path: path, row: row, inputs: inputs, tag: tag)
        }
        // A trigger's payload is the real occurrence the interpreter already built into
        // `inputs` — pass it through rather than fabricating fingerprinted content (P3-MD-01).
        if row.task == "On File" || row.task == "On Schedule" || row.task == "On Flow" {
            return inputs.first ?? Asset(items: [])
        }
        // `Read Context` must return parseable journal JSON (the interpreter hydrates the
        // run's live journal from it) — a real empty journal is the deterministic mock value.
        if row.task == "Read Context" {
            return Asset(items: [Item(kind: .context, value: "[]", path: nil, sourceText: nil)])
        }
        // A human row with a timeout default (not `wait=forever` — the interpreter parks
        // those) resolves to its `default=` tag / unchanged text and discloses F002.
        if row.task == "Ask Human" {
            let s = FlowSettings(row.settings)
            let output = inputs.first ?? Asset(items: [])
            let timeout = s.value(for: "timeout") ?? ""
            let dflt = s.value(for: "default") ?? ""
            lock.withLock {
                _lastTag = dflt.isEmpty ? nil : dflt
                if !timeout.isEmpty && !dflt.isEmpty {
                    _lastTimeoutFlag = ("F002", "Nobody answered by \(timeout) — proceeded as `\(dflt)`, unreviewed.")
                }
            }
            return output
        }
        if row.task == "Human Input" {
            let s = FlowSettings(row.settings)
            let output = inputs.first ?? Asset(items: [])
            let timeout = s.value(for: "timeout") ?? ""
            lock.withLock {
                if !timeout.isEmpty {
                    _lastTimeoutFlag = ("F002", "Nobody answered by \(timeout) — proceeded as `unchanged`, unreviewed.")
                }
            }
            return output
        }
        // Stage Send / Stage Post queue a real outbox entry (deterministic content-digest id).
        if row.task == "Stage Send" || row.task == "Stage Post" {
            let kind = row.task == "Stage Send" ? "send" : "post"
            let s = FlowSettings(row.settings)
            let destination = s.firstBare() ?? "outbox"
            let text = inputs.first?.items.first?.value ?? ""
            let digest = SHA256.hash(data: Data("\(row.task ?? "")|\(row.settings ?? "")|\(text)".utf8))
                .map { String(format: "%02x", $0) }.joined()
            let id = String(digest.prefix(10))
            let preview = text.count <= 60 ? text : String(text.prefix(60)) + "…"
            let output = Asset(items: [Item(kind: .status,
                                            value: "queued to \(destination) (\(id))",
                                            path: nil, sourceText: nil)])
            lock.withLock { _lastStaged = (id, kind, "to \(destination): \(preview)") }
            return output
        }
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

    // MARK: - Deciders (`_pick_tag` + `_decider_payload`)

    /// `_declared_tags` — `row.tags` if declared, else the decide clause's edge tags.
    private func declaredTags(_ row: Row) -> [String] {
        if let tags = row.tags, !tags.isEmpty { return tags }
        if case .decide(let edges)? = row.clause { return edges.map(\.tag) }
        return []
    }

    /// `_pick_tag` — the Python's `_pick_tag` (engines/mock.py): a decider's identity is the
    /// bare row path (activation suffix stripped); the scripted sequence for that base wins,
    /// else the first declared tag.
    private func pickTag(path: String, row: Row) -> String? {
        let base = path.split(separator: "@", maxSplits: 1).first.map(String.init) ?? path
        let n: Int = lock.withLock {
            let k = _deciderVisits[base] ?? 0
            _deciderVisits[base] = k + 1
            return k
        }
        if let scripted = deciderScript[base], !scripted.isEmpty {
            return scripted[min(n, scripted.count - 1)]
        }
        return declaredTags(row).first
    }

    /// `_decider_payload` — Spec §7.4 R1–R3.
    private func deciderPayload(path: String, row: Row, inputs: [Asset], tag: String?) -> Asset {
        if row.task == "Judge" { return judgePayload(row: row, inputs: inputs, tag: tag) }
        if row.task == "Think" { return thinkPayload(path: path, row: row, inputs: inputs, tag: tag) }
        return inputs.first ?? Asset(items: [])   // R1: unchanged passthrough
    }

    /// R2 (Judge) — flatten every ref's items, take the trailing `len(tags)` as candidates,
    /// return the one the fired tag names positionally.
    private func judgePayload(row: Row, inputs: [Asset], tag: String?) -> Asset {
        let tags = declaredTags(row)
        let items = inputs.flatMap { $0.items }
        let candidates = tags.isEmpty ? items : Array(items.suffix(tags.count))
        if let tag, tags.contains(tag), let idx = tags.firstIndex(of: tag), idx < candidates.count {
            return Asset(items: [candidates[idx]])
        }
        return inputs.first ?? Asset(items: [])
    }

    /// R3 (Think) — the model's own `input:` line (a tool tag) or final answer, synthesized
    /// deterministically and keyed additionally by the fired tag.
    private func thinkPayload(path: String, row: Row, inputs: [Asset], tag: String?) -> Asset {
        guard let first = inputs.first?.items.first else { return Asset(items: []) }
        let kind = first.kind
        let basis = "\(fingerprint(path: path, row: row, inputs: inputs))|tag=\(tag ?? "")"
        return Asset(items: [mockItem(kind: kind, basis: basis, path: path, index: 0)])
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
