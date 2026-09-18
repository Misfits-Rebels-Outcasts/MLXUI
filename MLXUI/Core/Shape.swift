import Foundation

/// The reference kind of a row's implementation — how its refs are type-checked. Ported
/// from `catflow-mlx/src/catflow/catalog/tasks.py` (`RefKind`, 3 cases). `singleCompatible`
/// / `bundleCompatible` check `rk == .frame` **before** the `AnyKind()` branch, so a frame
/// row (Summarize, Rewrite, …) imposes its stricter "every ref must be text" rule on top of
/// whatever its `accepts` shape says. `TaskCatalog` (CFM-R2-1) carries this per task.
nonisolated enum RefKind: String, Sendable {
    case tool, engine, frame
}

/// The shape of an asset — ported verbatim from `catflow-mlx/src/catflow/core/kinds.py`
/// (`Single`/`ListOf`/`TupleOf`/`UnionOf`/`AnyKind`/`SameAsInput`). The type system
/// `catflow check` checks; FlowKit's validator, runner, and step picker all share it, so
/// "check and run can't drift".
nonisolated enum Shape: Equatable, Sendable {
    /// A single asset of one kind, e.g. `text`.
    case single(Kind)
    /// A homogeneous list, e.g. `[text]`.
    case listOf(Kind)
    /// A fixed-arity heterogeneous bundle, e.g. `[text, vector]`.
    case tupleOf([Kind])
    /// Accepts any one of several kinds, e.g. `audio|video` (Trim).
    case unionOf([Kind])
    /// `anything` — the Save family: any single-asset input.
    case anyKind
    /// `same` — output kind mirrors whichever input kind was actually given (Trim).
    case sameAsInput

    /// Mirrors the Python shapes' `__str__`: `text`, `[image]`, `[text, vector]`,
    /// `audio|video`, `anything`, `same`.
    var signatureText: String {
        switch self {
        case .single(let k):        return k.rawValue
        case .listOf(let k):        return "[\(k.rawValue)]"
        case .tupleOf(let ks):      return "[" + ks.map(\.rawValue).joined(separator: ", ") + "]"
        case .unionOf(let ks):      return ks.map(\.rawValue).joined(separator: "|")
        case .anyKind:              return "anything"
        case .sameAsInput:          return "same"
        }
    }
}

/// A row's block kind — `list` | `each` | `parallel`. Ported from `core/model.py`
/// (`Row.block_kind`). Used by `signature` to compute a block's accepts/gives from its
/// first/last child and by FlowDocument (`FlowKit/FlowDocument.swift`, CFM-R1-3).
nonisolated enum BlockKind: String, Sendable {
    case list, each, parallel
}

/// The minimal view of a row `signature` needs — the Core/FlowKit seam. `FlowDocument.Row`
/// conforms to it in CFM-R1-3; the shared compatibility logic stays in Core so the
/// validator, runner, and step picker can never drift (the `shapes.py` isolation rule).
protocol RowShape: Sendable {
    nonisolated var task: String? { get }
    nonisolated var blockKind: BlockKind? { get }
    nonisolated var children: [any RowShape] { get }
}

nonisolated extension Shape {
    /// `signature(row)` from `catflow-mlx/src/catflow/core/shapes.py` — the (accepts, gives)
    /// pair for a task row (catalog lookup) or a block row (first-child/last-child
    /// inference, `<each>` lifting `Single → ListOf`). `nil` if the task/block signature
    /// can't be determined. `lookup` is the task-catalog (CFM-R2-1) accepts/gives source.
    static func signature(
        _ row: any RowShape,
        lookup: (String) -> (Shape, Shape)?
    ) -> (Shape, Shape)? {
        if let blockKind = row.blockKind {
            guard let first = row.children.first, let last = row.children.last else { return nil }
            guard let firstSig = signature(first, lookup: lookup),
                  let lastSig = signature(last, lookup: lookup) else { return nil }
            var accepts = firstSig.0
            var gives = lastSig.1
            if blockKind == .each {
                if case .single(let k) = accepts { accepts = .listOf(k) }
                if case .single(let k) = gives { gives = .listOf(k) }
            }
            return (accepts, gives)
        }
        guard let task = row.task else { return nil }
        return lookup(task)
    }

    /// `base_kind(shape)` — the kind of a `Single` or `ListOf`, `nil` otherwise.
    static func baseKind(_ shape: Shape) -> Kind? {
        switch shape {
        case .single(let k), .listOf(let k): return k
        default: return nil
        }
    }

    /// `single_compatible(accepts, given, rk)` from `core/shapes.py` — whether one given
    /// shape satisfies an accepts shape, under the ref-kind rules (SPEC-Q71a/b lift a
    /// `Single`/homogeneous `TupleOf` into a one-item `ListOf`).
    static func singleCompatible(accepts: Shape, given: Shape, rk: RefKind?) -> Bool {
        if rk == .frame { return baseKind(given) == .text }
        switch accepts {
        case .anyKind: return true
        case .unionOf(let kinds):
            if case .single(let k) = given { return kinds.contains(k) }
            return false
        case .single(let acceptsKind):
            if case .single(let givenKind) = given { return givenKind == acceptsKind }
            return false
        case .listOf(let acceptsKind):
            // SPEC-Q71(a): a single `Single(K)` given lifts to a one-item `[K]`.
            if case .single(let givenKind) = given, givenKind == acceptsKind { return true }
            if case .listOf(let givenKind) = given, givenKind == acceptsKind { return true }
            // SPEC-Q71(b): a homogeneous `TupleOf` given fills a `ListOf` accepts.
            if case .tupleOf(let givenKinds) = given,
               givenKinds.allSatisfy({ $0 == acceptsKind }) { return true }
            return false
        case .tupleOf, .sameAsInput:
            // TupleOf accepts needs an explicit multi-ref bundle; `same` is a gives, not an accepts.
            return false
        }
    }

    /// `bundle_compatible(accepts, given, rk)` from `core/shapes.py` — whether a bundle of
    /// given shapes satisfies an accepts shape (SPEC-Q117: a ref that's itself `ListOf(K)`
    /// fills one position, same as a `Single`).
    static func bundleCompatible(accepts: Shape, given: [Shape], rk: RefKind?) -> Bool {
        if rk == .frame { return given.allSatisfy { baseKind($0) == .text } }
        if accepts == .anyKind { return true }
        if given.count == 1 { return singleCompatible(accepts: accepts, given: given[0], rk: rk) }
        switch accepts {
        case .tupleOf(let kinds):
            guard given.count == kinds.count else { return false }
            return zip(given, kinds).allSatisfy { baseKind($0) == $1 }
        case .listOf(let acceptsKind):
            return given.allSatisfy { baseKind($0) == acceptsKind }
        case .single, .unionOf, .anyKind, .sameAsInput:
            return false
        }
    }
}
