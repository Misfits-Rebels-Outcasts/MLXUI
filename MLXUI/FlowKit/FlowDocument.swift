import Foundation

/// CAT Flow parse-tree document — ported from `catflow-mlx/src/catflow/core/model.py`
/// (`Flow`/`Row`/`Ref`), shaped per the design doc §A.4 item 2. The parse tree comes from
/// the Swift `CatParser` (CFM-R5-2, byte-identical to the Python corpus) — the pre-parsed
/// `*.parse.json` resources are retired. `FlowDocument`'s Codable boundary still
/// round-trips the Python parse-tree shape (numbers on the wire, ids in the model) for
/// tests and storage.
///
/// **Identity contract.** `Row.id` is a stable `UUID` minted at decode time. Row numbers
/// are rendering only, computed at display time; references store **row ids**, never
/// numbers. During decode, the Python's numeric `refs` (`{"kind":"row","number":2}`) are
/// resolved against a number→id map built per scope (top-level rows, or a block's own
/// children) — so no number-shaped reference survives into the model. This is the whole
/// reason delete-and-reorder can be made safe later.
///
/// **Codable boundary.** `FlowDocument` is the Codable boundary and round-trips the
/// Python parse-tree shape (numbers on the wire, ids in the model). `Row` itself is not
/// `Codable` — resolving a ref needs its containing scope, which a standalone row can't
/// provide; `CompositeDef`'s nested rows use an id-based `RowCodec` instead (R5 parser
/// populates composites; nothing in R1–R4 exercises them).
nonisolated struct FlowDocument: Codable, Sendable, Equatable {
    var version: String
    var fileKind: FileKind
    var rows: [Row]

    // The remaining fields are the design-sketch shape; the pre-parsed JSON only ever
    // carries `version` + `rows` (+ optional `flags`/`definitions`/`models`), so these
    // default and are populated by the R5 parser.
    var flags: Set<CapabilityFlag>
    /// Source order of the header flags — the R6 `CatSerializer` renders `; code; events` in
    /// the order the file wrote them (the Python's flags are an ordered tuple; the Set isn't).
    var flagsOrder: [CapabilityFlag]
    var uses: [String: String]
    var models: [String: String]
    var transforms: [String: TransformDef]
    var definitions: [String: CompositeDef]
    var accepts: [Kind]?
    var gives: String?
    var params: [ParamDecl]
    var presets: [String: PresetDecl]
    /// `.catpipeline` declaration name (`pipeline <Name>`), kept for the R6 serializer.
    var pipelineName: String?
    /// Source order of the section keys — the R6 `CatSerializer` needs it to reproduce a
    /// file byte-for-byte (the Python's dicts are insertion-ordered; Swift's are not).
    var modelsOrder: [String]
    var usesOrder: [String]
    var definitionsOrder: [String]
    var transformsOrder: [String]
    var presetsOrder: [String]

    init(
        version: String,
        fileKind: FileKind = .catflow,
        rows: [Row] = [],
        flags: Set<CapabilityFlag> = [],
        flagsOrder: [CapabilityFlag] = [],
        uses: [String: String] = [:],
        models: [String: String] = [:],
        transforms: [String: TransformDef] = [:],
        definitions: [String: CompositeDef] = [:],
        accepts: [Kind]? = nil,
        gives: String? = nil,
        params: [ParamDecl] = [],
        presets: [String: PresetDecl] = [:],
        pipelineName: String? = nil,
        modelsOrder: [String] = [],
        usesOrder: [String] = [],
        definitionsOrder: [String] = [],
        transformsOrder: [String] = [],
        presetsOrder: [String] = []
    ) {
        self.version = version
        self.fileKind = fileKind
        self.rows = rows
        self.flags = flags
        self.flagsOrder = flagsOrder
        self.uses = uses
        self.models = models
        self.transforms = transforms
        self.definitions = definitions
        self.accepts = accepts
        self.gives = gives
        self.params = params
        self.presets = presets
        self.pipelineName = pipelineName
        self.modelsOrder = modelsOrder
        self.usesOrder = usesOrder
        self.definitionsOrder = definitionsOrder
        self.transformsOrder = transformsOrder
        self.presetsOrder = presetsOrder
    }

    // MARK: Codable — parse-tree shape, numbers on the wire

    init(from decoder: Decoder) throws {
        let raw = try RawFlow(from: decoder)
        self.version = raw.version ?? ""
        self.fileKind = .catflow
        self.flags = Set((raw.flags ?? []).compactMap { CapabilityFlag(rawValue: $0) })
        self.flagsOrder = (raw.flagsOrder ?? []).compactMap { CapabilityFlag(rawValue: $0) }
        self.models = raw.models ?? [:]
        self.definitions = [:]
        self.transforms = [:]
        self.uses = [:]
        self.accepts = nil
        self.gives = nil
        self.params = []
        self.presets = [:]
        self.pipelineName = raw.pipelineName
        self.modelsOrder = raw.modelsOrder ?? []
        self.usesOrder = raw.usesOrder ?? []
        self.definitionsOrder = raw.definitionsOrder ?? []
        self.transformsOrder = raw.transformsOrder ?? []
        self.presetsOrder = raw.presetsOrder ?? []
        self.rows = try FlowDocument.resolve(raw.rows, scopeLabel: "the flow")
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: RawFlow.CodingKeys.self)
        try container.encode(version, forKey: .version)
        let rawRows = try FlowDocument.serialize(rows, scope: Dictionary(rows.enumerated().map { ($1.id, $0 + 1) }, uniquingKeysWith: { a, _ in a }))
        try container.encode(rawRows, forKey: .rows)
        if !flags.isEmpty {
            try container.encode(flags.map(\.rawValue), forKey: .flags)
        }
        if !models.isEmpty {
            try container.encode(models, forKey: .models)
        }
        // Swift-only round-trip keys: the parser-order fields the Python wire shape doesn't
        // carry. Encoded additively (Python ignores them), decoded when present, so a
        // Swift-encoded document re-serializes with its sections in source order and its
        // `pipeline` line (CFM-R8-FIX-6: the decoder no longer zeroes them).
        if !flagsOrder.isEmpty {
            try container.encode(flagsOrder.map(\.rawValue), forKey: .flagsOrder)
        }
        if let pipelineName {
            try container.encode(pipelineName, forKey: .pipelineName)
        }
        if !modelsOrder.isEmpty {
            try container.encode(modelsOrder, forKey: .modelsOrder)
        }
        if !usesOrder.isEmpty {
            try container.encode(usesOrder, forKey: .usesOrder)
        }
        if !definitionsOrder.isEmpty {
            try container.encode(definitionsOrder, forKey: .definitionsOrder)
        }
        if !transformsOrder.isEmpty {
            try container.encode(transformsOrder, forKey: .transformsOrder)
        }
        if !presetsOrder.isEmpty {
            try container.encode(presetsOrder, forKey: .presetsOrder)
        }
    }

    // MARK: - Parse-tree serialization (the `flow_to_dict` wire shape)

    /// Emit this document as the Python `flow_to_dict` shape — `version`, `rows`, and
    /// additively `flags`/`definitions`/`models` (only when non-empty). Used by the
    /// R5-2 corpus test to compare the Swift parser's tree against the conformance
    /// goldens. Key order follows `tests/conftest.py::flow_to_dict`.
    func toParseTreeJSON() throws -> Data {
        var d: [String: Any] = ["version": version.isEmpty ? NSNull() : version]
        d["rows"] = try FlowDocument.parseTreeRows(rows, scope: Dictionary(rows.enumerated().map { ($1.id, $0 + 1) }, uniquingKeysWith: { a, _ in a }))
        if !flags.isEmpty {
            // Python's `flow_to_dict` emits `flags` as an ordered list; the Swift `Set`
            // loses source order, so use the parser-recorded `flagsOrder` (a stable sort for
            // JSON-decoded documents) — multi-flag flows (53-MorningBriefing, 58-NightlyLedger)
            // otherwise mismatch their goldens (CFM-R6-FIX-5).
            let ordered = flagsOrder.isEmpty ? flags.map(\.rawValue).sorted() : flagsOrder.map(\.rawValue)
            d["flags"] = ordered
        }
        if !definitions.isEmpty {
            d["definitions"] = try Dictionary(definitions.map { ($0.key, try FlowDocument.parseTreeComposite($0.value)) }, uniquingKeysWith: { a, _ in a })
        }
        if !models.isEmpty {
            d["models"] = models
        }
        return try JSONSerialization.data(withJSONObject: d, options: [])
    }

    private static func parseTreeComposite(_ def: CompositeDef) throws -> [String: Any] {
        let childScope = Dictionary(def.rows.enumerated().map { ($1.id, $0 + 1) }, uniquingKeysWith: { a, _ in a })
        return [
            "name": def.name,
            "signature": def.signature ?? NSNull(),
            "params": def.params.map { ["name": $0.name, "kind": $0.kind ?? NSNull(), "default": $0.defaultValue ?? NSNull()] } as [[String: Any]],
            "rows": try parseTreeRows(def.rows, scope: childScope),
        ]
    }

    private static func parseTreeRows(_ rows: [Row], scope: [UUID: Int]) throws -> [[String: Any]] {
        try rows.map { row in
            let childScope = Dictionary(row.children.enumerated().map { ($1.id, $0 + 1) }, uniquingKeysWith: { a, _ in a })
            var d: [String: Any] = [
                "task": row.task ?? NSNull(),
                "model": row.model ?? NSNull(),
                "settings": row.settings ?? NSNull(),
                "refs": try row.refs.map { try parseTreeRef($0, scope: scope) },
                "chain_break": row.chainBreak,
                "block_kind": row.blockKind?.rawValue ?? NSNull(),
                "block_name": row.blockName ?? NSNull(),
                "children": try parseTreeRows(row.children, scope: childScope),
            ]
            if let clause = row.clause {
                d["clause"] = try parseTreeClause(clause)
            }
            if let tags = row.tags, !tags.isEmpty {
                d["tags"] = tags
            }
            if let visitsLeq = row.visitsLeq {
                d["visits_leq"] = visitsLeq
            }
            if let onBudget = row.onBudget {
                d["on_budget"] = onBudget
            }
            if let comment = row.comment {
                d["comment"] = comment
            }
            if let declaredSignature = row.declaredSignature {
                d["declared_signature"] = declaredSignature
            }
            if !row.leadingComments.isEmpty {
                d["leading_comments"] = row.leadingComments
            }
            return d
        }
    }

    private static func parseTreeRef(_ ref: Ref, scope: [UUID: Int]) throws -> [String: Any] {
        switch ref {
        case .rowRef(let id):
            guard let number = scope[id] else {
                throw FlowDocumentError.unresolvedReference(id: id)
            }
            return ["kind": "row", "number": number]
        case .inputRef(let position):
            return ["kind": "input", "position": position]
        case .paramRef(let name):
            return ["kind": "param", "name": name]
        }
    }

    private static func parseTreeClause(_ clause: Clause) throws -> [String: Any] {
        func target(_ t: ClauseTarget) -> [String: Any] {
            switch t {
            case .row(let number): return ["kind": "row", "number": number]
            case .call(let number): return ["kind": "call", "number": number]
            case .resume: return ["kind": "resume"]
            case .done: return ["kind": "done"]
            }
        }
        switch clause {
        case .goto(let t):
            return ["kind": "goto", "target": target(t)]
        case .fork(let targets):
            return ["kind": "fork", "targets": targets.map(target)]
        case .decide(let edges):
            return ["kind": "decide", "edges": edges.map { ["tag": $0.tag, "target": target($0.target)] }]
        case .call(let number):
            return ["kind": "call", "target": ["kind": "call", "number": number]]
        case .resume:
            return ["kind": "resume"]
        }
    }

    // MARK: Scope-aware conversion (Python numbers ↔ Swift ids)

    /// Mint ids for `rawRows`, resolving numeric refs against this scope's number→id map.
    private static func resolve(_ rawRows: [RawRow], scopeLabel: String) throws -> [Row] {
        let ids = rawRows.enumerated().map { (number: $0.offset + 1, id: UUID()) }
        let byNumber = Dictionary(ids.map { ($0.number, $0.id) }, uniquingKeysWith: { a, _ in a })
        return try rawRows.enumerated().map { index, raw in
            let id = ids[index].id
            return try raw.toRow(id: id, byNumber: byNumber, scopeLabel: scopeLabel)
        }
    }

    /// Emit `rows` as raw parse-tree rows; `scope` maps each row id to its 1-based number.
    private static func serialize(_ rows: [Row], scope: [UUID: Int]) throws -> [RawRow] {
        try rows.map { row in
            let childScope = Dictionary(row.children.enumerated().map { ($1.id, $0 + 1) }, uniquingKeysWith: { a, _ in a })
            return try RawRow(
                task: row.task,
                model: row.model,
                settings: row.settings,
                refs: row.refs.map { try $0.raw(scope: scope) },
                chainBreak: row.chainBreak,
                blockKind: row.blockKind?.rawValue,
                blockName: row.blockName,
                children: try serialize(row.children, scope: childScope),
                declaredSignature: row.declaredSignature,
                tags: row.tags,
                visitsLeq: row.visitsLeq,
                onBudget: row.onBudget,
                comment: row.comment,
                leadingComments: row.leadingComments.isEmpty ? nil : row.leadingComments
            )
        }
    }
}

// MARK: - Row

/// One row of a flow: a task (or a block), optional model, settings text, references, a
/// chain-break flag, and (for blocks) a body of children. Ported from `core/model.py::Row`.
nonisolated struct Row: Identifiable, Sendable, Equatable {
    let id: UUID
    var task: String?
    var blockKind: BlockKind?
    var blockName: String?
    var model: String?
    var settings: String?
    var refs: [Ref]
    var chainBreak: Bool
    var children: [Row]
    var clause: Clause?
    var tags: [String]?
    var visitsLeq: Int?
    var onBudget: String?
    var declaredSignature: String?
    var comment: String?
    var leadingComments: [String]

    init(
        id: UUID = UUID(),
        task: String? = nil,
        blockKind: BlockKind? = nil,
        blockName: String? = nil,
        model: String? = nil,
        settings: String? = nil,
        refs: [Ref] = [],
        chainBreak: Bool = false,
        children: [Row] = [],
        clause: Clause? = nil,
        tags: [String]? = nil,
        visitsLeq: Int? = nil,
        onBudget: String? = nil,
        declaredSignature: String? = nil,
        comment: String? = nil,
        leadingComments: [String] = []
    ) {
        self.id = id
        self.task = task
        self.blockKind = blockKind
        self.blockName = blockName
        self.model = model
        self.settings = settings
        self.refs = refs
        self.chainBreak = chainBreak
        self.children = children
        self.clause = clause
        self.tags = tags
        self.visitsLeq = visitsLeq
        self.onBudget = onBudget
        self.declaredSignature = declaredSignature
        self.comment = comment
        self.leadingComments = leadingComments
    }
}

// MARK: - Ref

/// A reference from one row to another (`rowRef`), to a block input (`inputRef`), or to a
/// bound parameter (`paramRef`). Ported from `core/model.py::Ref`. The Python's `RowRef`
/// (a bare row number) is resolved to a `rowRef(UUID)` **during decode** — no number-shaped
/// reference survives into the model.
nonisolated enum Ref: Hashable, Sendable {
    case rowRef(UUID)
    case inputRef(Int)
    case paramRef(String)

    /// The `(2)` / `(1,2)` display the `.cat` file uses, computed from ids at display time.
    /// `scope` maps row id → 1-based number within the referencing row's scope.
    fileprivate func raw(scope: [UUID: Int]) throws -> RawRef {
        switch self {
        case .rowRef(let id):
            guard let number = scope[id] else {
                throw FlowDocumentError.unresolvedReference(id: id)
            }
            return RawRef(kind: "row", number: number, position: nil, name: nil, activation: nil)
        case .inputRef(let position):
            return RawRef(kind: "input", number: nil, position: position, name: nil, activation: nil)
        case .paramRef(let name):
            return RawRef(kind: "param", number: nil, position: nil, name: name, activation: nil)
        }
    }
}

// MARK: - Supporting types (design-sketch shape)

/// `.cat` vs `.catpipeline`. The release train only ships `.cat`; pipelines are a
/// `.catpipeline` superset (model/latent rows) that the R5 parser will distinguish.
nonisolated enum FileKind: String, Codable, Sendable, Equatable {
    case catflow
    case catpipeline
}

/// Header capability flags: `network`, `events`, `improvise`, `code`, `offdevice`.
/// The `code`/`improvise`/`offdevice` doors refuse to run under `APPSTORE_BUILD`
/// (CFM-R5-6); the rest are ordinary declarations.
nonisolated enum CapabilityFlag: String, Codable, Sendable, Equatable, CaseIterable {
    case network, events, improvise, code, offdevice
}

/// A continuation clause (`-> N`, `-> {tag: N}`, `call N`, `resume`). Phase-2 grammar,
/// absent from every R1–R4 gallery flow; defined here to keep `Row` the agreed shape.
nonisolated enum Clause: Sendable, Equatable {
    case goto(target: ClauseTarget)
    case fork(targets: [ClauseTarget])
    case decide(edges: [ClauseEdge])
    case call(number: Int)
    case resume

    /// The decide edges (nil for a non-decide clause) — the R9 inspector's `[tag | target]`
    /// pairs.
    var edges: [ClauseEdge]? {
        if case .decide(let edges) = self { return edges }
        return nil
    }
}

/// A clause edge target: a row, a `call`, `resume`, or `done`.
nonisolated enum ClauseTarget: Sendable, Equatable {
    case row(number: Int)
    case call(number: Int)
    case resume
    case done
}

/// One `tag: TARGET` entry of a decide clause.
nonisolated struct ClauseEdge: Sendable, Equatable {
    var tag: String
    var target: ClauseTarget
}

/// A `definitions:` composite: a named block registered as a task (Spec §1.4).
nonisolated struct CompositeDef: Sendable, Equatable {
    var name: String
    var signature: String?
    var params: [ParamDecl]
    var rows: [Row]

    // Nested composite rows carry ids — a scope-free, id-based codec (see the
    // `FlowDocument` header note). Composites arrive with the R5 parser.
    init(name: String, signature: String? = nil, params: [ParamDecl] = [], rows: [Row] = []) {
        self.name = name
        self.signature = signature
        self.params = params
        self.rows = rows
    }
}

/// A `transforms:` entry (Spec §1.6): a named external script, declared not inferred.
nonisolated struct TransformDef: Sendable, Equatable {
    var name: String
    var signature: String?
    var run: String?
    var timeout: String?
    var workdir: String?
    var params: [ParamDecl]
}

/// One entry of a composite's `params:` line: a name, an optional `(kind)` hint, and an
/// optional `= default` raw text value.
nonisolated struct ParamDecl: Sendable, Equatable {
    var name: String
    var kind: String?
    var defaultValue: String?
}

/// A `.catpipeline` `presets:` entry — name → ordered (key, value) raw-text pairs.
nonisolated struct PresetDecl: Sendable, Equatable {
    var name: String
    var bindings: [PresetBinding]
}

/// One ordered `key=value` raw-text binding of a preset.
nonisolated struct PresetBinding: Sendable, Equatable {
    var key: String
    var value: String
}

// MARK: - Errors

/// Decoding/encoding failures for `FlowDocument`. The error voice names the row, states
/// the problem in one plain sentence, and implies the fix (standing rule 6).
nonisolated enum FlowDocumentError: Error, CustomStringConvertible, Equatable {
    case unresolvedReference(id: UUID)
    case unknownReferenceKind(String)
    case missingReferenceTarget(Int, scope: String)

    var description: String {
        switch self {
        case .unresolvedReference(let id):
            return "A row references \(id.uuidString), which isn't in this scope — re-check the flow's references."
        case .unknownReferenceKind(let kind):
            return "A reference of kind '\(kind)' isn't one catflow knows — update the flow file."
        case .missingReferenceTarget(let number, let scope):
            return "Row \(number) is referenced but doesn't exist in \(scope) — the reference points past the last row."
        }
    }
}

// MARK: - Raw parse-tree DTOs (the wire shape)

/// The Python's `flow_to_dict`/`conformance` parse-tree shape. `kind`-discriminated refs
/// carry a `number` (row ref), `position` (input ref), or `name` (param ref).
private struct RawFlow: Decodable {
    var version: String?
    var rows: [RawRow]
    var flags: [String]?
    var definitions: [String: RawComposite]?
    var models: [String: String]?
    var pipelineName: String?
    var flagsOrder: [String]?
    var modelsOrder: [String]?
    var usesOrder: [String]?
    var definitionsOrder: [String]?
    var transformsOrder: [String]?
    var presetsOrder: [String]?

    enum CodingKeys: String, CodingKey {
        case version, rows, flags, definitions, models
        case pipelineName = "pipeline_name"
        case flagsOrder = "flags_order"
        case modelsOrder = "models_order"
        case usesOrder = "uses_order"
        case definitionsOrder = "definitions_order"
        case transformsOrder = "transforms_order"
        case presetsOrder = "presets_order"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(String.self, forKey: .version)
        rows = try c.decodeIfPresent([RawRow].self, forKey: .rows) ?? []
        flags = try c.decodeIfPresent([String].self, forKey: .flags)
        definitions = try c.decodeIfPresent([String: RawComposite].self, forKey: .definitions)
        models = try c.decodeIfPresent([String: String].self, forKey: .models)
        pipelineName = try c.decodeIfPresent(String.self, forKey: .pipelineName)
        flagsOrder = try c.decodeIfPresent([String].self, forKey: .flagsOrder)
        modelsOrder = try c.decodeIfPresent([String].self, forKey: .modelsOrder)
        usesOrder = try c.decodeIfPresent([String].self, forKey: .usesOrder)
        definitionsOrder = try c.decodeIfPresent([String].self, forKey: .definitionsOrder)
        transformsOrder = try c.decodeIfPresent([String].self, forKey: .transformsOrder)
        presetsOrder = try c.decodeIfPresent([String].self, forKey: .presetsOrder)
    }
}

private struct RawComposite: Decodable {
    var name: String
    var signature: String?
    var params: [RawParamDecl]
    var rows: [RawRow]

    enum CodingKeys: String, CodingKey {
        case name, signature, params, rows
    }
}

private struct RawParamDecl: Decodable {
    var name: String
    var kind: String?
    var defaultValue: String?

    enum CodingKeys: String, CodingKey {
        case name, kind
        case defaultValue = "default"
    }
}

private struct RawRow: Codable {
    var task: String?
    var model: String?
    var settings: String?
    var refs: [RawRef]
    var chainBreak: Bool
    var blockKind: String?
    var blockName: String?
    var children: [RawRow]
    var declaredSignature: String?
    var tags: [String]?
    var visitsLeq: Int?
    var onBudget: String?
    var comment: String?
    var leadingComments: [String]?

    enum CodingKeys: String, CodingKey {
        case task, model, settings, refs, children, tags
        case chainBreak = "chain_break"
        case blockKind = "block_kind"
        case blockName = "block_name"
        case declaredSignature = "declared_signature"
        case visitsLeq = "visits_leq"
        case onBudget = "on_budget"
        case comment
        case leadingComments = "leading_comments"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        task = try c.decodeIfPresent(String.self, forKey: .task)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        settings = try c.decodeIfPresent(String.self, forKey: .settings)
        refs = try c.decodeIfPresent([RawRef].self, forKey: .refs) ?? []
        chainBreak = try c.decodeIfPresent(Bool.self, forKey: .chainBreak) ?? false
        blockKind = try c.decodeIfPresent(String.self, forKey: .blockKind)
        blockName = try c.decodeIfPresent(String.self, forKey: .blockName)
        children = try c.decodeIfPresent([RawRow].self, forKey: .children) ?? []
        declaredSignature = try c.decodeIfPresent(String.self, forKey: .declaredSignature)
        tags = try c.decodeIfPresent([String].self, forKey: .tags)
        visitsLeq = try c.decodeIfPresent(Int.self, forKey: .visitsLeq)
        onBudget = try c.decodeIfPresent(String.self, forKey: .onBudget)
        comment = try c.decodeIfPresent(String.self, forKey: .comment)
        leadingComments = try c.decodeIfPresent([String].self, forKey: .leadingComments)
    }

    init(
        task: String?, model: String?, settings: String?, refs: [RawRef],
        chainBreak: Bool, blockKind: String?, blockName: String?,
        children: [RawRow], declaredSignature: String?, tags: [String]?,
        visitsLeq: Int?, onBudget: String?, comment: String?, leadingComments: [String]?
    ) {
        self.task = task
        self.model = model
        self.settings = settings
        self.refs = refs
        self.chainBreak = chainBreak
        self.blockKind = blockKind
        self.blockName = blockName
        self.children = children
        self.declaredSignature = declaredSignature
        self.tags = tags
        self.visitsLeq = visitsLeq
        self.onBudget = onBudget
        self.comment = comment
        self.leadingComments = leadingComments
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(task, forKey: .task)
        try c.encode(model, forKey: .model)
        try c.encode(settings, forKey: .settings)
        try c.encode(refs, forKey: .refs)
        try c.encode(chainBreak, forKey: .chainBreak)
        try c.encode(blockKind, forKey: .blockKind)
        try c.encode(blockName, forKey: .blockName)
        try c.encode(children, forKey: .children)
        if let declaredSignature {
            try c.encode(declaredSignature, forKey: .declaredSignature)
        }
        if let tags {
            try c.encode(tags, forKey: .tags)
        }
        if let visitsLeq {
            try c.encode(visitsLeq, forKey: .visitsLeq)
        }
        if let onBudget {
            try c.encode(onBudget, forKey: .onBudget)
        }
        if let comment {
            try c.encode(comment, forKey: .comment)
        }
        if let leadingComments {
            try c.encode(leadingComments, forKey: .leadingComments)
        }
    }

    /// Convert this raw row into a model `Row`, resolving numeric refs against `byNumber`
    /// (this scope's number→id map) and minting ids for children with their own scope.
    func toRow(id: UUID, byNumber: [Int: UUID], scopeLabel: String) throws -> Row {
        let resolvedRefs = try refs.map { ref -> Ref in
            switch ref.kind {
            case "row":
                guard let number = ref.number, let targetID = byNumber[number] else {
                    throw FlowDocumentError.missingReferenceTarget(ref.number ?? 0, scope: scopeLabel)
                }
                return .rowRef(targetID)
            case "input":
                return .inputRef(ref.position ?? 1)
            case "param":
                return .paramRef(ref.name ?? "")
            default:
                throw FlowDocumentError.unknownReferenceKind(ref.kind)
            }
        }
        let childIDs = children.enumerated().map { (number: $0.offset + 1, id: UUID()) }
        let childByNumber = Dictionary(childIDs.map { ($0.number, $0.id) }, uniquingKeysWith: { a, _ in a })
        let childRows = try children.enumerated().map { index, raw in
            try raw.toRow(id: childIDs[index].id, byNumber: childByNumber, scopeLabel: "the block '\(blockName ?? "")'")
        }
        return Row(
            id: id,
            task: task,
            blockKind: blockKind.flatMap(BlockKind.init(rawValue:)),
            blockName: blockName,
            model: model,
            settings: settings,
            refs: resolvedRefs,
            chainBreak: chainBreak,
            children: childRows,
            tags: tags,
            visitsLeq: visitsLeq,
            onBudget: onBudget,
            declaredSignature: declaredSignature,
            comment: comment,
            leadingComments: leadingComments ?? []
        )
    }
}

private struct RawRef: Codable {
    var kind: String
    var number: Int?
    var position: Int?
    var name: String?
    var activation: Int?

    init(kind: String, number: Int?, position: Int?, name: String?, activation: Int?) {
        self.kind = kind
        self.number = number
        self.position = position
        self.name = name
        self.activation = activation
    }
}
