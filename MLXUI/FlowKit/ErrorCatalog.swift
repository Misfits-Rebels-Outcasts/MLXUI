import Foundation

/// One entry of the CAT Flow error catalog — ported from
/// `catflow-mlx/src/catflow/core/errors_catalog.py` (CFM-R5-4). Every ⚠/⚑ wording
/// lives here, byte-identical to the docs. `catalogV07` is the v0.1/v0.4/v0.7
/// wording; `catalogV08` is v0.8's (marker spelling the message quotes verbatim,
/// plus E108, which only exists at v0.8).
nonisolated struct CatErrorSpec: Sendable, Equatable {
    var code: String
    var name: String
    var citation: String?
    var template: String
}

nonisolated enum ErrorCatalog {
    // GENERATED — do not hand-edit. Regenerate via the repo's generator;
    // source of truth: catflow-mlx/src/catflow/core/errors_catalog.py.

    /// 61 codes, generated from `core/errors_catalog.py` (`catalogV07`).
    static let catalogV07: [String: CatErrorSpec] = [
        "E101": CatErrorSpec(code: "E101", name: "unknown version", citation: "Spec §1.1", template: "This flow needs CAT Flow {version} — it uses {feature-example}. This runtime speaks {supported}. Nothing was run."),
        "E102": CatErrorSpec(code: "E102", name: "unknown header flag", citation: "§1.1", template: "This flow declares `· {flag}`, which this runtime doesn't support. Nothing was run."),
        "E103": CatErrorSpec(code: "E103", name: "missing `network` flag", citation: "§1.2", template: "Row {n} ({task}) talks to the internet, but the header doesn't say `· network`. Add it — shared flows must show their network use on line one. (fmt will do this for you.)"),
        "E104": CatErrorSpec(code: "E104", name: "model not pinned", citation: "§1.5", template: "Row {n} names \"{display}\", but the `models:` section doesn't pin an id for it. Pick the model again to pin it, or add the line yourself."),
        "E105": CatErrorSpec(code: "E105", name: "unparseable row", citation: nil, template: "Row {n} couldn't be read from \"{fragment}…\". A row is: `N. Task  model · settings  (refs)`. The first thing that confused the parser: {detail}."),
        "E106": CatErrorSpec(code: "E106", name: "unclosed quote", citation: nil, template: "Row {n}'s quoted text opens with `\"` but never closes. If the text should contain a quote, write `\\\"`."),
        "E107": CatErrorSpec(code: "E107", name: "duplicate position", citation: nil, template: "Two rows are numbered {n} in the same scope. Positions must be sequential — running fmt will renumber and re-aim references."),
        "E201": CatErrorSpec(code: "E201", name: "type mismatch (auto-chain)", citation: "§5.1", template: "Row {n} ({task}) needs {wanted}, but row {n-1} produces {got}. Between them you likely want {suggestion}."),
        "E202": CatErrorSpec(code: "E202", name: "type mismatch (reference)", citation: "§5.2", template: "Row {n}'s reference ({m}) delivers {got}, but its {position} input needs {wanted}."),
        "E203": CatErrorSpec(code: "E203", name: "reference to a missing row", citation: "§5.2", template: "Row {n} references row {m}, which no longer exists. It was deleted — pick a new source, or undo."),
        "E204": CatErrorSpec(code: "E204", name: "reference doesn't dominate", citation: "§5.3, ruling R5", template: "Row {n} references row {m}, but row {m} doesn't run on every path that reaches row {n} — when {decider-row}'s tag `{tag}` fires, row {m} is skipped. Reference something both paths produce, or move the reference behind the paths' converging row."),
        "E205": CatErrorSpec(code: "E205", name: "too many references", citation: "§5.2, R14", template: "Row {n} bundles {count} references; the limit is 4. Combine some upstream (Template and Join Text both bundle), then reference the result."),
        "E206": CatErrorSpec(code: "E206", name: "implicit broadcasting", citation: "§6.5", template: "Row {n} ({task}) takes one {kind}, but receives a list of {count}. To run it once per item, wrap it in `<each>`. To combine the items first, use {suggestion}."),
        "E207": CatErrorSpec(code: "E207", name: "unresolvable placeholder", citation: "§12.3", template: "Row {n}'s pattern uses `{{{ph}}}`, but its input has only {count} position(s). Positions here: {list}."),
        "E208": CatErrorSpec(code: "E208", name: "circular reference", citation: nil, template: "Row {n} references row {m}, which (through {chain}) references row {n} back. References read completed outputs — one of these needs to come first."),
        "E301": CatErrorSpec(code: "E301", name: "unreachable row", citation: "check 1", template: "Row {n} can never run: nothing falls into it, no edge aims at it, and no reference reads it. Delete it, or connect it."),
        "E302": CatErrorSpec(code: "E302", name: "tag without an edge", citation: "check 2", template: "Row {n} declares tags {tags}, but `{tag}` has no destination. Every tag needs an edge: add `{tag}: <row>` to the clause."),
        "E303": CatErrorSpec(code: "E303", name: "edge with an unknown tag", citation: "check 2", template: "Row {n}'s clause routes `{tag}`, but the row's tags are {tags}. Closest declared tag: `{closest}`."),
        "E304": CatErrorSpec(code: "E304", name: "edge payload mismatch", citation: "check 2", template: "Row {n}'s tag `{tag}` carries {got} to row {m}, which needs {wanted}. (A decider hands on its own input — see what row {n} receives.)"),
        "E305": CatErrorSpec(code: "E305", name: "cycle without an exit", citation: "check 3", template: "Rows {cycle} form a loop with no way out — no decider in the loop has an edge leaving it. Add an exit tag, or aim one edge past the loop."),
        "E306": CatErrorSpec(code: "E306", name: "cycle without a budget", citation: "check 4", template: "Rows {cycle} form a loop, but no row in it carries `visits≤N`. A loop must say how many times it may run — put the budget on the decider (row {suggested})."),
        "E307": CatErrorSpec(code: "E307", name: "budget without `on_budget`", citation: "check 4", template: "Row {n} has `visits≤{N}` but doesn't say what running out means. Add `on_budget={tag}` to proceed with the best so far, or `on_budget=fail` to stop instead."),
        "E308": CatErrorSpec(code: "E308", name: "`on_budget` names an unknown tag", citation: "check 4", template: "Row {n}'s `on_budget={tag}` isn't one of its tags ({tags}) or `fail`."),
        "E309": CatErrorSpec(code: "E309", name: "`resume` without a caller", citation: "check 5", template: "Row {n} ends its chain with `resume`, but no `call` can reach it. At the top level of a flow this is fine (resume means done) — inside a block it means the tool chain is orphaned. Aim a `call` at row {first-of-chain}, or end the chain another way."),
        "E401": CatErrorSpec(code: "E401", name: "unbound block input", citation: "§6.1", template: "Inside {block}, row {n} reads `(input:{k})`, but the block row binds only {count} input(s). Add a reference on the block's row, or drop to `(input:{count})`."),
        "E402": CatErrorSpec(code: "E402", name: "block return types disagree", citation: "§6.3, R4", template: "{block} can end at row {a} (producing {type-a}) or at row {b} (producing {type-b}). A block's exits must agree — converge them on one row."),
        "E403": CatErrorSpec(code: "E403", name: "internal reference escapes the block", citation: "§6.1", template: "Inside {block}, row {n} references row {m} of the outer flow. Blocks are sealed — pass it in: add `({m})` to the block's row and read it as `(input:{next})`."),
        "E404": CatErrorSpec(code: "E404", name: "composite parameter unbound", citation: "check 9", template: "Row {n} uses {composite} but doesn't set `{param}`, which has no default. Add `{param}=…` to the row."),
        "E405": CatErrorSpec(code: "E405", name: "composite definition missing", citation: "check 9", template: "Row {n} uses {composite}, but this file's `definitions:` section doesn't define it. The flow isn't whole — re-insert the composite from the library, or paste its definition."),
        "E406": CatErrorSpec(code: "E406", name: "unknown parameter", citation: nil, template: "Row {n} passes `{key}=…`, but {composite}'s parameters are {params}."),
        "E501": CatErrorSpec(code: "E501", name: "human row without a waiting policy", citation: "check 6", template: "Row {n} ({task}) waits for a person but doesn't say what happens if nobody answers. Add `timeout=… · default=…` to proceed unattended (flagged), or `wait=forever` to park until answered."),
        "E502": CatErrorSpec(code: "E502", name: "default isn't a declared tag", citation: "check 6", template: "Row {n}'s `default={value}` isn't one of its tags ({tags}). The timeout must pick a real choice."),
        "E503": CatErrorSpec(code: "E503", name: "context read without marker", citation: "check 7", template: "Row {n}'s pattern mentions the journal, but the row doesn't carry `· ctx`. Context is never ambient — add the marker, and the journal becomes a visible input."),
        "E504": CatErrorSpec(code: "E504", name: "`<parallel>` chain without a tail type", citation: "check 8", template: "Chain {k} of {block} ends in a row that produces {got}, which doesn't fit the block's declared bundle. Each chain's last row is one slot of the output."),
        "E505": CatErrorSpec(code: "E505", name: "Stage row outside the outbox model", citation: nil, template: "Row {n} would send directly. Direct sends don't exist — use `Stage Send`, and commit from the outbox after the run."),
        "E601": CatErrorSpec(code: "E601", name: "trigger not at row 1", citation: "check 10", template: "`{trigger}` sits at row {n}, but a trigger is the flow's front door — it must be row 1, and there can be only one."),
        "E602": CatErrorSpec(code: "E602", name: "trigger inside a block", citation: "check 10", template: "{block} contains `{trigger}`. Doors open from the outside; a block can't contain one. Move it to row 1 of its own flow."),
        "E603": CatErrorSpec(code: "E603", name: "edge aims at the trigger", citation: "check 10", template: "Row {n}'s edge targets row 1, which is a trigger. Nothing inside a flow can knock on its own front door — aim at row 2."),
        "E604": CatErrorSpec(code: "E604", name: "missing `events` flag", citation: "check 11", template: "Row 1 is a trigger, but the header doesn't say `· events`. Add it — a reader must see from line one that this flow can be armed. (fmt will do this for you.)"),
        "E605": CatErrorSpec(code: "E605", name: "rate budget required", citation: "check 12", template: "This flow has `{trigger}` and uses {costly-rows} — it must say how often it may fire. Add `runs≤N/hour` or `runs≤N/day` to row 1."),
        "E701": CatErrorSpec(code: "E701", name: "unknown setting", citation: nil, template: "`{key}` isn't a setting {model} knows. Its settings: {list}. {Closest: `{closest}`.}"),
        "E702": CatErrorSpec(code: "E702", name: "value off the enum", citation: nil, template: "`{key}={value}` — {model} has no such {key}. Closest: `{closest}`. All values: {list}."),
        "E703": CatErrorSpec(code: "E703", name: "value out of range", citation: nil, template: "`{key}={value}` is outside {model}'s range ({min}–{max})."),
        "E704": CatErrorSpec(code: "E704", name: "model not installed", citation: nil, template: "This flow pins {id}, which isn't installed. Download ({size}, {license})? Until then this row is held."),
        "E705": CatErrorSpec(code: "E705", name: "pinned id unavailable", citation: nil, template: "{id} isn't in the registry and isn't downloadable from a curated source. The flow won't substitute a same-named model — install the id, or pick a model on the row (which re-pins)."),
        "E706": CatErrorSpec(code: "E706", name: "display-name collision", citation: "registry load, not a row", template: "Two models claim the name \"{display}\" ({id-a}, {id-b}). Registry not loaded — rename one manifest's display."),
        "E707": CatErrorSpec(code: "E707", name: "embedder mismatch", citation: "Catalog §5", template: "Row {n} retrieves from {index}, which was built with {embedder-a}, but row {m} embeds with {embedder-b}. Same text, different spaces — results would be silently wrong. Match them, or rebuild the index."),
        "E801": CatErrorSpec(code: "E801", name: "local data into a net row", citation: "§14.2", template: "Row {n} sends text derived from {source-file} to the web ({task}). If that's intended, add `allow_upload` to the row — the flow must say your data leaves."),
        "E802": CatErrorSpec(code: "E802", name: "Think mixing index and web", citation: "Catalog §8.4", template: "{block}'s Think uses both your index and the web. Fetched pages could steer what it searches for next — add `allow_upload` to the block to accept that, visibly."),
        "R901": CatErrorSpec(code: "R901", name: "model failed to load", citation: nil, template: "{model} needs ~{ram} GB free; {available} GB available. Close something, or swap row {n} to a smaller model ({suggestion})."),
        "R902": CatErrorSpec(code: "R902", name: "empty output", citation: nil, template: "Row {n} produced nothing. Its input is on the row — usually the story is there (an empty transcript, a filter that matched nothing)."),
        "R903": CatErrorSpec(code: "R903", name: "file unreadable", citation: nil, template: "Row {n} couldn't read {path}: {reason}. Fix or re-point the row; rows 1–{n-1} are cached."),
        "R904": CatErrorSpec(code: "R904", name: "provider error (remote row)", citation: nil, template: "{provider} answered: {status}. The row is held; nothing was retried without you. Your key is configured in Settings, never in the flow."),
        "R905": CatErrorSpec(code: "R905", name: "step cap reached", citation: "Spec §8.3, R18", template: "This run hit the global cap of {N} steps and was stopped. If this flow legitimately needs more, raise the cap in Settings — budgets on rows are the better fix."),
        "F001": CatErrorSpec(code: "F001", name: "budget exhausted", citation: nil, template: "Out of visits ({N}) — proceeded on `{tag}` with the best so far."),
        "F002": CatErrorSpec(code: "F002", name: "timeout default taken", citation: nil, template: "Nobody answered by {time} — proceeded as `{default}`, unreviewed."),
        "F003": CatErrorSpec(code: "F003", name: "item skipped", citation: "`on_error=skip`", template: "Item {k} of {N} failed ({reason}) and was skipped. {N-1} delivered."),
        "F004": CatErrorSpec(code: "F004", name: "constrained decoding unavailable", citation: "Spec §12.2", template: "{provider} can't constrain output; used strict parsing + one retry. The tag is valid, but read the frame if the stakes are high."),
        "F005": CatErrorSpec(code: "F005", name: "occurrence held", citation: "events", template: "{count} occurrence(s) beyond `runs≤{budget}` are held; they'll run when the window reopens."),
        "F006": CatErrorSpec(code: "F006", name: "settings recomputed", citation: "Registry §4, M2", template: "Recomputed: {model}'s manifest changed a default ({key} {old} → {new})."),
        "F007": CatErrorSpec(code: "F007", name: "forced substitution shown", citation: "Spec §7.5, R17", template: "This activation's first input arrived from row {m}'s edge, overriding the `({ref})` reference for this pass."),
    ]

    /// 98 codes, generated from `core/errors_catalog.py` (`catalogV08`).
    static let catalogV08: [String: CatErrorSpec] = [
        "E101": CatErrorSpec(code: "E101", name: "unknown version", citation: "Spec §1.1, §1.1a", template: "This flow says `mlxflow {version}`. This runtime speaks 0.8 only, and there is no converter. Nothing was run."),
        "E102": CatErrorSpec(code: "E102", name: "unknown header flag", citation: "§1.1", template: "This flow declares `; {flag}`, which this runtime doesn't support. Nothing was run."),
        "E103": CatErrorSpec(code: "E103", name: "missing `network` flag", citation: "§1.2", template: "Row {n} ({task}) talks to the internet, but the header doesn't say `; network`. Add it — shared flows must show their network use on line one. (fmt will do this for you.)"),
        "E104": CatErrorSpec(code: "E104", name: "model not pinned", citation: "§1.5", template: "Row {n} names \"{display}\", but the `models:` section doesn't pin an id for it. Pick the model again to pin it, or add the line yourself."),
        "E105": CatErrorSpec(code: "E105", name: "unparseable row", citation: nil, template: "Row {n} couldn't be read from \"{fragment}…\". A row is: `N. Task  (refs)  model; settings`. The first thing that confused the parser: {detail}."),
        "E106": CatErrorSpec(code: "E106", name: "unclosed quote", citation: nil, template: "Row {n}'s quoted text opens with `\"` but never closes. If the text should contain a quote, write `\\\"`."),
        "E107": CatErrorSpec(code: "E107", name: "duplicate position", citation: nil, template: "Two rows are numbered {n} in the same scope. Positions must be sequential — running fmt will renumber and re-aim references."),
        "E108": CatErrorSpec(code: "E108", name: "pre-0.8 character", citation: "§2.1a", template: "{subject} uses `{char}`, which mlx-workflow 0.8 replaced with `{ascii}`. Run `mlxflow fmt --upgrade` to convert this file. Nothing was run."),
        "E109": CatErrorSpec(code: "E109", name: "missing `improvise` flag", citation: "Spec §1.2, §14.4a", template: "Row {n} improvises with a shell, but the header doesn't say `; improvise`. Add it — shared flows must show what they can do on line one. (fmt will do this for you.)"),
        "E110": CatErrorSpec(code: "E110", name: "`Improvise` row without a bound", citation: "§14.4b; checks 13–14", template: "Row {n} improvises, but doesn't say when to stop. Add `max_actions=N` (how many steps it may take) and `timeout=<duration>` (how long any one step may run)."),
        "E111": CatErrorSpec(code: "E111", name: "`Improvise` row without a `workdir`", citation: "§14.4c; check 15", template: "Row {n} improvises, but doesn't say where. Add `workdir=<dir>` — the agent reads and writes there and nowhere else."),
        "E112": CatErrorSpec(code: "E112", name: "`improvise` and `events` together", citation: "§14.4f; check 16", template: "This flow both improvises and starts itself. Those can't combine: triggers run unattended, and an improvising row is the one thing that shouldn't. Remove the trigger, or the `Improvise` row."),
        "E113": CatErrorSpec(code: "E113", name: "unresolved task", citation: "Spec §1.4a; check 17", template: "Row {n} names \"{task}\", which isn't a catalog task, a composite in `definitions:`, or an entry in `uses:`. If it's another flow in this folder, add `{task} = ./{task}.cat` to `uses:`."),
        "E114": CatErrorSpec(code: "E114", name: "`uses:` path outside the folder", citation: "§1.4a; check 18", template: "`uses:` points at `{path}`, which resolves outside this flow's own folder. A flow may only use flows beside it — copy it in, or move it here."),
        "E115": CatErrorSpec(code: "E115", name: "`uses:` cycle", citation: "§1.4a; check 19", template: "`{a}` uses `{b}`, which (through {chain}) uses `{a}` back. A flow can't use itself, directly or at a distance."),
        "E116": CatErrorSpec(code: "E116", name: "used flow is invalid", citation: "§1.4a; check 18", template: "Row {n} uses `{path}`, but that file doesn't validate on its own: {first-error}. Fix it there — every used flow is also an ordinary flow, so opening and checking it works normally."),
        "E117": CatErrorSpec(code: "E117", name: "undeclared inherited capability", citation: "Spec §1.4b; check 20", template: "This flow uses `{path}`, which {chain} improvises with a shell — but this header doesn't say `; improvise`. Add it. What a flow can do has to be readable on line one, even when it happens two files away."),
        "E118": CatErrorSpec(code: "E118", name: "missing `code` flag", citation: "Spec §1.2, §1.6", template: "This flow declares `transforms:`, but the header doesn't say `; code`. Add it — shared flows must show what they can do on line one. (fmt will do this for you.)"),
        "E119": CatErrorSpec(code: "E119", name: "transform without a `workdir`", citation: "§14.6; check 24", template: "Row {n} uses {transform}, but its `transforms:` entry doesn't declare `workdir:`. Every transform names the directory its script may touch — add `workdir: <dir>` to its entry."),
        "E120": CatErrorSpec(code: "E120", name: "missing `offdevice` flag", citation: "Spec §1.2; RA-09/SPEC-Q203", template: "Row {n} ({task}) binds {provider} — this row's input {egress} — but the header doesn't say `; offdevice`. Add it, so line one shows when data leaves this machine. (fmt will do this for you.)"),
        "E201": CatErrorSpec(code: "E201", name: "type mismatch (auto-chain)", citation: "§5.1", template: "Row {n} ({task}) needs {wanted}, but row {n-1} produces {got}. Between them you likely want {suggestion}."),
        "E202": CatErrorSpec(code: "E202", name: "type mismatch (reference)", citation: "§5.2", template: "Row {n}'s reference ({m}) delivers {got}, but its {position} input needs {wanted}."),
        "E203": CatErrorSpec(code: "E203", name: "reference to a missing row", citation: "§5.2", template: "Row {n} references row {m}, which no longer exists. It was deleted — pick a new source, or undo."),
        "E204": CatErrorSpec(code: "E204", name: "reference doesn't dominate", citation: "§5.3, ruling R5", template: "Row {n} references row {m}, but row {m} doesn't run on every path that reaches row {n} — when {decider-row}'s tag `{tag}` fires, row {m} is skipped. Reference something both paths produce, or move the reference behind the paths' converging row."),
        "E205": CatErrorSpec(code: "E205", name: "too many references", citation: "§5.2, R14", template: "Row {n} bundles {count} references; the limit is 4. Combine some upstream (Template and Join Text both bundle), then reference the result."),
        "E206": CatErrorSpec(code: "E206", name: "implicit broadcasting", citation: "§6.5", template: "Row {n} ({task}) takes one {kind}, but receives a list of {count}. To run it once per item, wrap it in `<each>`. To combine the items first, use {suggestion}."),
        "E207": CatErrorSpec(code: "E207", name: "unresolvable placeholder", citation: "§12.3", template: "Row {n}'s pattern uses `{{{ph}}}`, but its input has only {count} position(s). Positions here: {list}."),
        "E208": CatErrorSpec(code: "E208", name: "circular reference", citation: nil, template: "Row {n} references row {m}, which (through {chain}) references row {n} back. References read completed outputs — one of these needs to come first."),
        "E209": CatErrorSpec(code: "E209", name: "mismatched embedder", citation: "Spec §3.1a; check 21", template: "Row {n} queries an index built with {index-embedder}, but row {m} embeds with {row-embedder}. Vectors from different models aren't comparable — re-embed the query with {index-embedder}, or rebuild the index."),
        "E210": CatErrorSpec(code: "E210", name: "latent space mismatch", citation: "P6-G4-05; Spec §3.1a, check 21", template: "Row {n} produces a {space-a} latent ({channels-a} channels) via {model-a}, but row {m} consumes it as {space-b} ({channels-b} channels) from {model-b}. Channel count alone doesn't make two latents interchangeable — a latent is only meaningful in the space that wrote it. Match the producing and consuming models."),
        "E301": CatErrorSpec(code: "E301", name: "unreachable row", citation: "check 1", template: "Row {n} can never run: nothing falls into it, no edge aims at it, and no reference reads it. Delete it, or connect it."),
        "E302": CatErrorSpec(code: "E302", name: "tag without an edge", citation: "check 2", template: "Row {n} declares tags {tags}, but `{tag}` has no destination. Every tag needs an edge: add `{tag}: <row>` to the clause."),
        "E303": CatErrorSpec(code: "E303", name: "edge with an unknown tag", citation: "check 2", template: "Row {n}'s clause routes `{tag}`, but the row's tags are {tags}. Closest declared tag: `{closest}`."),
        "E304": CatErrorSpec(code: "E304", name: "edge payload mismatch", citation: "check 2", template: "Row {n}'s tag `{tag}` carries {got} to row {m}, which needs {wanted}. (A decider hands on its own input — see what row {n} receives.)"),
        "E305": CatErrorSpec(code: "E305", name: "cycle without an exit", citation: "check 3", template: "Rows {cycle} form a loop with no way out — no decider in the loop has an edge leaving it. Add an exit tag, or aim one edge past the loop."),
        "E306": CatErrorSpec(code: "E306", name: "cycle without a budget", citation: "check 4", template: "Rows {cycle} form a loop, but no row in it carries `max_visits=N`. A loop must say how many times it may run — put the budget on the decider (row {suggested})."),
        "E307": CatErrorSpec(code: "E307", name: "budget without `on_budget`", citation: "check 4", template: "Row {n} has `max_visits={N}` but doesn't say what running out means. Add `on_budget={tag}` to proceed with the best so far, or `on_budget=fail` to stop instead."),
        "E308": CatErrorSpec(code: "E308", name: "`on_budget` names an unknown tag", citation: "check 4", template: "Row {n}'s `on_budget={tag}` isn't one of its tags ({tags}) or `fail`."),
        "E309": CatErrorSpec(code: "E309", name: "`resume` without a caller", citation: "check 5", template: "Row {n} ends its chain with `resume`, but no `call` can reach it. At the top level of a flow this is fine (resume means done) — inside a block it means the tool chain is orphaned. Aim a `call` at row {first-of-chain}, or end the chain another way."),
        "E310": CatErrorSpec(code: "E310", name: "`Compare` with a tag count other than two", citation: "check 22", template: "Row {n} declares {count} tags ({tags}), but `Compare` always describes a yes/no split — exactly two. A three-way split is two `Compare` rows in sequence, not a wider tag set."),
        "E311": CatErrorSpec(code: "E311", name: "`Range` bounds invalid", citation: "check 23", template: "Row {n}'s `Range {bounds}` isn't legal: {detail}."),
        "E401": CatErrorSpec(code: "E401", name: "unbound block input", citation: "§6.1", template: "Inside {block}, row {n} reads `(input:{k})`, but the block row binds only {count} input(s). Add a reference on the block's row, or drop to `(input:{count})`."),
        "E402": CatErrorSpec(code: "E402", name: "block return types disagree", citation: "§6.3, R4", template: "{block} can end at row {a} (producing {type-a}) or at row {b} (producing {type-b}). A block's exits must agree — converge them on one row."),
        "E403": CatErrorSpec(code: "E403", name: "internal reference escapes the block", citation: "§6.1", template: "Inside {block}, row {n} references row {m} of the outer flow. Blocks are sealed — pass it in: add `({m})` to the block's row and read it as `(input:{next})`."),
        "E404": CatErrorSpec(code: "E404", name: "composite parameter unbound", citation: "check 9", template: "Row {n} uses {composite} but doesn't set `{param}`, which has no default. Add `{param}=…` to the row."),
        "E405": CatErrorSpec(code: "E405", name: "composite definition missing", citation: "check 9", template: "Row {n} uses {composite}, but this file's `definitions:` section doesn't define it. The flow isn't whole — re-insert the composite from the library, or paste its definition."),
        "E406": CatErrorSpec(code: "E406", name: "unknown parameter", citation: nil, template: "Row {n} passes `{key}=…`, but {composite}'s parameters are {params}."),
        "E407": CatErrorSpec(code: "E407", name: "transform without a `run:`", citation: "§1.6", template: "Row {n} uses {transform}, but its `transforms:` entry doesn't declare `run:`. Every transform names an executable — add `run: <path>` to its entry."),
        "E408": CatErrorSpec(code: "E408", name: "transform without a `timeout:`", citation: "§1.6", template: "Row {n} uses {transform}, but its `transforms:` entry doesn't declare `timeout:`. Add a bound — every transform declares how long its script may run."),
        "E409": CatErrorSpec(code: "E409", name: "transform script unresolved", citation: "§1.6", template: "Row {n} uses {transform}, whose `run:` script ({path}) doesn't exist, or isn't executable. Until it does, this row is held."),
        "E501": CatErrorSpec(code: "E501", name: "human row without a waiting policy", citation: "check 6", template: "Row {n} ({task}) waits for a person but doesn't say what happens if nobody answers. Add `timeout=…; default=…` to proceed unattended (flagged), or `wait=forever` to park until answered."),
        "E502": CatErrorSpec(code: "E502", name: "default isn't a declared tag", citation: "check 6", template: "Row {n}'s `default={value}` isn't one of its tags ({tags}). The timeout must pick a real choice."),
        "E503": CatErrorSpec(code: "E503", name: "context read without marker", citation: "check 7", template: "Row {n}'s pattern mentions the journal, but the row doesn't carry `; ctx`. Context is never ambient — add the marker, and the journal becomes a visible input."),
        "E504": CatErrorSpec(code: "E504", name: "`<parallel>` chain without a tail type", citation: "check 8", template: "Chain {k} of {block} ends in a row that produces {got}, which doesn't fit the block's declared bundle. Each chain's last row is one slot of the output."),
        "E505": CatErrorSpec(code: "E505", name: "Stage row outside the outbox model", citation: nil, template: "Row {n} would send directly. Direct sends don't exist — use `Stage Send`, and commit from the outbox after the run."),
        "E601": CatErrorSpec(code: "E601", name: "trigger not at row 1", citation: "check 10", template: "`{trigger}` sits at row {n}, but a trigger is the flow's front door — it must be row 1, and there can be only one."),
        "E602": CatErrorSpec(code: "E602", name: "trigger inside a block", citation: "check 10", template: "{block} contains `{trigger}`. Doors open from the outside; a block can't contain one. Move it to row 1 of its own flow."),
        "E603": CatErrorSpec(code: "E603", name: "edge aims at the trigger", citation: "check 10", template: "Row {n}'s edge targets row 1, which is a trigger. Nothing inside a flow can knock on its own front door — aim at row 2."),
        "E604": CatErrorSpec(code: "E604", name: "missing `events` flag", citation: "check 11", template: "Row 1 is a trigger, but the header doesn't say `; events`. Add it — a reader must see from line one that this flow can be armed. (fmt will do this for you.)"),
        "E605": CatErrorSpec(code: "E605", name: "rate budget required", citation: "check 12", template: "This flow has `{trigger}` and uses {costly-rows} — it must say how often it may fire. Add `max_runs=N/hour` or `max_runs=N/day` to row 1."),
        "E701": CatErrorSpec(code: "E701", name: "unknown setting", citation: nil, template: "`{key}` isn't a setting {model} knows. Its settings: {list}. {Closest: `{closest}`.}"),
        "E702": CatErrorSpec(code: "E702", name: "value off the enum", citation: nil, template: "`{key}={value}` — {model} has no such {key}. Closest: `{closest}`. All values: {list}."),
        "E703": CatErrorSpec(code: "E703", name: "value out of range", citation: nil, template: "`{key}={value}` is outside {model}'s range ({min}–{max})."),
        "E708": CatErrorSpec(code: "E708", name: "rejected setting", citation: "P6-G1-07", template: "{model} rejects `{key}=`: {reason}. Remove it, or switch the row to a model that supports {key}."),
        "E709": CatErrorSpec(code: "E709", name: "adapter doesn't fit the model", citation: "P6-G3-02/03", template: "{adapter} is a {type} for {base}, but this row runs {model}. They don't share a base, so applying it would produce noise — remove the `{setting}=`, or switch the row to {base}."),
        "E710": CatErrorSpec(code: "E710", name: "refused inside a pipeline", citation: "P6-G4-01", template: "{task} isn't allowed in a pipeline: {reason}."),
        "E711": CatErrorSpec(code: "E711", name: "a model pipeline reads nothing", citation: "P6-G4-01/02", template: "{name} gives a model, so it reads nothing — a recipe's parents, method and ratios are bound from `params:`, never from a flow-computed input. Remove the `accepts:` line."),
        "E712": CatErrorSpec(code: "E712", name: "extension and header disagree", citation: "P6-G4-01", template: "This file is named *{suffix} but its header says `{kind} {version}` — the extension and line one are both the answer to \"what is this file\". Rename it to *{correct} or fix the header."),
        "E713": CatErrorSpec(code: "E713", name: "model space is pipeline-only", citation: "P6-G4-02", template: "{task} is model space: a row that produces or consumes a model lives only inside a `.catpipeline`. `.cat` is \"assets flow, models are declared\" — put this row in a pipeline and reference the result from `models:`."),
        "E714": CatErrorSpec(code: "E714", name: "not a pinned adapter", citation: "P6-G3-02/03 review fix", template: "`{adapter}` isn't a pinned adapter — {setting}= applies weights pinned in `models:`. Add `{adapter} = <lora|controlnet id>` to `models:`."),
        "E715": CatErrorSpec(code: "E715", name: "preset not defined in this pipeline", citation: "Q199", template: "{preset} isn't a preset this pipeline defines. Its presets: {list}."),
        "E716": CatErrorSpec(code: "E716", name: "presets are pipeline-only", citation: "Q199", template: "`presets:` is pipeline-only — a named settings bundle lives inside a `.catpipeline`, never in a `.cat` or a `.transform`."),
        "E704": CatErrorSpec(code: "E704", name: "model not installed", citation: nil, template: "This flow pins {id}, which isn't installed. Download ({size}, {license})? Until then this row is held."),
        "E705": CatErrorSpec(code: "E705", name: "pinned id unavailable", citation: nil, template: "{id} isn't in the registry and isn't downloadable from a curated source. The flow won't substitute a same-named model — install the id, or pick a model on the row (which re-pins)."),
        "E706": CatErrorSpec(code: "E706", name: "display-name collision", citation: "registry load, not a row", template: "Two models claim the name \"{display}\" ({id-a}, {id-b}). Registry not loaded — rename one manifest's display."),
        "E707": CatErrorSpec(code: "E707", name: "embedder mismatch", citation: "Catalog §5", template: "Row {n} retrieves from {index}, which was built with {embedder-a}, but row {m} embeds with {embedder-b}. Same text, different spaces — results would be silently wrong. Match them, or rebuild the index."),
        "E801": CatErrorSpec(code: "E801", name: "local data into a net row", citation: "§14.2", template: "Row {n} sends text derived from {source-file} to the web ({task}). If that's intended, add `allow_upload` to the row — the flow must say your data leaves."),
        "E802": CatErrorSpec(code: "E802", name: "Think mixing index and web", citation: "Catalog §8.4", template: "{block}'s Think uses both your index and the web. Fetched pages could steer what it searches for next — add `allow_upload` to the block to accept that, visibly."),
        "R901": CatErrorSpec(code: "R901", name: "model failed to load", citation: nil, template: "{model} needs ~{ram} GB free; {available} GB available. Close something, or swap row {n} to a smaller model ({suggestion})."),
        "R902": CatErrorSpec(code: "R902", name: "empty output", citation: nil, template: "Row {n} produced nothing. Its input is on the row — usually the story is there (an empty transcript, a filter that matched nothing)."),
        "R903": CatErrorSpec(code: "R903", name: "file unreadable", citation: nil, template: "Row {n} couldn't read {path}: {reason}. Fix or re-point the row; rows 1–{n-1} are cached."),
        "R904": CatErrorSpec(code: "R904", name: "provider error (remote row)", citation: nil, template: "{provider} answered: {status}. The row is held; nothing was retried without you. Your key is configured in Settings, never in the flow."),
        "R905": CatErrorSpec(code: "R905", name: "step cap reached", citation: "Spec §8.3, R18", template: "This run hit the global cap of {N} steps and was stopped. If this flow legitimately needs more, raise the cap in Settings — budgets on rows are the better fix."),
        "R906": CatErrorSpec(code: "R906", name: "improvised outside the fence", citation: "Spec §14.4c", template: "Row {n} tried to {read|write|delete} `{path}`, which is outside its `workdir={dir}`. The run stopped and nothing at that path was touched. Everything row {n} changed before this can be undone with `mlxflow undo {run} {n}`."),
        "R907": CatErrorSpec(code: "R907", name: "improvised action failed", citation: "Spec §14.4", template: "Row {n}'s action failed: {detail}. The agent had {k} of {N} actions left. Everything it changed can be undone with `mlxflow undo {run} {n}`."),
        "R908": CatErrorSpec(code: "R908", name: "non-numeric comparison", citation: nil, template: "Row {n}'s `Compare` couldn't read \"{value}\" as a number ({detail}). `Compare` only reads numbers — clean the text first (`Extract`, `Filter`) if it needs it."),
        "R909": CatErrorSpec(code: "R909", name: "transform failed", citation: "§14.6", template: "Row {n}'s transform {transform} ({script}) {detail}. Nothing downstream of row {n} ran."),
        "F001": CatErrorSpec(code: "F001", name: "budget exhausted", citation: nil, template: "Out of visits ({N}) — proceeded on `{tag}` with the best so far."),
        "F002": CatErrorSpec(code: "F002", name: "timeout default taken", citation: nil, template: "Nobody answered by {time} — proceeded as `{default}`, unreviewed."),
        "F003": CatErrorSpec(code: "F003", name: "item skipped", citation: "`on_error=skip`", template: "Item {k} of {N} failed ({reason}) and was skipped. {N-1} delivered."),
        "F004": CatErrorSpec(code: "F004", name: "constrained decoding unavailable", citation: "Spec §12.2", template: "{provider} can't constrain output; used strict parsing + one retry. The tag is valid, but read the frame if the stakes are high."),
        "F005": CatErrorSpec(code: "F005", name: "occurrence held", citation: "events", template: "{count} occurrence(s) beyond `max_runs={budget}` are held; they'll run when the window reopens."),
        "F006": CatErrorSpec(code: "F006", name: "settings recomputed", citation: "Registry §4, M2", template: "Recomputed: {model}'s manifest changed a default ({key} {old} -> {new})."),
        "F007": CatErrorSpec(code: "F007", name: "forced substitution shown", citation: "Spec §7.5, R17", template: "This activation's first input arrived from row {m}'s edge, overriding the `({ref})` reference for this pass."),
        "F008": CatErrorSpec(code: "F008", name: "actions exhausted", citation: "Spec §14.4b", template: "Row {n} used all {N} of its actions and stopped there. What it finished is kept; what it didn't isn't. Raise `max_actions=` if the work was genuinely bigger than the budget."),
        "F009": CatErrorSpec(code: "F009", name: "files changed by an improvised row", citation: "Spec §14.4d", template: "Row {n} changed {k} file(s) in `{workdir}`: {paths}. Undo with `mlxflow undo {run} {n}`."),
        "F010": CatErrorSpec(code: "F010", name: "remote decider tag parsed, not guaranteed", citation: "Spec §12.2", template: "Row {n} ({task}) binds {provider}, which can't constrain its tag — the row that routes this flow runs on a parsed tag, not a guaranteed one. Prefer a local model for the routing step, or read the frame after the run."),
        "F011": CatErrorSpec(code: "F011", name: "seed ignored", citation: "Spec §13.1, §12.5", template: "{model} accepted `seed={seed}` but doesn't honour it — this row is not reproducible. Nothing was retried without you."),
    ]

    // MARK: - fill()

    /// SPEC-Q50: E701 is the doc's only bracketed-optional-clause placeholder
    /// (`{Closest: `{closest}`.}`) rather than a bare `{name}` token — special-cased.
    private static let e701Optional = " {Closest: `{closest}`.}"

    /// Render *code*'s template with *values*. `isV08` picks the v0.8 catalog
    /// (the only one a `catflow 0.8` runtime ever needs). Throws for a code the
    /// catalog doesn't know or a placeholder no value supplies — the Python
    /// equivalent is KeyError, and this must fail loudly, never render
    /// `{unfilled}`.
    static func fill(code: String, values: [String: String] = [:], isV08: Bool = false) throws -> String {
        let catalog = isV08 ? catalogV08 : catalogV07
        guard let spec = catalog[code] else {
            throw ErrorCatalogError.unknownCode(code)
        }
        var merged = values
        var template = spec.template
        if code == "E701" {
            let closest = merged.removeValue(forKey: "closest")
            let replacement = closest.map { " Closest: `\($0)`." } ?? ""
            template = template.replacingOccurrences(of: e701Optional, with: replacement)
        }
        return try render(template, values: merged)
    }

    /// `str.format_map` on a plain-name template: `{name}` fields, `{{`/`}}` escapes.
    /// Field names are looked up verbatim (dict semantics — `{n-1}` is a key named
    /// `n-1`), mirroring the Python `format_map` the catalog's callers rely on.
    private static func render(_ template: String, values: [String: String]) throws -> String {
        var out = ""
        var i = template.startIndex
        while i < template.endIndex {
            let c = template[i]
            if c == "{" {
                let next = template.index(after: i)
                if next < template.endIndex && template[next] == "{" {
                    out.append("{")
                    i = template.index(next, offsetBy: 1)
                    continue
                }
                guard let close = template[i...].firstIndex(of: "}") else {
                    throw ErrorCatalogError.unclosedBrace
                }
                let name = String(template[template.index(after: i)..<close])
                guard let value = values[name] else {
                    throw ErrorCatalogError.missingValue(name)
                }
                out.append(value)
                i = template.index(close, offsetBy: 1)
            } else if c == "}" {
                let next = template.index(after: i)
                if next < template.endIndex && template[next] == "}" {
                    out.append("}")
                    i = template.index(next, offsetBy: 1)
                } else {
                    out.append("}")
                    i = next
                }
            } else {
                out.append(c)
                i = template.index(after: i)
            }
        }
        return out
    }
}

nonisolated enum ErrorCatalogError: Error, Equatable, CustomStringConvertible {
    case unknownCode(String)
    case missingValue(String)
    case unclosedBrace

    var description: String {
        switch self {
        case .unknownCode(let code):
            return "ErrorCatalog.fill: no template for code \(code)"
        case .missingValue(let name):
            return "ErrorCatalog.fill: template needs `{\(name)}` but no value was supplied"
        case .unclosedBrace:
            return "ErrorCatalog.fill: template has an unclosed `{`"
        }
    }
}
