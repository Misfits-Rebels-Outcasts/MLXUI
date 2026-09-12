import Foundation

/// The seven task classes — ported verbatim from `catflow-mlx/src/catflow/catalog/tasks.py`
/// (`TaskClass`). Deciders are **not** a `TaskClass`; they are `model` tasks held in
/// `DECIDER_TASKS`, exactly as in the Python (five call sites branch on decider-specific
/// behavior — tag-constraint, never-cache-under-mock, transcript handling).
nonisolated enum TaskClass: String, Sendable, CaseIterable {
    case instant, model, human, trigger, staged, net, agent
}

/// The reference kind of a task's implementation — `tool` (pure code), `engine` (raw model
/// primitive), or `frame` (a published prompt frame owned by the runtime). Ported from
/// `catalog/tasks.py::RefKind` (the same type Core/Shape.swift defines for the shared
/// compatibility functions).
typealias TaskRefKind = RefKind

/// One task's static contract. Ported from `catalog/tasks.py::TaskDescriptor`. **Data only**
/// — no dispatch, no validator. `refName` stays a plain dotted string resolved by a dispatch
/// table in CFM-R2-6, never anything import-like (the Python's core-isolation rule).
nonisolated struct TaskDescriptor: Sendable, Equatable {
    var name: String
    var accepts: Shape
    var gives: Shape
    var taskClass: TaskClass
    var refKind: TaskRefKind
    var refName: String

    init(_ name: String, _ accepts: Shape, _ gives: Shape,
         _ taskClass: TaskClass, _ refKind: TaskRefKind, _ refName: String) {
        self.name = name
        self.accepts = accepts
        self.gives = gives
        self.taskClass = taskClass
        self.refKind = refKind
        self.refName = refName
    }
}

/// The task catalog — the static Swift table mirroring `catflow-mlx/src/catflow/catalog/tasks.py`
/// (`CATALOG`, 106 entries) plus the six deciders held separately (`DECIDER_TASKS`), giving
/// `allTasks() == 112`. Counts and class histogram are pinned by `CatFlowTaskCatalogTests`.
/// See `RSI/DelegateMergeBacklog.md` CFM-R2-1.
nonisolated enum TaskCatalog {

    static let catalog: [String: TaskDescriptor] = {
        var byName: [String: TaskDescriptor] = [:]
        for entry in entries {
            precondition(byName[entry.name] == nil, "duplicate task name: \(entry.name)")
            byName[entry.name] = entry
        }
        return byName
    }()

    static let deciderTasks: [String: TaskDescriptor] = {
        var byName: [String: TaskDescriptor] = [:]
        for entry in deciders {
            precondition(byName[entry.name] == nil, "duplicate decider name: \(entry.name)")
            byName[entry.name] = entry
        }
        return byName
    }()

    /// The 106 catalog entries (non-decider).
    static var entries: [TaskDescriptor] {
        _entries
    }

    /// The 6 deciders, held separately (SPEC-Q55(a)).
    static var deciders: [TaskDescriptor] {
        _deciders
    }

    /// Every listable task — `CATALOG` plus the deciders (the Python's `all_tasks()`).
    static func allTasks() -> [TaskDescriptor] {
        _entries + _deciders
    }

    static func get(_ name: String) -> TaskDescriptor? {
        catalog[name] ?? deciderTasks[name]
    }

    /// RM-2 (SPEC-Q204) — expand a named `tasks:` group a provider manifest can reference
    /// (`"tasks": ["@frames-text", "Decide"]`, e.g. `macstudio-qwen3-32b.json`, ported
    /// verbatim in RM-1). Ported from `catflow-mlx/src/catflow/catalog/tasks.py
    /// ::task_group_members`: membership is *derived* from the catalog's own signatures,
    /// never a hand-written inventory, so a new frame task inherits membership the day it
    /// lands — a manifest is a policy file, not a maintained list that drifts silently
    /// (the exact trap AFM-2 sidestepped by NOT using this group for `apple-foundation
    /// .json`, per its own comment — that manifest's curated list predates this function).
    /// `"frames-text"` is the only group the reference defines: the bare `Generate`
    /// primitive plus every frame-backed model task whose output is text (`.single`/
    /// `.listOf` of `.text`). `deciders` are never members — `CATALOG` in the Python
    /// excludes them too (SPEC-Q55(a)'s split), and a `Decide`/`Gate`/… row is picked by
    /// its own explicit `tasks:` entry, not swept in by a group.
    static func taskGroupMembers(_ group: String) -> [String] {
        guard group == "frames-text" else { return [] }
        var members: Set<String> = ["Generate"]
        for entry in entries where entry.taskClass == .model && entry.refKind == .frame {
            switch entry.gives {
            case .single(.text), .listOf(.text):
                members.insert(entry.name)
            default:
                break
            }
        }
        return members.sorted()
    }

    /// A manifest's raw `tasks` array, with any `@group` entries expanded (RM-2) — the
    /// list `TaskModels.providerModels`/`systemModels` actually key their registry by.
    static func expandTaskNames(_ tasks: [String]) -> [String] {
        var out: [String] = []
        for task in tasks {
            if task.hasPrefix("@") {
                out.append(contentsOf: taskGroupMembers(String(task.dropFirst())))
            } else {
                out.append(task)
            }
        }
        return out
    }

    // MARK: - The table (ported verbatim from tasks.py `_ENTRIES`)

    /// `_t(kind)` → `Shape.single(kind)`.
    private static func t(_ kind: Kind) -> Shape { .single(kind) }

    private static let _entries: [TaskDescriptor] = [
        // -- §1 Files in and out (instant) --
        TaskDescriptor("Read Text", t(.file), t(.text), .instant, .tool, "tools.files.read_text"),
        TaskDescriptor("Read Audio", t(.file), t(.audio), .instant, .tool, "tools.files.read_audio"),
        TaskDescriptor("Read Image", t(.file), t(.image), .instant, .tool, "tools.files.read_image"),
        TaskDescriptor("Read Video", t(.file), t(.video), .instant, .tool, "tools.files.read_video"),
        TaskDescriptor("Read Images", t(.folder), .listOf(.image), .instant, .tool, "tools.files.read_images"),
        TaskDescriptor("Read Files", t(.folder), .listOf(.file), .instant, .tool, "tools.files.read_files"),
        TaskDescriptor("Read PDF", t(.file), t(.text), .instant, .tool, "tools.files.read_pdf"),
        TaskDescriptor("Read Index", t(.file), t(.index), .instant, .tool, "tools.index_store.read_index"),
        TaskDescriptor("Save Text", .anyKind, t(.status), .instant, .tool, "tools.files.save_text"),
        TaskDescriptor("Save Audio", .anyKind, t(.status), .instant, .tool, "tools.files.save_audio"),
        TaskDescriptor("Save Image", .anyKind, t(.status), .instant, .tool, "tools.files.save_image"),
        TaskDescriptor("Save Video", .anyKind, t(.status), .instant, .tool, "tools.files.save_video"),
        TaskDescriptor("Save Images", .listOf(.image), t(.status), .instant, .tool, "tools.files.save_images"),
        TaskDescriptor("Store Index", .tupleOf([.text, .vector]), t(.index), .instant, .tool, "tools.index_store.store_index"),

        // -- §2 Text (instant) --
        TaskDescriptor("Split", t(.text), .listOf(.text), .instant, .tool, "tools.text.split"),
        TaskDescriptor("Template", .listOf(.text), t(.text), .instant, .tool, "tools.text.template"),
        TaskDescriptor("Join Text", .listOf(.text), t(.text), .instant, .tool, "tools.text.join_text"),
        TaskDescriptor("Filter", .listOf(.text), .listOf(.text), .instant, .tool, "tools.text.filter_"),
        TaskDescriptor("Sort", .listOf(.text), .listOf(.text), .instant, .tool, "tools.text.sort_"),
        TaskDescriptor("Dedupe", .listOf(.text), .listOf(.text), .instant, .tool, "tools.text.dedupe"),
        TaskDescriptor("Extract", t(.text), .listOf(.text), .instant, .tool, "tools.text.extract"),
        TaskDescriptor("Count", .listOf(.text), t(.text), .instant, .tool, "tools.text.count"),
        TaskDescriptor("Diff", .tupleOf([.text, .text]), t(.text), .instant, .tool, "tools.text.diff"),

        // -- §3 Language models, Phase-1 subset (primitives + non-decider framed) --
        TaskDescriptor("Generate", t(.text), t(.text), .model, .engine, "engines.llm.generate"),
        TaskDescriptor("Summarize", t(.text), t(.text), .model, .frame, "frames/Summarize.frame.txt"),
        TaskDescriptor("Translate", t(.text), t(.text), .model, .frame, "frames/Translate.frame.txt"),
        TaskDescriptor("Answer", t(.text), t(.text), .model, .frame, "frames/Answer.frame.txt"),
        TaskDescriptor("Rewrite", t(.text), t(.text), .model, .frame, "frames/Rewrite.frame.txt"),
        TaskDescriptor("Draft", t(.text), t(.text), .model, .frame, "frames/Draft.frame.txt"),
        TaskDescriptor("Ask", t(.text), .listOf(.text), .model, .frame, "frames/Ask.frame.txt"),
        TaskDescriptor("Title", t(.text), t(.text), .model, .frame, "frames/Title.frame.txt"),
        TaskDescriptor("Critique", t(.text), t(.text), .model, .frame, "frames/Critique.frame.txt"),
        TaskDescriptor("Verify", t(.text), t(.text), .model, .frame, "frames/Verify.frame.txt"),
        TaskDescriptor("Revise", .listOf(.text), t(.text), .model, .frame, "frames/Revise.frame.txt"),
        TaskDescriptor("Merge", .listOf(.text), t(.text), .model, .frame, "frames/Merge.frame.txt"),
        TaskDescriptor("Extract Structured", t(.text), t(.table), .model, .engine, "engines.llm.extract_structured"),
        TaskDescriptor("Improvise", t(.text), t(.text), .agent, .engine, "engines.agent.improvise"),

        // -- §4 Audio, image, video --
        TaskDescriptor("Transcribe", t(.audio), t(.text), .model, .engine, "engines.asr.transcribe"),
        TaskDescriptor("Speak", t(.text), t(.audio), .model, .engine, "engines.tts.speak"),
        TaskDescriptor("Describe Image", t(.image), t(.text), .model, .engine, "engines.vlm.describe_image"),
        TaskDescriptor("OCR", t(.image), t(.text), .model, .engine, "engines.vlm.ocr"),
        TaskDescriptor("Generate Image", t(.text), t(.image), .model, .engine, "engines.diffusion.generate_image"),
        TaskDescriptor("Edit Image", .tupleOf([.image, .text]), t(.image), .model, .engine, "engines.diffusion.edit_image"),
        TaskDescriptor("Instruct Edit", .tupleOf([.image, .text]), t(.image), .model, .engine, "engines.diffusion.edit_image"),
        TaskDescriptor("Inpaint", .tupleOf([.image, .image, .text]), t(.image), .model, .engine, "engines.diffusion.inpaint"),
        TaskDescriptor("Upscale", t(.image), t(.image), .model, .engine, "engines.diffusion.upscale"),
        TaskDescriptor("Detect Edges", t(.image), t(.image), .instant, .tool, "tools.media.detect_edges"),
        TaskDescriptor("Estimate Depth", t(.image), t(.image), .model, .engine, "engines.diffusion.estimate_depth"),
        TaskDescriptor("Detect Pose", t(.image), t(.image), .instant, .tool, "tools.pose.detect_pose"),
        TaskDescriptor("Segment", t(.image), t(.image), .model, .engine, "engines.diffusion.segment"),
        TaskDescriptor("Generate Video", t(.text), t(.video), .model, .engine, "engines.diffusion.generate_video"),
        TaskDescriptor("Animate", t(.image), t(.video), .model, .engine, "engines.diffusion.animate"),
        TaskDescriptor("Generate Sound", t(.text), t(.audio), .model, .engine, "engines.diffusion.generate_sound"),
        TaskDescriptor("Mux", .tupleOf([.video, .audio]), t(.video), .instant, .tool, "tools.media.mux"),
        TaskDescriptor("Load Checkpoint", .anyKind, t(.model), .model, .engine, "engines.diffusion.load_checkpoint"),
        TaskDescriptor("Blend", t(.model), t(.model), .model, .engine, "engines.diffusion.blend"),
        TaskDescriptor("Bake LoRA", t(.model), t(.model), .model, .engine, "engines.diffusion.bake_lora"),
        TaskDescriptor("Pin Model", t(.model), t(.model), .model, .engine, "engines.diffusion.pin_model"),
        TaskDescriptor("Init Latent", .anyKind, t(.latent), .model, .engine, "engines.diffusion.init_latent"),
        TaskDescriptor("Encode Latent", t(.image), t(.latent), .model, .engine, "engines.diffusion.encode_latent"),
        TaskDescriptor("Decode Latent", t(.latent), t(.image), .model, .engine, "engines.diffusion.decode_latent"),
        TaskDescriptor("Denoise", t(.latent), t(.latent), .model, .engine, "engines.diffusion.denoise"),
        TaskDescriptor("Extract Frame", t(.video), t(.image), .instant, .tool, "tools.media.extract_frame"),
        TaskDescriptor("Extract Audio", t(.video), t(.audio), .instant, .tool, "tools.media.extract_audio"),
        TaskDescriptor("Trim", .unionOf([.audio, .video]), .sameAsInput, .instant, .tool, "tools.media.trim"),
        TaskDescriptor("Join Video", .listOf(.video), t(.video), .instant, .tool, "tools.media.join_video"),
        TaskDescriptor("Resize", t(.image), t(.image), .instant, .tool, "tools.media.resize"),
        TaskDescriptor("Crop", t(.image), t(.image), .instant, .tool, "tools.media.crop"),
        TaskDescriptor("Convert", t(.image), t(.image), .instant, .tool, "tools.media.convert"),
        TaskDescriptor("Watermark", t(.image), t(.image), .instant, .tool, "tools.media.watermark"),
        TaskDescriptor("Overlay Text", t(.image), t(.image), .instant, .tool, "tools.media.overlay_text"),
        TaskDescriptor("Contact Sheet", .tupleOf([.image, .text]), t(.image), .instant, .tool, "tools.media.contact_sheet"),

        // -- §5 Search and retrieval (local) --
        TaskDescriptor("Embed", .listOf(.text), .listOf(.vector), .model, .engine, "engines.embed.embed"),
        TaskDescriptor("Retrieve", .tupleOf([.index, .vector]), .listOf(.text), .instant, .tool, "tools.index_store.retrieve"),
        TaskDescriptor("Rerank", .listOf(.text), .listOf(.text), .model, .engine, "engines.rerank.rerank"),
        TaskDescriptor("Keyword Search", .tupleOf([.index, .text]), .listOf(.text), .instant, .tool, "tools.index_store.keyword_search"),

        // -- §6 Data --
        TaskDescriptor("Read CSV", t(.file), t(.table), .instant, .tool, "tools.data.read_csv"),
        TaskDescriptor("Read JSON", t(.file), t(.table), .instant, .tool, "tools.data.read_json"),
        TaskDescriptor("Query Table", t(.table), t(.table), .instant, .tool, "tools.data.query_table"),
        TaskDescriptor("Table to Text", t(.table), t(.text), .instant, .tool, "tools.data.table_to_text"),
        TaskDescriptor("Text to Table", t(.text), t(.table), .model, .engine, "engines.llm.text_to_table"),
        TaskDescriptor("Calculate", t(.text), t(.text), .instant, .tool, "tools.calc.calculate"),
        TaskDescriptor("Compare", t(.text), .sameAsInput, .instant, .tool, "tools.compare.compare"),
        TaskDescriptor("Range", .anyKind, .listOf(.text), .instant, .tool, "tools.range_.range_"),
        TaskDescriptor("Chart", t(.table), t(.image), .instant, .tool, "tools.data.chart"),

        // -- §7 Networked tools --
        TaskDescriptor("Web Search", t(.text), .listOf(.text), .net, .tool, "tools.net.web_search"),
        TaskDescriptor("Web Fetch", t(.text), t(.text), .net, .tool, "tools.net.web_fetch"),
        TaskDescriptor("Fetch Feed", t(.text), .listOf(.text), .net, .tool, "tools.net.fetch_feed"),
        TaskDescriptor("Download File", t(.text), t(.file), .net, .tool, "tools.net.download_file"),
        TaskDescriptor("HTTP Get", t(.text), t(.text), .net, .tool, "tools.net.http_get"),

        // -- §11 v0.5 additions: humans, context, staged effects --
        TaskDescriptor("Ask Human", t(.text), .sameAsInput, .human, .tool, "tools.human.ask_human"),
        TaskDescriptor("Human Input", t(.text), t(.text), .human, .tool, "tools.human.human_input"),
        TaskDescriptor("Save Context", t(.context), t(.status), .instant, .tool, "tools.context.save_context"),
        TaskDescriptor("Read Context", t(.file), t(.context), .instant, .tool, "tools.context.read_context"),
        TaskDescriptor("Count Context", t(.context), t(.text), .instant, .tool, "tools.context.count_context"),
        TaskDescriptor("Stage Send", t(.text), t(.status), .staged, .tool, "tools.outbox.stage_send"),
        TaskDescriptor("Stage Post", t(.text), t(.status), .staged, .tool, "tools.outbox.stage_post"),

        // -- §12 v0.7 additions: triggers --
        TaskDescriptor("On File", t(.occurrence), t(.file), .trigger, .tool, "tools.triggers.on_file"),
        TaskDescriptor("On Schedule", t(.occurrence), t(.text), .trigger, .tool, "tools.triggers.on_schedule"),
        TaskDescriptor("On Flow", t(.occurrence), t(.text), .trigger, .tool, "tools.triggers.on_flow"),

        // -- §13 v0.8 additions: the store door --
        TaskDescriptor("Store Query", t(.file), t(.table), .instant, .tool, "tools.store.store_query"),
        TaskDescriptor("Store Read", t(.file), t(.table), .instant, .tool, "tools.store.store_read"),
        TaskDescriptor("Store Write", t(.table), t(.status), .instant, .tool, "tools.store.store_write"),
        TaskDescriptor("Set Field", t(.table), t(.table), .instant, .tool, "tools.entity.set_field"),
        TaskDescriptor("Append Row", t(.table), t(.table), .instant, .tool, "tools.entity.append_row"),
        TaskDescriptor("Merge Record", .tupleOf([.table, .table]), t(.table), .instant, .tool, "tools.entity.merge_record"),
    ]

    private static let _deciders: [TaskDescriptor] = [
        TaskDescriptor("Decide", .anyKind, .sameAsInput, .model, .engine, "engines.llm.decide"),
        TaskDescriptor("Classify", .anyKind, .sameAsInput, .model, .frame, "frames/Classify.frame.txt"),
        TaskDescriptor("Gate", .anyKind, .sameAsInput, .model, .frame, "frames/Gate.frame.txt"),
        TaskDescriptor("Score", .anyKind, .sameAsInput, .model, .frame, "frames/Score.frame.txt"),
        TaskDescriptor("Judge", .anyKind, .sameAsInput, .model, .frame, "frames/Judge.frame.txt"),
        TaskDescriptor("Think", .anyKind, .sameAsInput, .model, .frame, "frames/Think.frame.txt"),
    ]
}

// MARK: - Families (catalog/families.py)

/// Task families — the section-grouping from `catflow-mlx/src/catflow/catalog/families.py`
/// (`FAMILIES` + `family_of`). Data only; used by the step picker (R7) and diagnostics.
nonisolated enum TaskFamilies {
    static let families: [String: [String]] = [
        "Files": ["Read Text", "Read Audio", "Read Image", "Read Video", "Read Images",
                  "Read Files", "Read PDF", "Read Index", "Save Text", "Save Audio",
                  "Save Image", "Save Video", "Save Images", "Store Index"],
        "Text": ["Split", "Template", "Join Text", "Filter", "Sort", "Dedupe",
                 "Extract", "Count", "Diff"],
        "Language models": ["Generate", "Decide", "Summarize", "Translate", "Answer",
                            "Rewrite", "Draft", "Ask", "Title", "Critique", "Verify",
                            "Revise", "Merge", "Extract Structured", "Classify", "Gate",
                            "Score", "Judge", "Think", "Improvise"],
        "Audio, image, video": ["Transcribe", "Speak", "Describe Image", "OCR", "Extract Frame",
                                "Extract Audio", "Trim", "Join Video", "Resize", "Crop",
                                "Convert", "Watermark", "Overlay Text", "Contact Sheet",
                                "Generate Image", "Edit Image", "Instruct Edit", "Inpaint",
                                "Upscale", "Detect Edges", "Estimate Depth", "Detect Pose",
                                "Segment", "Init Latent", "Encode Latent", "Decode Latent",
                                "Denoise", "Generate Video", "Animate", "Generate Sound", "Mux"],
        "Search and retrieval": ["Embed", "Retrieve", "Rerank", "Keyword Search"],
        "Data": ["Read CSV", "Read JSON", "Query Table", "Table to Text",
                 "Text to Table", "Calculate", "Compare", "Range", "Chart"],
        "Networked tools": ["Web Search", "Web Fetch", "Fetch Feed", "Download File", "HTTP Get"],
        "Humans": ["Ask Human", "Human Input"],
        "Context": ["Save Context", "Read Context", "Count Context"],
        "Staged effects": ["Stage Send", "Stage Post"],
        "Triggers": ["On File", "On Schedule", "On Flow"],
        "Stores": ["Store Query", "Store Read", "Store Write", "Set Field", "Append Row", "Merge Record"],
        "Model space": ["Load Checkpoint", "Blend", "Bake LoRA", "Pin Model"],
    ]

    static func family(of task: String) -> String? {
        for (family, tasks) in families where tasks.contains(task) {
            return family
        }
        return nil
    }
}
