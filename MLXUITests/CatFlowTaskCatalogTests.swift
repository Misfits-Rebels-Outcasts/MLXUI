import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R2-1: the `TaskCatalog` data port of `catflow-mlx/src/catflow/catalog/tasks.py`
/// (CATALOG 106 + DECIDER_TASKS 6 = allTasks 112) and `catalog/families.py`. Counts,
/// histogram, and the ten spot-checks are verified against the Python by running the same
/// queries through `catflow/catalog/tasks.py`. See `RSI/DelegateMergeBacklog.md` CFM-R2-1.
struct CatFlowTaskCatalogTests {

    // MARK: - Counts

    @Test func catalogCounts() {
        #expect(TaskCatalog.catalog.count == 106)
        #expect(TaskCatalog.deciderTasks.count == 6)
        #expect(TaskCatalog.allTasks().count == 112)
    }

    @Test func noDuplicateTaskNames() {
        let names = TaskCatalog.allTasks().map(\.name)
        #expect(Set(names).count == names.count)
    }

    // MARK: - Class histogram (verified against the Python)

    @Test func classHistogram() {
        let histogram = Dictionary(grouping: TaskCatalog.allTasks(), by: \.taskClass)
            .mapValues(\.count)
        #expect(histogram == [
            .instant: 55, .model: 44, .net: 5, .trigger: 3,
            .human: 2, .staged: 2, .agent: 1,
        ])
    }

    @Test func decidersAreModelClassHeldSeparately() {
        let deciderNames = Set(TaskCatalog.deciderTasks.keys)
        #expect(deciderNames == ["Classify", "Decide", "Gate", "Judge", "Score", "Think"])
        // Deciders are model tasks, but not in CATALOG.
        for name in deciderNames {
            #expect(TaskCatalog.deciderTasks[name]?.taskClass == .model)
            #expect(TaskCatalog.catalog[name] == nil)
        }
    }

    // MARK: - Spot-checks (from the backlog; expected values from the Python)

    @Test func spotCheckReadAudio() {
        let d = try! #require(TaskCatalog.get("Read Audio"))
        #expect(d.accepts == .single(.file))
        #expect(d.gives == .single(.audio))
        #expect(d.taskClass == .instant)
        #expect(d.refKind == .tool)
        #expect(d.refName == "tools.files.read_audio")
    }

    @Test func spotCheckSaveText() {
        let d = try! #require(TaskCatalog.get("Save Text"))
        #expect(d.accepts == .anyKind)
        #expect(d.gives == .single(.status))
        #expect(d.taskClass == .instant)
    }

    @Test func spotCheckStoreIndex() {
        let d = try! #require(TaskCatalog.get("Store Index"))
        #expect(d.accepts == .tupleOf([.text, .vector]))
        #expect(d.gives == .single(.index))
        #expect(d.taskClass == .instant)
    }

    @Test func spotCheckSplit() {
        let d = try! #require(TaskCatalog.get("Split"))
        #expect(d.accepts == .single(.text))
        #expect(d.gives == .listOf(.text))
        #expect(d.taskClass == .instant)
    }

    @Test func spotCheckTemplate() {
        let d = try! #require(TaskCatalog.get("Template"))
        #expect(d.accepts == .listOf(.text))
        #expect(d.gives == .single(.text))
        #expect(d.taskClass == .instant)
    }

    @Test func spotCheckSummarizeIsFrameBacked() {
        let d = try! #require(TaskCatalog.get("Summarize"))
        #expect(d.accepts == .single(.text))
        #expect(d.gives == .single(.text))
        #expect(d.taskClass == .model)
        #expect(d.refKind == .frame)
        #expect(d.refName == "frames/Summarize.frame.txt")
    }

    @Test func spotCheckTranscribe() {
        let d = try! #require(TaskCatalog.get("Transcribe"))
        #expect(d.accepts == .single(.audio))
        #expect(d.gives == .single(.text))
        #expect(d.taskClass == .model)
        #expect(d.refKind == .engine)
        #expect(d.refName == "engines.asr.transcribe")
    }

    @Test func spotCheckSpeak() {
        let d = try! #require(TaskCatalog.get("Speak"))
        #expect(d.accepts == .single(.text))
        #expect(d.gives == .single(.audio))
        #expect(d.taskClass == .model)
        #expect(d.refKind == .engine)
    }

    @Test func spotCheckSegment() {
        let d = try! #require(TaskCatalog.get("Segment"))
        #expect(d.accepts == .single(.image))
        #expect(d.gives == .single(.image))
        #expect(d.taskClass == .model)
        #expect(d.refKind == .engine)
    }

    @Test func spotCheckGate() {
        let d = try! #require(TaskCatalog.get("Gate"))
        #expect(d.accepts == .anyKind)
        #expect(d.gives == .sameAsInput)
        #expect(d.taskClass == .model)
        #expect(d.refKind == .frame)
        #expect(d.refName == "frames/Gate.frame.txt")
    }

    @Test func unknownTaskIsNil() {
        #expect(TaskCatalog.get("Nope") == nil)
    }

    // MARK: - Families (catalog/families.py)

    @Test func familiesCoverEveryTask() {
        let allNames = Set(TaskCatalog.allTasks().map(\.name))
        let familyTasks = Set(TaskFamilies.families.values.flatMap { $0 })
        #expect(familyTasks == allNames, "every task must belong to exactly one family")
    }

    @Test func familyOfKnownAndUnknown() {
        #expect(TaskFamilies.family(of: "Summarize") == "Language models")
        #expect(TaskFamilies.family(of: "Read Audio") == "Files")
        #expect(TaskFamilies.family(of: "Web Search") == "Networked tools")
        #expect(TaskFamilies.family(of: "Nope") == nil)
    }
}
