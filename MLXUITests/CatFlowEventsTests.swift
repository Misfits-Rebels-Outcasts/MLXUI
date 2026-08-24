import Testing
import Foundation
@testable import MLXUI

/// CFM-R10-Events — trigger arming. `On File` / `On Schedule` / `On Flow` flows are now
/// runnable; the `FlowArmSession` arms them and fires the flow with an occurrence the
/// interpreter turns into the trigger row's payload.
struct CatFlowEventsTests {

    private func parse(_ text: String) throws -> FlowDocument {
        try CatParser.parse(text)
    }

    // MARK: - canRun

    @Test func triggerFlowsAreRunnable() throws {
        let doc = try parse("catflow 0.8; events\n1. On File   inbox/; pattern=*.pdf\n2. Save Text   out.md")
        #expect(FlowRunner.canRun(doc) == .runnable)
    }

    @Test func inboxIngestIsRunnableNow() throws {
        let doc = try GalleryLoader.loadDocument(flowID: "51-InboxIngest")
        #expect(FlowRunner.canRun(doc) == .runnable)
        let meta = try #require(GalleryLoader.loadMetadata().first { $0.flowID == "51-InboxIngest" })
        #expect(meta.isRunnable)
    }

    // MARK: - Arming

    @Test func armingAnOnFileFlowStartsTheWatcher() throws {
        let doc = try parse("catflow 0.8; events\n1. On File   inbox/; pattern=*.pdf\n2. Save Text   out.md")
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-events-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: root)
        let arm = FlowArmSession()
        var fired: FlowInterpreter.Occurrence?
        arm.arm(flowID: "t", doc: doc, workspace: ws) { fired = $0 }
        #expect(arm.isArmed)
        #expect(arm.kind == .onFile)
        #expect(arm.armedDescription == "When a file is added to this folder (trigger)")
        arm.disarm()
        #expect(!arm.isArmed)
    }

    @Test func armingAnOnScheduleFlowUsesTheFriendlyWording() throws {
        let doc = try parse("catflow 0.8; events\n1. On Schedule   every=day; at=06:30\n2. Save Text   out.md")
        let arm = FlowArmSession()
        arm.arm(flowID: "t", doc: doc,
                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory)) { _ in }
        #expect(arm.isArmed)
        #expect(arm.kind == .onSchedule)
        #expect(arm.armedDescription == "Run every morning at 06:30 (trigger)")
        arm.disarm()
    }

    @Test func armingAnOnFlowFlowUsesTheNamedFlow() throws {
        let doc = try parse("catflow 0.8; events\n1. On Flow   OtherFlow.cat\n2. Save Text   out.md")
        let arm = FlowArmSession()
        arm.arm(flowID: "t", doc: doc,
                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory)) { _ in }
        #expect(arm.armedDescription == "Run after OtherFlow.cat finishes (trigger)")
        var fired = false
        arm.arm(flowID: "t", doc: doc,
                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory)) { _ in fired = true }
        arm.notifyFlowCompleted("OtherFlow.cat")
        #expect(fired)
        arm.disarm()
    }

    @Test func aNonTriggerFlowDoesNotArm() throws {
        let doc = try parse("catflow 0.8\n1. Read Text   memo.txt\n2. Save Text   out.md")
        let arm = FlowArmSession()
        arm.arm(flowID: "t", doc: doc,
                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory)) { _ in }
        #expect(!arm.isArmed)
        #expect(arm.kind == .none)
    }

    // MARK: - CFM-R10-FIX-1: arming a door-carrying trigger flow refuses (§14.4)

    @Test func armingRefusesATriggerFlowCarryingImprovise() throws {
        let doc = try parse("catflow 0.8; events\n1. On File   inbox/\n2. Improvise   \"fix the headers\"")
        let arm = FlowArmSession()
        arm.inspect(doc: doc)
        #expect(arm.isTriggerFlow)
        let fired = arm.arm(flowID: "t", doc: doc,
                            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory)) { _ in
            Issue.record("a door-carrying trigger flow must never fire unattended")
        }
        #expect(fired == false)
        #expect(!arm.isArmed)
        #expect(!arm.isArmable)
        #expect(arm.armRefusal?.contains("unattended") == true)
    }

    @Test func armingRefusesATriggerFlowCarryingTransforms() throws {
        var doc = try parse("catflow 0.8; events\n1. On File   inbox/\n2. CustomTool")
        doc.transforms = ["CustomTool": TransformDef(name: "CustomTool", signature: nil, run: "echo hi",
                                                     timeout: nil, workdir: nil, params: [])]
        let arm = FlowArmSession()
        let fired = arm.arm(flowID: "t", doc: doc,
                            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory)) { _ in }
        #expect(fired == false)
        #expect(arm.armRefusal != nil)
    }

    @Test func armingRefusesATriggerFlowWithTheImproviseHeaderFlag() throws {
        var doc = try parse("catflow 0.8; events; improvise\n1. On Schedule   at=06:30\n2. Save Text   out.md")
        doc.flags = [.events, .improvise]
        let arm = FlowArmSession()
        let fired = arm.arm(flowID: "t", doc: doc,
                            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory)) { _ in }
        #expect(fired == false)
        #expect(arm.armRefusal != nil)
    }

    // MARK: - CFM-R10-FIX-1: the runner consults canRun (no UI path around it)

    @Test func runnerRefusesADoorFlowBeforeStarting() async throws {
        // In the App Store tier, a `transforms:` flow is refused by canRun — `FlowRunner.run`
        // must surface that as a `.failed`, never start the run.
        var doc = try parse("catflow 0.8\n1. CustomTool\n2. Save Text   out.md")
        doc.transforms = ["CustomTool": TransformDef(name: "CustomTool", signature: nil, run: "echo hi",
                                                     timeout: nil, workdir: nil, params: [])]
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-fix1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let context = FlowRunner.RunContext(flowID: "t", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: mock)
        var failed = false
        for await event in FlowRunner().run(doc, context: context) {
            if case .failed = event { failed = true }
        }
        #expect(failed)
    }

    // MARK: - The interpreter runs a trigger with an occurrence

    @Test func onFileRowPassesTheOccurrenceThrough() async throws {
        let doc = try parse("catflow 0.8; events\n1. On File   inbox/\n2. Save Text   out.md")
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-events-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let occurrence = FlowInterpreter.Occurrence(path: base.appendingPathComponent("inbox/order.pdf"),
                                                    tick: nil, payload: nil)
        let events = try await FlowInterpreter.run(doc, executor: mock, occurrence: occurrence)
        #expect(events.contains { $0.kind == .rowCompleted && $0.path == "1" })
        #expect(events.last?.kind == .runCompleted)
    }
}
