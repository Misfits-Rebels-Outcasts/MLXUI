import Foundation
import Observation

/// CFM-R10-Events — the trigger arming session. A flow whose first row is `On File` /
/// `On Schedule` / `On Flow` can be **armed**: the app watches the folder, waits for the
/// schedule, or hooks the named flow's completion, and fires the flow with the occurrence
/// the interpreter turns into the trigger row's payload (P3-MD-01).
///
/// The friendly armed wording is the `g3` answer: "When a file is added to this folder
/// (trigger)", "Run every morning at 6:30 (trigger)", "Run after this flow finishes
/// (trigger)".
///
/// **Entitlement note (ships alone):** watching a folder *outside* the sandboxed app's
/// container needs the `files.user-selected.read-write` entitlement plus a security-scoped
/// bookmark (the flow's On File folder resolves through `FlowWorkspace`, which is the
/// container; a user-picked folder is a later step).
@Observable
final class FlowArmSession {

    enum TriggerKind: String { case onFile, onSchedule, onFlow, none }

    private(set) var isArmed = false
    /// The friendly armed description ("When a file is added to this folder (trigger)").
    private(set) var armedDescription: String?
    /// Recent firings, newest last ("06:30 tick", "inbox/orders.pdf", …).
    private(set) var firingLog: [String] = []
    /// The trigger kind of the armed flow's first row.
    private(set) var kind: TriggerKind = .none

    private var fileSource: DispatchSourceFileSystemObject?
    private var fileDescriptor: Int32 = -1
    private var scheduleTimer: Timer?
    private var flowName: String?
    private var fireCallback: ((FlowInterpreter.Occurrence) -> Void)?

    var isTriggerFlow: Bool { kind != .none }
    /// CFM-R10-FIX-1 (§14.4): whether the flow combines a trigger with a door. A flow that
    /// fires on events and runs fenced code must never run unattended — it is not armable.
    private(set) var armRefusal: String?
    var isArmable: Bool { isTriggerFlow && armRefusal == nil }

    /// The door in `doc` — an `Improvise` row, a `transforms:` call, or a `code`/`improvise`
    /// header flag — **at any depth, inside blocks, and through `uses:`** — or `nil` when
    /// there is none. §14.4 forbids arming (unattended execution) when one is present, and the
    /// returned sentence names where the door is. `workspace`/`flowID` resolve the `uses:`
    /// chain; without them only this file is inspected (CFM-R17-FIX-3).
    static func doorIn(_ doc: FlowDocument, workspace: FlowWorkspace? = nil, flowID: String? = nil) -> String? {
        // Header flags — this flow, plus any inherited through `uses:` when the chain resolves.
        let flags: [String] = (workspace != nil && flowID != nil)
            ? CapabilityGate.effectiveFlags(doc, workspace: workspace, flowID: flowID)
            : doc.flags.map(\.rawValue)
        if let flag = flags.first(where: { $0 == "code" || $0 == "improvise" }) {
            return "§14.4: this flow declares `\(flag)` and fires on events — it must never run unattended."
        }
        // Rows in this flow, including block children.
        if let sentence = doorInRows(doc.rows, transforms: doc.transforms, where: "this flow") {
            return sentence
        }
        // Used flows, recursively (their rows and their own nested `uses:`).
        if let workspace, let flowID, !doc.uses.isEmpty {
            for (name, used) in UsesResolver.resolve(doc, workspace: workspace, flowID: flowID) {
                if let sentence = usedFlowDoor(used, calledAs: name) { return sentence }
            }
        }
        return nil
    }

    /// The first `Improvise` row or `transforms:` call in `rows` or any block child.
    private static func doorInRows(_ rows: [Row], transforms: [String: TransformDef],
                                   where label: String) -> String? {
        for row in rows {
            if row.task == "Improvise" {
                return "§14.4: \(label) runs an `Improvise` row and fires on events — it must never run unattended."
            }
            if let task = row.task, transforms[task] != nil {
                return "§14.4: \(label) calls the transform `\(task)` and fires on events — it must never run unattended."
            }
            if !row.children.isEmpty,
               let nested = doorInRows(row.children, transforms: transforms, where: label) {
                return nested
            }
        }
        return nil
    }

    /// An `Improvise` row inside a used flow (its rows, block children, and nested `uses:`).
    /// A used flow's `transforms:` section is dropped by `UsesResolver`, so a transform call
    /// there surfaces to `canRun` as an unknown task, not here.
    private static func usedFlowDoor(_ used: FlowInterpreter.UsedFlow, calledAs name: String) -> String? {
        func scan(_ rows: [Row]) -> Bool {
            for row in rows {
                if row.task == "Improvise" { return true }
                if !row.children.isEmpty, scan(row.children) { return true }
            }
            return false
        }
        if scan(used.rows) {
            return "§14.4: the used flow `\(name)` runs an `Improvise` row — a trigger flow must never run one unattended."
        }
        for (sub, nested) in used.nested {
            if let sentence = usedFlowDoor(nested, calledAs: "\(name) → \(sub)") { return sentence }
        }
        return nil
    }

    /// Establish the trigger kind (and the §14.4 refusal, when a door is present) without
    /// starting any watcher — the header's Arm button visibility depends on it.
    func inspect(doc: FlowDocument, workspace: FlowWorkspace? = nil, flowID: String? = nil) {
        guard let first = doc.rows.first, let settings = first.settings,
              let task = first.task, task.hasPrefix("On ") else {
            kind = .none
            return
        }
        kind = .none
        armRefusal = nil
        if task == "On File" || task == "On Schedule" || task == "On Flow" {
            if let door = Self.doorIn(doc, workspace: workspace, flowID: flowID) {
                kind = .onFile
                armRefusal = door
            } else {
                switch task {
                case "On File": kind = .onFile
                case "On Schedule": kind = .onSchedule
                case "On Flow": kind = .onFlow
                default: break
                }
                let s = FlowSettings(settings)
                armedDescription = task == "On File" ? "When a file is added to this folder (trigger)"
                    : task == "On Schedule" ? friendlySchedule(s)
                    : "Run after \(s.firstBare() ?? "another flow") finishes (trigger)"
            }
        }
    }

    // MARK: - Arming

    /// Arm `doc` if its first row is a trigger. `onFire` runs the flow with the occurrence.
    /// Returns false (refusing) when the flow carries a door — §14.4: never unattended.
    @discardableResult
    func arm(flowID: String, doc: FlowDocument, workspace: FlowWorkspace,
             onFire: @escaping (FlowInterpreter.Occurrence) -> Void) -> Bool {
        disarm()
        fireCallback = onFire
        guard let first = doc.rows.first, let settings = first.settings else {
            kind = .none
            return false
        }
        let s = FlowSettings(settings)
        switch first.task {
        case "On File", "On Schedule", "On Flow":
            if let door = Self.doorIn(doc, workspace: workspace, flowID: flowID) {
                // §14.4: this flow would run fenced code unattended — refuse to arm.
                kind = .none
                armRefusal = door
                isArmed = false
                return false
            }
        default:
            kind = .none
            return false
        }
        switch first.task {
        case "On File":
            kind = .onFile
            armedDescription = "When a file is added to this folder (trigger)"
            armFileWatcher(settings: s, flowID: flowID, workspace: workspace)
        case "On Schedule":
            kind = .onSchedule
            armedDescription = friendlySchedule(s)
            armSchedule(settings: s)
        case "On Flow":
            kind = .onFlow
            flowName = s.firstBare()
            armedDescription = "Run after \(flowName ?? "another flow") finishes (trigger)"
            // The hook is `notifyFlowCompleted(_:)` — the app calls it when a run ends.
        default:
            kind = .none
        }
        isArmed = kind != .none
        return isArmed
    }

    func disarm() {
        fileSource?.cancel()
        fileSource = nil
        if fileDescriptor >= 0 {
            close(fileDescriptor)
            fileDescriptor = -1
        }
        scheduleTimer?.invalidate()
        scheduleTimer = nil
        flowName = nil
        fireCallback = nil
        isArmed = false
        kind = .none
        armedDescription = nil
    }

    /// On Flow hook: when a run of `completedFlowID` finishes, fire the armed flow.
    func notifyFlowCompleted(_ completedFlowID: String) {
        guard isArmed, kind == .onFlow, let expected = flowName, expected == completedFlowID else { return }
        fire(FlowInterpreter.Occurrence(path: nil, tick: timestamp(), payload: nil),
             label: "after \(expected)")
    }

    /// A manual fire (the armed flow's "Fire now" affordance).
    func fireNow() {
        guard isArmed else { return }
        switch kind {
        case .onFile:
            // Re-scan the watched folder and fire for the newest matching file.
            if let file = lastMatchingFile {
                fire(FlowInterpreter.Occurrence(path: file, tick: nil, payload: nil),
                     label: file.lastPathComponent)
            } else {
                fire(FlowInterpreter.Occurrence(path: nil, tick: timestamp(), payload: nil),
                     label: "manual")
            }
        case .onSchedule:
            fire(FlowInterpreter.Occurrence(path: nil, tick: timestamp(), payload: nil), label: timestamp())
        case .onFlow:
            fire(FlowInterpreter.Occurrence(path: nil, tick: timestamp(), payload: nil), label: "manual")
        case .none:
            break
        }
    }

    // MARK: - The watchers

    private func armFileWatcher(settings: FlowSettings, flowID: String, workspace: FlowWorkspace) {
        guard let rawPath = settings.pathValue(), let folder = try? workspace.resolve(rawPath, flowID: flowID) else {
            kind = .none
            return
        }
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        watchedFolder = folder
        let fd = open(folder.path, O_EVTONLY)
        guard fd >= 0 else { return }
        fileDescriptor = fd
        let source = DispatchSource.makeFileSystemObjectSource(fileDescriptor: fd,
                                                               eventMask: [.write, .rename],
                                                               queue: .global(qos: .utility))
        fileSource = source
        source.setEventHandler { [weak self] in
            self?.folderChanged(settings: settings, folder: folder)
        }
        source.setCancelHandler { close(fd) }
        source.resume()
    }

    /// The settle delay (`settle=30s`) and the pattern (`pattern=*.pdf`).
    private var settleSeconds: TimeInterval {
        // Stored when the On File row arms.
        _settleSeconds
    }

    private var _settleSeconds: TimeInterval = 0
    private var _pattern: String?

    private func folderChanged(settings: FlowSettings, folder: URL) {
        _settleSeconds = settings.value(for: "settle").flatMap { durationSeconds($0) } ?? 0
        _pattern = settings.value(for: "pattern")
        let settle = _settleSeconds
        let fireBlock = { [weak self] in
            guard let self else { return }
            if let file = self.lastMatchingFile {
                self.fire(FlowInterpreter.Occurrence(path: file, tick: nil, payload: nil),
                          label: file.lastPathComponent)
            }
        }
        if settle > 0 {
            DispatchQueue.global().asyncAfter(deadline: .now() + settle, execute: fireBlock)
        } else {
            fireBlock()
        }
    }

    /// The newest file in the watched folder matching `pattern=`.
    private var lastMatchingFile: URL? {
        guard fileSource != nil else { return nil }
        // The watched folder path is the file descriptor's original; recover from the
        // pending occurrence bookkeeping by re-walking the settings path stored on arm.
        guard let folder = watchedFolder else { return nil }
        let files = (try? FileManager.default.contentsOfDirectory(at: folder,
                                                                  includingPropertiesForKeys: [.contentModificationDateKey],
                                                                  options: [.skipsHiddenFiles])) ?? []
        let pattern = _pattern ?? "*"
        let matching = files.filter { glob($0.lastPathComponent, pattern: pattern) }
        return matching.sorted { a, b in
            let ad = try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
            let bd = try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
            return (ad ?? .distantPast) > (bd ?? .distantPast)
        }.first
    }

    private var watchedFolder: URL?

    private func armSchedule(settings: FlowSettings) {
        guard let at = settings.value(for: "at") else {
            kind = .none
            return
        }
        let next = nextFireDate(at: at)
        let timer = Timer(fire: next, interval: 86400, repeats: true) { [weak self] _ in
            self?.fire(FlowInterpreter.Occurrence(path: nil, tick: self?.timestamp(), payload: nil),
                       label: self?.timestamp() ?? "tick")
        }
        scheduleTimer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    // MARK: - Fire

    private func fire(_ occurrence: FlowInterpreter.Occurrence, label: String) {
        firingLog.append(label)
        if firingLog.count > 20 { firingLog.removeFirst(firingLog.count - 20) }
        fireCallback?(occurrence)
    }

    private func timestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm"
        return f.string(from: Date())
    }

    /// `every=day; at=06:30` → the next `HH:MM` (the friendly wording, g3).
    private func friendlySchedule(_ settings: FlowSettings) -> String {
        let every = settings.value(for: "every", default: "day") ?? "day"
        let at = settings.value(for: "at") ?? ""
        switch every {
        case "day": return "Run every morning at \(at) (trigger)"
        case "hour": return "Run every hour (trigger)"
        default: return "Run \(every) at \(at) (trigger)"
        }
    }

    /// The next `Date` at `HH:MM` (today if still ahead, else tomorrow).
    private func nextFireDate(at: String) -> Date {
        let parts = at.split(separator: ":").compactMap { Int($0) }
        let hour = parts.first ?? 0
        let minute = parts.count > 1 ? parts[1] : 0
        let cal = Calendar.current
        var comps = cal.dateComponents([.year, .month, .day], from: Date())
        comps.hour = hour
        comps.minute = minute
        var date = cal.date(from: comps) ?? Date()
        if date <= Date() {
            date = cal.date(byAdding: .day, value: 1, to: date) ?? date
        }
        return date
    }

    /// `settle=30s` → 30.
    private func durationSeconds(_ raw: String) -> TimeInterval? {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        if let s = Double(trimmed) { return s }
        if trimmed.hasSuffix("s"), let s = Double(trimmed.dropLast()) { return s }
        if trimmed.hasSuffix("m"), let s = Double(trimmed.dropLast()) { return s * 60 }
        return nil
    }

    /// A minimal `fnmatch`-style glob (`*` and `?`).
    private func glob(_ name: String, pattern: String) -> Bool {
        var regex = "^"
        for c in pattern {
            switch c {
            case "*": regex += ".*"
            case "?": regex += "."
            default: regex += NSRegularExpression.escapedPattern(for: String(c))
            }
        }
        regex += "$"
        return name.range(of: regex, options: .regularExpression) != nil
    }
}
