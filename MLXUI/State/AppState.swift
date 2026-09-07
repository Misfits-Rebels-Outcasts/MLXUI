import SwiftUI
import Observation
import UniformTypeIdentifiers

@Observable
final class AppState {
    // ── Optional domain hiding ──────────────────────────────────────────────────
    /// Set to `true` to hide the "Video Generation" domain (and every model listed
    /// under it, e.g. WAN 2.1) from the catalog UI. Flip back to `false` to restore.
    static let hideVideoGeneration = true

    /// Set to `true` to hide the "Image Segmentation" domain (and every model listed
    /// under it, e.g. SAM3) from the catalog UI. Mirrors `hideVideoGeneration`.
    static let hideImageSegmentation = false

    /// Set to `true` to hide the "Image Upscaling" domain (and every model listed
    /// under it, e.g. SeedVR2 3B) from the catalog UI. Mirrors `hideVideoGeneration`.
    static let hideSeedVR2 = true

    /// Set to `true` to hide the entire "Flows" section (sidebar, gallery, run views).
    /// When `true`, `galleryEntries` is empty and the section does not render — the app
    /// returns to its pre-Flows behavior exactly. Mirrors `hideVideoGeneration`.
    static let hideFlows = false

    /// Set to `true` to hide the **Automate → AI Workflows** gallery specifically (the
    /// sidebar section and the bundled flow badges), independently of the rest of Flows.
    /// When `true`, `galleryEntries` is empty. Mirrors `hideFlows`.
    static let hideAutomate = false

    /// Set to `true` to hide the "My Workflows" shelf (the user's own saved flows and the
    /// New Flow badge) on the Automate → AI Workflows page. The two bundled shelves —
    /// Basic Gallery and Advance Gallery — still render.
    static let hideMyWorkflows = true

    /// Set to `true` to hide the "My Workspace" shelf (the New Workspace badge and every
    /// workspace, bundled or saved) on the Automate → AI Workflows page. Flip to `false`
    /// to show the section. Mirrors `hideMyWorkflows`.
    static let hideMyWorkspace = true

    /// Set to `true` to hide the "Rerank" domain (and its one model, Qwen3-Reranker-0.6B)
    /// from Browse → Infrastructure. The Infrastructure section itself still renders while
    /// Embeddings has models. Mirrors `hideVideoGeneration`.
    static let hideQwenReranker = true

    /// Set to `true` to hide the "Qwen3.5-9B" model badge from Browse → Chat & Text. Only
    /// the text (`llm`) card is hidden — "Qwen3.5-9B Vision" under Vision is a separate
    /// entry and is unaffected. Mirrors `hideVideoGeneration`.
    static let hideQwen359B = true

    /// Set to `true` to hide the "Advance Gallery" shelf (every bundled flow that isn't
    /// basic) on the Automate → AI Workflows page. My Workflows and Basic Gallery still
    /// render.
    static let hideAdvanceGallery = true

    /// Hide individual gallery flows (badges) by gallery number. Numbers are the
    /// `_metadata.json` `number` field (1–71 today), stable across renames. Basic Gallery's
    /// filenames restart at `1-`/`2-` for a friendly per-shelf look, but their `number` values
    /// (70, 71) stay in the single global sequence — hiding by number never accidentally
    /// catches both an Advance Gallery flow and a same-numbered Basic Gallery one. Ranges
    /// read naturally — hide flows 60–69:
    /// `static let hiddenFlowNumbers: Set<Int> = Set(60...69)`
    /// …and also hide flow 50:
    /// `static let hiddenFlowNumbers: Set<Int> = [50] + Set(60...69)` (or `Set([50])`).
    /// Empty = show everything.
    //static let hiddenFlowNumbers: Set<Int> = Set(60...69)
    static let hiddenFlowNumbers: Set<Int> = []
    
    /// The bundled gallery flows, in gallery order. Empty when `hideFlows` is true.
    var galleryEntries: [GalleryFlowMetadata] = []

    /// The "Basic Gallery" shelf — the simple bundled flows, in gallery order.
    var basicGalleryEntries: [GalleryFlowMetadata] {
        galleryEntries.filter(\.isBasic)
    }

    /// The "Advance Gallery" shelf — every bundled flow that isn't basic.
    var advanceGalleryEntries: [GalleryFlowMetadata] {
        galleryEntries.filter { !$0.isBasic }
    }

    /// CFM-R12-1: the user's saved flows (the "My Workflows" shelf), newest first. Loaded by
    /// `reloadUserFlows()` — never trusted to be current across an editor save.
    private(set) var userFlowEntries: [UserFlowStore.Entry] = []

    /// CFM-R17-3: the workspaces on the "My Workflows" shelf — every directory under
    /// `workspaces/` that holds one or more `.cat` files, plus the bundled ones materialised
    /// on first launch. Loaded by `reloadWorkspaces()`.
    private(set) var workspaceEntries: [WorkspaceStore.Workspace] = []
    /// A workspace whose page is pushed onto the detail stack (CFM-R17-3).
    var selectedWorkspace: WorkspaceStore.Workspace?
    var workspaceImportError: String?
    var workspaceRemoveError: String?

    /// CFM-R17-FIX-1: bundled workspace ids the user has removed. Mirrored from `UserDefaults`
    /// so the shelf and the "Restore bundled workspaces" affordance react; `reloadWorkspaces()`
    /// will not re-materialise these. A *user* workspace never has an entry here.
    private(set) var removedBundledWorkspaceIDs: Set<String> = []

    /// CFM-R12-4: the bundled flows `FlowRunner.canRun` refuses — the gallery's ⚠ badge tells
    /// the truth for every blocked flow (net/staged/agent channels, unported instant tools),
    /// not just the ~10 the metadata's `notRunnableReason` happens to name. Computed on
    /// gallery appear (`refreshGalleryBlocked()`), never at launch.
    private(set) var galleryBlocked: Set<String> = []

    var browserData: BrowserData?
    var loadError: String?
    var systemInfo = SystemInfo.detect()

    // Sidebar
    var selectedSection: SidebarItem = .home

    // Filters
    var filterSource: ModelSource?
    var filterRAMLimitGB: Double
    var filterCapabilities: Set<String> = []
    var sortOrder: SortOrder = .ramLowToHigh

    // Navigation
    var selectedModel: ModelEntry?

    // Run / chat — non-nil presents the Run chat sheet for this model
    var runningModel: ModelEntry?

    // Non-nil presents the registry-driven run sheet (e.g. Whisper ASR) for this model.
    var asrRunModel: ModelEntry?

    // Comparison
    var comparisonSet: [ModelEntry] = []

    // Installation
    var installManager = InstallManager()
    var modelRunner = ModelRunner()

    // Model module registry — resolves non-LLM models (e.g. Whisper ASR) to their run UI.
    let registry = ModelRegistry()
    var installedModelIDs: Set<String> = []

    /// CFM-R14-2 — the catalog models the registry can claim (`bestModule(for:) != nil`),
    /// precomputed once after `browserData` loads. The nonisolated model-pool derivation
    /// (`TaskModels.derivedModels`) reads this instead of touching the `@MainActor` registry —
    /// the deliberate choice the R14-2 contract offers ("AppState precomputes a claim table
    /// the nonisolated path can read").
    var claimableModelIDs: Set<String> = []
    var installedModels: InstalledModels?
    private let installedURL: URL

    // Search
    var showCommandPalette = false
    var searchQuery = ""

    // Settings sheet
    var showSettings = false

    // Automate → AI Workflows: a gallery flow chosen from the badge grid. Non-nil pushes
    // that flow's detail (title, rows, inspector) onto the detail NavigationStack.
    var selectedFlow: FlowSelection?

    /// Non-nil surfaces a "Remove Flow" failure (unreadable / access denied) as an alert.
    var flowRemoveError: String?

    /// Non-nil surfaces a My Workflows Import failure as an alert.
    var flowImportError: String?

    /// Non-nil surfaces a My Workflows Export failure as an alert.
    var flowExportError: String?

    /// CFM-R11-0: a flow the editor is editing — nil document = a fresh flow (the old
    /// "New Flow" route). Non-nil pushes the editor onto the detail stack.
    var editingFlow: FlowEditTarget?

    /// A stable identity for the editor's navigation destination.
    struct EditorNavigation: Hashable { var id = UUID() }

    /// Flow ids whose required models are currently downloading. Lives here (not in the
    /// per-view `FlowRunSession`) so the "Installing Models" state survives navigating away
    /// from the flow and back (smoke-30 finding).
    var installingFlowIDs: Set<String> = []

    // R5-5: a `.cat` file the user opened from disk. Non-nil shows `OpenedFlowView`
    // in the Flows detail area. Read-only — parsed + validated, never written.
    var openedCatFlow: OpenedCatFlow?
    var showOpenCatPanel = false
    /// Non-nil surfaces an Open .cat… failure (unreadable / access denied / invalid).
    var openCatFlowError: String?

    init() {
        filterRAMLimitGB = SystemInfo.detect().totalRAMGB
        installedURL = ModelStore.shared.installedRegistryURL
        loadInstalledModels()
        for module in installedModules { module.register(into: registry) }
        if !Self.hideFlows, !Self.hideAutomate {
            galleryEntries = GalleryLoader.loadMetadata()
                .filter { !Self.hiddenFlowNumbers.contains($0.number) }
        }
        reloadUserFlows()
        removedBundledWorkspaceIDs = BundledWorkspaceTombstones.load()
        reloadWorkspaces()
    }

    /// CFM-R12-1: refresh the "My Workflows" shelf from the flow folder. Called at launch and
    /// every time the gallery reappears, so a just-saved or just-duplicated flow shows up.
    func reloadUserFlows() {
        let bundled = Set(galleryEntries.map(\.flowID))
        userFlowEntries = UserFlowStore.scan(workspace: FlowWorkspace.shared,
                                             bundledFlowIDs: bundled)
    }

    /// CFM-R17-3: materialise every bundled workspace into `workspaces/` (idempotent — a
    /// user's edits are left alone), then refresh the workspace shelf. A bundled workspace's
    /// scratch copy lists like any other, so it is **not** excluded here (unlike a gallery
    /// flow's working directory) — it is a real, editable workspace the user can run.
    func reloadWorkspaces() {
        let ws = FlowWorkspace(root: ModelStore.shared.workspacesDirectory)
        // CFM-R17-FIX-1: a tombstoned bundled workspace is not re-materialised — Remove means
        // removed until the user restores it.
        BundledWorkspaces.prepareAll(into: ws, removed: removedBundledWorkspaceIDs)
        workspaceEntries = WorkspaceStore.scan(workspace: ws)
    }

    /// Delete a workspace's whole directory, refresh the shelf, and clear any navigation
    /// pointing into it. For a *bundled* workspace this also writes a tombstone
    /// (CFM-R17-FIX-1) so `reloadWorkspaces()` does not put it straight back; the user brings
    /// it back with `restoreBundledWorkspaces()`.
    func removeWorkspace(workspaceID: String) {
        do {
            try WorkspaceStore.remove(workspaceID: workspaceID,
                                      workspace: FlowWorkspace(root: ModelStore.shared.workspacesDirectory))
            if BundledWorkspaces.isBundled(workspaceID) {
                removedBundledWorkspaceIDs.insert(workspaceID)
                BundledWorkspaceTombstones.save(removedBundledWorkspaceIDs)
            }
            reloadWorkspaces()
            if selectedWorkspace?.workspaceID == workspaceID { selectedWorkspace = nil }
            if selectedFlow?.workspace?.workspaceID == workspaceID { selectedFlow = nil }
            if editingFlow?.workspace?.workspaceID == workspaceID { editingFlow = nil }
        } catch {
            workspaceRemoveError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
        }
    }

    /// CFM-R17-FIX-1: bring back every bundled workspace the user has removed. Their flows and
    /// stock corpus are re-materialised on the next `reloadWorkspaces()`; a bundled workspace
    /// the user still has on disk (never removed) is untouched, as is every user workspace.
    func restoreBundledWorkspaces() {
        guard !removedBundledWorkspaceIDs.isEmpty else { return }
        removedBundledWorkspaceIDs.removeAll()
        BundledWorkspaceTombstones.save([])
        reloadWorkspaces()
    }

    /// Delete a user flow's folder from disk, refresh the shelf, and clear any navigation
    /// (detail or editor) that points at it — the current view pops back to the gallery.
    /// Only user flows reach here: the My Workflows shelf and a user flow's list.
    func removeUserFlow(flowID: String) {
        do {
            try UserFlowStore.remove(flowID: flowID, workspace: FlowWorkspace.shared)
            reloadUserFlows()
            if selectedFlow?.flowID == flowID { selectedFlow = nil }
            if editingFlow?.flowID == flowID { editingFlow = nil }
        } catch {
            flowRemoveError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
        }
    }

    /// CFM-R12-4: recompute which bundled flows are refused (the ⚠ badges) from the **live**
    /// gates — `FlowRunner.canRun` + the model preflight, never a stale metadata string
    /// (CFM-R12-FIX-1). Called when the gallery appears.
    func refreshGalleryBlocked() {
        let catalog = browserData?.domains.flatMap { $0.allModels } ?? []
        var blocked: Set<String> = []
        for flow in galleryEntries {
            if let doc = try? GalleryLoader.loadDocument(flowID: flow.flowID),
               FlowRunnability.refusalReason(for: doc, catalog: catalog,
                                             installed: installedModelIDs,
                                             totalRAMGB: systemInfo.totalRAMGB) != nil {
                blocked.insert(flow.flowID)
            }
        }
        galleryBlocked = blocked
    }

    // ── Visible sidebar sections ──
    var visibleSections: [SidebarSection] {
        guard let data = browserData else { return [] }
        return data.sidebarSections.filter { $0.modelCount(in: data) > 0 }
    }

    /// Domain ids for a browse section, minus domains hidden by `hideVideoGeneration` /
    /// `hideImageSegmentation` / `hideSeedVR2` / `hideQwenReranker`.
    private func domainIDsForSection(_ sectionID: String) -> [String] {
        guard let data = browserData else { return [] }
        let ids = data.sidebarSections.first(where: { $0.id == sectionID })?.domainIds ?? []
        return ids.filter {
            !(Self.hideVideoGeneration && $0 == "videogen")
                && !(Self.hideImageSegmentation && $0 == "segmentation")
                && !(Self.hideSeedVR2 && $0 == "upscale")
                && !(Self.hideQwenReranker && $0 == "rerank")
        }
    }

    // ── Sources actually present in the catalog ──
    // The bundled catalog is mlx-only today, so the source picker collapses to a
    // single option (and the UI hides it). Computed from the data so it adapts if
    // non-mlx models are ever added.
    var availableSources: [ModelSource] {
        guard let data = browserData else { return [] }
        let present = Set(data.domains.flatMap { $0.allModels }.map { $0.source })
        return [.mlx, .coreai, .coreml, .research].filter { present.contains($0) }
    }

    // ── Installed models (catalog entries the user has downloaded) ──
    var installedEntries: [ModelEntry] {
        guard let data = browserData else { return [] }
        var seen = Set<String>()
        return data.domains.flatMap { $0.allModels }
            .filter { installedModelIDs.contains($0.id) }
            .filter { seen.insert($0.id).inserted }
            .sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }

    // ── RAM slider upper bound, scaled to this machine ──
    var ramSliderMax: Double {
        max(8, systemInfo.totalRAMGB.rounded(.up))
    }

    // ── Filtered models ──
    var filteredModels: [ModelEntry] {
        guard let data = browserData else { return [] }
        var models: [ModelEntry] = []
        if case .browse(let sectionID) = selectedSection {
            let domainIds = domainIDsForSection(sectionID)
            for domain in data.domains where domainIds.contains(domain.id) {
                models.append(contentsOf: domain.allModels)
            }
            if Self.hideQwen359B {
                models.removeAll { $0.id == "mlx-community--Qwen3.5-9B-MLX-4bit" }
            }
        }
        if let source = filterSource {
            models = models.filter { $0.source == source }
        }
        if !filterCapabilities.isEmpty {
            models = models.filter { model in
                guard let tags = model.taskTags else { return false }
                return !filterCapabilities.isDisjoint(with: Set(tags))
            }
        }
        models.sort(by: sortOrder.comparator)
        return models
    }

    // ── Grouped models ──
    var groupedModels: [(domain: DomainNode, models: [ModelEntry])] {
        guard let data = browserData else { return [] }
        var result: [(DomainNode, [ModelEntry])] = []
        if case .browse(let sectionID) = selectedSection {
            let domainIds = domainIDsForSection(sectionID)
            let modelSet = Set(filteredModels.map { $0.id })
            for domain in data.domains where domainIds.contains(domain.id) {
                for leaf in domain.leafDomains {
                    let leafModels = (leaf.models?.filter { modelSet.contains($0.id) } ?? [])
                        .sorted(by: sortOrder.comparator)
                    if !leafModels.isEmpty {
                        result.append((leaf, leafModels))
                    }
                }
            }
        }
        return result
    }

    // ── Available capabilities for current section ──
    var availableCapabilities: [String] {
        var tags = Set<String>()
        if case .browse = selectedSection {
            for m in filteredModels {
                if let t = m.taskTags { tags.formUnion(t) }
            }
        }
        return Array(tags).sorted()
    }

    // ── Model count label ──
    var modelCountLabel: String {
        let count = filteredModels.count
        let dimmed = filteredModels.filter { $0.ramGB > filterRAMLimitGB }.count
        if dimmed > 0 {
            return "\(count) models · \(dimmed) exceed RAM"
        }
        return "\(count) models"
    }

    // ── Load ──
    func loadBrowserData() {
        // Load the curated MVP catalog (browser.json).
        guard let url = Bundle.main.url(forResource: "browser", withExtension: "json") else {
            loadError = "browser.json not found in bundle"
            return
        }
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            browserData = try decoder.decode(BrowserData.self, from: data)
            recomputeClaimableModelIDs()
            // MoC-5-FIX-1: `loadInstalledModels()` first ran in `init()` before the catalog
            // existed, so any card whose on-disk directory is repo-keyed away from its own id
            // (`Qwen3.5 9B Vision`, sharing the `Qwen3.5 9B` download) was checked at the
            // wrong path and dropped. Re-run now that the catalog can map id → repo; a
            // second pass is a no-op for every card that was already resolved (MoC-5-3).
            loadInstalledModels()
            loadFilters()
        } catch let DecodingError.keyNotFound(key, context) {
            let path = context.codingPath.map { $0.stringValue }.joined(separator: " → ")
            loadError = "Missing key '\(key.stringValue)' at \(path)"
            print("DECODE ERROR: \(loadError!)")
            print("  Context: \(context.debugDescription)")
        } catch let DecodingError.typeMismatch(type, context) {
            let path = context.codingPath.map { $0.stringValue }.joined(separator: " → ")
            loadError = "Type mismatch: expected \(type) at \(path)"
            print("DECODE ERROR: \(loadError!)")
            print("  Context: \(context.debugDescription)")
        } catch let DecodingError.valueNotFound(type, context) {
            let path = context.codingPath.map { $0.stringValue }.joined(separator: " → ")
            loadError = "Null value for non-optional \(type) at \(path)"
            print("DECODE ERROR: \(loadError!)")
            print("  Context: \(context.debugDescription)")
        } catch let DecodingError.dataCorrupted(context) {
            let path = context.codingPath.map { $0.stringValue }.joined(separator: " → ")
            loadError = "Corrupted data at \(path): \(context.debugDescription)"
            print("DECODE ERROR: \(loadError!)")
        } catch {
            loadError = "Failed to parse: \(error.localizedDescription)"
            print("DECODE ERROR: \(error)")
        }
    }

    /// CFM-R14-2 — recompute which catalog models the registry can claim. Called after
    /// `browserData` decodes (the registry is populated at init). The claim table is what the
    /// nonisolated model-pool derivation reads instead of the `@MainActor` registry.
    func recomputeClaimableModelIDs() {
        let catalog = browserData?.domains.flatMap { $0.allModels } ?? []
        claimableModelIDs = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    // ── Installed models persistence ──
    func loadInstalledModels() {
        print("[AppState] Loading installed from \(installedURL.path)")
        guard FileManager.default.fileExists(atPath: installedURL.path),
              let data = try? Data(contentsOf: installedURL) else {
            print("[AppState] No installed.json found")
            return
        }
        installedModels = try? JSONDecoder().decode(InstalledModels.self, from: data)
        if let models = installedModels?.models {
            print("[AppState] Loaded \(models.count) models from registry: \(models.keys)")
            // MoC-5-FIX-1: at launch this runs from `init()` before `browserData` is decoded,
            // so `catalog` is empty and every id resolves card-keyed (correct for the 38
            // single-card entries). `loadBrowserData()` re-runs this once the catalog exists,
            // which is when a shared-repo card (`Qwen3.5 9B Vision`) gets resolved by repo.
            let catalog = browserData?.domains.flatMap { $0.allModels } ?? []
            installedModelIDs = installManager.loadInstalled(modelIDs: Set(models.keys), catalog: catalog)
            print("[AppState] Verified \(installedModelIDs.count) still on disk")
        }
    }

    func saveInstalledModels() {
        installManager.saveRegistry(installedIDs: installedModelIDs, browserData: browserData)
    }

    func installModel(_ model: ModelEntry) {
        installManager.install(model) { [weak self] modelId in
            self?.installedModelIDs.insert(modelId)
            self?.saveInstalledModels()
        }
    }

    func cancelInstall(_ model: ModelEntry) {
        installManager.cancel(model)
    }

    /// MoC-5-2: reference-counted — `installManager.uninstall` derives whether another
    /// installed card shares this model's HF repo from the live catalog + the current
    /// installed-ids set, both already at hand here.
    func uninstallModel(_ model: ModelEntry) {
        let catalog = browserData?.domains.flatMap { $0.allModels } ?? []
        installManager.uninstall(model, catalog: catalog, installedModelIDs: installedModelIDs)
        installedModelIDs.remove(model.id)
        saveInstalledModels()
    }

    func runModel(_ model: ModelEntry) {
        // LLMs open the chat sheet. Everything else routes through the registry: a
        // registered module (e.g. Whisper ASR) opens its run sheet; otherwise fall back
        // to the legacy ModelRunner, which shows the "unsupported" alert.
        if model.modelType == .llm {
            modelRunner.prepare(for: model)
            runningModel = model
        } else if registry.bestModule(for: model) != nil {
            asrRunModel = model
        } else {
            modelRunner.run(model)
        }
    }

    func sendPrompt(_ text: String) {
        guard let model = runningModel else { return }
        modelRunner.send(text, for: model)
    }

    func stopModel() {
        modelRunner.stop()
    }

    // ── Comparison ──
    func toggleComparison(_ model: ModelEntry) {
        if let idx = comparisonSet.firstIndex(where: { $0.id == model.id }) {
            comparisonSet.remove(at: idx)
        } else if comparisonSet.count < 5 {
            comparisonSet.append(model)
        }
    }

    func isInComparison(_ model: ModelEntry) -> Bool {
        comparisonSet.contains(where: { $0.id == model.id })
    }

    func resetFilters() {
        filterSource = nil
        filterRAMLimitGB = systemInfo.totalRAMGB
        filterCapabilities = []
        saveFilters()
    }

    func saveFilters() {
        let defaults = UserDefaults.standard
        defaults.set(filterSource?.rawValue, forKey: "filterSource")
        defaults.set(filterRAMLimitGB, forKey: "filterRAMLimitGB")
        defaults.set(Array(filterCapabilities), forKey: "filterCapabilities")
        defaults.set(sortOrder.rawValue, forKey: "sortOrder")
    }

    func loadFilters() {
        let defaults = UserDefaults.standard
        if let raw = defaults.string(forKey: "filterSource") {
            filterSource = ModelSource(rawValue: raw)
        }
        filterRAMLimitGB = defaults.double(forKey: "filterRAMLimitGB")
        if filterRAMLimitGB < 2 { filterRAMLimitGB = systemInfo.totalRAMGB }
        if let caps = defaults.array(forKey: "filterCapabilities") as? [String] {
            filterCapabilities = Set(caps)
        }
        if let raw = defaults.string(forKey: "sortOrder"),
           let order = SortOrder(rawValue: raw) {
            sortOrder = order
        }
    }

    // ── R5-5: Open .cat… ──────────────────────────────────────────────────────────

    /// Parse + validate a `.cat` file from disk and stage it as `openedCatFlow`.
    /// `bookmarkData` (if any) is kept so a later launch can re-grant access; the file's
    /// contents are captured here, so the security-scoped access can be released after.
    /// Throws `OpenCatFlowError` on any failure — the caller surfaces `description`.
    func openCatFlow(at url: URL, bookmarkData: Data? = nil) throws {
        let rawText: String
        do {
            rawText = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw OpenCatFlowError.unreadable(path: url.path, reason: error.localizedDescription)
        }
        let parsed: ParsedFlow
        do {
            parsed = try CatParser.parseForValidation(rawText)
        } catch {
            throw OpenCatFlowError.parseFailed(path: url.path, message: String(describing: error))
        }
        openedCatFlow = OpenedCatFlow(
            url: url,
            displayName: url.deletingPathExtension().lastPathComponent,
            rawText: rawText,
            parsed: parsed,
            issues: FlowValidator.checkFlow(parsed),
            bookmarkData: bookmarkData
        )
    }

    /// Present the read-only Open panel, then parse + validate the picked file.
    /// A security-scoped bookmark is created so the choice survives a relaunch.
    func presentOpenCatPanel() {
        let panel = NSOpenPanel()
        panel.title = "Open a CAT Flow"
        panel.prompt = "Open"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.plainText, .data]
        panel.allowedFileTypes = ["cat", "catpipeline", "txt"]
        panel.begin { [weak self] response in
            guard response == .OK, let self, let url = panel.url else { return }
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            var bookmark: Data?
            do {
                bookmark = try url.bookmarkData(
                    options: .withSecurityScope,
                    includingResourceValuesForKeys: nil,
                    relativeTo: nil
                )
            } catch {
                bookmark = nil  // non-fatal: the file is still readable this session
            }
            do {
                try self.openCatFlow(at: url, bookmarkData: bookmark)
            } catch {
                self.openCatFlowError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
            }
        }
    }
}

// ── Supporting Types ──

enum SidebarItem: Hashable {
    case home
    case browse(String)
    case overview
    case aiWorkflows
    var isHome: Bool { if case .home = self { return true }; return false }
}

/// A gallery flow selected from the "AI Workflows" badge grid, pushed onto the detail
/// stack so the sidebar selection ("AI Workflows") stays highlighted and a back button
/// is available.
struct FlowSelection: Hashable, Identifiable {
    let flowID: String
    /// CFM-R12-1: whether this is a user flow (the shelf) or a bundled gallery flow.
    let isUserFlow: Bool
    /// CFM-R17-3: set when this flow lives inside a workspace — `FlowListView` then resolves
    /// its `.cat` paths against the shared workspace directory, not `flows/<flowID>/`.
    let workspace: WorkspaceRef?
    /// CFM-R17-4: `FlowListView` runs the flow once as soon as it loads — the workspace
    /// index card's Build / Ask verbs open the flow and start it.
    let autoRun: Bool

    init(flowID: String, isUserFlow: Bool = false, workspace: WorkspaceRef? = nil,
         autoRun: Bool = false) {
        self.flowID = flowID
        self.isUserFlow = isUserFlow
        self.workspace = workspace
        self.autoRun = autoRun
    }

    var id: String {
        let run = autoRun ? "!" : ""
        if let workspace { return "w\(run)-\(workspace.workspaceID)/\(workspace.flowFile)" }
        return "\(isUserFlow ? "u" : "g")\(run)-\(flowID)"
    }
}

enum SortOrder: String, CaseIterable, Identifiable {
    case ramLowToHigh, ramHighToLow, name
    var id: String { rawValue }
    var label: String {
        switch self {
        case .ramLowToHigh: "RAM: Low→High"
        case .ramHighToLow: "RAM: High→Low"
        case .name: "Name"
        }
    }
    func comparator(_ a: ModelEntry, _ b: ModelEntry) -> Bool {
        switch self {
        case .ramLowToHigh: a.ramGB < b.ramGB
        case .ramHighToLow: a.ramGB > b.ramGB
        case .name: a.displayName.localizedStandardCompare(b.displayName) == .orderedAscending
        }
    }
}

struct InstalledModels: Codable {
    var version: Int
    var models: [String: InstalledModel]
}

struct InstalledModel: Codable {
    let installedAt: String
    let variant: String
    let path: String
    let sizeBytes: Int
}
