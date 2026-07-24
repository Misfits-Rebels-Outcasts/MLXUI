import SwiftUI
import Observation

@Observable
final class AppState {
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
    var installedModels: InstalledModels?
    private let installedURL: URL

    // Search
    var showCommandPalette = false
    var searchQuery = ""

    // Settings sheet
    var showSettings = false

    init() {
        filterRAMLimitGB = SystemInfo.detect().totalRAMGB
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        installedURL = appSupport.appendingPathComponent("AI Browser/installed.json")
        loadInstalledModels()
        for module in installedModules { module.register(into: registry) }
    }

    // ── Visible sidebar sections ──
    var visibleSections: [SidebarSection] {
        guard let data = browserData else { return [] }
        return data.sidebarSections.filter { $0.modelCount(in: data) > 0 }
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
            let domainIds = data.sidebarSections.first(where: { $0.id == sectionID })?.domainIds ?? []
            for domain in data.domains where domainIds.contains(domain.id) {
                models.append(contentsOf: domain.allModels)
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
            let domainIds = data.sidebarSections.first(where: { $0.id == sectionID })?.domainIds ?? []
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
            installedModelIDs = installManager.loadInstalled(modelIDs: Set(models.keys))
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

    func cancelInstall(_ modelId: String) {
        installManager.cancel(modelId)
    }

    func uninstallModel(_ modelId: String) {
        installManager.uninstall(modelId)
        installedModelIDs.remove(modelId)
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
}

// ── Supporting Types ──

enum SidebarItem: Hashable {
    case home
    case browse(String)
    var isHome: Bool { if case .home = self { return true }; return false }
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
