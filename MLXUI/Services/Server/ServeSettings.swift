import Foundation

/// The Local Server's persisted settings (S1-2), stored the way `AgentToolSettings` is: a
/// `Codable` value blob in `UserDefaults`, value-semantics "settingX" copies, a thin
/// `load`/`save`. S1 carries exactly two fields (backlog S1-2): the port and the served set.
///
/// `servedModelIDs` holds the ids clients see in `/v1/models` — an MLX model's HF repo id, or
/// `apple-foundation`. It survives `hideLocalServer = true` (kept, not deleted), so flipping the
/// flag back restores what the user had served.
nonisolated struct ServeSettings: Codable, Sendable, Equatable {
    /// Owner ruling R10 (2026-09-27): "For OQ-1 use port 1212".
    static let defaultPort = 1212
    /// Unprivileged ports only. Implementer's call, pending owner confirmation — the plan says the
    /// port is "user-changeable" but sets no bounds.
    static let portRange = 1024...65535

    private(set) var port: Int
    private(set) var servedModelIDs: Set<String>

    init(port: Int = ServeSettings.defaultPort, servedModelIDs: Set<String> = []) {
        self.port = Self.clampedPort(port)
        self.servedModelIDs = servedModelIDs
    }

    /// Back-compat decoding: tolerate a missing field so a later build can add one.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            port: try container.decodeIfPresent(Int.self, forKey: .port) ?? Self.defaultPort,
            servedModelIDs: try container.decodeIfPresent(Set<String>.self, forKey: .servedModelIDs) ?? [])
    }

    private enum CodingKeys: String, CodingKey { case port, servedModelIDs }

    static func clampedPort(_ port: Int) -> Int {
        min(max(port, portRange.lowerBound), portRange.upperBound)
    }

    func isServing(_ id: String) -> Bool { servedModelIDs.contains(id) }

    /// A copy with `id` served (or not).
    func serving(_ id: String, _ on: Bool) -> ServeSettings {
        var next = servedModelIDs
        if on { next.insert(id) } else { next.remove(id) }
        return ServeSettings(port: port, servedModelIDs: next)
    }

    /// A copy with the port set (clamped to `portRange`).
    func settingPort(_ port: Int) -> ServeSettings {
        ServeSettings(port: port, servedModelIDs: servedModelIDs)
    }

    // MARK: Persistence

    private static let defaultsKey = "localServerSettings"

    static func load(from defaults: UserDefaults = .standard) -> ServeSettings {
        guard let data = defaults.data(forKey: defaultsKey),
              let decoded = try? JSONDecoder().decode(ServeSettings.self, from: data)
        else { return ServeSettings() }
        return decoded
    }

    func save(to defaults: UserDefaults = .standard) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        defaults.set(data, forKey: Self.defaultsKey)
    }
}
