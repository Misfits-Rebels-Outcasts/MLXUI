import Foundation

struct SystemInfo {
    let modelIdentifier: String
    let chipName: String
    let totalRAMGB: Double
    let memoryBandwidthGBps: Double
    let macOSVersion: String
    let availableDiskGB: Double

    static func detect() -> SystemInfo {
        let memSize = ProcessInfo.processInfo.physicalMemory
        let totalRAM = Double(memSize) / 1_000_000_000.0

        let osVersion = ProcessInfo.processInfo.operatingSystemVersion
        let osStr = "\(osVersion.majorVersion).\(osVersion.minorVersion)"

        let modelId = getModelIdentifier()
        let chip = detectChip(modelIdentifier: modelId)
        let bw = chipBandwidth(chip)

        let home = FileManager.default.homeDirectoryForCurrentUser
        let diskFree = (try? home.resourceValues(forKeys: [.volumeAvailableCapacityKey])
            .volumeAvailableCapacity).map { Double($0) / 1_000_000_000.0 } ?? 0

        return SystemInfo(
            modelIdentifier: modelId,
            chipName: chip,
            totalRAMGB: totalRAM,
            memoryBandwidthGBps: bw,
            macOSVersion: osStr,
            availableDiskGB: diskFree
        )
    }

    private static func getModelIdentifier() -> String {
        var size: size_t = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        return String(cString: model)
    }

    private static func detectChip(modelIdentifier: String) -> String {
        let map: [String: String] = [
            "Mac16,7": "M4 Max", "Mac16,8": "M4 Max", "Mac16,9": "M4 Max",
            "Mac16,10": "M4 Pro", "Mac16,11": "M4 Pro",
            "Mac16,1": "M4", "Mac16,2": "M4",
            "Mac15,8": "M3 Max", "Mac15,9": "M3 Max", "Mac15,10": "M3 Max",
            "Mac15,6": "M3 Pro", "Mac15,7": "M3 Pro",
            "Mac15,3": "M3", "Mac15,4": "M3", "Mac15,5": "M3",
            "Mac14,13": "M2 Max", "Mac14,14": "M2 Max",
            "Mac14,9": "M2 Pro", "Mac14,10": "M2 Pro",
            "Mac14,2": "M2", "Mac14,7": "M2",
            "Mac14,5": "M2 Max", "Mac14,6": "M2 Max",
            "Mac14,3": "M2 Pro", "Mac14,4": "M2 Pro",
        ]
        return map[modelIdentifier] ?? "Apple Silicon"
    }

    private static func chipBandwidth(_ chip: String) -> Double {
        let map: [String: Double] = [
            "M4 Max": 546, "M4 Pro": 273, "M4": 120,
            "M3 Max": 400, "M3 Pro": 200, "M3": 100,
            "M2 Ultra": 800, "M2 Max": 400, "M2 Pro": 200, "M2": 100,
            "M1 Ultra": 800, "M1 Max": 400, "M1 Pro": 200, "M1": 68,
        ]
        return map[chip] ?? 100.0
    }

    var formattedRAM: String {
        String(format: "%.0f GB", totalRAMGB)
    }

    var formattedDisk: String {
        if availableDiskGB >= 1000 {
            return String(format: "%.1f TB", availableDiskGB / 1000)
        }
        return String(format: "%.0f GB", availableDiskGB)
    }
}
