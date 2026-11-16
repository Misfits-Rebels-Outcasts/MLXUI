//
//  HomeStatusLine.swift
//  MLXUI — Home chat, HC-2.
//
//  The one source of the "Your Mac" facts: `HomeView`'s badges and the Home chat page's footer line both
//  format through here, so the wording is never written twice. Pure.
//

import Foundation

nonisolated struct HomeStatusLine: Equatable, Sendable {
    struct Item: Equatable, Sendable {
        let icon: String
        let label: String
    }

    let chip: Item
    let ram: Item
    let disk: Item
    let macOS: Item

    init(chipName: String, formattedRAM: String, formattedDisk: String, macOSVersion: String) {
        chip = Item(icon: "cpu.fill", label: chipName)
        // HC-FIX-1 ruling K: the memory figure says what it is.
        ram = Item(icon: "memorychip.fill", label: "\(formattedRAM) memory")
        disk = Item(icon: "internaldrive.fill", label: "\(formattedDisk) free")
        macOS = Item(icon: "macbook", label: "macOS \(macOSVersion)")
    }

    /// Every badge `HomeView` shows, in order.
    var badges: [Item] { [chip, ram, disk, macOS] }

    /// The footer's system sentence: chip · RAM · free disk.
    var sentence: String { [chip, ram, disk].map(\.label).joined(separator: " · ") }

    /// The footer: the system sentence, then the Local Server card sentence when there is one.
    static func footer(system: String, serverSentence: String?) -> String {
        guard let serverSentence, !serverSentence.isEmpty else { return system }
        return "\(system) · \(serverSentence)"
    }
}
