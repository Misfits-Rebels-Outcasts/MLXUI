import SwiftUI

/// MC5-T-notice (owner ruling MC5-L, 2026-11-21): the licence notices for ACE-Step, shown on its model page.
/// The text is the BUNDLED files (`Resources/AceStep/NOTICE.txt`, `StabilityAICommunityLicense.txt`), never a copy in Swift.
nonisolated enum AceStepLicenceNotices {
    static let noticeFile = "NOTICE"
    static let licenseFile = "StabilityAICommunityLicense"

    /// Whether the model page for this entry carries the disclosure (the ace-step family only).
    static func applies(toModelID id: String) -> Bool { id.lowercased().hasPrefix("ace-step/") }

    /// The bundled file's text, or nil when missing (the disclosure then says so rather than showing nothing).
    static func load(_ name: String, bundle: Bundle = .main) -> String? {
        guard let url = bundle.url(forResource: name, withExtension: "txt"),
              let data = try? Data(contentsOf: url) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Sections in display order: the NOTICE (which carries the Stability line and the Qwen Apache-2.0 notice), then the full licence.
    static func sections(bundle: Bundle = .main) -> [(title: String, text: String)] {
        [("Notice", load(noticeFile, bundle: bundle) ?? "NOTICE.txt is missing from this build."),
         ("Stability AI Community License", load(licenseFile, bundle: bundle) ?? "StabilityAICommunityLicense.txt is missing from this build.")]
    }

    /// Collapsed until the person opens it.
    static let expandedByDefault = false
}

struct AceStepLicenceNoticesView: View {
    @State private var expanded = AceStepLicenceNotices.expandedByDefault

    var body: some View {
        DisclosureGroup("Licence notices", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 12) {
                ForEach(AceStepLicenceNotices.sections(), id: \.title) { section in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(section.title).font(.subheadline.weight(.semibold))
                        ScrollView {
                            Text(section.text)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 240)
                    }
                }
            }
            .padding(.top, 6)
        }
        .font(.subheadline)
    }
}
